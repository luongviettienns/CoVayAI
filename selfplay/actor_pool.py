"""Actor pool managing multi-process self-play workers with centralized batched inference."""

import os
import random
import time
import multiprocessing as mp
from typing import List, Dict, Any, Optional, Tuple, Callable
import numpy as np

from selfplay.inference_server import BatchedInferenceServer, InferenceClient
from selfplay.worker import play_selfplay_game, GameResult


def _actor_worker_main(
    worker_id: int,
    task_queue: Any,
    result_queue: Any,
    request_queue: Any,
    response_queue: Any,
    seed: int,
) -> None:
    """
    Worker process entry point. Runs Go self-play games sequentially,
    forwarding NN evaluation requests to the central batched inference server.
    """
    # Seed per-worker RNG independently
    random.seed(seed)
    np.random.seed(seed)

    # Initialize client proxy for batched GPU inference
    client = InferenceClient(
        client_id=worker_id,
        request_queue=request_queue,
        response_queue=response_queue,
    )

    while True:
        try:
            task = task_queue.get()
        except (EOFError, BrokenPipeError, KeyboardInterrupt):
            break

        if task is None:  # Shutdown sentinel
            break

        game_idx, game_kwargs = task
        try:
            result = play_selfplay_game(model=client, **game_kwargs)
            result.seed = seed
            result_queue.put((game_idx, result, None))
        except Exception as exc:
            result_queue.put((game_idx, None, exc))


class SelfPlayActorPool:
    """
    Manages a pool of CPU worker processes running MCTS self-play games,
    communicating with a central BatchedInferenceServer on GPU/CPU.
    """

    def __init__(
        self,
        model: Any,
        num_workers: int = 4,
        max_batch_size: int = 32,
        max_wait_ms: float = 2.0,
        device: Optional[Any] = None,
        use_amp: bool = False,
        base_seed: Optional[int] = None,
    ):
        self.model = model
        self.num_workers = max(1, num_workers)
        self.max_batch_size = max_batch_size
        self.max_wait_ms = max_wait_ms
        self.device = device
        self.use_amp = use_amp
        self.base_seed = base_seed if base_seed is not None else int(time.time() * 1000) % 1_000_000

        # Use 'spawn' context for cross-platform reliability (Windows + CUDA safety)
        self.ctx = mp.get_context("spawn")

        self.request_queue = self.ctx.Queue()
        self.task_queue = self.ctx.Queue()
        self.result_queue = self.ctx.Queue()

        self.response_queues: Dict[int, Any] = {
            i: self.ctx.Queue() for i in range(self.num_workers)
        }

        # Start the centralized inference server
        self.server = BatchedInferenceServer(
            model=self.model,
            request_queue=self.request_queue,
            response_queues=self.response_queues,
            max_batch_size=self.max_batch_size,
            max_wait_ms=self.max_wait_ms,
            device=self.device,
            use_amp=self.use_amp,
        )
        self.server.start()

        # Start worker processes
        self.workers: List[mp.Process] = []
        for i in range(self.num_workers):
            worker_seed = self.base_seed + (i * 10007) + 1
            p = self.ctx.Process(
                target=_actor_worker_main,
                name=f"SelfPlayActor-{i}",
                args=(
                    i,
                    self.task_queue,
                    self.result_queue,
                    self.request_queue,
                    self.response_queues[i],
                    worker_seed,
                ),
            )
            p.daemon = True
            p.start()
            self.workers.append(p)

        self._closed = False

    def generate_games(
        self,
        num_games: int,
        board_size: int = 9,
        num_simulations: int = 200,
        c_puct: float = 1.5,
        dirichlet_alpha: float = 0.25,
        dirichlet_eps: float = 0.25,
        temp_moves: int = 12,
        resign_threshold: float = -0.9,
        resign_disable_frac: float = 0.10,
        disable_resign: Optional[bool] = None,
        max_moves: Optional[int] = None,
        komi: float = 7.5,
        adjudicate_truncated: Optional[bool] = None,
        progress_bar: bool = False,
        game_callback: Optional[Callable[[int, GameResult], None]] = None,
        seed: Optional[int] = None,
    ) -> List[GameResult]:
        """
        Generate games across the actor pool concurrently.
        """
        if self._closed:
            raise RuntimeError("SelfPlayActorPool is closed.")

        try:
            from tqdm import tqdm
        except ImportError:
            tqdm = None

        # Enqueue all tasks
        for game_idx in range(num_games):
            effective_disable_resign = (
                disable_resign if disable_resign is not None
                else (random.random() < resign_disable_frac)
            )
            game_kwargs = {
                "board_size": board_size,
                "num_simulations": num_simulations,
                "c_puct": c_puct,
                "dirichlet_alpha": dirichlet_alpha,
                "dirichlet_eps": dirichlet_eps,
                "temp_moves": temp_moves,
                "resign_threshold": resign_threshold,
                "disable_resign": effective_disable_resign,
                "max_moves": max_moves,
                "komi": komi,
                "adjudicate_truncated": adjudicate_truncated,
            }
            self.task_queue.put((game_idx, game_kwargs))

        pbar = None
        if progress_bar and tqdm is not None:
            pbar = tqdm(total=num_games, desc="Parallel Self-Play", unit="game", dynamic_ncols=True)

        results: Dict[int, GameResult] = {}
        total_samples = 0

        while len(results) < num_games:
            # Check worker health to avoid hanging on crashes
            for p in self.workers:
                if not p.is_alive() and len(results) < num_games:
                    # Give queue a short moment to flush any final result
                    time.sleep(0.1)
                    if self.result_queue.empty():
                        raise RuntimeError(f"SelfPlayActor process {p.pid} terminated unexpectedly (exitcode {p.exitcode}).")

            try:
                game_idx, result, exc = self.result_queue.get(timeout=1.0)
            except (mp.queues.Empty, TimeoutError):
                continue

            if exc is not None:
                raise RuntimeError(f"Error during self-play game {game_idx}: {exc}") from exc

            results[game_idx] = result
            total_samples += len(result.samples)

            winner_str = "B" if result.winner == 1 else ("W" if result.winner == -1 else "Draw")
            if pbar is not None:
                pbar.update(1)
                pbar.set_postfix({
                    "moves": result.moves,
                    "winner": winner_str,
                    "reason": result.end_reason,
                    "samples": total_samples,
                    "avg_batch": f"{self.server.avg_batch_size:.1f}",
                })
            elif progress_bar:
                print(f"   -> [Self-Play {len(results)}/{num_games}] {result.moves} moves ({result.end_reason}) | Winner: {winner_str} | Total samples: {total_samples}")

            if game_callback is not None:
                game_callback(game_idx, result)

        if pbar is not None:
            pbar.close()

        # Return ordered list by game_idx
        return [results[i] for i in range(num_games)]

    def close(self, timeout: float = 5.0) -> None:
        """Gracefully shut down all worker processes and the inference server."""
        if self._closed:
            return
        self._closed = True

        # Send stop sentinel to all workers
        for _ in self.workers:
            try:
                self.task_queue.put(None)
            except Exception:
                pass

        # Join worker processes
        for p in self.workers:
            p.join(timeout=timeout)
            if p.is_alive():
                p.terminate()
                p.join(timeout=1.0)

        # Stop server
        self.server.stop(timeout=timeout)

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()
