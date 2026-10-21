"""High-throughput batched inference server and client proxy for parallel self-play."""

import time
import queue
import threading
from typing import Dict, Any, Optional, Tuple, List
import numpy as np
import torch


class InferenceClient:
    """
    Client proxy presenting the same `.predict()` interface as a neural network model.
    Sends evaluation requests across a queue to BatchedInferenceServer and awaits the response.
    """

    def __init__(
        self,
        client_id: int,
        request_queue,
        response_queue,
        timeout: float = 60.0,
    ):
        self.client_id = client_id
        self.request_queue = request_queue
        self.response_queue = response_queue
        self.timeout = timeout
        self._next_req_id = 0

    def predict(
        self,
        state_features: np.ndarray,
        legal_moves_mask: Optional[np.ndarray] = None,
        device: Optional[Any] = None,
    ) -> Tuple[np.ndarray, float]:
        """
        Evaluate single board state features with legal moves mask via batched server.

        Args:
            state_features: NumPy array of shape (C, N, N)
            legal_moves_mask: Optional NumPy array of shape (N*N + 1,)
            device: Ignored (device is managed by inference server)

        Returns:
            Tuple of (probs: np.ndarray [N*N + 1], value: float in [-1, 1])
        """
        req_id = self._next_req_id
        self._next_req_id += 1

        # Enqueue request: (client_id, req_id, state_features, legal_moves_mask)
        self.request_queue.put((self.client_id, req_id, state_features, legal_moves_mask))

        # Await response from server
        try:
            resp = self.response_queue.get(timeout=self.timeout)
        except queue.Empty:
            raise TimeoutError(
                f"InferenceClient {self.client_id} timed out waiting for server response after {self.timeout}s"
            )

        resp_req_id, probs, value = resp
        if isinstance(value, Exception):
            raise value
        if resp_req_id != req_id:
            raise RuntimeError(
                f"InferenceClient {self.client_id}: Request ID mismatch: expected {req_id}, got {resp_req_id}"
            )

        return probs, value


class BatchedInferenceServer:
    """
    Central inference server aggregating evaluation requests from multiple self-play actors
    into dynamic batches for maximum GPU throughput.
    """

    def __init__(
        self,
        model: Any,
        request_queue: Any,
        response_queues: Dict[int, Any],
        max_batch_size: int = 32,
        max_wait_ms: float = 2.0,
        device: Optional[Any] = None,
        use_amp: bool = False,
    ):
        self.model = model
        self.request_queue = request_queue
        self.response_queues = response_queues
        self.max_batch_size = max(1, max_batch_size)
        self.max_wait_ms = max(0.0, max_wait_ms)
        self.device = device
        self.use_amp = use_amp

        self.total_samples: int = 0
        self.total_batches: int = 0
        self.avg_batch_size: float = 0.0

        self._stop_event = threading.Event()
        self._thread: Optional[threading.Thread] = None

    def start(self) -> None:
        """Start the background inference loop thread."""
        if self._thread is not None and self._thread.is_alive():
            return
        self._stop_event.clear()
        self._thread = threading.Thread(target=self._run_loop, name="BatchedInferenceServer", daemon=True)
        self._thread.start()

    def stop(self, timeout: float = 5.0) -> None:
        """Signal the inference server to stop and join its thread."""
        self._stop_event.set()
        try:
            self.request_queue.put(None)
        except Exception:
            pass

        if self._thread is not None:
            self._thread.join(timeout=timeout)
            self._thread = None

    def register_client(self, client_id: int, response_queue: Any) -> None:
        """Register a new actor client response queue."""
        self.response_queues[client_id] = response_queue

    def unregister_client(self, client_id: int) -> None:
        """Unregister an actor client response queue."""
        self.response_queues.pop(client_id, None)

    def _run_loop(self) -> None:
        """Main batching and inference loop."""
        # Ensure model is in eval mode
        if hasattr(self.model, "eval"):
            self.model.eval()

        max_wait_sec = self.max_wait_ms / 1000.0

        while not self._stop_event.is_set():
            # 1. Fetch first request with a short timeout to permit checking stop_event
            try:
                first_item = self.request_queue.get(timeout=0.05)
            except (queue.Empty, TimeoutError):
                continue
            except Exception:
                if self._stop_event.is_set():
                    break
                continue

            if first_item is None or (isinstance(first_item, tuple) and first_item[0] == "STOP"):
                break

            batch: List[Tuple[int, int, np.ndarray, Optional[np.ndarray]]] = [first_item]
            start_time = time.perf_counter()

            # 2. Dynamic batching: gather subsequent requests until max_batch_size or max_wait_ms
            while len(batch) < self.max_batch_size:
                rem_time = max_wait_sec - (time.perf_counter() - start_time)
                if rem_time <= 0:
                    break

                try:
                    next_item = self.request_queue.get(timeout=max(1e-4, rem_time))
                except (queue.Empty, TimeoutError):
                    break
                except Exception:
                    break

                if next_item is None or (isinstance(next_item, tuple) and next_item[0] == "STOP"):
                    self._stop_event.set()
                    break

                batch.append(next_item)

            if len(batch) == 0:
                continue

            # 3. Execute batched inference
            try:
                states = np.stack([item[2] for item in batch], axis=0)
                has_masks = batch[0][3] is not None
                if has_masks:
                    masks = np.stack([item[3] for item in batch], axis=0)
                else:
                    masks = None

                if hasattr(self.model, "predict_batch"):
                    probs_batch, values_batch = self.model.predict_batch(
                        states,
                        legal_moves_masks=masks,
                        device=self.device,
                        use_amp=self.use_amp,
                    )
                else:
                    # Fallback for models only implementing predict()
                    probs_list, values_list = [], []
                    for idx in range(len(batch)):
                        p, v = self.model.predict(
                            states[idx],
                            legal_moves_mask=masks[idx] if masks is not None else None,
                            device=self.device,
                        )
                        probs_list.append(p)
                        values_list.append(v)
                    probs_batch = np.stack(probs_list, axis=0)
                    values_batch = np.array(values_list, dtype=np.float32)

                # 4. Dispatch individual responses back to respective clients
                for idx, (cid, req_id, _, _) in enumerate(batch):
                    resp_q = self.response_queues.get(cid)
                    if resp_q is not None:
                        val_scalar = float(values_batch[idx].item()) if hasattr(values_batch[idx], "item") else float(values_batch[idx])
                        resp_q.put((req_id, probs_batch[idx], val_scalar))

                self.total_samples += len(batch)
                self.total_batches += 1
                self.avg_batch_size = self.total_samples / self.total_batches

            except Exception as exc:
                # Dispatch error to all waiting clients in this batch to prevent deadlocks
                for cid, req_id, _, _ in batch:
                    resp_q = self.response_queues.get(cid)
                    if resp_q is not None:
                        resp_q.put((req_id, None, exc))
