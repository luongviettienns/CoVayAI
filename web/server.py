"""Web server for interactive CoVayAI Go gameplay."""

import os
import sys
import json
import mimetypes
from http.server import HTTPServer, SimpleHTTPRequestHandler
from socketserver import ThreadingMixIn
from typing import Optional, Dict, Any, List
import numpy as np
import torch

# Ensure repository root is on sys.path
REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from go.board import Board
from go.zobrist import BLACK, WHITE, EMPTY, opponent
from go.scoring import compute_area_score, get_winner_value
from nn.model import AlphaZeroNet
from mcts.search import MCTS, select_action


class ThreadedHTTPServer(ThreadingMixIn, HTTPServer):
    """Multi-threaded HTTP server handling concurrent requests."""
    daemon_threads = True


class GameSession:
    """Encapsulates active game state, AI model, and MCTS tree."""

    def __init__(self, board_size: int = 9, komi: float = 7.5, sims: int = 128, device_str: Optional[str] = None):
        self.board_size = board_size
        self.komi = komi
        self.sims = sims
        self.user_color = BLACK
        self.board = Board(size=board_size, max_moves=2 * board_size * board_size)
        self.move_history: List[int] = []
        self.last_ai_value: float = 0.0
        self.last_ai_policy: Optional[np.ndarray] = None

        if device_str is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device_str)

        self.model: Optional[AlphaZeroNet] = None
        self.mcts: Optional[MCTS] = None
        self._load_best_model()

    def _load_best_model(self) -> None:
        """Load the latest/best trained checkpoint for this board size."""
        ckpt_dir = os.path.join(REPO_ROOT, "checkpoints", "run9_prod" if self.board_size == 9 else "smoke5_v1")
        best_path = os.path.join(ckpt_dir, "best_model.pt")

        num_res_blocks = 6 if self.board_size == 9 else 2
        num_filters = 96 if self.board_size == 9 else 32
        loaded_state = None
        self.loaded_iteration = None

        if os.path.exists(best_path):
            try:
                ckpt = torch.load(best_path, map_location=self.device)
                state = ckpt["model_state_dict"] if isinstance(ckpt, dict) and "model_state_dict" in ckpt else ckpt
                if isinstance(ckpt, dict) and "iteration" in ckpt:
                    self.loaded_iteration = str(ckpt["iteration"])
                if "conv_in.weight" in state:
                    num_filters = state["conv_in.weight"].shape[0]
                    num_res_blocks = sum(1 for k in state.keys() if k.startswith("res_blocks.") and k.endswith(".conv1.weight"))
                loaded_state = state
            except Exception as e:
                print(f"[-] Web Server: Warning reading {best_path}: {e}")

        self.model = AlphaZeroNet(
            board_size=self.board_size,
            num_res_blocks=num_res_blocks,
            num_filters=num_filters,
        ).to(self.device)
        self.model.eval()

        if loaded_state is not None:
            try:
                self.model.load_state_dict(loaded_state)
                iter_info = f"iter {self.loaded_iteration}" if self.loaded_iteration is not None else "weights"
                print(f"[+] Web Server: Loaded model ({num_res_blocks}b x {num_filters}f, {iter_info}) from {best_path}")
            except Exception as e:
                print(f"[-] Web Server: Warning loading weights from {best_path}: {e}")

        self.mcts = MCTS(
            model=self.model,
            num_simulations=self.sims,
            c_puct=1.5,
            add_noise=False,
            komi=self.komi,
            device=self.device,
        )

    def reset(self, board_size: int = 9, user_color: int = BLACK, komi: Optional[float] = None, sims: int = 64) -> None:
        """Reset the game with new parameters."""
        self.board_size = board_size
        self.user_color = user_color
        self.komi = komi if komi is not None else (7.5 if board_size >= 9 else 0.5)
        self.sims = sims
        self.board = Board(size=board_size, max_moves=2 * board_size * board_size)
        self.move_history.clear()
        self.last_ai_value = 0.0
        self.last_ai_policy = None
        self._load_best_model()

    def play_ai_turn(self) -> Optional[int]:
        """Compute and execute AI move using MCTS."""
        if self.board.is_game_over():
            return None

        pi = self.mcts.search(self.board)
        self.last_ai_policy = pi
        self.last_ai_value = float(self.mcts.root.value_score)

        pass_action = self.board_size * self.board_size
        # Early-pass avoidance heuristic:
        # If opponent did not pass and board is still in active play, force AI to play on board
        if self.board.consecutive_passes == 0 and len(self.move_history) < 2 * self.board_size:
            pi_candidate = pi.copy()
            pi_candidate[pass_action] = 0.0
            sum_prob = np.sum(pi_candidate)
            if sum_prob > 0:
                pi_candidate /= sum_prob
                action = int(select_action(pi_candidate, temperature=0.0))
            else:
                action = pass_action
        else:
            action = int(select_action(pi, temperature=0.0))

        if self.board.play(action):
            self.move_history.append(action)
            self.mcts.advance_root(action)
            return action
        return None

    def get_state_dict(self) -> Dict[str, Any]:
        """Serialize current game state for JSON response."""
        b_score, w_score, score_winner = compute_area_score(self.board, komi=self.komi)
        is_over = self.board.is_game_over()

        # Format policy heatmap top moves
        top_moves = []
        if self.last_ai_policy is not None:
            indices = np.argsort(self.last_ai_policy)[::-1][:5]
            for idx in indices:
                prob = float(self.last_ai_policy[idx])
                if prob > 0.01:
                    top_moves.append({"action": int(idx), "prob": round(prob, 3)})

        return {
            "board_size": self.board_size,
            "grid": self.board.grid.tolist(),
            "current_player": self.board.current_player,
            "user_color": self.user_color,
            "consecutive_passes": self.board.consecutive_passes,
            "move_count": len(self.move_history),
            "move_history": self.move_history,
            "is_game_over": is_over,
            "komi": self.komi,
            "black_score": b_score,
            "white_score": w_score,
            "winner": score_winner if is_over else None,
            "ai_value": round(self.last_ai_value, 3),
            "ai_winrate": round((self.last_ai_value + 1.0) / 2.0 * 100.0, 1),
            "top_moves": top_moves,
            "legal_moves": np.where(self.board.get_legal_moves() == 1)[0].tolist(),
        }


# Global active game session
SESSION = GameSession()


class GoRequestHandler(SimpleHTTPRequestHandler):
    """HTTP request handler supporting static assets and Go API endpoints."""

    def __init__(self, *args, **kwargs):
        static_dir = os.path.join(os.path.dirname(__file__), "static")
        super().__init__(*args, directory=static_dir, **kwargs)

    def _send_json(self, data: Dict[str, Any], status: int = 200) -> None:
        payload = json.dumps(data).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(payload)))
        self.send_header("Access-Control-Allow-Origin", "*")
        self.end_headers()
        self.wfile.write(payload)

    def do_OPTIONS(self) -> None:
        self.send_response(200)
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Methods", "GET, POST, OPTIONS")
        self.send_header("Access-Control-Allow-Headers", "Content-Type")
        self.end_headers()

    def do_GET(self) -> None:
        if self.path == "/api/state":
            self._send_json(SESSION.get_state_dict())
            return
        if self.path == "/api/info":
            info = {
                "board_size": SESSION.board_size,
                "device": str(SESSION.device),
                "cuda_available": torch.cuda.is_available(),
                "device_name": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU",
                "iteration": SESSION.loaded_iteration,
            }
            self._send_json(info)
            return
        super().do_GET()

    def do_POST(self) -> None:
        content_length = int(self.headers.get("Content-Length", 0))
        body = self.rfile.read(content_length).decode("utf-8")
        data = json.loads(body) if body else {}

        if self.path == "/api/new_game":
            size = int(data.get("board_size", 9))
            color_str = data.get("user_color", "B")
            user_color = BLACK if color_str == "B" else WHITE
            komi = float(data.get("komi", 7.5 if size >= 9 else 0.5))
            sims = int(data.get("sims", 128))

            SESSION.reset(board_size=size, user_color=user_color, komi=komi, sims=sims)

            # If user plays White, AI makes the first move as Black
            if user_color == WHITE:
                SESSION.play_ai_turn()

            self._send_json(SESSION.get_state_dict())
            return

        if self.path == "/api/play":
            action = int(data.get("action", -1))
            pass_move = SESSION.board_size * SESSION.board_size

            # Ensure it is user's turn
            if SESSION.board.current_player != SESSION.user_color:
                self._send_json({"error": "Not your turn"}, status=400)
                return

            # Execute user move
            if not SESSION.board.play(action):
                self._send_json({"error": "Illegal move"}, status=400)
                return

            SESSION.move_history.append(action)
            SESSION.mcts.advance_root(action)

            # If game is not over, AI replies
            if not SESSION.board.is_game_over():
                SESSION.play_ai_turn()

            self._send_json(SESSION.get_state_dict())
            return

        if self.path == "/api/pass":
            pass_move = SESSION.board_size * SESSION.board_size
            if not SESSION.board.play(pass_move):
                self._send_json({"error": "Cannot pass"}, status=400)
                return

            SESSION.move_history.append(pass_move)
            SESSION.mcts.advance_root(pass_move)

            if not SESSION.board.is_game_over():
                SESSION.play_ai_turn()

            self._send_json(SESSION.get_state_dict())
            return

        if self.path == "/api/resign":
            # Current player resigns
            self._send_json({"resigned": True, **SESSION.get_state_dict()})
            return

        self._send_json({"error": "Endpoint not found"}, status=404)


def run_server(port: int = 8080) -> None:
    server_address = ("127.0.0.1", port)
    httpd = ThreadedHTTPServer(server_address, GoRequestHandler)
    print(f"\n=======================================================")
    print(f"[*] CoVayAI Web Arena running at: http://localhost:{port}")
    print(f"[*] Device: {SESSION.device} | Board: {SESSION.board_size}x{SESSION.board_size}")
    print(f"=======================================================\n")
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        print("\nShutting down server...")
        httpd.server_close()


if __name__ == "__main__":
    port = int(sys.argv[1]) if len(sys.argv) > 1 else 8080
    run_server(port)
