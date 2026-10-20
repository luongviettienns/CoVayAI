"""Go Text Protocol (GTP v2) Server for CoVayAI.

Enables seamless integration with Go GUIs and analysis tools
such as Sabaki, Lizzie, Katrain, and tournament managers.
"""

import sys
import re
from typing import Optional, List, Tuple
import numpy as np
import torch

from go.board import Board
from go.zobrist import BLACK, WHITE, EMPTY, opponent
from go.scoring import compute_area_score
from mcts.search import MCTS, select_action


COLUMNS = "ABCDEFGHJKLMNOPQRST"  # Note: 'I' is intentionally omitted in Go notation


def action_to_vertex(action: int, board_size: int = 9) -> str:
    """Convert grid action index to standard Go vertex notation (e.g. 0 -> 'A9', 81 -> 'pass')."""
    if action == board_size * board_size:
        return "pass"

    row = action // board_size
    col = action % board_size

    col_letter = COLUMNS[col]
    row_num = board_size - row
    return f"{col_letter}{row_num}"


def vertex_to_action(vertex: str, board_size: int = 9) -> int:
    """Convert standard Go vertex notation to grid action index (e.g. 'A9' -> 0, 'pass' -> 81)."""
    cleaned = vertex.strip().lower()
    if cleaned in ("pass", "p"):
        return board_size * board_size
    if cleaned in ("resign", "r"):
        return -1

    if len(cleaned) < 2:
        raise ValueError(f"Invalid vertex format: {vertex}")

    col_char = cleaned[0].upper()
    if col_char not in COLUMNS[:board_size]:
        raise ValueError(f"Invalid column: {col_char}")

    col = COLUMNS.index(col_char)
    try:
        row_num = int(cleaned[1:])
    except ValueError:
        raise ValueError(f"Invalid row number: {cleaned[1:]}")

    if not (1 <= row_num <= board_size):
        raise ValueError(f"Row {row_num} out of bounds for size {board_size}")

    row = board_size - row_num
    return row * board_size + col


def parse_color(color_str: str) -> int:
    """Parse color string into BLACK (1) or WHITE (2)."""
    c = color_str.strip().lower()
    if c in ("b", "black", "1"):
        return BLACK
    if c in ("w", "white", "2"):
        return WHITE
    raise ValueError(f"Unrecognized color: {color_str}")


class GTPError(Exception):
    """Raised for invalid or illegal GTP commands."""
    pass


class GTPEngine:
    """GTP v2 protocol engine for CoVayAI AlphaZero."""

    def __init__(
        self,
        model,
        board_size: int = 9,
        komi: float = 7.5,
        num_sims: int = 200,
        c_puct: float = 1.5,
        device: Optional[torch.device] = None,
    ):
        self.model = model
        self.board_size = board_size
        self.komi = komi
        self.num_sims = num_sims
        self.c_puct = c_puct
        self.device = device
        self.running = True

        self.board = Board(size=self.board_size)
        self.mcts = MCTS(
            model=self.model,
            num_simulations=self.num_sims,
            c_puct=self.c_puct,
            add_noise=False,
            komi=self.komi,
            device=self.device,
        )
        self.move_history: List[Tuple[int, int]] = []

        self.commands = {
            "protocol_version": self.cmd_protocol_version,
            "name": self.cmd_name,
            "version": self.cmd_version,
            "known_command": self.cmd_known_command,
            "list_commands": self.cmd_list_commands,
            "boardsize": self.cmd_boardsize,
            "clear_board": self.cmd_clear_board,
            "komi": self.cmd_komi,
            "play": self.cmd_play,
            "genmove": self.cmd_genmove,
            "undo": self.cmd_undo,
            "final_score": self.cmd_final_score,
            "showboard": self.cmd_showboard,
            "quit": self.cmd_quit,
        }

    # --- Command implementations ---

    def cmd_protocol_version(self, args: List[str]) -> str:
        return "2"

    def cmd_name(self, args: List[str]) -> str:
        return "CoVayAI"

    def cmd_version(self, args: List[str]) -> str:
        return "1.0"

    def cmd_known_command(self, args: List[str]) -> str:
        if not args:
            return "false"
        cmd = args[0].lower()
        return "true" if cmd in self.commands else "false"

    def cmd_list_commands(self, args: List[str]) -> str:
        return "\n".join(sorted(self.commands.keys()))

    def cmd_boardsize(self, args: List[str]) -> str:
        if not args:
            raise GTPError("syntax error: boardsize <size>")
        try:
            size = int(args[0])
        except ValueError:
            raise GTPError("syntax error: boardsize must be integer")

        if size != self.board_size:
            raise GTPError(f"unacceptable size {size} (configured for {self.board_size})")

        self.board = Board(size=self.board_size)
        self.mcts.reset()
        self.move_history.clear()
        return ""

    def cmd_clear_board(self, args: List[str]) -> str:
        self.board = Board(size=self.board_size)
        self.mcts.reset()
        self.move_history.clear()
        return ""

    def cmd_komi(self, args: List[str]) -> str:
        if not args:
            raise GTPError("syntax error: komi <float>")
        try:
            self.komi = float(args[0])
            self.mcts.komi = self.komi
            return ""
        except ValueError:
            raise GTPError("syntax error: invalid komi")

    def cmd_play(self, args: List[str]) -> str:
        if len(args) < 2:
            raise GTPError("syntax error: play <color> <vertex>")

        try:
            color = parse_color(args[0])
        except ValueError as e:
            raise GTPError(str(e))

        try:
            action = vertex_to_action(args[1], self.board_size)
        except ValueError as e:
            raise GTPError(f"illegal move: {e}")

        if action == -1:
            # Resign
            return ""

        # If move order is non-alternating, reset MCTS tree to prevent perspective corruption
        if self.board.current_player != color:
            self.board.current_player = color
            self.mcts.reset()

        legal_moves = self.board.get_legal_moves()
        if legal_moves[action] == 0:
            raise GTPError("illegal move")

        self.board.play(action)
        self.mcts.advance_root(action)
        self.move_history.append((color, action))
        return ""

    def cmd_genmove(self, args: List[str]) -> str:
        if not args:
            raise GTPError("syntax error: genmove <color>")

        try:
            color = parse_color(args[0])
        except ValueError as e:
            raise GTPError(str(e))

        if self.board.current_player != color:
            self.board.current_player = color
            self.mcts.reset()

        if self.board.is_game_over():
            return "pass"

        pi = self.mcts.search(self.board)
        action = select_action(pi, temperature=0.0)

        self.board.play(action)
        self.mcts.advance_root(action)
        self.move_history.append((color, action))

        return action_to_vertex(action, self.board_size)

    def cmd_undo(self, args: List[str]) -> str:
        """Undo the last move by replaying history from scratch on a clean board."""
        if not self.move_history:
            raise GTPError("cannot undo: no moves played")

        self.move_history.pop()
        self.board = Board(size=self.board_size)
        for color, action in self.move_history:
            self.board.current_player = color
            self.board.play(action)

        self.mcts.reset()
        return ""

    def cmd_final_score(self, args: List[str]) -> str:
        b_score, w_score, winner = compute_area_score(self.board, komi=self.komi)
        if winner == BLACK:
            diff = b_score - w_score
            return f"B+{diff:.1f}"
        else:
            diff = w_score - b_score
            return f"W+{diff:.1f}"

    def cmd_showboard(self, args: List[str]) -> str:
        symbols = {EMPTY: ".", BLACK: "X", WHITE: "O"}
        lines = ["\n   " + " ".join(COLUMNS[:self.board_size])]
        for r in range(self.board_size):
            row_num = self.board_size - r
            row_str = " ".join(symbols[self.board.grid[r, c]] for c in range(self.board_size))
            lines.append(f"{row_num:2d} {row_str}")
        return "\n".join(lines)

    def cmd_quit(self, args: List[str]) -> str:
        self.running = False
        return ""

    # --- Line handling & protocol formatting ---

    def handle_line(self, line: str) -> str:
        """Process one GTP command line and return formatted response string."""
        # Strip comments starting with '#'
        line = re.sub(r"#.*", "", line).strip()
        if not line:
            return ""

        tokens = line.split()
        cmd_id = ""

        # Check for optional integer command ID
        if tokens[0].isdigit():
            cmd_id = tokens[0]
            tokens = tokens[1:]

        if not tokens:
            return ""

        command = tokens[0].lower()
        args = tokens[1:]

        if command in self.commands:
            try:
                res = self.commands[command](args)
                prefix = f"={cmd_id}"
                return f"{prefix} {res}\n\n" if res else f"{prefix} \n\n"
            except GTPError as e:
                return f"?{cmd_id} {str(e)}\n\n"
            except Exception as e:
                return f"?{cmd_id} {str(e)}\n\n"
        else:
            return f"?{cmd_id} unknown command\n\n"

    def run(self) -> None:
        """Run interactive GTP loop over standard input/output."""
        while self.running:
            try:
                line = sys.stdin.readline()
                if not line:
                    break
                response = self.handle_line(line)
                if response:
                    sys.stdout.write(response)
                    sys.stdout.flush()
            except (KeyboardInterrupt, SystemExit):
                break


def main() -> None:
    """CLI entrypoint for running the GTP server."""
    import argparse
    from nn.model import AlphaZeroNet

    parser = argparse.ArgumentParser(description="CoVayAI GTP v2 Server")
    parser.add_argument("--size", type=int, default=9, help="Board size (default: 9)")
    parser.add_argument("--sims", type=int, default=200, help="MCTS simulations per move (default: 200)")
    parser.add_argument("--komi", type=float, default=7.5, help="Komi (default: 7.5)")
    parser.add_argument("--checkpoint", type=str, default=None, help="Path to model checkpoint .pt")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu", help="Device")
    args = parser.parse_args()

    device = torch.device(args.device)
    model = AlphaZeroNet(board_size=args.size).to(device)

    if args.checkpoint:
        ckpt = torch.load(args.checkpoint, map_location=device)
        state_dict = ckpt["model_state_dict"] if "model_state_dict" in ckpt else ckpt
        model.load_state_dict(state_dict)

    model.eval()

    engine = GTPEngine(
        model=model,
        board_size=args.size,
        komi=args.komi,
        num_sims=args.sims,
        device=device,
    )
    engine.run()


if __name__ == "__main__":
    main()
