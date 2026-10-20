"""Unit tests for GTP (Go Text Protocol v2) server."""

import pytest
import torch
from nn.model import AlphaZeroNet
from gtp.gtp_server import GTPEngine, vertex_to_action, action_to_vertex


def test_vertex_action_conversion():
    """Verify coordinate conversion between Go vertex format and grid actions."""
    board_size = 9
    # Corners
    # Top-left: row 0, col 0 -> A9
    assert action_to_vertex(0, board_size) == "A9"
    assert vertex_to_action("A9", board_size) == 0
    assert vertex_to_action("a9", board_size) == 0

    # Bottom-left: row 8, col 0 -> A1
    bottom_left_action = 8 * 9 + 0
    assert action_to_vertex(bottom_left_action, board_size) == "A1"
    assert vertex_to_action("A1", board_size) == bottom_left_action

    # Center tengen: row 4, col 4 -> E5
    tengen_action = 4 * 9 + 4
    assert action_to_vertex(tengen_action, board_size) == "E5"
    assert vertex_to_action("E5", board_size) == tengen_action

    # Letter 'I' is skipped: column 8 (9th letter) is 'J'
    top_right_action = 8  # row 0, col 8
    assert action_to_vertex(top_right_action, board_size) == "J9"
    assert vertex_to_action("J9", board_size) == top_right_action

    # Pass action
    pass_action = 81
    assert action_to_vertex(pass_action, board_size) == "pass"
    assert vertex_to_action("pass", board_size) == pass_action
    assert vertex_to_action("PASS", board_size) == pass_action


@pytest.fixture
def engine():
    """Create a lightweight GTPEngine instance for testing."""
    torch.manual_seed(42)
    model = AlphaZeroNet(board_size=9, num_res_blocks=1, num_filters=16)
    model.eval()
    return GTPEngine(model=model, board_size=9, num_sims=5)


def test_basic_gtp_commands(engine):
    """Test standard identification and inquiry commands."""
    assert engine.handle_line("protocol_version") == "= 2\n\n"
    assert engine.handle_line("name") == "= CoVayAI\n\n"
    assert engine.handle_line("version") == "= 1.0\n\n"

    # Known commands
    assert engine.handle_line("known_command play") == "= true\n\n"
    assert engine.handle_line("known_command foobar") == "= false\n\n"

    # Command IDs
    assert engine.handle_line("42 name") == "=42 CoVayAI\n\n"
    assert engine.handle_line("100 protocol_version") == "=100 2\n\n"


def test_board_configuration_commands(engine):
    """Test komi, clear_board, and boardsize commands."""
    assert engine.handle_line("komi 6.5") == "= \n\n"
    assert engine.komi == 6.5

    assert engine.handle_line("boardsize 9") == "= \n\n"
    # Unacceptable size when model is 9x9
    res = engine.handle_line("boardsize 19")
    assert res.startswith("?")


def test_play_and_genmove(engine):
    """Test playing moves and generating engine moves."""
    engine.handle_line("clear_board")

    # Black plays D4
    res = engine.handle_line("play black D4")
    assert res == "= \n\n"
    assert engine.board.move_count == 1

    # White plays E5
    res = engine.handle_line("play white E5")
    assert res == "= \n\n"
    assert engine.board.move_count == 2

    # Black generates move
    res = engine.handle_line("genmove black")
    assert res.startswith("= ")
    move_str = res.split()[1]
    assert move_str == "pass" or (len(move_str) >= 2 and move_str[0].isalpha())
    assert engine.board.move_count == 3


def test_illegal_move_handling(engine):
    """Test proper GTP error response for illegal moves."""
    engine.handle_line("clear_board")
    engine.handle_line("play B E5")

    # Playing on top of existing stone should fail
    res = engine.handle_line("play W E5")
    assert res.startswith("?")
    assert "illegal" in res.lower()


def test_final_score(engine):
    """Test final_score calculation."""
    engine.handle_line("clear_board")
    res = engine.handle_line("final_score")
    assert res.startswith("= ")
    # Tromp-Taylor score should start with B+ or W+
    score_str = res.split()[1]
    assert score_str.startswith("B+") or score_str.startswith("W+")


def test_undo(engine):
    """Test undo command replaying history from scratch."""
    from go.zobrist import BLACK, WHITE
    engine.handle_line("clear_board")
    engine.handle_line("play B D4")
    assert engine.board.move_count == 1
    engine.handle_line("play W E5")
    assert engine.board.move_count == 2

    # Undo move 2 (White E5)
    res = engine.handle_line("undo")
    assert res == "= \n\n"
    assert engine.board.move_count == 1
    assert engine.board.current_player == WHITE
    assert len(engine.move_history) == 1

    # Undo move 1 (Black D4)
    res = engine.handle_line("undo")
    assert res == "= \n\n"
    assert engine.board.move_count == 0
    assert engine.board.current_player == BLACK
    assert len(engine.move_history) == 0

    # Undo on empty board should return error
    res = engine.handle_line("undo")
    assert res.startswith("?")
