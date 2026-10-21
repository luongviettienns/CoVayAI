"""Unit and integration tests for web server and Go API endpoints."""

import json
import threading
import time
from urllib.request import urlopen, Request
from urllib.error import HTTPError
import pytest
import torch

from web.server import GameSession, ThreadedHTTPServer, GoRequestHandler
from go.zobrist import BLACK, WHITE


@pytest.fixture(scope="module")
def web_test_server():
    """Spin up a temporary local test server on an ephemeral port."""
    server = ThreadedHTTPServer(("127.0.0.1", 0), GoRequestHandler)
    port = server.server_port
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    time.sleep(0.1)
    base_url = f"http://127.0.0.1:{port}"
    yield base_url
    server.shutdown()
    server.server_close()


def test_game_session_init_and_state():
    """Verify GameSession initialization and serialization structure."""
    session = GameSession(board_size=5, komi=0.5, sims=4, device_str="cpu")
    state = session.get_state_dict()

    assert state["board_size"] == 5
    assert len(state["grid"]) == 5
    assert state["current_player"] == BLACK
    assert state["user_color"] == BLACK
    assert state["is_game_over"] is False
    assert state["komi"] == 0.5
    assert isinstance(state["legal_moves"], list)
    assert 25 in state["legal_moves"]  # Pass move is legal


def test_game_session_play_and_ai_turn():
    """Verify playing a valid move advances the state and AI responds."""
    session = GameSession(board_size=5, komi=0.5, sims=4, device_str="cpu")
    # Play Tengen on 5x5 (action 12)
    assert session.board.play(12)
    session.move_history.append(12)
    session.mcts.advance_root(12)

    assert session.board.current_player == WHITE
    ai_action = session.play_ai_turn()

    assert ai_action is not None
    assert session.board.current_player == BLACK
    state = session.get_state_dict()
    assert state["move_count"] == 2


def test_http_api_endpoints(web_test_server):
    """Verify all REST API endpoints via HTTP requests."""
    base = web_test_server

    # 1. GET /api/info
    with urlopen(f"{base}/api/info") as res:
        assert res.status == 200
        info = json.loads(res.read().decode())
        assert "board_size" in info
        assert "device" in info

    # 2. POST /api/new_game (5x5, user Black)
    req = Request(
        f"{base}/api/new_game",
        data=json.dumps({"board_size": 5, "user_color": "B", "sims": 4}).encode(),
        headers={"Content-Type": "application/json"},
    )
    with urlopen(req) as res:
        assert res.status == 200
        state = json.loads(res.read().decode())
        assert state["board_size"] == 5
        assert state["user_color"] == BLACK
        assert state["current_player"] == BLACK

    # 3. POST /api/play (center move 12)
    req = Request(
        f"{base}/api/play",
        data=json.dumps({"action": 12}).encode(),
        headers={"Content-Type": "application/json"},
    )
    with urlopen(req) as res:
        assert res.status == 200
        state = json.loads(res.read().decode())
        # AI replied, so move_count should be 2, current_player should be Black again
        assert state["move_count"] == 2
        assert state["current_player"] == BLACK

    # 4. POST /api/pass
    req = Request(
        f"{base}/api/pass",
        data=json.dumps({}).encode(),
        headers={"Content-Type": "application/json"},
    )
    with urlopen(req) as res:
        assert res.status == 200
        state = json.loads(res.read().decode())
        assert state["move_count"] >= 3

    # 5. GET /api/state
    with urlopen(f"{base}/api/state") as res:
        assert res.status == 200
        state = json.loads(res.read().decode())
        assert "grid" in state

    # 6. Static files
    with urlopen(f"{base}/index.html") as res:
        assert res.status == 200
        html = res.read().decode()
        assert "CoVayAI" in html

    with urlopen(f"{base}/style.css") as res:
        assert res.status == 200
        css = res.read().decode()
        assert "goCanvas" in css

    with urlopen(f"{base}/app.js") as res:
        assert res.status == 200
        js = res.read().decode()
        assert "GoWebArena" in js
