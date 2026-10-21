"""Tests for BatchedInferenceServer, InferenceClient, and SelfPlayActorPool."""

import time
import threading
import numpy as np
import pytest
import torch

from go.board import Board
from nn.features import extract_features
from nn.model import AlphaZeroNet
from selfplay.inference_server import BatchedInferenceServer, InferenceClient
from selfplay.actor_pool import SelfPlayActorPool
from selfplay.worker import play_selfplay_game, generate_selfplay_games


class MockLinearModel:
    """Deterministic mock model for verifying exact batch and single correspondence."""
    def __init__(self, action_size: int = 82):
        self.action_size = action_size

    def predict(self, state_features: np.ndarray, legal_moves_mask: np.ndarray = None, device=None):
        probs = np.ones(self.action_size, dtype=np.float32)
        if legal_moves_mask is not None:
            probs = probs * (legal_moves_mask > 0).astype(np.float32)
        total = np.sum(probs)
        if total > 0:
            probs /= total
        # Value is derived deterministically from the mean of state_features
        v = float(np.tanh(np.mean(state_features)))
        return probs, v

    def predict_batch(self, state_features: np.ndarray, legal_moves_masks: np.ndarray = None, device=None, use_amp: bool = False):
        b = state_features.shape[0]
        probs = np.ones((b, self.action_size), dtype=np.float32)
        if legal_moves_masks is not None:
            probs = probs * (legal_moves_masks > 0).astype(np.float32)
        totals = np.sum(probs, axis=1, keepdims=True)
        totals[totals == 0] = 1.0
        probs /= totals
        values = np.tanh(np.mean(state_features, axis=(1, 2, 3))).astype(np.float32)
        return probs, values


def test_batched_inference_server_single_client():
    """Verify that a single client through InferenceClient gets results identical to direct model.predict()."""
    import queue
    model = MockLinearModel(action_size=82)
    req_queue = queue.Queue()
    resp_queue = queue.Queue()
    resp_queues = {0: resp_queue}

    server = BatchedInferenceServer(
        model=model,
        request_queue=req_queue,
        response_queues=resp_queues,
        max_batch_size=8,
        max_wait_ms=5.0,
    )
    server.start()

    try:
        client = InferenceClient(client_id=0, request_queue=req_queue, response_queue=resp_queue)
        board = Board(size=9)
        features = extract_features(board)
        mask = board.get_legal_moves()

        expected_probs, expected_v = model.predict(features, legal_moves_mask=mask)
        actual_probs, actual_v = client.predict(features, legal_moves_mask=mask)

        np.testing.assert_allclose(actual_probs, expected_probs, atol=1e-6)
        assert np.isclose(actual_v, expected_v, atol=1e-6)
    finally:
        server.stop()


def test_batched_inference_server_multi_client_dynamic_batching():
    """Verify concurrent requests from multiple clients are aggregated into batches."""
    import queue
    model = MockLinearModel(action_size=82)
    req_queue = queue.Queue()
    num_clients = 4
    resp_queues = {i: queue.Queue() for i in range(num_clients)}

    server = BatchedInferenceServer(
        model=model,
        request_queue=req_queue,
        response_queues=resp_queues,
        max_batch_size=8,
        max_wait_ms=10.0,
    )
    server.start()

    errors = []

    def client_worker(cid: int):
        try:
            client = InferenceClient(client_id=cid, request_queue=req_queue, response_queue=resp_queues[cid])
            for step in range(10):
                # Unique feature per client and step
                features = np.full((17, 9, 9), fill_value=float(cid * 10 + step), dtype=np.float32)
                mask = np.ones(82, dtype=np.int8)
                expected_probs, expected_v = model.predict(features, legal_moves_mask=mask)
                actual_probs, actual_v = client.predict(features, legal_moves_mask=mask)

                np.testing.assert_allclose(actual_probs, expected_probs, atol=1e-5)
                assert np.isclose(actual_v, expected_v, atol=1e-5)
        except Exception as e:
            errors.append((cid, e))

    threads = [threading.Thread(target=client_worker, args=(i,)) for i in range(num_clients)]
    for t in threads:
        t.start()
    for t in threads:
        t.join(timeout=10.0)

    server.stop()

    assert len(errors) == 0, f"Client thread errors: {errors}"
    assert server.total_samples == num_clients * 10
    assert server.total_batches > 0
    # Average batch size should be > 1 because 4 concurrent threads hammer the server
    assert server.avg_batch_size >= 1.0


def test_batched_inference_server_with_real_alphazero_net():
    """Verify BatchedInferenceServer works seamlessly with real AlphaZeroNet."""
    import queue
    net = AlphaZeroNet(board_size=9, num_res_blocks=2, num_filters=16)
    net.eval()
    req_queue = queue.Queue()
    resp_queue = queue.Queue()
    resp_queues = {0: resp_queue}

    server = BatchedInferenceServer(
        model=net,
        request_queue=req_queue,
        response_queues=resp_queues,
        max_batch_size=4,
        max_wait_ms=5.0,
    )
    server.start()

    try:
        client = InferenceClient(client_id=0, request_queue=req_queue, response_queue=resp_queue)
        board = Board(size=9)
        features = extract_features(board)
        mask = board.get_legal_moves()

        expected_probs, expected_v = net.predict(features, legal_moves_mask=mask)
        actual_probs, actual_v = client.predict(features, legal_moves_mask=mask)

        np.testing.assert_allclose(actual_probs, expected_probs, atol=1e-5)
        assert np.isclose(actual_v, expected_v, atol=1e-5)
    finally:
        server.stop()


def test_actor_pool_multiprocess_selfplay():
    """Verify SelfPlayActorPool runs games across multiple worker processes without deadlock."""
    net = AlphaZeroNet(board_size=5, num_res_blocks=2, num_filters=16)
    net.eval()

    pool = SelfPlayActorPool(
        model=net,
        num_workers=2,
        max_batch_size=4,
        max_wait_ms=5.0,
    )

    try:
        results = pool.generate_games(
            num_games=2,
            board_size=5,
            num_simulations=5,
            temp_moves=2,
            max_moves=6,
            disable_resign=True,
            seed=42,
        )

        assert len(results) == 2
        for game in results:
            assert game.board_size == 5
            assert len(game.samples) > 0
            for s, pi, z in game.samples:
                assert s.shape == (17, 5, 5)
                assert pi.shape == (26,)
                assert np.isclose(np.sum(pi), 1.0, atol=1e-4)
                assert z in (-1.0, 0.0, 1.0)
    finally:
        pool.close()


def test_generate_selfplay_games_num_workers_parameter():
    """Verify generate_selfplay_games transparently dispatches to actor pool when num_workers > 1."""
    net = AlphaZeroNet(board_size=5, num_res_blocks=2, num_filters=16)
    net.eval()

    samples = generate_selfplay_games(
        model=net,
        num_games=2,
        board_size=5,
        num_simulations=5,
        temp_moves=2,
        max_moves=6,
        disable_resign=True,
        num_workers=2,
    )

    assert len(samples) > 0
    for s, pi, z in samples:
        assert s.shape == (17, 5, 5)
        assert pi.shape == (26,)
        assert np.isclose(np.sum(pi), 1.0, atol=1e-4)
        assert z in (-1.0, 0.0, 1.0)
