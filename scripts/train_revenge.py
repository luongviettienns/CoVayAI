"""
Targeted Fine-Tuning: Train model to learn from human user game (Revenge & Curriculum Learning).
Injects human match experience, branches MCTS scenario self-play, and fine-tunes policy & value heads.
"""

import os
import sys
import copy
import numpy as np
import torch
from tqdm import tqdm

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from go.board import Board, BLACK, WHITE
from go.scoring import compute_area_score
from nn.model import AlphaZeroNet
from nn.features import extract_features, get_symmetries
from selfplay.replay_buffer import ReplayBuffer
from selfplay.worker import play_selfplay_game
from train.trainer import Trainer
from mcts.search import MCTS

LETTERS = {'A':0, 'B':1, 'C':2, 'D':3, 'E':4, 'F':5, 'G':6, 'H':7, 'J':8}

# Exact 24-move game played by the user
USER_MOVES_STR = [
    'E5', 'D8', 'E3', 'D6', 'G6', 'E7', 'D5', 'C5',
    'C4', 'D4', 'C3', 'B4', 'B3', 'E4', 'F4', 'D3',
    'D2', 'D4', 'F7', 'Pass', 'C6', 'Pass', 'E6', 'Pass'
]


def coord_to_action(m: str, size: int = 9) -> int:
    if m == 'Pass':
        return size * size
    c = LETTERS[m[0]]
    r = size - int(m[1])
    return r * size + c


def action_to_coord(action: int, size: int = 9) -> str:
    if action == size * size:
        return "Pass"
    r = action // size
    c = action % size
    rev_letters = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'J']
    return f"{rev_letters[c]}{size - r}"


def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print("=" * 65)
    print("[*] CoVayAI: TARGETED CURRICULUM TRAINING ('REVENGE LEARNING')")
    print(f"[*] Target Device: {device}")
    print("=" * 65)

    ckpt_dir = os.path.join(REPO_ROOT, "checkpoints", "run9_prod")
    best_path = os.path.join(ckpt_dir, "best_model.pt")
    buffer_path = os.path.join(ckpt_dir, "buffer.pkl")

    # 1. Load active best model
    ckpt = torch.load(best_path, map_location=device)
    state = ckpt["model_state_dict"] if isinstance(ckpt, dict) and "model_state_dict" in ckpt else ckpt
    num_filters = state["conv_in.weight"].shape[0]
    num_res_blocks = sum(1 for k in state.keys() if k.startswith("res_blocks.") and k.endswith(".conv1.weight"))
    base_iter = ckpt.get("iteration", 69) if isinstance(ckpt, dict) else 69

    model = AlphaZeroNet(board_size=9, num_res_blocks=num_res_blocks, num_filters=num_filters).to(device)
    model.load_state_dict(state)
    print(f"[+] Loaded base model ({num_res_blocks}b x {num_filters}f) at Iteration {base_iter}")

    # 2. Check initial policy before learning
    board_test = Board(size=9)
    for m in USER_MOVES_STR[:7]:
        board_test.play(coord_to_action(m))
    mcts_before = MCTS(model=model, num_simulations=128, c_puct=1.5, add_noise=False, device=device)
    pi_before = mcts_before.search(board_test)
    top_move_before = np.argmax(pi_before)
    print(f"[*] Before learning: At move 8 (White to move after D5), AI preferred: {action_to_coord(top_move_before)} (prob={pi_before[top_move_before]:.3f})")

    # 3. Load Replay Buffer
    buffer = ReplayBuffer(board_size=9, capacity=50000)
    if os.path.exists(buffer_path):
        buffer.load(buffer_path)
        print(f"[+] Loaded Replay Buffer with {len(buffer)} existing samples.")

    # 4. Step 1: Reconstruct human game & inject with 8 D4 symmetries
    print("\n[*] Step 1/3: Injecting human user match trajectory into training buffer...")
    board = Board(size=9)
    human_samples = []

    # In user's game: User played Black, AI played White. Black won.
    winner = BLACK
    action_size = 9 * 9 + 1

    for step_idx, m_str in enumerate(USER_MOVES_STR):
        act = coord_to_action(m_str)
        curr_player = board.current_player
        features = extract_features(board)

        # Target policy: sharp probability on the winning move
        pi = np.zeros(action_size, dtype=np.float32)
        pi[act] = 1.0

        # Target value: +1 for winner (Black), -1 for loser (White)
        z = 1.0 if curr_player == winner else -1.0

        # Data augmentation: generate all 8 dihedral symmetries (rotations and flips)
        symmetries = get_symmetries(features, pi)
        for sym_feat, sym_pi in symmetries:
            human_samples.append((sym_feat, sym_pi, z))

        board.play(act)

    buffer.push_game(human_samples)
    print(f"    [+] Injected {len(human_samples)} augmented human game samples (24 moves x 8 symmetries).")

    # 5. Step 2: Scenario-based Self-Play (Branching from user opening)
    print("\n[*] Step 2/3: Generating 10 Scenario Self-Play games from user's opening...")
    # Fork point: moves 1 to 6 (E5, D8, E3, D6, G6, E7)
    fork_moves = [coord_to_action(m) for m in USER_MOVES_STR[:6]]
    scenario_samples = []

    for game_idx in range(10):
        res = play_selfplay_game(
            model=model,
            board_size=9,
            num_simulations=96,
            c_puct=1.5,
            dirichlet_alpha=0.25,
            dirichlet_eps=0.25,
            temp_moves=8,
            resign_threshold=-0.9,
            device=device,
            initial_moves=fork_moves,
        )
        # Augment each selfplay game with random D4 symmetry
        for s, p, z in res.samples:
            scenario_samples.append((s, p, z))

    buffer.push_game(scenario_samples)
    print(f"    [+] Generated and added {len(scenario_samples)} scenario self-play samples. Total buffer: {len(buffer)}")

    # 6. Step 3: Targeted Gradient Descent Fine-Tuning
    print("\n[*] Step 3/3: Running targeted fine-tuning (80 steps, lr=0.0005)...")
    trainer = Trainer(
        model=model,
        lr=0.0005,
        weight_decay=1e-4,
        device=device,
    )

    losses = []
    p_losses = []
    v_losses = []

    for _ in tqdm(range(80), desc="Fine-tuning"):
        states, pis, zs = buffer.sample_batch(batch_size=64, augment=True)
        m = trainer.train_step(states, pis, zs)
        losses.append(m["total_loss"])
        p_losses.append(m["policy_loss"])
        v_losses.append(m["value_loss"])

    print(f"    [+] Fine-tuning complete! Avg Loss: {np.mean(losses):.4f} (Policy: {np.mean(p_losses):.4f}, Value: {np.mean(v_losses):.4f})")

    # 7. Check updated policy after learning
    mcts_after = MCTS(model=model, num_simulations=128, c_puct=1.5, add_noise=False, device=device)
    pi_after = mcts_after.search(board_test)
    top_move_after = np.argmax(pi_after)
    print(f"\n[*] After learning: At move 8, AI now chooses: {action_to_coord(top_move_after)} (prob={pi_after[top_move_after]:.3f})")

    # 8. Save updated model checkpoint
    new_iter_tag = f"{base_iter}-Revenge"
    ckpt_save = {
        "iteration": new_iter_tag,
        "board_size": 9,
        "num_filters": num_filters,
        "num_res_blocks": num_res_blocks,
        "model_state_dict": model.state_dict(),
    }
    revenge_ckpt_path = os.path.join(ckpt_dir, "checkpoint_revenge.pt")
    torch.save(ckpt_save, revenge_ckpt_path)
    torch.save(ckpt_save, best_path)
    buffer.save(buffer_path)
    print(f"[+] Saved updated checkpoint to {revenge_ckpt_path} and {best_path}")
    print("=" * 65)
    print("[+] REVENGE CURRICULUM TRAINING FINISHED SUCCESSFULLY!")
    print("=" * 65)


if __name__ == "__main__":
    main()
