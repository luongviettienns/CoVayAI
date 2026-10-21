"""Configuration for 9x9 AlphaZero Go engine."""

# --- Luật cờ ---
BOARD_SIZE: int = 9
KOMI: float = 7.5          # area scoring (Tromp-Taylor), superko (0.5 cho bàn 5x5)
MAX_MOVES: int = 162       # 2 * N^2, quá số này thì ép kết thúc ván

# --- Mạng nơ-ron ---
INPUT_PLANES: int = 17     # 8 nước gần nhất mỗi bên + 1 plane lượt đi
NUM_BLOCKS: int = 6
NUM_FILTERS: int = 96      # ~1M tham số
USE_AMP: bool = True       # fp16 mixed precision

# --- MCTS (self-play) ---
NUM_SIMS: int = 200
C_PUCT: float = 1.5
DIRICHLET_ALPHA: float = 0.30
DIRICHLET_EPS: float = 0.25
TEMP_MOVES: int = 12       # 12 nước đầu: tau = 1, sau đó tau -> 0
RESIGN_THRESHOLD: float = -0.9
RESIGN_DISABLE_FRAC: float = 0.10  # 10% ván không cho resign để kiểm tra false-positive

# --- Playout Cap Randomization ---
PLAYOUT_CAP_RANDOMIZATION: bool = False
FAST_SIMS: int = 60
FULL_SEARCH_PROB: float = 0.25
TRAIN_ONLY_FULL_SEARCH: bool = True

# --- Self-play song song ---
NUM_WORKERS: int = 5       # số luồng/tiến trình sinh ván
INFER_BATCH_SIZE: int = 64 # gom leaf thành batch cho GPU
GAMES_PER_ITER: int = 100

# --- Huấn luyện ---
OPTIMIZER: str = "adamw"
LR: float = 1e-3
WEIGHT_DECAY: float = 1e-2 # 0.01 cho AdamW (chỉ áp dụng trên Conv2D/Linear weights)
LR_MILESTONES = [30, 60]   # Giảm LR x0.1 tại iter 30 và 60
LR_GAMMA: float = 0.1
BATCH_SIZE: int = 256
TRAIN_STEPS: int = 400     # số batch mỗi iteration
BUFFER_SIZE: int = 200_000 # số mẫu (state, pi, z)
AUGMENT: bool = True       # 8 phép đối xứng D4, áp dụng lúc lấy batch
NUM_ITERS: int = 150

# --- Arena ---
ARENA_GAMES: int = 50
ARENA_SIMS: int = 200
OPENING_PLIES: int = 8     # 8 nước đầu lấy mẫu tau=1 để đa dạng hóa thế trận
WIN_THRESHOLD: float = 0.55

# --- Lưu trữ ---
CHECKPOINT_EVERY: int = 1  # lưu mỗi iteration để resume được
