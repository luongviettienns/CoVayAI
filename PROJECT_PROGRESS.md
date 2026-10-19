# Kế Hoạch & Tiến Độ Dự Án GoAI (AlphaZero cho Cờ Vây)

## 1. Tổng quan Dự án
- **Tên dự án:** `go_ai` (AlphaZero Go Engine)
- **Mục tiêu:** Xây dựng hệ sinh thái AI chơi Cờ vây hoàn chỉnh theo thuật toán AlphaZero từ tabula rasa (không dùng dữ liệu con người), hỗ trợ bàn 9x9 (và mở rộng 19x19), luật Trung Quốc (Area Scoring, Positional Superko), đấu trường Arena Elo, và chuẩn giao thức GTP.
- **Ngôn ngữ & Công nghệ dự kiến:** Python 3.13, PyTorch, NumPy, Pytest.

---

## 2. Lộ trình Triển khai (Roadmap & Phases)

### Phase 0: Phân tích Yêu cầu & Kiến trúc (Requirements & Architectural Blueprint)
- [x] Đọc và nắm vững kiến trúc 7 module cốt lõi và thuật toán AlphaZero cho cờ vây.
- [x] Thiết lập file theo dõi tiến độ dự án (`PROJECT_PROGRESS.md`).
- [x] Cài đặt bộ kỹ năng chuyên sâu được chọn lọc vào `.agents/skills/`:
  - `ml-engineer`: Kiến trúc và huấn luyện hệ thống PyTorch 2.x
  - `python-performance-optimization` & `performance-profiling`: Tối ưu hiệu năng, profiling MCTS simulation
  - `python-testing-patterns`: Kiểm thử chuyên sâu, TDD, fixtures
  - `python-patterns` & `python-pro`: Thiết kế module theo chuẩn Python 3.12+
- [ ] Xác nhận với người dùng về môi trường (PyTorch / GPU / CPU) và mục tiêu giai đoạn tiếp theo.

### Phase 1: Go Engine Core (`go/`)
- [ ] `go/zobrist.py`: Mã hóa Zobrist hash cho bàn cờ $N \times N$, phát hiện Positional Superko.
- [ ] `go/board.py`: Cấu trúc dữ liệu bàn cờ, chuỗi quân (strings/chains), khí (liberties), bắt quân (capture), cấm tự sát, lịch sử nước đi.
- [ ] `go/scoring.py`: Thuật toán tính điểm diện tích (Area scoring - Tromp-Taylor / Chinese rules), komi, xác định thắng/thua $z \in \{+1, -1\}$.
- [ ] `tests/test_go/`: Bộ unit test đầy đủ cho các tình huống cờ phức tạp (bắt quân, ko, superko, mắt cờ, tự sát).

### Phase 2: Neural Network Architecture (`nn/`)
- [ ] `nn/features.py`: Trích xuất không gian đặc trưng đa kênh $N \times N \times C$ (16 plane lịch sử + 1 plane lượt đi), chuẩn hóa perspective (canonical form).
- [ ] `nn/model.py`: Mạng Residual Dual-Head (Backbone $B$ Residual blocks, Policy Head $N^2+1$, Value Head $\tanh$).
- [ ] `tests/test_nn/`: Kiểm tra kích thước tensor, forward pass, backward pass, đối xứng tensor.

### Phase 3: Monte Carlo Tree Search (`mcts/`)
- [ ] `mcts/node.py`: Cấu trúc nút MCTS ($N, W, Q, P$, liên kết cha-con).
- [ ] `mcts/search.py`: Thuật toán PUCT (Select, Expand & Masking, Backup đảo dấu, Dirichlet noise tại Root, Temperature $\tau$, Subtree reuse).
- [ ] `tests/test_mcts/`: Test tìm kiếm MCTS với mock model, xác minh hội tụ nước đi tối ưu cơ bản.

### Phase 4: Self-Play & Replay Buffer (`selfplay/`)
- [ ] `selfplay/replay_buffer.py`: Bộ nhớ đệm trượt (100k-500k mẫu), Data Augmentation với 8 phép đối xứng nhóm $D_4$ (xoay + lật).
- [ ] `selfplay/worker.py`: Vòng lặp tự chơi sinh ván đấu $(s, \pi, z)$, cơ chế resign threshold (-0.9), giới hạn số nước đi ($2N^2$).
- [ ] `tests/test_selfplay/`: Test worker sinh dữ liệu mẫu, kiểm tra tính toàn vẹn của Replay Buffer.

### Phase 5: Training Pipeline (`train/`)
- [ ] `train/trainer.py`: Vòng lặp huấn luyện, Loss function $(z-v)^2 - \pi^T \log p + c \|\theta\|^2$, Optimizer (SGD Momentum/Adam), LR scheduler, checkpointing.
- [ ] `tests/test_train/`: Test 1 step train trên dummy batch để xác nhận loss giảm và không nổ gradient.

### Phase 6: Arena & Elo Rating (`arena/`)
- [ ] `arena/match.py`: Đấu đối kháng giữa $\theta_{\text{new}}$ và $\theta_{\text{best}}$ (luân phiên Đen/Trắng, deterministic $\tau \to 0$).
- [ ] `arena/elo.py`: Tính tỷ lệ thắng (ngưỡng 55%) và ước tính cập nhật điểm Elo.
- [ ] `tests/test_arena/`: Test trận đấu giữa 2 agent đơn giản.

### Phase 7: GTP Server & Tích hợp GUI (`gtp/`)
- [ ] `gtp/gtp_server.py`: Cài đặt chuẩn Go Text Protocol (GTP v2) để kết nối Sabaki, Lizzie, Katrain, hoặc GNU Go.
- [ ] `tests/test_gtp/`: Test tương tác GTP qua stdin/stdout.

### Phase 8: AlphaZero Pipeline End-to-End & Tối ưu hóa nâng cao
- [ ] Tích hợp script điều phối tổng thể `main.py` (Self-play $\to$ Train $\to$ Arena).
- [ ] Cải tiến tiềm năng: Batch inference, Gumbel AlphaZero, Ownership/Score head (KataGo style).
