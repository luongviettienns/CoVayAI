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
- [x] Xác nhận môi trường phần cứng: GPU NVIDIA RTX 3050 (4GB VRAM, CUDA 12.7), Python 3.13.15, PyTorch 2.6.0+cu124.
- [x] Khởi tạo kho Git cục bộ và đẩy mã nguồn ban đầu lên GitHub: `https://github.com/luongviettienns/CoVayAI.git` (nhánh `main`).

### Phase 1: Go Engine Core (`go/`) [HOÀN THÀNH - 18/18 Tests Passed]
- [x] `go/zobrist.py`: Mã hóa Zobrist hash 64-bit cho bàn cờ $N \times N$, hỗ trợ tính toán gia tăng (XOR incremental update) và phát hiện Positional Superko.
- [x] `go/board.py`: Cấu trúc dữ liệu bàn cờ, chuỗi quân liên thông (chains/strings), khí (liberties), bắt quân (capture), cấm tự sát, Positional Superko, clone/copy cho MCTS.
- [x] `go/scoring.py`: Thuật toán tính điểm diện tích (Area scoring - Tromp-Taylor rules), komi 7.5, xác định thắng/thua $z \in \{+1, -1\}$.
- [x] `tests/test_go/`: Bộ kiểm thử toàn diện gồm 18 bài test (Zobrist determinism/uniqueness/incremental, bắt đơn/đa quân, cấm tự sát & ngoại lệ bắt quân, superko, seki, occupied point rejection, v.v.).

### Phase 2: Neural Network Architecture (`nn/`) [HOÀN THÀNH - 8/8 Tests Passed & CUDA Verified]
- [x] `config.py`: Khởi tạo hệ thống siêu tham số AlphaZero chuẩn (9x9, 6 ResBlocks, 96 filters, fp16, c_puct 1.5, batch size 256, v.v.).
- [x] `nn/features.py`: Trích xuất đặc trưng đa kênh 17-plane (8 frames quân ta + 8 frames quân địch + 1 frame lượt đi), chuẩn hóa canonical view; tạo 8 phép đối xứng nhóm $D_4$ giữ bất biến xác suất pass.
- [x] `nn/model.py`: Mạng Residual Dual-Head $f_\theta(s) = (p, v)$ chuẩn AlphaZero (~1M parameters), hỗ trợ GPU RTX 3050 CUDA 12.7, fast inference `predict()` với illegal moves masking.
- [x] `tests/test_nn/`: Bộ 8 bài test kiểm thử tensor shapes, bounds, gradient backprop, masking xác suất, đối xứng $D_4$ và thực thi CUDA trên phần cứng thật.

### Phase 3: Monte Carlo Tree Search (`mcts/`) [HOÀN THÀNH - 8/8 Tests Passed]
- [x] `mcts/node.py`: Cấu trúc Node lưu trữ $N(s,a), W(s,a), Q(s,a), P(s,a)$, công thức điểm PUCT UCB, quan hệ cha-con.
- [x] `mcts/search.py`: Thuật toán MCTS hoàn chỉnh:
  - Chọn nhánh (Select) theo công thức PUCT.
  - Mở rộng (Expand) và đánh giá bằng mạng nơ-ron hoặc kiểm tra kết thúc bằng Area Scoring.
  - Lan truyền ngược (Backup) đảo dấu $v \gets -v$ theo tính chất zero-sum hai người chơi luân phiên.
  - Bổ sung Dirichlet Noise tại root ($\alpha = 0.25, \varepsilon = 0.25$) cho các nước đi hợp lệ.
  - Quản lý nhiệt độ Temperature $\tau$ ($\tau \to 0$ cho nước đi tối ưu greedy, $\tau = 1$ cho khám phá).
  - Tái sử dụng cây con (Subtree reuse qua `advance_root`).
- [x] `tests/test_mcts/`: Bộ kiểm thử toàn diện gồm 8 bài test (cấu trúc node, PUCT formula, visit counts sum invariant, temperature, Dirichlet noise, subtree reuse, và test tích hợp trực tiếp với AlphaZeroNet trên GPU).

### Phase 4: Self-Play & Replay Buffer (`selfplay/`) [HOÀN THÀNH - 5/5 Tests Passed]
- [x] `selfplay/replay_buffer.py`: Bộ nhớ đệm trượt ReplayBuffer dung lượng 200k mẫu (deque sliding window), trích xuất batch ngẫu nhiên kèm Data Augmentation trực tiếp với 8 phép đối xứng $D_4$, hỗ trợ lưu/tải buffer ra đĩa.
- [x] `selfplay/worker.py`: Vòng lặp tự chơi sinh ván đấu `play_selfplay_game`:
  - Khám phá mở màn với nhiệt độ $\tau = 1.0$ cho 12 nước đầu, sau đó chuyển sang $\tau \to 0$ khai thác tối ưu.
  - Cơ chế Resign threshold ($-0.9$) sau 10 nước đi, kết hợp tỉ lệ $10\%$ ván không cho resign để chống false-positive.
  - Gán nhãn $z \in \{+1.0, -1.0\}$ từ góc nhìn của từng người chơi tại thời điểm $s$.
  - Hàm `generate_selfplay_games` hỗ trợ sinh hàng loạt ván đấu.
- [x] `tests/test_selfplay/`: Bộ 5 bài test kiểm thử sức chứa buffer, lấy mẫu batch tensor, serialization lưu/tải file và kiểm thử quy trình chơi trọn vẹn một ván cờ tự đấu có xử lý xin thua (resignation).

### Phase 5: Training Pipeline (`train/`) [HOÀN THÀNH - 4/4 Tests Passed]
- [x] `train/trainer.py`: Lớp Trainer quản lý toàn bộ quy trình tối ưu hóa:
  - Hàm mất mát kết hợp chuẩn AlphaZero: $L = (z - v)^2 - \pi^T \log p + c \|\theta\|^2$ (MSE cho Value head + Cross-Entropy cho Policy head + L2 weight decay).
  - Tích hợp kỹ thuật tăng tốc Mixed Precision FP16 (`torch.amp.autocast` và `GradScaler`) trên GPU RTX 3050.
  - Gradient Clipping (`clip_grad_norm_`) chống nổ gradient trong quá trình huấn luyện sâu.
  - Quản lý checkpointing (`save_checkpoint`, `load_checkpoint`) lưu/nạp trọng số mạng và optimizer state.
  - Vòng lặp `train_iteration` huấn luyện $K$ batch từ ReplayBuffer kèm D4 Augmentation.
- [x] `tests/test_train/`: Bộ 4 bài test kiểm thử single step loss metrics, kiểm chứng gradient descent hội tụ giảm loss trên fixed batch, kiểm thử train iteration trên buffer và kiểm tra tính toàn vẹn khi lưu/tải checkpoint.

### Phase 6: Arena & Elo Rating (`arena/`) [HOÀN THÀNH - 8/8 Tests Passed]
- [x] `arena/elo.py`: Hệ thống tính điểm và xếp hạng Elo chuẩn:
  - Công thức xác suất kỳ vọng chuẩn $E_A = 1 / (1 + 10^{(R_B - R_A)/400})$.
  - Cập nhật điểm $R'_A = R_A + K(S_A - E_A)$ bảo toàn tổng điểm đối kháng.
  - Ước lượng độ chênh lệch Elo dựa trên tỷ lệ thắng đối đầu $\Delta R = 400 \log_{10}(W / (1 - W))$ kèm kẹp biên số học an toàn chống chia cho 0.
  - Lớp `EloTracker` lưu trữ lịch sử đối đầu, bảng xếp hạng (leaderboard) và serialization JSON bền vững (`save`/`load`).
- [x] `arena/match.py`: Đấu trường đối kháng chuẩn quy cách:
  - Hàm `play_arena_game`: thi đấu 1 ván trực tiếp giữa 2 model với MCTS deterministic ($\tau = 0$, không noise), tái sử dụng cây con (`advance_root`), hỗ trợ ngưỡng xin thua (resignation threshold).
  - Lớp `Arena`: điều phối giải đấu $N$ ván luân phiên màu quân (vòng chẵn candidate cầm Đen, vòng lẻ cầm Trắng), tính tỷ lệ thắng và quyết định phê duyệt mô hình mới (ngưỡng $\ge 55\%$).
  - Cấu trúc kết quả `ArenaResult` cung cấp thống kê chi tiết (thắng khi cầm Đen/Trắng, hòa, tỷ lệ thắng, trạng thái duyệt).
- [x] `tests/test_arena/`: Bộ 8 bài test kiểm thử công thức Elo, bảo toàn tổng điểm, ước lượng chênh lệch, lưu/tải tracker, đấu cờ 2 model, luân phiên màu quân và tính xác định (deterministic replay).

### Phase 7: GTP Server & Tích hợp GUI (`gtp/`) [HOÀN THÀNH - 6/6 Tests Passed]
- [x] `gtp/gtp_server.py`: Cài đặt chuẩn Go Text Protocol (GTP v2) tương thích các GUI cờ vây chuyên nghiệp (Sabaki, Lizzie, Katrain, GNU Go):
  - Ánh xạ tọa độ chuẩn bàn cờ (A-T bỏ chữ 'I', số hàng từ 1 đến N, xử lý nước đi `pass` và `resign`).
  - Hỗ trợ đầy đủ bộ lệnh cốt lõi: `protocol_version`, `name`, `version`, `known_command`, `list_commands`, `boardsize`, `clear_board`, `komi`, `play`, `genmove`, `final_score`, `showboard`, `quit`.
  - Quản lý Command ID (`={id} <response>\n\n`, `?{id} <error>\n\n`), lọc comment ký tự `#`.
  - CLI entrypoint (`python -m gtp.gtp_server --checkpoint <path> --size 9 --sims 200`) sẵn sàng nạp checkpoint trọng số mạng để đấu trực tiếp qua GUI.
- [x] `tests/test_gtp/`: Bộ 6 bài test kiểm thử ánh xạ tọa độ 2 chiều (A9, A1, E5, J9, pass), chuỗi lệnh định danh & truy vấn, cấu hình bàn cờ/komi, sinh nước đi thông minh `genmove`, bắt lỗi nước đi không hợp lệ và chấm điểm ván cờ `final_score`.

### Phase 8: AlphaZero Pipeline End-to-End & Tối ưu hóa nâng cao
- [ ] Tích hợp script điều phối tổng thể `main.py` (Self-play $\to$ Train $\to$ Arena).
- [ ] Cải tiến tiềm năng: Batch inference, Gumbel AlphaZero, Ownership/Score head (KataGo style).
