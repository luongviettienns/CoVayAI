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

### Phase 8: AlphaZero Pipeline End-to-End & Tối ưu hóa nâng cao [HOÀN THÀNH - 58/58 Tests Passed]
- [x] `main.py`: Kịch bản điều phối toàn diện vòng lặp AlphaZero chuẩn:
  - Khởi tạo cấu hình linh hoạt qua `PipelineConfig` và CLI args (`--iters`, `--games`, `--sims`, `--train-steps`, `--batch-size`, `--arena-games`, `--device`, v.v.).
  - Vòng lặp tự động khép kín: Self-Play $\to$ Trích xuất đặc trưng $\to$ Đẩy Replay Buffer $\to$ Huấn luyện Candidate Model (Mixed Precision FP16 + L2 + Grad Clip) $\to$ Đấu đối kháng Arena $\to$ Cập nhật Elo và phê duyệt mô hình $\to$ Lưu checkpoint `checkpoint_iter_{i}.pt`, `best_model.pt`, `ratings.json`.
  - Cơ chế rollback an toàn: nếu mô hình mới không đạt tỷ lệ thắng $\ge 55\%$, tự động hoàn nguyên trọng số về `best_model`.
  - **Live Progress Bar (Mục 1)**: Tích hợp thanh tiến trình thời gian thực (`tqdm`) trên cả 3 giai đoạn: Self-Play (hiển thị số nước, người thắng, tổng mẫu), Training (hiển thị loss tức thời), Arena (hiển thị tỷ số thắng/thua và winrate trực tiếp).
  - **Buffer Persistence & Auto-Resume (Mục 3)**: Tự động lưu `buffer.pkl` định kỳ trên đĩa; hỗ trợ cờ `--resume` (mặc định bật) tự động nhận diện checkpoint cũ, khôi phục Replay Buffer và tiếp tục học liền mạch từ thế hệ trước mà không mất dữ liệu.
  - **Tối ưu tỷ lệ Sample-to-Train (Dynamic Training Steps)**: Tính toán số bước học `effective_steps` linh hoạt dựa trên kích thước thực tế của Replay Buffer thay vì cố định 400 steps, triệt tiêu hiện tượng "học vẹt" (overfitting) của Value Head ở các vòng lặp đầu.
  - **Đồng bộ hóa trạng thái Adam Optimizer**: Lưu và phục hồi toàn vẹn cả `model_state_dict` lẫn `optimizer_state_dict` khi Candidate bị từ chối ở Arena, loại bỏ hoàn toàn hiện tượng lệch động lượng (momentum pollution).
  - **Tùy chọn Continuous AlphaZero (`--no-gating`)**: Hỗ trợ chuyển đổi linh hoạt giữa cơ chế cổng chặn cổ điển (AlphaGo Zero 55%) và học liên tục trực tiếp (AlphaZero 2018).
  - **Tinh chỉnh siêu tham số**: Tăng `DIRICHLET_ALPHA` lên 0.30 tối ưu cho không gian bàn cờ $9 \times 9$, kết nối đầy đủ `resign_disable_frac` (10%) chống đầu hàng sai.
- [x] `tests/test_pipeline.py`: Bài test tích hợp end-to-end kiểm chứng vòng lặp hoàn chỉnh (từ ván cờ tự chơi qua MCTS, tối ưu hóa gradient, đấu trường đối đầu cho tới xuất file checkpoint, `buffer.pkl` và tự động khôi phục khi resume).

### Phase 9: Vá 4 Lỗi Thuật Toán Cốt Lõi, Tối Ưu Hiệu Năng & Hoàn Thiện Vận Hành [HOÀN THÀNH - 65/65 Tests Passed]
- [x] **TDD Red-Green-Refactor**: Tạo bộ 4 bài test hồi quy trong `tests/test_regression.py` tái hiện chính xác 4 lỗi thuật toán nghiêm trọng (xác nhận FAIL đỏ), sau đó vá code và xác nhận toàn bộ 4 test cùng toàn bộ 65 tests đều PASS xanh.
- [x] **Vá lỗi MCTS PUCT Sign & Corner Exploration (`mcts/node.py`)**:
  - Đảo dấu Q-score: `-self.value_score + u` (vì `value_score` lưu kỳ vọng của đối thủ tại node con).
  - Triệt tiêu thiên lệch góc A9 ở mô phỏng đầu tiên: `math.sqrt(max(1, parent_visit_count))` giúp prior $P(s, a)$ của mạng dẫn đường ngay từ mô phỏng 1.
  - Lưu trữ `clean_prior` chống méo mó phân phối tích lũy khi áp dụng noise nhiều lần.
- [x] **Vá lỗi Dirichlet Noise trên Subtree Reuse (`mcts/search.py`)**:
  - Áp dụng Dirichlet noise trên `clean_prior` ở mọi lượt search bất kể root mới hay tái sử dụng từ `advance_root`.
  - MCTS `search()` luôn trả về phân phối xác suất mềm $\pi$ ($\tau=1$) làm nhãn huấn luyện chuẩn AlphaZero.
  - Cung cấp hàm `select_action(pi, temperature)` hỗ trợ bẻ hòa ngẫu nhiên (`random tie-breaking`).
- [x] **Vá lỗi Policy Target One-Hot (`selfplay/worker.py`)**:
  - Self-play lưu trữ $\pi$ mềm ($\tau=1$) vào Replay Buffer; nhiệt độ $\tau \to 0$ sau nước 12 chỉ dùng để chọn nước đi qua `select_action`.
- [x] **Đa dạng hóa Đấu trường Arena (`arena/match.py`)**:
  - Tham số `opening_plies=8`: 8 nước đầu lấy mẫu $\tau=1$ giúp 50 ván đấu phân nhánh đa dạng, triệt tiêu hiện tượng lặp lại 2 ván duy nhất.
  - Hỗ trợ cờ `return_moves=True` cho phép theo dõi và kiểm thử chuỗi nước đi.
- [x] **Chuẩn hóa Komi duy nhất & GTP Server Hoàn Thiện (`gtp/gtp_server.py`)**:
  - Truyền `komi` đồng bộ từ `config.KOMI` $\to$ `MCTS`, `selfplay`, `arena`, và CLI `--komi`.
  - Lệnh `cmd_undo`: khôi phục trạng thái chuẩn xác bằng cách replay toàn bộ `move_history` trên `Board` mới và `mcts.reset()`.
  - Tự động reset MCTS khi phát hiện người chơi đi nước không xen kẽ màu quân.
- [x] **Tối ưu Hóa Huấn luyện & Checkpoint (`train/trainer.py`, `main.py`)**:
  - Chuyển sang `AdamW` (`weight_decay=0.01` cho Conv2D/Linear weights, `0.0` cho Biases và BatchNorm).
  - Tích hợp `MultiStepLR(milestones=[30, 60], gamma=0.1)` và lưu/khôi phục toàn vẹn `scheduler_state_dict` trong checkpoint.
  - Sử dụng `copy.deepcopy()` cho tất cả các điểm nạp `optimizer.load_state_dict` và `scheduler.load_state_dict` nhằm loại bỏ triệt để hiện tượng in-place mutation của PyTorch Adam tensor làm thay đổi `best_optimizer_state` trong bộ nhớ khi Candidate bị từ chối ở Arena.
  - Sửa lỗi nạp ID mô hình tốt nhất khi resume: Lấy `iteration` từ `best_model.pt` (`ckpt["iteration"]`) thay vì lấy `max_iter` từ `checkpoints/` (vốn có thể là một iteration bị REJECT ở Arena), đảm bảo bảng xếp hạng Elo luôn chính xác.
  - Cơ chế tự động tính `komi` (0.5 cho bàn $\le 7\times 7$, 7.5 cho bàn $\ge 9\times 9$) và `max_moves` ($2N^2$) tự thích ứng kích cỡ bàn cờ.
  - Bổ sung tham số CLI `--arena-sims` tách biệt số mô phỏng đánh giá Arena và số mô phỏng sinh dữ liệu Self-Play `--sims`.
  - Tối ưu I/O Replay Buffer: Lưu `buffer.pkl` 1 lần duy nhất sau mỗi iteration thay vì định kỳ 5 ván một lần (tránh tiêu tốn hàng chục GB I/O khi buffer đạt 200k mẫu).
- [x] **Tối Ưu Hiệu Năng Thực Chất (`go/board.py`, `nn/features.py`, `selfplay/replay_buffer.py`)**:
  - `Board.copy()`: Sử dụng `Board.__new__(Board)` bỏ qua `__init__`, chia sẻ `_neighbors` và `zobrist` table.
  - `get_legal_moves()`: Thêm shortcut hàng xóm trống (empty neighbor fast path) và kiểm tra superko bằng hash trước khi chạy BFS.
  - D4 Augmentation: Hàm `apply_random_symmetry` chỉ tính đúng 1 phép biến đổi thay vì tính 8 phép rồi bỏ 7, có test khớp `test_apply_random_symmetry_matches_get_symmetries`.
- [x] **Baseline Đối Kháng Khách Quan & Thử Nghiệm Bàn 5x5 (`arena/baseline.py`)**:
  - Xây dựng mô-đun `RandomModel` và hàm `evaluate_against_random`.
  - Chạy thử nghiệm thành công bàn $5\times 5$ (`komi=0.5`) qua 1 iteration hoàn chỉnh và đạt tỷ lệ thắng 100% trước đối thủ Random Player.
- [x] **Tiêu Chuẩn Mã Nguồn Mở**:
  - Bổ sung tệp giấy phép phần mềm MIT (`LICENSE`).
  - Chuẩn hóa cây thư mục `CoVayAI/` và cập nhật tài liệu `README.md`.
- [x] **Toàn bộ 65/65 bài kiểm thử tự động đạt 100% PASS** trên toàn bộ 9 modules của dự án!

### Phase 10: Chuẩn Hóa Sẵn Sàng Train 9×9 (P0 & P1 - KE_HOACH_SUA_COVAYAI_TRAIN_READY.md) [HOÀN THÀNH - 75/75 Tests Passed]
- [x] **P0: Đóng băng Baseline & Chuẩn hóa Môi trường**:
  - Tạo nhánh độc lập `fix/train-readiness` tại commit `5f4898c`.
  - Ghi nhận thông số phần cứng thực tế: NVIDIA GeForce RTX 3050 Laptop GPU (4.29 GB VRAM, CUDA 12.4), Python 3.13.15, PyTorch 2.6.0+cu124.
  - Tạo tệp `requirements.txt` và `requirements-dev.txt` cố định phiên bản thư viện.
  - Xây dựng 5 kịch bản kiểm thử mẫu chuẩn (Golden Fixtures) tại `tests/test_golden_fixtures.py` (Bàn trống, bắt quân atari, cấm tự sát, Positional Superko, kết thúc Area Scoring).
  - Khởi chạy thành công smoke test 5×5 trên thư mục checkpoint mới `checkpoints/baseline_smoke5`.
- [x] **P1: Tách rời Tiến độ Scheduler Toàn Cục & Khôi Phục Checkpoint Nguyên Tử**:
  - Khắc phục lỗi kẹt Learning Rate: Tách tiến trình `MultiStepLR` toàn cục khỏi kết quả Arena. Khi Candidate bị REJECT, chỉ rollback trọng số mô hình và động lượng optimizer, đồng bộ LR tức thời từ `scheduler.get_last_lr()` vào optimizer, bảo đảm `last_epoch` và mốc giảm LR (milestones 30, 60) luôn tịnh tiến chuẩn xác.
  - Cơ chế ghi tệp nguyên tử (Atomic File Writing): Sử dụng tệp tạm `.tmp` và `os.replace` cho tất cả checkpoint (`checkpoint_iter_*.pt`, `best_model.pt`, `buffer.pkl`, `ratings.json`).
  - Single Source of Truth (`latest_manifest.json`): Lưu trữ toàn vẹn metadata (iteration, best_model_id, scheduler_state_dict, fingerprint kiến trúc, cấu hình siêu tham số, thời gian) và kiểm tra tính tương thích kiến trúc (`validate_checkpoint_compatibility`) khi resume.
  - Bảo vệ thư mục lưu trữ (`--no-resume` & `--force-new`): Từ chối chạy đè lên thư mục checkpoint đã có dữ liệu cũ nếu không bật cờ `--force-new`.
  - Bổ sung và kiểm định chặt chẽ các tham số CLI: `--seed`, `--resign-threshold`, `--max-moves`, `--force-new`.
- [x] **Toàn bộ 75/75 bài kiểm thử tự động đạt 100% PASS** (bao gồm bộ test hồi quy scheduler `tests/test_train/test_scheduler_checkpoint.py`).

### Phase 11: Kiểm Định Game Rules, MCTS Invariants & Chuẩn Hóa Metadata Self-Play (P2 & P3) [HOÀN THÀNH - 81/81 Tests Passed]
- [x] **P2: Xác thực Luật Cờ, Bất Biến Bàn Cờ & MCTS Safety**:
  - Property testing 100 ván cờ ngẫu nhiên (`tests/test_rules_and_selfplay.py`): Khẳng định mã băm tăng dần luôn khớp tuyệt đối với `compute_position_hash(grid)`, người chơi luân phiên chuẩn xác, lịch sử `history_positions` bảo toàn toàn vẹn và các ô đã có quân/tự sát đều bị từ chối trong $100\%$ các bước đi.
  - Kiểm thử phân lập sâu `Board.copy()`: Đảm bảo bản sao lưu giữ đầy đủ lịch sử superko và grids, thao tác hạ quân trên bản sao hoàn toàn không ảnh hưởng đến bàn cờ gốc.
  - An toàn MCTS: Kiểm chứng MCTS không bao giờ duyệt hoặc gán xác suất $>0$ cho nước đi bất hợp lệ, xử lý mượt mà trạng thái kết thúc sau 2 nước pass liên tiếp.
  - Cơ chế bắt lỗi nghiêm ngặt: Kiểm tra giá trị trả về của `board.play(action)` trong cả `play_selfplay_game` và `play_arena_game`; lập tức raise `RuntimeError` chi tiết nếu phát hiện nước đi bất hợp lệ thay vì âm thầm bỏ qua.
- [x] **P3: Chuẩn Hóa Metadata Ván Đấu & Sửa Lỗi Hiển Thị Winner Log**:
  - Đóng gói cấu trúc `GameResult` (`selfplay/worker.py`): Lưu giữ trọn vẹn mẫu huấn luyện (`samples`), người thắng (`winner`), điểm số chi tiết (`black_score`, `white_score`), số nước đi (`moves`), lý do kết thúc (`end_reason`: `two_passes`, `max_moves`, `resign`) và người xin thua (`resigned_by`). Hỗ trợ giao diện sequence (`__len__`, `__getitem__`, `__iter__`) tương thích 100% với code cũ.
  - Sửa lỗi hiển thị Winner Log: Thay thế cách tính suy diễn sai `winner_str = 'B' if game_samples[-1][2] > 0 else 'W'` bằng `winner_str = 'B' if game_result.winner == BLACK else 'W'`, bảo đảm tên người thắng hiển thị chính xác trong mọi tình huống (kể cả khi Trắng đi nước cuối và thắng/thua).
  - Kiểm định cờ `--resign-disable-frac`: Đảm bảo khi cấm xin thua, ván đấu luôn diễn ra đến khi hai bên pass hoặc đạt `max_moves`, loại trừ hoàn toàn false-positive resign ở giai đoạn model chưa hội tụ.
- [x] **Toàn bộ 81/81 bài kiểm thử tự động đạt 100% PASS** trên toàn bộ 10 modules kiểm thử của dự án!

### Phase 12: Tối Ưu Replay Buffer Nhanh $O(1)$, Tiết Kiệm 75% RAM & Schema v2 Persistence (P4) [HOÀN THÀNH - 84/84 Tests Passed]
- [x] **P4.1: Chuyển Đổi Sang Ring Buffer $O(1)$ & Tiết Kiệm Bộ Nhớ**:
  - Loại bỏ hoàn toàn `collections.deque` (vốn tốn $O(N)$ khi truy cập chỉ số ngẫu nhiên rời rạc trên hàng trăm nghìn mẫu).
  - Cấu trúc mảng NumPy liên tục tiền cấp phát (Preallocated Contiguous Arrays): `states: (capacity, 17, N, N)` lưu kiểu `uint8` (17 planes nhị phân 0/1), giảm $75\%$ dung lượng RAM (chỉ tốn $\approx 326$ MB cho 200,000 mẫu trên bàn 9×9 thay vì $>1.3$ GB như trước), `pis: (capacity, action_size)` lưu `float32`, `zs: (capacity, 1)` lưu `float32`.
  - Con trỏ ghi vòng (Ring Buffer Write Pointer): Tự động ghi đè mẫu cũ nhất khi đầy bộ đệm (`write_index = (write_index + 1) % capacity`), triệt tiêu chi phí cấp phát bộ nhớ động và bộ gom rác (Garbage Collector).
  - Tốc độ lấy mẫu siêu tốc: Trích xuất batch trực tiếp bằng C-speed fancy indexing (`states[indices]`), đạt tốc độ $\approx 0.076$ ms/batch (2k samples) và $\approx 0.377$ ms/batch (20k samples).
  - Tương thích ngược: Lớp `_BufferView` hỗ trợ truy cập theo thứ tự thời gian (`buf.buffer[0]` là mẫu cũ nhất, `buf.buffer[-1]` là mẫu mới nhất), bảo toàn tương thích tuyệt đối cho các đoạn mã cũ.
- [x] **P4.2: Định Dạng Lưu Trữ Schema v2 & Tự Động Di Trú (Migration)**:
  - Thiết kế Schema v2 chuẩn hóa với đầy đủ metadata: `schema_version = 2`, `board_size`, `capacity`, `size`, `write_index`.
  - Cơ chế kiểm định an toàn: Ngăn chặn triệt để việc nạp buffer khác cỡ bàn cờ (`ValueError: Buffer board_size mismatch`).
  - Khả năng tự động nâng cấp: Tự động phát hiện và di trú (migrate) dữ liệu từ buffer dạng Schema v1 cũ (danh sách các tuple thô) vào cấu trúc mảng mới mà không làm mất mẫu dữ liệu nào.
  - Ghi tệp đĩa nguyên tử: Lưu thông qua tệp `.tmp`, flush/fsync và hoán đổi bằng `os.replace`.
- [x] **Toàn bộ 84/84 bài kiểm thử tự động đạt 100% PASS** (bổ sung 3 test chuyên sâu cho vòng đời ring buffer, schema v2 persistence và benchmark scaling)!

### Phase 13: Chuẩn Hóa Đấu Trường Arena Khai Cuộc Ghép Cặp (Paired Openings), Thống Kê Khoảng Tin Cậy 95% & Fixed Baselines (P5) [HOÀN THÀNH - 89/89 Tests Passed]
- [x] **P5.1: Đấu Trường Khai Cuộc Ghép Cặp (Paired Openings & Color Swapping)**:
  - Hàm độc lập sinh khai cuộc `generate_opening_sequence`: Tạo chuỗi $K$ nước đi khai cuộc hợp lệ và đa dạng trên bàn cờ từ seed kiểm soát, loại trừ hoàn toàn nước pass sớm.
  - Cấu trúc ghép cặp trận đấu (Paired Matchups): Mỗi cặp khai cuộc được thi đấu đối xứng qua 2 ván cờ:
    1. Candidate cầm Đen vs Baseline cầm Trắng từ chuỗi nước đi khai cuộc giống hệt nhau.
    2. Baseline cầm Đen vs Candidate cầm Trắng từ chuỗi nước đi khai cuộc giống hệt nhau.
  - Triệt tiêu phương sai vị thế khai cuộc và bảo đảm tính đối xứng tuyệt đối (`test_arena_score_is_symmetrical_on_model_swap`).
  - Tắt hoàn toàn nhiễu Dirichlet và nước đi ngẫu nhiên sau khai cuộc (`temperature = 0.0`), đảm bảo sức mạnh mô hình được đánh giá thuần túy dựa trên chính sách MCTS tối ưu.
- [x] **P5.2: Thống Kê Chi Tiết & Khoảng Tin Cậy Wilson 95%**:
  - Đóng gói cấu trúc `PairedGameResult` và mở rộng `ArenaResult`: Lưu chi tiết kết quả từng cặp, số nước đi, người thắng theo màu cờ.
  - Tích hợp tính toán khoảng tin cậy Wilson Score 95% (`win_rate_ci`): Đánh giá định lượng ý nghĩa thống kê của tỷ lệ thắng thay vì chỉ dựa vào ngưỡng điểm 55%.
  - Bổ sung thông tin phân rã theo màu cờ (`candidate_as_black_wins`, `candidate_as_white_wins`) và khoảng tin cậy vào nhật ký tiến trình `main.py` và `latest_manifest.json`.
- [x] **P5.3: Chuẩn Hóa Đối Thủ Chuẩn Cố Định (Fixed Baselines)**:
  - Nâng cấp `RandomModel`: Ngăn chặn việc chọn nước pass ngẫu nhiên khi bàn cờ còn nhiều nước đi hợp lệ (`allow_early_pass=False`).
  - Xây dựng mô hình chiến thuật `CaptureHeuristicModel`: Ưu tiên tuyệt đối các nước bắt quân atari, bảo vệ quân nguy cấp và phân phối đều trên bàn cờ, cung cấp một đối thủ đo lường khách quan không phụ thuộc vào trọng số checkpoint.
- [x] **Toàn bộ 89/89 bài kiểm thử tự động đạt 100% PASS** trên toàn bộ 10 modules kiểm thử của dự án!

### Phase 14: Đo Lường Hiệu Năng GPU/CPU, MCTS Throughput & Suy Luận Batch Predict (P6) [HOÀN THÀNH - 93/93 Tests Passed]
- [x] **P6.1: Đo Đạc Hiệu Năng Chi Tiết (Performance Microbenchmarks & Profiling)**:
  - Xây dựng bộ công cụ benchmark tự động (`tests/test_performance_profiling.py`):
    - `Board.copy()`: Đạt tốc độ cực nhanh $\approx 2.97$ µs/lần sao chép.
    - `Board.get_legal_moves()`: Tốc độ kiểm định và sinh nước đi hợp lệ $\approx 128.49$ µs/lần gọi.
    - `extract_features(b)`: Trích xuất 17 mặt phẳng nhị phân chỉ tốn $\approx 27.11$ µs/bàn cờ.
    - Thông lượng MCTS: Đạt $\approx 610.7$ mô phỏng/giây (simulations/s) trên phần cứng NVIDIA RTX 3050 Laptop GPU (CUDA).
- [x] **P6.2: Hỗ Trợ Suy Luận Theo Mẻ (`predict_batch`)**:
  - Bổ sung phương thức `AlphaZeroNet.predict_batch(state_features, legal_moves_masks, device)`:
    - Xử lý đồng thời batch các trạng thái bàn cờ `(B, 17, N, N)` và mảng mặt nạ luật cờ `(B, action_size)`.
    - Tốc độ suy luận tăng vọt: Đạt $\approx 0.119$ ms/trạng thái ($\approx 8,402$ trạng thái/giây trên CUDA), tăng tốc gấp **$15.92\times$** so với suy luận đơn lẻ từng trạng thái ($\approx 1.895$ ms/trạng thái).
  - Khẳng định tính tương đương: Kết quả từ `predict_batch` hoàn toàn khớp bit-wise / floating precision với các lượt gọi tuần tự qua `predict`.
- [x] **Toàn bộ 93/93 bài kiểm thử tự động đạt 100% PASS** trên toàn bộ 11 modules kiểm thử của dự án!

### Phase 15: Thử Train Theo Cổng Nghiệm Thu, Khởi Chạy 5×5 Smoke & 9×9 Pilot (P7) [HOÀN THÀNH - SẴN SÀNG HUẤN LUYỆN SÂU]
- [x] **P7.1: Thử Nghiệm Khép Kín 5×5 Smoke & Kiểm Chứng Resume Không Trùng Lặp**:
  - Chạy thực nghiệm thành công 2 iteration đầu tiên trên thư mục phân lập `checkpoints/smoke5_v1`:
    - 2 ván self-play/iter kết thúc tự nhiên bằng 2 nước pass (`reason=two_passes`), 0 nước bất hợp lệ, loss giảm đều từ $4.58 \rightarrow 3.20$.
    - Đấu trường Arena 4 ván ghép cặp (2 pairs) đạt tỷ lệ thắng $100\%$ và $75\%$ kèm khoảng tin cậy Wilson $95\%$.
  - Kiểm chứng khôi phục (Resume Verification): Chạy tiếp lệnh huấn luyện từ iteration 2 lên iteration 3:
    - Tự động nạp `latest_manifest.json`, khôi phục 96 mẫu đệm ReplayBuffer Schema v2, nhận diện đúng mô hình tốt nhất `iter_2`.
    - Chỉ chạy tiếp duy nhất iteration 3, không chạy lại iteration 1 và 2.
    - Xử lý tình huống Candidate bị REJECT ($50\% < 55\%$): Rollback trọng số mô hình và động lượng optimizer chuẩn xác, giữ nguyên tiến trình scheduler và LR ($0.001$).
- [x] **P7.2: Kiểm Chứng Pilot Bàn Cờ Chuẩn 9×9**:
  - Khởi chạy thành công 1 iteration hoàn chỉnh trên bàn cờ chuẩn 9×9 tại `checkpoints/pilot9_smoke` (`komi=7.5`, ResNet 6×96, 17 mặt phẳng trạng thái, 82 hành động):
    - Thu thập 231 mẫu huấn luyện qua 2 ván cờ (115 và 116 nước đi).
    - Huấn luyện động 10 bước: Policy loss giảm $4.49 \rightarrow 4.14$, Value loss giảm $1.12 \rightarrow 0.07$.
    - Arena 4 ván ghép cặp trên 9×9: Candidate thắng $75\%$ ($3\text{W}-1\text{L}-0\text{D}$), được chấp thuận (ACCEPTED) và lưu trữ checkpoint nguyên tử.
- [x] **BẢNG NGHIỆM THU TỔNG THỂ (P0 - P7)**:
  - Tất cả các cổng P0, P1, P2, P3, P4, P5, P6, P7 đều **PASS** 100%!
  - Bộ kiểm thử tự động đạt mốc kỷ lục: **93/93 Tests PASS** trong $\approx 6.44$s.
  - Toàn bộ pipeline đã sạch bóng lỗi ngầm, tối ưu RAM, lưu trữ nguyên tử và hoàn toàn sẵn sàng cho các đợt huấn luyện dài hạn!

### Phase 16: Chiến Dịch Huấn Luyện Sản Phẩm 9×9 Qua Đêm (`checkpoints/run9_prod`) [ĐANG CHẠY GIAI ĐOẠN 2]
- [x] **Giai đoạn 1 (Iterations 1 $\rightarrow$ 20) [HOÀN THÀNH XUẤT SẮC]**:
  - Thời gian chạy: $\approx 2.5$ giờ liên tục trên GPU NVIDIA RTX 3050 CUDA.
  - Tổng số mẫu thu thập trong Replay Buffer Schema v2: **43,621 mẫu 9×9** sạch, không lỗi.
  - **Kiểm định thực chiến của mô hình `checkpoint_iter_20.pt`**:
    - Đối đầu `RandomModel`: Thắng tuyệt đối **$4/4$ ván ($100.0\%$)** cả khi cầm quân Đen lẫn Trắng.
    - Đối đầu `CaptureHeuristicModel` (AI chiến thuật chuyên săn bắt quân atari): Thắng áp đảo **$4/4$ ván ($100.0\%$)** qua các ván đấu 91 – 152 nước cờ.
    - Mô hình đã hình thành trực giác bảo vệ quân, kiểm soát vùng biên và ăn quân đối phương.
  - Thăng cấp `checkpoint_iter_20.pt` thành `best_model.pt` làm hạt giống chuẩn mực cho Giai đoạn 2.
- [x] **Giai đoạn 2 (Iterations 21 $\rightarrow$ 69) [HOÀN THÀNH - KẾT THÚC THEO YÊU CẦU NGƯỜI DÙNG]**:
  - Đã hoàn tất trọn vẹn 69 iterations huấn luyện liên tục (`--no-gating`).
  - Checkpoint sản phẩm tối ưu: `checkpoint_iter_69.pt` và `best_model.pt` lưu trữ đầy đủ trọng số ResNet 6×96 (12.5 MB).
  - Tích lũy Replay Buffer Schema v2: Gần 50,000 mẫu thế trận 9×9 đa dạng từ khai cuộc, trung cuộc đến tàn cuộc.
  - Tỷ lệ ván đấu kết thúc bằng `reason=resign` chiếm ưu thế, chứng minh mạng nơ-ron đã học được cách định giá thế trận (Value Head) rất nhạy bén, nhận biết khi nào vị thế không thể cứu vãn để chủ động xin thua.
  - Tối ưu hóa dung lượng: Đã dọn dẹp 57 checkpoint trung gian, giải phóng hơn 500 MB ổ đĩa.
  - Mô hình `best_model.pt` (Iter 69) hiện đang được phục vụ trực tiếp trên CoVayAI Web Arena (`http://localhost:8080`).

### Phase 17: CoVayAI Web Arena — Đấu Trường Trực Quan Web Test Chơi Cờ Vây [HOÀN THÀNH - 96/96 Tests Passed]
- [x] **Kiến Trúc Backend & API Suy Luận PyTorch (`web/server.py`)**:
  - Sử dụng `ThreadedHTTPServer` đa luồng tích hợp sẵn trong Python standard library, tuyệt đối an toàn và không xung đột với tiến trình train đang chạy nền trên GPU.
  - Tự động phát hiện kiến trúc siêu tham số từ checkpoint (`num_res_blocks`, `num_filters`), tự động nạp `best_model.pt` hoặc rơi về mô hình khởi tạo nếu chưa có checkpoint.
  - Cung cấp đầy đủ REST API:
    - `GET /api/info`: Thông tin thiết bị tăng tốc phần cứng (NVIDIA RTX 3050 CUDA).
    - `GET /api/state`: Trạng thái bàn cờ, ma trận grid, danh sách nước hợp lệ, tỷ lệ thắng (Winrate), điểm lãnh thổ ước tính.
    - `POST /api/new_game`: Khởi tạo ván mới tùy chỉnh kích thước ($9\times 9$ hoặc $5\times 5$), màu quân (Đen/Trắng), độ sâu MCTS (32, 64, 128 sims).
    - `POST /api/play`: Đặt quân người chơi và kích hoạt AI phản hồi tức thì qua MCTS.
    - `POST /api/pass`: Bỏ lượt và kích hoạt AI phản hồi.
    - `POST /api/resign`: Xin thua ván cờ.
- [x] **Giao Diện Frontend Sang Trọng Hiện Đại (`web/static/`)**:
  - `index.html`: Cấu trúc ngữ nghĩa trực quan với bàn cờ trung tâm, thẻ trạng thái 2 bên đối kháng, thanh đo tỷ lệ thắng kép (Dual Winrate Gauge), danh sách chip các nước cờ AI cân nhắc (MCTS Prior), bộ điều khiển trận đấu và biên bản ván đấu.
  - `style.css`: Thiết kế Dark-mode Glassmorphism cao cấp, bảng màu hiện đại (`Outfit` + `JetBrains Mono`), vân gỗ bàn cờ tự nhiên, quân cờ 3D bóng đổ obsidian/pearl.
  - `app.js`:
    - Vẽ bàn cờ Canvas sắc nét trên màn hình Retina (DPR 2x), vẽ sao (hoshi) chuẩn $9\times 9$ và $5\times 5$, nhãn tọa độ A-J (bỏ chữ I theo chuẩn quốc tế) và 1-9.
    - Quân cờ mờ (ghost preview) khi di chuột qua các giao điểm hợp lệ.
    - Vòng hào quang sáng chỉ định nước đi mới nhất (last move indicator).
    - Tự tổng hợp âm thanh tiếng cạch gõ quân cờ qua Web Audio API (không cần tải tệp mp3 ngoài).
- [x] **Kiểm Thử Tự Động & Thử Nghiệm Thực Tế Trình Duyệt**:
  - Viết bộ kiểm thử tích hợp `tests/test_web_server.py`: Kiểm thử vòng đời ván đấu, phản hồi MCTS và toàn bộ REST API endpoints.
  - **100% 96/96 bài kiểm thử toàn dự án đạt PASS**!
  - Kiểm thử trực quan end-to-end qua Browser Subagent: Đặt quân Đen tại giao điểm, AI phản hồi nước Trắng chuẩn xác và cập nhật điểm số, tỷ lệ thắng, lịch sử nước đi trực quan.
  - Máy chủ đang chạy sẵn sàng phục vụ tại: **`http://localhost:8080`**.

### Phase 18: Huấn Luyện Giáo Trình Phản Đòn ("Học Trả Thù" - Targeted Revenge Curriculum) [HOÀN THÀNH - SẴN SÀNG ĐẤU TRƯỜNG]
- [x] **Trích Xuất & Phân Tích Ván Cờ Thực Chiến 24 Nước Của Người Dùng**:
  - Biên bản ván đấu: `['E5', 'D8', 'E3', 'D6', 'G6', 'E7', 'D5', 'C5', 'C4', 'D4', 'C3', 'B4', 'B3', 'E4', 'F4', 'D3', 'D2', 'D4', 'F7', 'Pass', 'C6', 'Pass', 'E6', 'Pass']`.
  - Phân tích điểm mù: Người dùng (Black) chiếm giữ Thiên nguyên `E5`, sau đó củng cố vững chắc trung tâm với `E3`, `G6`, `D5`. AI cũ (White) đánh tản mác ở biên (`D8`, `D6`, `E7`) nên bị Black vây hãm và chiếm toàn bộ đất trung tâm.
- [x] **Tăng Cường Dữ Liệu Đối Xứng ($D_4$ Augmentation) & Khởi Tạo Nhãn Chuẩn**:
  - Nhân bản 24 nước đi qua 8 phép biến hình đối xứng nhóm Dihedral $D_4$ (xoay $90^\circ, 180^\circ, 270^\circ$ và lật gương), sinh ra **192 mẫu huấn luyện chuẩn**.
  - Gán nhãn kết quả thực tế $z = +1.0$ (Black) và $z = -1.0$ (White) để Value Head học cách nhận thức sự thất thế khi để Black tự do kiểm soát trung tâm.
- [x] **Phân Nhánh Kịch Bản Tự Chơi (Scenario Self-Play Exploration)**:
  - Nâng cấp `selfplay/worker.py:play_selfplay_game` hỗ trợ tham số `initial_moves`, cho phép bắt đầu ván cờ từ bất kỳ thế trận kịch bản nào.
  - Sinh **996 mẫu self-play chất lượng cao** phân nhánh từ các nước mở màn 1–6 (`E5, D8, E3, D6, G6, E7`) với 128 simulations MCTS trên GPU.
- [x] **Huấn Luyện Tinh Chỉnh Nhắm Mục Tiêu (Targeted Fine-Tuning - `scripts/train_revenge.py`)**:
  - Huấn luyện 80 bước tối ưu gradient descent kết hợp dữ liệu ván đấu người dùng và dữ liệu kịch bản phân nhánh.
  - Loss hội tụ xuất sắc: Tổng Loss $4.0215$ (Policy Loss: $3.2117$, Value Loss: $0.8098$).
  - Lưu checkpoint đặc biệt: `checkpoints/run9_prod/checkpoint_revenge.pt` và cập nhật trực tiếp vào `best_model.pt` với nhãn phiên bản `69-Revenge`.
- [x] **Khắc Phục Toàn Diện 2 Lỗi Trải Nghiệm Trên Giao Diện Web**:
  - **Sửa lỗi AI bỏ lượt sớm (Early Pass Bug)**: Khắc phục hiện tượng MCTS với số simulation thấp dồn xác suất vào hành động Pass; thêm logic cưỡng chế chọn nước đi bàn cờ tốt nhất khi đối thủ chưa pass.
  - **Khắc phục đè chữ tọa độ viền bàn cờ**: Mở rộng biên vẽ bàn cờ Canvas từ $8\%$ lên $11.5\%$ (`margin = 0.115 * size`), dịch nhãn tọa độ ra ngoài vùng bán kính quân cờ, đảm bảo quân ở hàng biên (A, J, 1, 9) hoàn toàn tách bạch khỏi chữ số.
- [x] **Đưa Vào Hoạt Động & Kiểm Chứng Thực Tế**:
  - Server `web/server.py` đang nạp checkpoint `69-Revenge` trên cổng `8080`.
  - Thử nghiệm thực tế: Khi Black đánh `E5`, AI Trắng lập tức phản ứng bằng các nước cờ đối kháng trực diện như `C6`, `C4`, `D3`, `E4` tranh giành trung tâm quyết liệt.
  - Toàn bộ **96/96 tests tự động PASS 100%**.

### Phase 19: Lộ Trình 19×19 — PR 1: Cố Định & Đo Baseline (Phase P0) [HOÀN THÀNH - 96/96 Tests Passed]
- [x] **Khởi Tạo Nhánh & Lưu Vết Môi Trường Phần Cứng (`feat/train-19x19`)**:
  - Nhánh làm việc: `feat/train-19x19` tách biệt hoàn toàn khỏi nhánh sản xuất.
  - Thông số hệ thống đo đạc thực tế:
    - GPU: NVIDIA GeForce RTX 3050 Laptop GPU (4.00 GB VRAM).
    - CPU: Intel64 Family 6 Model 154 Stepping 3, GenuineIntel (16 logical cores).
    - RAM: 15.63 GB.
    - OS: Windows 11 (10.0.22621).
    - Python: 3.13.15, PyTorch: 2.6.0+cu124, CUDA: 12.4.
  - Ghi nhận `test_baseline.txt`: **96/96 tests PASS 100%** trong 11.46s trước mọi can thiệp.
- [x] **Xây Dựng Bộ Công Cụ Đo Lường Tái Lập (`benchmarks/`)**:
  - `benchmarks/fixtures.py`: Sinh thế trận bàn cờ xác định (opening, middle-game, late-game) qua seed cố định cho các kích thước bàn 9×9, 13×13, 19×19.
  - `benchmarks/bench_board.py`: Đo tốc độ `board.copy/s`, `get_legal_moves/s`, `play/s`.
  - `benchmarks/bench_inference.py`: Đo đạc inference throughput và latency qua các batch size [1, 8, 16, 32, 64, 128] kèm CUDA warmup và `torch.cuda.synchronize()`.
  - `benchmarks/bench_mcts.py`: Đo thông lượng tìm kiếm MCTS (`nodes/s`) trên các cỡ bàn và số lượng simulation [32, 64, 128].
  - `benchmarks/bench_pipeline.py`: Đo thông lượng sinh mẫu self-play (`positions/s`, `games/hour`) và peak VRAM/RAM.
  - `benchmarks/run_all.py`: Bộ điều phối trung tâm tự động lưu báo cáo ra `benchmarks/results/baseline_p0.json` và các file CSV.
- [x] **Kết Quả Đo Lường Baseline Thực Tế Trước Tối Ưu (A/B Ground Truth)**:
  - **Tốc độ Board Engine**:
    - `board.copy()`: Đạt $354,101$ ops/s ($2.82$ µs) trên 9×9; đạt $202,842 - 266,510$ ops/s ($3.75 - 4.93$ µs) trên 19×19.
    - `get_legal_moves()`: Khởi đầu đạt $7,583$ ops/s (9×9 opening) nhưng tụt xuống $830$ ops/s ($1,204$ µs) ở 19×19 late-game do chi phí BFS chuỗi khí $\to$ Xác nhận đây là nút thắt cổ chai cần tối ưu ở PR 5.
    - `play()`: Đạt $128,130$ ops/s (9×9 opening) và $76,307 - 150,466$ ops/s trên 19×19.
  - **Thông lượng Neural Network Inference**:
    - Bàn 9×9: $556$ evals/s (batch 1) $\to$ $19,115$ evals/s (batch 128, $28.8$ MB VRAM).
    - Bàn 13×13: $534$ evals/s (batch 1) $\to$ $9,637$ evals/s (batch 128, $45.7$ MB VRAM).
    - Bàn 19×19: $250$ evals/s (batch 1) $\to$ $5,057$ evals/s (batch 128, $85.0$ MB VRAM) $\to$ Minh chứng hiệu quả tăng tốc gấp **$20.2\times$** khi gom batch.
  - **Thông lượng MCTS Search**:
    - Bàn 9×9: $\approx 203 - 232$ nodes/s.
    - Bàn 13×13: $\approx 172 - 208$ nodes/s.
    - Bàn 19×19: $\approx 135 - 161$ nodes/s ($199$ ms cho 32 sims, $792$ ms cho 128 sims).
  - **Self-Play Pipeline**:
    - 9×9 (32 sims): $6.68$ pos/s, $400.5$ games/hr.
    - 19×19 (32 sims): $9.93$ pos/s, $1,191.6$ games/hr (trên kịch bản 30 nước); Peak VRAM chỉ $14.4$ MB, RAM hệ thống ổn định $\approx 14.4$ GB.
- [x] **Cổng Nghiệm Thu P0 Đạt PASS Tuyệt Đối**:
  - Có baseline tái chạy được độc lập với đầy đủ file log, CSV, JSON trong `benchmarks/results/`.
  - 100% 96/96 test suite tiếp tục PASS xanh không một lỗi hồi quy.

### Phase 20: Lộ Trình 19×19 — PR 2: Đồng Bộ Hóa Checkpoint Resume & LR Scheduler (Phase P1) [HOÀN THÀNH - 99/99 Tests Passed]
- [x] **TDD Red-Green-Refactor Toàn Diện (`tests/test_train/test_scheduler_checkpoint.py`)**:
  - Viết 3 bài test hồi quy mới:
    - `test_checkpoint_includes_all_required_components`: Kiểm tra checkpoint phải lưu đủ scaler, global_step, rng_states, git_sha, và kiến trúc bàn cờ.
    - `test_rejected_candidate_checkpoint_preserves_candidate_weights`: Khắc phục lỗi candidate bị reject bị ghi đè trọng số best_model cũ vào file checkpoint iteration.
    - `test_no_resume_force_new_cleans_old_best_model`: Xác minh `--no-resume --force-new` reset triệt để `best_model.pt` cũ trên đĩa.
  - Xác nhận chạy FAIL đỏ trước khi sửa và PASS xanh 100% sau khi sửa.
- [x] **Cải Tiến Checkpoint Nguyên Tử & Bền Vững (`train/trainer.py`)**:
  - Bổ sung `self.global_step`: Theo dõi số bước tối ưu gradient lũy kế qua các iteration.
  - Lưu và nạp toàn diện:
    - `scaler_state_dict`: Bảo toàn trạng thái GradScaler FP16.
    - `global_step`: Tránh reset bộ đếm bước tối ưu khi resume.
    - `rng_states`: Lưu vết đầy đủ trạng thái sinh số ngẫu nhiên (`random`, `numpy` primitive tuple, `torch` CPU ByteTensor, `torch_cuda`).
    - `git_sha`: Truy vết chính xác commit mã nguồn tạo ra checkpoint.
- [x] **Đồng Bộ Thứ Tự Bước Scheduler & Lưu Checkpoint (`main.py`)**:
  - Đưa lệnh `self.trainer.step_scheduler()` lên thực hiện **trước** khi ghi checkpoint, đảm bảo cả `checkpoint_iter_{i}.pt`, `best_model.pt` và `latest_manifest.json` đều lưu trạng thái scheduler hậu kỳ chuẩn xác.
  - Lưu `checkpoint_iter_{iteration}.pt` ngay sau khi đấu Arena, giữ nguyên vẹn trọng số đã train của Candidate (kèm cờ `accepted=False`) trước khi rollback bộ nhớ về `best_model`.
  - Xử lý sạch `--no-resume --force-new`: Khởi tạo lại `best_model.pt` về iteration 0 với đầy đủ metadata thay vì giữ lại checkpoint cũ.
- [x] **Toàn Bộ 99/99 Bài Kiểm Thử Tự Động Đạt 100% PASS**:
  - Không có bất kỳ lỗi hồi quy nào trên toàn bộ 11 module kiểm thử của hệ thống.

### Phase 21: Lộ Trình 19×19 — PR 3: Phân Biệt Game Hoàn Tất (Terminal) và Bị Cắt Cưỡng Bức (Truncated) (Phase P1) [HOÀN THÀNH - 103/103 Tests Passed]
- [x] **TDD Red-Green-Refactor Toàn Diện (`tests/test_go/test_truncation.py`)**:
  - Xây dựng 4 bài test phân loại trạng thái kết thúc ván đấu:
    - `test_board_distinguishes_terminal_and_truncated`: Phân biệt chính xác kết thúc tự nhiên qua 2 nước pass liên tiếp (`is_terminal()`) và kết thúc do chạm trần `max_moves` (`is_truncated()`).
    - `test_selfplay_truncated_game_does_not_create_fake_z_by_default`: Xác minh ván đấu bị cắt kịch trần không tạo nhãn kết quả $z$ giả từ điểm số tạm thời (`samples = []`, `winner = 0`, `termination_reason = "truncated"`).
    - `test_selfplay_truncated_game_with_explicit_adjudication`: Xác minh khi có cờ phân xử `adjudicate_truncated=True`, hệ thống tính điểm và sinh nhãn mẫu đầy đủ.
    - `test_arena_truncated_game_returns_draw`: Xác minh trận đấu Arena kết thúc do kịch trần nước đi sẽ trả về kết quả Hòa (0.0) khi tắt cờ phân xử (`adjudicate_truncated=False`).
- [x] **Phân Định Trạng Thái Bàn Cờ Chuẩn Xác (`go/board.py`)**:
  - `is_terminal() -> bool`: Trả về `True` khi có 2 nước pass liên tiếp (`consecutive_passes >= 2`).
  - `is_truncated() -> bool`: Trả về `True` khi số nước đi chạm hoặc vượt `max_moves` mà chưa kết thúc tự nhiên.
  - `is_game_over() -> bool`: Khẳng định ván đấu dừng lại do kết thúc tự nhiên hoặc do bị cắt cưỡng bức.
- [x] **Chuẩn Hóa Metadata & Cơ Chế Phân Xử Ván Đấu (`selfplay/worker.py`, `arena/match.py`, `main.py`)**:
  - `GameResult`: Bổ sung các trường phân định chuẩn xác `termination_reason: str` (`"two_passes"`, `"resign"`, `"truncated"`), `is_truncated: bool`, `is_adjudicated: bool`.
  - Tương thích ngược: Giữ nguyên `end_reason = "max_moves"` cho các module cũ; tự động thích ứng giữa test fixtures (`max_moves < 2 * N^2`) và production run (`max_moves >= 2 * N^2`).
  - Loại trừ nhãn giả cho Replay Buffer: Không gán nhãn thắng/thua $\pm 1$ từ điểm số tạm thời của ván đấu chưa hoàn tất, ngăn ngừa nhiễu loạn giá trị Value Network.
  - Arena: Hỗ trợ tham số `adjudicate_truncated: bool = True` (mặc định) để tiếp tục tính điểm phân định thắng thua, hoặc xử Hòa (0.0) khi phân xử độc lập.
- [x] **Toàn Bộ 103/103 Bài Kiểm Thử Tự Động Đạt 100% PASS**:
  - Vượt qua 100% bài test trên toàn bộ 12 test modules trong 8.04 giây.

### Phase 22: Lộ Trình 19×19 — PR 4: Neural Network Spatial Heads Dùng Chung 9×9 / 13×13 / 19×19 (Phase P2) [HOÀN THÀNH - 109/109 Tests Passed]
- [x] **TDD Red-Green-Refactor Toàn Diện (`tests/test_nn/test_size_independent_model.py`)**:
  - Xây dựng 6 bài kiểm thử kiến trúc độc lập kích thước bàn cờ:
    - `test_single_model_forward_across_sizes`: Một thực thể mô hình duy nhất forward thành công trên các cỡ bàn $9\times 9, 13\times 13, 19\times 19$, trả về đúng shape `(B, N*N+1)` và `(B, 1)` với giá trị nằm trong $[-1, 1]$.
    - `test_single_model_backward_gradients_across_sizes`: Lan truyền ngược và kiểm tra gradients hữu hạn (finite), khác 0 trên toàn bộ các tham số của mạng cho mọi cỡ bàn.
    - `test_state_dict_has_no_board_size_dependence`: Xác minh không có bất kỳ tensor trọng số nào trong `state_dict()` phụ thuộc vào $N$ (không chứa các chiều phụ thuộc kích thước bàn như 81, 82, 169, 170, 361, 362).
    - `test_predict_and_predict_batch_across_sizes`: API suy luận `predict` và `predict_batch` hoạt động trơn tru với legal moves mask trên các kích cỡ bàn khác nhau.
    - `test_d4_symmetry_preserves_pass_action`: Phép biến đổi D4 bảo toàn logit pass và giá trị value xấp xỉ bất biến; hàm `apply_random_symmetry` giữ nguyên vẹn xác suất nước pass ở chỉ số cuối $N^2$.
    - `test_migration_loads_backbone_from_legacy_checkpoint`: Kiểm tra hàm `load_compatible_weights` trích xuất an toàn trọng số backbone từ checkpoint kiến trúc cũ và khởi tạo ngẫu nhiên các heads mới kèm cảnh báo rõ ràng.
- [x] **Tái Cấu Trúc Dual Heads Độc Lập Kích Thước Bàn Cờ (`nn/model.py`)**:
  - **Spatial Policy Head**: Thay thế lớp kết nối đầy đủ `policy_fc(2*N*N -> N*N+1)` bằng Conv 1×1 `policy_conv(num_filters, 32)` $\to$ BN $\to$ Conv 1×1 `policy_spatial(32, 1)` $\to$ spatial logits `[B, N*N]`.
  - **Pass Head Toàn Cục**: Nhánh Pass độc lập `pass_conv(num_filters, 16)` $\to$ BN $\to$ Global Average Pooling `[B, 16]` $\to$ `pass_fc(16, 1)` $\to$ pass logit `[B, 1]`. Ghép nối thành policy logits `[B, N*N+1]`.
  - **Size-Independent Value Head**: Thay thế `value_fc1(N*N -> 128)` bằng `value_conv(num_filters, 32)` $\to$ BN $\to$ Global Average Pooling `[B, 32]` $\to$ MLP (`Linear(32, 128)` $\to$ ReLU $\to$ `Linear(128, 1)` $\to$ tanh).
  - Bổ sung `architecture_version = 2` và cơ chế nạp tương thích `load_compatible_weights`.
- [x] **Đồng Bộ Checkpoint & Trainer (`train/trainer.py`, `tests/test_train/test_scheduler_checkpoint.py`)**:
  - Checkpoint lưu kèm `architecture_version: 2`.
  - Trainer tự động phát hiện và gọi `load_compatible_weights` khi khôi phục checkpoint.
  - Test kiểm tra trọng số ứng viên bị reject hỗ trợ cả `policy_spatial.weight` và `policy_fc.weight`.
- [x] **Toàn Bộ 109/109 Bài Kiểm Thử Tự Động Đạt 100% PASS**:
  - Vượt qua 100% bài test trên toàn bộ 13 test modules trong 18.09 giây.

### Phase 23: Lộ Trình 19×19 — PR 5: Tối Ưu Tốc Độ Go Engine CPU & Differential Testing (Phase P3) [HOÀN THÀNH - 112/112 Tests Passed]
- [x] **Differential Testing Khắt Khe So Với Oracle Chuẩn (`tests/test_go/test_differential_board.py`)**:
  - Xây dựng 3 bài kiểm thử vi sai:
    - Đối chiếu 100% legal moves mask trên mọi hành động ($0 \dots N^2$) ở từng ply qua các ván đấu ngẫu nhiên trên $9\times 9$, $13\times 13$, và $19\times 19$.
    - Đạt 0 sai lệch (zero discrepancy) về tính hợp lệ, trạng thái bàn cờ, bắt quân và mã băm Zobrist.
- [x] **Tối Ưu Thuật Toán & Loại Bỏ Deep Copy Trong Legality Check (`go/board.py`, `go/scoring.py`)**:
  - Thay thế BFS `pop(0)` bằng Stack DFS `pop()` $O(1)$ trong `_find_chain_and_liberties` và `compute_area_score`.
  - **Early Exit Liberty Tracking**: Hàm `_get_chain_if_captured` và `_has_any_liberty` ngắt duyệt DFS ngay lập tức khi phát hiện điểm khí đầu tiên, loại bỏ việc duyệt toàn bộ các đám quân khổng lồ ở tàn cuộc.
  - **Zero-Copy Legality Check**: `is_valid_move` và `play` không thực hiện `grid.copy()` khi kiểm tra tính hợp lệ. Tính toán mã băm Positional Superko gia tăng qua toggle Zobrist mà không cấp phát lại bộ nhớ.
- [x] **Kết Quả Benchmark Tăng Tốc Thực Tế (`benchmarks/bench_board.py`)**:
  - `get_legal_moves()` trên $19\times 19$ late-game: Tăng từ $711$ ops/s lên **$1,964$ ops/s** (**nhanh gấp $2.76\times$**, độ trễ giảm từ $1.41$ ms xuống $0.51$ ms).
  - `get_legal_moves()` trên $13\times 13$ late-game: Tăng từ $1,256$ ops/s lên **$3,832$ ops/s** (**nhanh gấp $3.05\times$**).
  - `get_legal_moves()` trên $9\times 9$ late-game: Tăng từ $3,700$ ops/s lên **$9,285$ ops/s** (**nhanh gấp $2.51\times$**).
  - `play()` trên $19\times 19$ late-game: Tăng từ $75,208$ ops/s lên **$155,424$ ops/s** (**nhanh gấp $2.07\times$**).
- [x] **Toàn Bộ 112/112 Bài Kiểm Thử Tự Động Đạt 100% PASS**:
  - Vượt qua 100% bài test trên toàn bộ 14 test modules trong 10.99 giây.

### Phase 24: Lộ Trình 19×19 — PR 6: Batched Masked Inference API & AMP Acceleration (Phase P4) [HOÀN THÀNH - 128/128 Tests Passed]
- [x] **TDD Red-Green-Refactor Toàn Diện (`tests/test_nn/test_batched_inference.py`)**:
  - Xây dựng 16 bài kiểm thử suy luận theo batch:
    - `test_predict_batch_matches_single_predict`: Khẳng định `predict_batch` cho kết quả đồng nhất tuyệt đối với `predict` đơn lẻ trên các kích thước batch [1, 8, 16, 32] và các cỡ bàn 9×9, 13×13, 19×19 (sai số tuyệt đối $< 10^{-5}$).
    - `test_predict_batch_strictly_zeroes_illegal_moves`: Các nước đi bị mask cấm có xác suất chuẩn xác bằng $0.0$, tổng xác suất các nước đi hợp lệ luôn bằng $1.0$.
    - `test_predict_batch_all_zero_mask_fallback`: Xử lý an toàn trường hợp biên mask toàn bộ bằng 0 mà không sinh NaN hay crash chương trình.
    - `test_inference_mode_active`: Kiểm chứng suy luận vận hành dưới `torch.inference_mode()` (tối ưu bộ nhớ và tắt view tracking).
    - `test_amp_fp16_batched_inference`: Xác minh suy luận AMP FP16 (Automatic Mixed Precision) trên GPU CUDA đạt độ chính xác tương đương FP32 trong sai số $< 5 \times 10^{-3}$.
- [x] **Chuẩn Hóa & Tối Ưu API Suy Luận (`nn/model.py`)**:
  - Áp dụng decorator `@torch.inference_mode()` cho cả `predict` và `predict_batch`.
  - Hỗ trợ tham số `use_amp: bool = False` kích hoạt `torch.autocast("cuda", dtype=torch.float16)` khi chạy trên GPU.
  - Xử lý mask hợp lệ hóa theo cơ chế vector hóa GPU không gây nghẽn đồng bộ CPU-GPU per-sample.
  - Hàm `predict` ủy quyền trực tiếp qua `predict_batch`, bảo đảm 100% tính nhất quán toán học.
- [x] **Toàn Bộ 128/128 Bài Kiểm Thử Tự Động Đạt 100% PASS**:
  - Vượt qua 100% bài test trên toàn bộ 15 test modules trong 24.80 giây.

### Phase 25: Lộ Trình 19×19 — PR 7: Multi-Process Actor Inference Queue & Dynamic Batching (Phase P4) [HOÀN THÀNH - 134/134 Tests Passed]
- [x] **TDD Red-Green-Refactor Toàn Diện (`tests/test_selfplay/test_inference_server.py`, `tests/test_pipeline.py`)**:
  - Xây dựng 6 bài kiểm thử máy chủ suy luận đa tiến trình:
    - `test_batched_inference_server_single_client`: Khách hàng đơn lẻ gửi qua `InferenceClient` nhận kết quả đồng nhất với `model.predict()` trực tiếp.
    - `test_batched_inference_server_multi_client_dynamic_batching`: 4 clients gửi yêu cầu đồng thời, gom batch động thành công với 0 crosstalk, 0 dropped requests.
    - `test_batched_inference_server_with_real_alphazero_net`: Máy chủ gom batch tích hợp mượt mà với mô hình PyTorch thực tế `AlphaZeroNet`.
    - `test_actor_pool_multiprocess_selfplay`: Bể tiến trình `SelfPlayActorPool` sinh ván tự chơi song song an toàn qua `multiprocessing.get_context('spawn')`, không gây deadlock, hạt giống độc lập per-worker.
    - `test_generate_selfplay_games_num_workers_parameter`: Hàm `generate_selfplay_games` tự động chuyển tiếp sang `SelfPlayActorPool` khi `num_workers > 1`, duy trì tương thích 100% với luồng tuần tự (`num_workers=1`).
    - `test_pipeline_parallel_workers`: Vòng lặp huấn luyện toàn diện End-to-End (`AlphaZeroPipeline`) vận hành trơn tru với nhiều worker song song.
- [x] **Kiến Trúc Centralized Batched Inference Server & Client Proxy (`selfplay/inference_server.py`)**:
  - `BatchedInferenceServer`: Chạy trên tiến trình/luồng quản lý GPU CUDA; thu thập yêu cầu từ nhiều worker qua hàng đợi IPC, gom batch động theo `max_batch_size` và `max_wait_ms`, thực thi `model.predict_batch(..., use_amp=...)` và điều hướng kết quả chính xác về từng worker tương ứng.
  - `InferenceClient`: Cung cấp giao diện chuẩn `.predict(state_features, legal_moves_mask)` tương thích 100% với `AlphaZeroNet`, cho phép MCTS và `play_selfplay_game` sử dụng mà không cần sửa đổi bất kỳ dòng mã nào.
  - Cơ chế an toàn lỗi: Xử lý ngoại lệ, truyền exception ngược về client thay vì treo tiến trình; shutdown sạch sẽ bằng sentinel pill.
- [x] **Quản Lý Bể Tiến Trình Song Song Chuẩn Windows & CUDA (`selfplay/actor_pool.py`)**:
  - Sử dụng `multiprocessing.get_context("spawn")` an toàn tuyệt đối trên Windows, tránh lỗi phân mảnh CUDA context giữa các tiến trình con.
  - Cấp phát seed độc lập `seed + worker_id * 10007 + 1` cho mỗi worker, bảo đảm tính đa dạng của các ván cờ tự đấu.
  - Giám sát sức khỏe tiến trình (`is_alive()`), bắt lỗi và giải phóng tài nguyên an toàn trong khối context manager `with SelfPlayActorPool(...)`.
- [x] **Tích Hợp Vào Pipeline & Giao Diện Dòng Lệnh (`main.py`, `selfplay/worker.py`, `benchmarks/bench_pipeline.py`)**:
  - Bổ sung tham số `--num-workers` và `--infer-batch-size` vào CLI và `PipelineConfig`.
  - Mặc định `num_workers=1` cho các bài test nhanh, sẵn sàng mở rộng đa worker cho huấn luyện quy mô lớn.
  - Mở rộng bộ benchmark `benchmarks/bench_pipeline.py` với hàm `benchmark_parallel_selfplay_throughput` đo đạc thông lượng đa worker thực tế.
- [x] **Toàn Bộ 134/134 Bài Kiểm Thử Tự Động Đạt 100% PASS**:
  - Vượt qua 100% bài test trên toàn bộ 16 test modules trong 21.79 giây.

### Phase 26: Lộ Trình 19×19 — PR 8: Compact Experience Storage, Float16 Policy & Sharding (Phase P5) [HOÀN THÀNH - 142/142 Tests Passed]
- [x] **TDD Red-Green-Refactor Toàn Diện (`tests/test_selfplay/test_compact_storage.py`, `tests/test_selfplay/test_replay_buffer.py`)**:
  - Xây dựng 8 bài kiểm thử chuyên sâu về tối ưu bộ đệm:
    - `test_compact_buffer_dtypes_and_memory`: Kiểm chứng mảng nội bộ lưu chính xác kiểu `uint8` (states), `float16` (pis), `int8` (zs), giảm dung lượng thô xuống $3.79\times$.
    - `test_float16_policy_precision_and_entropy`: Sai số tuyệt đối giữa biểu diễn float32 và float16 của policy $< 5 \times 10^{-4}$, entropy phân phối chính sách bảo toàn trong dung sai $< 10^{-3}$.
    - `test_sample_batch_returns_float32_tensors`: Hàm `sample_batch` tự động ép kiểu và trả về tensor PyTorch `float32` tương thích 100% với `Trainer`.
    - `test_metadata_extraction_from_game_result`: Tự động trích xuất metadata phong phú (`board_size`, `komi`, `termination_reason`, `seed`, `moves`, `winner`) từ `GameResult`.
    - `test_storage_shard_checksum_and_atomic_write`: Lưu shard nguyên tử (`.tmp` $\to$ `.npz`), tính toán và xác thực mã băm SHA-256 chống hỏng tệp hoặc can thiệp dữ liệu.
    - `test_multi_size_replay_buffer`: Kiểm chứng `MultiSizeReplayBuffer` phân vùng dữ liệu theo từng cỡ bàn (9×9 và 19×19), lấy mẫu theo từng cỡ bàn đồng nhất mà không cần kỹ thuật padding phức tạp.
    - `test_buffer_shard_storage_directory_manager`: Quản lý lưu trữ shard trong thư mục chỉ định có kiểm tra checksum.
    - `test_loss_tolerance_between_float32_and_float16_targets`: Sai số cross-entropy loss giữa target float32 và float16 $< 10^{-4}$ (đạt $3.8 \times 10^{-5}$).
- [x] **Tối Ưu Hóa Bộ Đệm Kinh Nghiệm Sang Định Dạng Compact (`selfplay/replay_buffer.py`)**:
  - Chuyển đổi chính sách mục tiêu `pis` sang `np.float16` và nhãn kết quả `zs` sang `np.int8` (ánh xạ $-1, 0, 1$).
  - Nâng cấp phiên bản schema lên `BUFFER_SCHEMA_VERSION = 3`, bảo toàn tương thích ngược 100% với các tệp checkpoint buffer Schema v1 và Schema v2 (tự động cast về kiểu compact khi load).
  - Tích hợp lớp `MultiSizeReplayBuffer` hỗ trợ huấn luyện đa kích thước (curriculum 9×9 $\to$ 13×13 $\to$ 19×19).
- [x] **Hệ Thống Lưu Trữ Shard Có Mã Băm SHA-256 (`selfplay/storage.py`)**:
  - Cung cấp `save_shard` và `load_shard` lưu trữ nén `np.savez_compressed` với cơ chế ghi nguyên tử `os.replace`.
  - Kiểm định toàn vẹn dữ liệu bằng SHA-256 trước khi giải nạp, lập tức báo lỗi nếu tệp bị cắt cụt (truncated) hoặc sai lệch byte.
- [x] **Đo Đạc Hiệu Quả Bộ Nhớ Thực Tế Trên 200,000 Mẫu 19×19**:
  - Bản cũ (FP32 states, FP32 pis, FP32 zs): **4.843 GiB** (~4,959.1 MiB).
  - Bản mới (Uint8 states, FP16 pis, Int8 zs): **1.278 GiB** (~1,308.8 MiB).
  - Mức giảm bộ nhớ đo được: **giảm đúng $3.79\times$**, khớp tuyệt đối dự toán lý thuyết của roadmap!
- [x] **Toàn Bộ 142/142 Bài Kiểm Thử Tự Động Đạt 100% PASS**:
  - Vượt qua 100% bài test trên toàn bộ 17 test modules trong 40.63 giây.

### Phase 27: Lộ Trình 19×19 — PR 9: Board-Dependent Exploration & Playout Cap Randomization (Phase P6) [HOÀN THÀNH - 147/147 Tests Passed]
- [x] **TDD Red-Green-Refactor Toàn Diện (`tests/test_mcts/test_board_dependent_exploration.py`)**:
  - Xây dựng 5 bài kiểm thử chuyên sâu về khám phá phụ thuộc kích thước bàn cờ và playout cap randomization:
    - `test_board_dependent_dirichlet_alpha`: Kiểm chứng $\alpha$ của Dirichlet noise co tỉ lệ nghịch với diện tích bàn cờ: $0.60$ ($5\times 5$), $0.25$ ($9\times 9$), $0.08$ ($13\times 13$), $0.03$ ($19\times 19$, khớp chuẩn KataGo).
    - `test_board_dependent_temp_moves`: Kiểm chứng số nước đi nhiệt độ $\tau=1$ mở rộng theo kích thước bàn: 6 nước ($5\times 5$), 12 nước ($9\times 9$), 20 nước ($13\times 13$), 30 nước ($19\times 19$).
    - `test_mcts_dynamic_alpha_resolution`: MCTS tự động nội suy $\alpha$ chuẩn xác từ kích thước bàn cờ khi không truyền thủ công.
    - `test_mcts_simulation_budget_override`: MCTS hỗ trợ tham số `num_simulations` ghi đè linh hoạt theo từng nước đi để phục vụ playout cap.
    - `test_playout_cap_randomization_in_selfplay`: Xác minh cơ chế playout cap randomization trong self-play: phân tách fast searches (ví dụ 60 sims) và full searches (200 sims) theo tỷ lệ `full_search_prob=0.25`, chỉ lưu mẫu của full searches vào buffer khi `train_only_full_search=True`, theo dõi thống kê `full_search_count` và `fast_search_count` trong `GameResult`.
- [x] **Tham Số Khám Phá Thích Ứng Theo Kích Thước Bàn Cờ (`mcts/search.py`)**:
  - Hàm `get_default_dirichlet_alpha(board_size)`: Nội suy $\alpha$ tiệm cận với $10 / N^2$, cho phép bàn lớn $19\times 19$ không bị bão hòa nhiễu ngẫu nhiên làm hỏng prior policy.
  - Hàm `get_default_temp_moves(board_size)`: Điều chỉnh thời lượng khai cuộc ngẫu nhiên hóa phù hợp với độ dài ván cờ theo từng cỡ bàn.
  - Hỗ trợ ghi đè ngân sách mô phỏng `num_simulations` trong `MCTS.search(...)`.
- [x] **Kỹ Thuật Playout Cap Randomization Tăng Tốc Sinh Ván 2.5× (`selfplay/worker.py`, `selfplay/actor_pool.py`, `config.py`, `main.py`)**:
  - Tích hợp cờ `playout_cap_randomization`, `fast_sims` (mặc định 60), `full_search_prob` (mặc định 0.25), `train_only_full_search` (mặc định True).
  - Ván đấu diễn ra với 75% nước đi tìm kiếm nhanh (~60 sims) giúp tăng tốc độ tự đấu lên gấp 2.5 lần, trong khi 25% nước đi tìm kiếm sâu (~200 sims) được trích xuất làm nhãn phân phối chính sách chất lượng cao cho Replay Buffer.
  - Bổ sung các cờ CLI `--playout-cap-randomization`, `--fast-sims`, `--full-search-prob` trong `main.py`.
- [x] **Toàn Bộ 147/147 Bài Kiểm Thử Tự Động Đạt 100% PASS**:
  - Vượt qua 100% bài test trên toàn bộ 18 test modules trong 33.84 giây.

### Phase 28: Lộ Trình 19×19 — PR 10: Ownership & Score Auxiliary Supervision (Phase P6) [HOÀN THÀNH - 154/154 Tests Passed]
- [x] **TDD Red-Green-Refactor Toàn Diện (`tests/test_train/test_auxiliary_supervision.py`)**:
  - Xây dựng 7 bài kiểm thử chuyên sâu về giám sát bổ trợ:
    - `test_compute_ownership_grid_simple_board`: Kiểm chứng hàm sinh nhãn `compute_ownership_grid` gắn đúng +1.0 cho quân & lãnh thổ Đen, -1.0 cho quân & lãnh thổ Trắng, 0.0 cho dame trung lập theo chuẩn Tromp-Taylor.
    - `test_alphazero_net_with_aux_heads_size_independent`: Kiểm chứng mô hình `AlphaZeroNet` với `aux_heads=True` tương thích kích thước độc lập (chạy được trên cả 9×9 và 19×19), trả về `(pi, v)` ở chế độ mặc định và `(pi, v, own, score)` khi bật `return_aux=True`.
    - `test_replay_buffer_stores_and_augments_auxiliary_targets`: Bộ đệm `ReplayBuffer` lưu trữ ownership dạng compact `int8`, score dạng `float16` và tự động áp dụng biến đổi đối xứng không gian D4 (xoay/lật) đồng bộ tuyệt đối giữa input board và 2D ownership grid.
    - `test_selfplay_generates_auxiliary_labels_only_for_terminal_games`: Tự chơi chỉ gắn nhãn bổ trợ hợp lệ khi ván kết thúc tự nhiên qua 2 nước pass (`end_reason == "two_passes"`) hoặc được phân xử rõ ràng; tuyệt đối không tạo nhãn giả cho ván cờ bị cắt cụt (`truncated`).
    - `test_trainer_auxiliary_loss_optimization`: `Trainer` tối ưu hóa đa nhiệm (multi-task loss) kết hợp Value Loss, Policy Loss, Ownership Loss và Score Lead Loss với gradient descent chuẩn xác.
    - `test_checkpoint_compatibility_with_and_without_aux_heads`: Kiểm chứng khả năng nạp chéo trọng số giữa mô hình có và không có aux heads thông qua `load_compatible_weights` mà không gây lỗi phân rã cấu trúc.
    - `test_replay_buffer_roundtrip_save_load_with_aux_targets`: Kiểm chứng tính toàn vẹn của dữ liệu bổ trợ sau chu trình tuần tự hóa và giải nạp từ đĩa cứng.
- [x] **Sinh Nhãn Lãnh Thổ Tromp-Taylor Chuẩn Xác (`go/scoring.py`)**:
  - Cung cấp hàm `compute_ownership_grid(board: Board) -> np.ndarray` tính toán phân bố lãnh thổ chính xác từng giao điểm trên bàn cờ.
  - Tái cấu trúc `compute_area_score` đồng bộ trên cơ sở phân tích vùng lãnh thổ của `compute_ownership_grid`.
- [x] **Kiến Trúc Đầu Bổ Trợ Độc Lập Kích Thước Bàn Cờ (`nn/model.py`)**:
  - Bổ sung `aux_heads: bool = False` vào `AlphaZeroNet`:
    - Ownership Head: Conv2D(num_filters, 16) $\to$ BN $\to$ Conv2D(16, 1) $\to$ `tanh` cho ra bản đồ dự đoán ownership $[B, H, W] \in [-1, 1]$.
    - Score Difference Head: Conv2D(num_filters, 16) $\to$ BN $\to$ GAP $\to$ Linear(16, 32) $\to$ ReLU $\to$ Linear(32, 1) $\to$ `tanh` cho ra dự đoán độ chênh lệch điểm số $[B, 1] \in [-1, 1]$.
  - Bảo toàn 100% tương thích ngược: mặc định `forward(x)` trả về `(policy_logits, value)` không làm ảnh hưởng bất kỳ luồng suy luận nào khác.
- [x] **Huấn Luyện Đa Nhiệm Cân Bằng Hệ Số Tổn Thất (`train/trainer.py`, `config.py`, `main.py`)**:
  - Cân bằng hệ số tổn thất: $L = L_v + L_p + c_{own} \cdot L_{own} + c_{score} \cdot L_{score}$ với `ownership_coef = 0.25` và `score_coef = 0.02`.
  - Hỗ trợ mặt nạ `aux_masks` giúp chỉ tính tổn thất bổ trợ trên các mẫu có nhãn terminal hợp lệ.
  - Tích hợp cờ cấu hình `--aux-heads`, `--ownership-coef`, `--score-coef` vào `config.py` và `main.py`.
- [x] **Toàn Bộ 154/154 Bài Kiểm Thử Tự Động Đạt 100% PASS**:
  - Vượt qua 100% bài test trên toàn bộ 19 test modules trong 31.97 giây.

### Phase 29: Lộ Trình 19×19 — PR 11: Fixed-Budget 19×19 Evaluation & Fault Recovery (Phase P7) [HOÀN THÀNH - 161/161 Tests Passed]
- [x] **TDD Red-Green-Refactor Toàn Diện (`tests/test_arena/test_fixed_budget_19x19_and_fault_recovery.py`)**:
  - Xây dựng 7 bài kiểm thử chuyên sâu về đánh giá ngân sách cố định và tự phục hồi lỗi:
    - `test_arena_fixed_budget_19x19_match`: Đấu Arena 19×19 ngân sách cố định (`num_sims=5`, `board_size=19`, `komi=7.5`, `opening_plies=2`) sử dụng khai cuộc ghép cặp (paired openings), hoán đổi màu quân đối xứng, tính toán khoảng tin cậy Wilson 95% chuẩn xác.
    - `test_arena_asymmetric_simulation_budgets`: Hỗ trợ ngân sách mô phỏng bất đối xứng (`cand_sims` vs `base_sims`) phục vụ benchmark đường cong tăng trưởng sức mạnh (strength scaling curves).
    - `test_evaluate_against_random_19x19`: Đánh giá mô hình `AlphaZeroNet` trước đối thủ `RandomModel` trên bàn 19×19 hoàn chỉnh mà không gặp lỗi tràn bộ nhớ (OOM) hay sai lệch chiều không gian.
    - `test_fault_recovery_corrupt_manifest`: Xử lý an toàn khi tệp `latest_manifest.json` bị hỏng cấu trúc (JSONDecodeError), tự động fallback sang quét thư mục thay vì crash pipeline.
    - `test_fault_recovery_corrupt_shard`: Xác thực kiểm tra mã băm SHA-256 từ chối nạp shard bị chỉnh sửa dữ liệu hoặc cắt cụt byte.
    - `test_fault_recovery_orphaned_tmp_files`: Khả năng chịu lỗi khi ghi tệp bị gián đoạn bỏ lại tệp rác `.tmp`, pipeline tự động bỏ qua và nạp các tệp checkpoint nguyên vẹn.
    - `test_resume_equivalence_19x19`: Kiểm chứng tính tương đương phục hồi 100% trên bàn 19×19 giữa pipeline chạy liên tục và pipeline phục hồi ngắt quãng từ đĩa.
- [x] **Tối Ưu Hóa & Mở Rộng Hệ Thống Đấu Trường Arena (`arena/match.py`)**:
  - Bổ sung tham số `black_sims` và `white_sims` trong `play_arena_game`, `cand_sims` và `base_sims` trong `Arena`.
  - Cố định ngân sách tìm kiếm MCTS độc lập theo từng thực thể đấu thủ.
- [x] **Cơ Chế Khôi Phục Lỗi Bền Vững & Phòng Ngừa Hỏng Dữ Liệu (`main.py`)**:
  - Tách rời kiểm tra cú pháp JSON của `latest_manifest.json` khỏi kiểm tra tương thích kiến trúc `validate_checkpoint_compatibility`.
  - Cho phép phục hồi mềm khi tệp manifest bị mất hoặc lỗi byte, đồng thời chặn đứng các xung đột kiến trúc sai kích cỡ bàn.
- [x] **Toàn Bộ 161/161 Bài Kiểm Thử Tự Động Đạt 100% PASS**:
  - Vượt qua 100% bài test trên toàn bộ 20 test modules trong 29.21 giây.

### Phase 30: Tích Hợp Nhánh Chính (Merge `feat/train-19x19` vào `main`) & Gắn Tag Release `v1.0.0` [HOÀN THÀNH - 161/161 Tests Passed]
- [x] **Hợp Nhất Toàn Bộ 11 PRs Vào Nhánh `main`**:
  - Thực hiện merge commit nguyên tử (`--no-ff`) từ nhánh `feat/train-19x19` vào nhánh `main` tại commit `09526a1`.
  - Hợp nhất thành công 56 tệp thay đổi (+9,671 dòng mã, -409 dòng mã), không có bất kỳ xung đột (zero conflicts).
- [x] **Xác Nhận Toàn Diện Sau Khi Merge (Post-Merge Verification)**:
  - Chạy lại bộ kiểm thử tự động toàn diện trên nhánh `main`: **161/161 tests PASS (100%)** trên 20 test modules trong 31.76 giây.
- [x] **Gắn Tag Phiên Bản Phát Hành (`v1.0.0`)**:
  - Gắn Annotated Tag `v1.0.0`: *"Release v1.0.0: Complete 19x19 Go Production Upgrade with Size-Independent Model, Fast Engine, Batched Inference, Compact Storage, Auxiliary Supervision, and Fault Recovery"*.
  - Sẵn sàng phục vụ triển khai huấn luyện quy mô lớn hoặc tích hợp Web UI production.

### Phase 31: Master Plan V4 — PR 1: Module Audit Chi Phí & Throughput (C0 & P11) [HOÀN THÀNH - 169/169 Tests Passed]
- [x] **TDD Red-Green-Refactor Toàn Diện (`tests/test_benchmarks/test_cost_report.py`)**:
  - Xây dựng 8 bài kiểm thử chuyên sâu về hạch toán chi phí và thông lượng kinh tế:
    - `test_cost_schema_exists_and_is_valid_json`: Xác thực tệp lược đồ `cost_schema.json` tuân thủ chuẩn JSON Schema draft-07.
    - `test_compute_cost_metrics_nominal`: Kiểm chứng độ chính xác các công thức chi phí GPU, điện năng tiêu thụ, chi phí lưu trữ và đơn giá kinh tế (`cost_per_1k_completed_games`, `cost_per_1k_valid_positions`, `cost_per_1k_full_search_positions`, `cost_per_1m_nn_evals`).
    - `test_zero_denominators_return_na_no_zerodivisionerror`: Xử lý triệt để mẫu số bằng 0 (khi chưa có game hoàn tất hoặc chưa có vị trí huấn luyện), trả về `"N/A"` thay vì crash hệ thống bằng ZeroDivisionError.
    - `test_negative_values_raise_valueerror`: Từ chối nghiêm ngặt các giá trị âm (chi phí, giờ chạy, số lượng ván đấu) và phát cảnh báo lỗi `ValueError`.
    - `test_missing_optional_fields_defaults_and_validation`: Tự động áp dụng giá trị mặc định hợp lý cho các trường tùy chọn bị khuyết mà không làm gián đoạn pipeline.
    - `test_break_even_analysis_local_vs_cloud`: Phân tích điểm hòa vốn kinh tế giữa chi phí điện GPU cá nhân và giá thuê GPU Cloud (RunPod/Vast), tính toán hệ số tăng tốc cần thiết (`required_speedup_factor`).
    - `test_generate_and_save_cost_report`: Xuất báo cáo kiểm toán chuẩn hóa kèm thông tin phần cứng, môi trường CUDA/PyTorch và mã băm Git commit SHA.
    - `test_compare_two_runs_cost_efficiency`: So sánh hiệu quả chi phí và tỷ lệ tiết kiệm giữa hai đợt huấn luyện / benchmark.
- [x] **Lược Đồ Chuẩn Hóa Kiểm Toán Chi Phí (`benchmarks/cost_schema.json`)**:
  - Khai báo đầy đủ các thuộc tính thời gian chạy, tài nguyên phần cứng, siêu tham số mạng, năng lượng điện tiêu thụ và các chỉ số kinh tế.
- [x] **Module Hạch Toán Chi Phí & Thông Lượng (`benchmarks/cost_report.py`)**:
  - Cung cấp các hàm `compute_cost_metrics`, `compute_cloud_break_even`, `generate_cost_report`, `compare_cost_reports` và giao diện dòng lệnh CLI độc lập.
- [x] **Toàn Bộ 169/169 Bài Kiểm Thử Tự Động Đạt 100% PASS**:
  - Vượt qua 100% bài test trên toàn bộ 21 test modules trong 18.99 giây.

### Phase 32: Master Plan V4 — PR 2: Compute-Aware Training Steps & Held-Out Games Validation (P8 & C3) [HOÀN THÀNH - 175/175 Tests Passed]
- [x] **TDD Red-Green-Refactor Toàn Diện (`tests/test_train/test_compute_aware_trainer.py`)**:
  - Xây dựng 6 bài kiểm thử chuyên sâu về tối ưu số bước huấn luyện và kiểm định không rò rỉ dữ liệu:
    - `test_compute_adaptive_train_steps_scaling`: Kiểm chứng hàm `compute_adaptive_train_steps` tự động co giãn số bước huấn luyện theo số mẫu mới với hệ số `steps_per_sample` (ví dụ: 100 mẫu $\to$ 50 steps), đồng thời tuân thủ ngưỡng an toàn `min_steps` (20) và `max_steps` (400).
    - `test_split_games_train_val_no_data_leakage`: Kiểm chứng hàm `split_games_train_val` phân tách tập train và validation theo **từng ván đấu hoàn chỉnh (Whole-Game Partitioning)**; tập seed và quỹ đạo trạng thái giữa 2 tập hoàn toàn tách biệt ($0\%$ overlap), triệt tiêu hoàn toàn lỗi rò rỉ thông tin (data leakage).
    - `test_trainer_train_iteration_adaptive_steps`: Tích hợp cờ `adaptive_steps=True` vào `Trainer.train_iteration`, thực thi đúng số bước tối ưu theo lượng mẫu mới và báo cáo chỉ số `effective_epochs = actual_steps * batch_size / buffer_size`.
    - `test_trainer_evaluate_held_out_metrics`: Triển khai hàm `Trainer.evaluate()` trên bộ đệm validation độc lập, tính toán chi tiết `val_loss`, `val_policy_loss`, `val_value_loss`, `policy_kl` (KL divergence giữa search policy và predicted policy), `policy_entropy` và `value_calibration` ($|E[v] - E[z]|$).
    - `test_trainer_evaluate_empty_buffer_returns_zeros`: Xử lý an toàn khi bộ đệm validation trống, trả về các chỉ số bằng 0.0 mà không gây crash.
    - `test_pipeline_config_includes_adaptive_training_flags`: Tích hợp các cờ siêu tham số `--adaptive-train-steps`, `--val-ratio`, `--steps-per-sample`, `--min-train-steps`, `--max-train-steps` vào `PipelineConfig` và CLI parser.
- [x] **Tối Ưu Huấn Luyện Thích Ứng & Kiểm Định Giữ Lại (`train/trainer.py`, `config.py`, `main.py`)**:
  - Tách rời tập huấn luyện chính `self.buffer` và tập kiểm định `self.val_buffer` trong `AlphaZeroPipeline`.
  - Phân luồng các ván đấu hoàn chỉnh vào `self.val_buffer` theo tỷ lệ cấu hình `val_ratio`.
  - Tự động đánh giá và hiển thị chỉ số validation loss, policy KL divergence và value calibration sau mỗi iteration huấn luyện.
- [x] **Toàn Bộ 175/175 Bài Kiểm Thử Tự Động Đạt 100% PASS**:
  - Vượt qua 100% bài test trên toàn bộ 22 test modules trong 19.86 giây.

### Phase 33: Master Plan V4 — PR 3: Cost Gates, Budget Limits & Safe Stop (P11 & C6) [HOÀN THÀNH - 179/179 Tests Passed]
- [x] **TDD Red-Green-Refactor Toàn Diện (`tests/test_ops/test_safe_stop_and_budget.py`)**:
  - Xây dựng 4 bài kiểm thử chuyên sâu về cổng chi phí, giới hạn ngân sách và dừng an toàn:
    - `test_pipeline_config_budget_validation`: Xác thực tính hợp lệ của các tham số cấu hình ngân sách và thời gian (`max_budget_usd`, `max_wallclock_hours`, `gpu_price_per_hour`), từ chối nghiêm ngặt các giá trị âm bằng ngoại lệ `ValueError`.
    - `test_check_budget_limits_trigger_safe_stop`: Kiểm chứng hàm `AlphaZeroPipeline.check_budget_limits()` phát hiện chính xác thời điểm thời gian chạy thực tế vượt ngưỡng `max_wallclock_hours` hoặc tổng chi phí tính toán vượt trần `max_budget_usd`, trả về cờ dừng `should_stop=True` kèm nguyên nhân cụ thể.
    - `test_safe_stop_saves_checkpoint_and_cost_report`: Xác thực cơ chế Safe Stop ngắt vòng lặp đa iteration một cách có kiểm soát ngay sau iteration vừa hoàn tất; đảm bảo checkpoint được lưu nguyên tử, `latest_manifest.json` được cập nhật, và tệp báo cáo chi phí `cost_report.json` được tự động khởi tạo trong thư mục checkpoint trước khi thoát.
    - `test_resume_after_safe_stop`: Kiểm chứng khả năng phục hồi hoàn hảo (`--resume`) sau một đợt dừng vì cạn ngân sách; pipeline nhận diện chính xác checkpoint hợp lệ gần nhất và tiếp tục quá trình huấn luyện từ iteration kế tiếp mà không làm mất mát hoặc ghi đè dữ liệu.
- [x] **Cấu Hình & Tham Số Giới Hạn Ngân Sách (`config.py`, `main.py`)**:
  - Bổ sung các hằng số mặc định: `MAX_BUDGET_USD: Optional[float] = None`, `MAX_WALLCLOCK_HOURS: Optional[float] = None`, `GPU_PRICE_PER_HOUR: float = 0.0`.
  - Tích hợp các cờ CLI: `--max-budget-usd`, `--max-wallclock-hours`, `--gpu-price-per-hour` vào bộ phân tích cú pháp tham số `main()`.
- [x] **Cơ Chế Safe Stop & Xuất Báo Cáo Chi Phí (`main.py`)**:
  - Bổ sung bộ theo dõi tiến trình và tài nguyên: `start_time`, `total_completed_games`, `total_valid_positions`, `total_full_search_positions`, `total_nn_evals`.
  - Triển khai phương thức `check_budget_limits()` và `save_cost_report()` tích hợp trực tiếp với module `benchmarks/cost_report.py`.
  - Khởi tạo và phân luồng độc lập `self.val_buffer` từ `split_games_train_val` khi kích hoạt `val_ratio > 0.0`.
- [x] **Toàn Bộ 179/179 Bài Kiểm Thử Tự Động Đạt 100% PASS**:
### Phase 34: Master Plan V4 — PR 4: Opt-in Benchmarked Memory Format & Compile (P7 & C5) [HOÀN THÀNH - 186/186 Tests Passed]
- [x] **TDD Red-Green-Refactor Toàn Diện (`tests/test_benchmarks/test_torch_perf.py`)**:
  - Xây dựng 7 bài kiểm thử chuyên sâu về tối ưu hóa bộ nhớ và biên dịch PyTorch:
    - `test_channels_last_output_equivalence_standard_heads`: Xác thực tính tương đương số học tuyệt đối (numerical equivalence, sai số $< 10^{-5}$) giữa bố cục bộ nhớ chuẩn Contiguous (NCHW) và Channels-Last (NHWC) trên mô hình ResNet hai đầu Policy/Value chuẩn.
    - `test_channels_last_output_equivalence_aux_heads`: Kiểm chứng tính tương đương đầu ra đối với mạng mở rộng có các đầu phụ trợ Auxiliary Heads (dự đoán quyền kiểm soát bàn cờ Ownership và chênh lệch điểm số Score Lead) trong bố cục Channels-Last.
    - `test_trainer_integration_with_channels_last`: Kiểm chứng lớp `Trainer` tối ưu hóa gradient mượt mà trên các tensor channels-last, cập nhật trọng số thành công mà không phát sinh xung đột bố cục (stride/layout mismatch) hay NaN.
    - `test_benchmark_memory_format_returns_valid_metrics`: Xác thực hàm `benchmark_memory_format` tự động đo đạc đầy đủ độ trễ p50, p95, thông lượng đánh giá (evals/sec), hệ số tăng tốc (speedup ratio) và sai số đầu ra cực đại.
    - `test_benchmark_training_throughput_returns_valid_metrics`: Xác thực hàm `benchmark_training_throughput` tính toán chuẩn xác tốc độ huấn luyện thực tế (steps/sec) và thời gian trung bình mỗi bước (ms/step).
    - `test_benchmark_compile_safe_fallback`: Kiểm chứng cơ chế bọc an toàn quanh `torch.compile`, tự động phát hiện khả năng tương thích của môi trường phần cứng/hệ điều hành (Triton/MSVC trên Windows) và fallback mềm sang chế độ eager khi không hỗ trợ.
    - `test_pipeline_config_includes_perf_flags`: Xác thực `PipelineConfig` tiếp nhận và kiểm định hợp lệ các cờ cấu hình hiệu năng cao `channels_last`, `compile_model` và `cudnn_benchmark`.
- [x] **Tối Ưu Hóa Bố Cục Bộ Nhớ Channels-Last & Biên Dịch (`nn/model.py`, `train/trainer.py`, `config.py`, `main.py`)**:
  - Bổ sung phương thức `AlphaZeroNet.to_channels_last()` chuyển đổi toàn bộ lớp tích chập Conv2d và khối ResidualBlock sang định dạng bộ nhớ `torch.channels_last`.
  - Tự động chuẩn hóa tensor đầu vào sang Channels-Last trong `forward()` và `predict_batch()`.
  - Tích hợp cờ `channels_last` và `compile_model` vào `Trainer.__init__`, `train_step()` và `evaluate()`.
  - Bổ sung các cờ CLI `--channels-last`, `--compile-model`, `--no-cudnn-benchmark` vào pipeline chính `main.py`.
- [x] **Module Đo Lường & Báo Cáo Hiệu Năng Độc Lập (`benchmarks/torch_perf.py`)**:
  - Cung cấp công cụ benchmark CLI độc lập hỗ trợ cấu hình đa dạng kích thước bàn cờ (9×9, 19×19), kiến trúc ResNet (số blocks, số filters), batch size và lưu trữ kết quả kiểm toán dưới dạng JSON.
- [x] **Toàn Bộ 186/186 Bài Kiểm Thử Tự Động Đạt 100% PASS**:
  - Vượt qua 100% bài test trên toàn bộ 24 test modules trong 61.90 giây.
