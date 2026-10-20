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
- [ ] **Giai đoạn 2 (Iterations 21 $\rightarrow$ 70) [ĐANG CHẠY NỀN]**:
  - Cấu hình: `--sims 64`, `--train-steps 100`, `--batch-size 64`, `--no-gating` (huấn luyện liên tục AlphaZero 2018).
  - Mục tiêu: Rèn luyện khả năng tạo mắt sống độc lập, tính toán sâu các thế vây bắt và tối ưu hóa điểm số lãnh thổ cuối trận (yose).





