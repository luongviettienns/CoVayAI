# KẾ HOẠCH SỬA CoVayAI ĐỂ SẴN SÀNG TRAIN 9×9

**Repository:** https://github.com/luongviettienns/CoVayAI  
**Mốc mã nguồn đã review:** `main` tại `5f4898c1f256d38cc3c4a06960799ab980c66a5b` (20/10/2026)  
**Phạm vi:** Python / PyTorch AlphaZero — Go 5×5, 9×9.  
**Trạng thái:** Kế hoạch kỹ thuật dựa trên code review tĩnh; **chưa phải chứng nhận dự án không có lỗi hoặc kết quả test chạy độc lập**.

## 0. NGUYÊN TẮC TRIỂN KHAI (BẮT BUỘC)

- [ ] Không sửa trực tiếp `main`. Tạo nhánh: `fix/train-readiness`.
- [ ] Mỗi pha là một PR/commit riêng; không gộp sửa thuật toán, refactor và tối ưu GPU trong một commit.
- [ ] Trước mỗi thay đổi: tạo test tái hiện lỗi (RED) → sửa ít nhất có thể → chạy test (GREEN) → chạy toàn bộ `pytest`.
- [ ] Ghi rõ file thay đổi, lý do, ảnh hưởng tương thích, kết quả test, số liệu trước/sau.
- [ ] Bất kỳ thay đổi nào làm mất khả năng resume checkpoint cũ phải kèm migration, hoặc ghi rõ không tương thích và yêu cầu chạy experiment mới.
- [ ] Không thay đổi `board_size`, `komi`, feature encoding, action indexing, config kiến trúc giữa chừng trong cùng một run/checkpoint.
- [ ] Không nhận kết luận "mạnh hơn" chỉ từ loss giảm hoặc vài trận Arena; benchmark phải có đối thủ cố định, luân phiên màu và đủ số trận.
- [ ] Với toàn bộ P0–P5: ưu tiên tính đúng, giữ API ổn định. P6 tối ưu tốc độ sau khi đã có baseline tin cậy.
- [ ] Mỗi pha chỉ được đánh dấu DONE khi các tiêu chí nghiệm thu đều hoàn thành.

**Quy ước trong tài liệu:** `P0.x` là các task cụ thể; `T...` là test bắt buộc; `DoD` = Definition of Done / Điều kiện qua pha.

---

## P0 — CHỐT BASELINE VÀ CHUẨN HÓA MÔI TRƯỜNG

### P0.1. Đóng băng mã nguồn

- [ ] Clone repo và xác nhận commit HEAD; nếu repo local có thay đổi chưa commit, sao lưu hoặc commit trước khi thao tác.
- [ ] `git switch -c fix/train-readiness` tại baseline (nếu tên nhánh đã có thì dùng tên mới).
- [ ] Lưu kết quả `git rev-parse HEAD` và `git status --short` trong nhật ký sửa lỗi.
- [ ] Bảo đảm thư mục `checkpoints/` của train cũ không bị ghi đè; mỗi thí nghiệm có `checkpoint-dir` riêng.
- [ ] Thêm các file checkpoint tạm, replay buffer, log và artifact train vào `.gitignore` phù hợp.

### P0.2. Xác định môi trường thật

- [ ] Ghi `python --version`, `python -m pip freeze`, phiên bản PyTorch, CUDA, GPU và dung lượng VRAM.
- [ ] Tạo môi trường ảo riêng; chạy lệnh cài PyTorch phù hợp đúng phiên bản Python/GPU từ tài liệu chính thức.
- [ ] Ghi `seed` và cấu hình vào metadata mỗi run; phân biệt GPU README (RTX 3050) với GPU thực tế đang train.
- [ ] Tạo `requirements.txt` hoặc file lock có version kiểm thử được; tách dependency dev (`pytest`, v.v.).

### P0.3. Baseline test + benchmark

- [ ] `python -m pytest -q`: ghi số test PASS/FAIL, không coi số 65/65 trong README là kết quả độc lập.
- [ ] Chạy smoke test 5×5 ở **thư mục checkpoint hoàn toàn mới**.
- [ ] Ghi thời gian/games-per-minute, moves/game, MCTS sims/s, peak RAM/VRAM, tỉ lệ ván kết thúc bằng pass/pass so với chạm `max_moves`.
- [ ] Lưu ít nhất 5 trạng thái bàn cờ định trước cùng kết quả MCTS/policy để làm golden fixtures.

**Lệnh tham khảo trên Windows PowerShell:**

```powershell
git clone https://github.com/luongviettienns/CoVayAI.git
cd CoVayAI
git rev-parse HEAD
git status --short
git switch -c fix/train-readiness
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install --upgrade pip
# Cài PyTorch bản tương thích CUDA/GPU + numpy, pytest, tqdm theo tài liệu môi trường
.\.venv\Scripts\python.exe -m pytest -q
.\.venv\Scripts\python.exe -c "import torch; print(torch.__version__); print(torch.cuda.is_available()); print(torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU')"
```

**DoD P0:** baseline commit và môi trường được ghi nhận; test chạy tái lập; không đụng tới dữ liệu train cũ.

---

## P1 — SỬA TRẠNG THÁI TRAIN, OPTIMIZER, SCHEDULER, CHECKPOINT VÀ RESUME (BLOCKER)

**Mã nguồn liên quan:** `main.py`, `train/trainer.py`, `config.py`, `tests/test_pipeline.py`, `tests/test_train/`.

### P1.1. Sửa lỗi MultiStepLR có thể reset sai sau Arena

**Hiện trạng:** `main.py` chụp `best_scheduler_state` trước khi gọi `self.trainer.step_scheduler()`, và đầu iteration kế tiếp lại restore từ state cũ. Khi candidate bị từ chối/được chấp nhận, các mốc LR có nguy cơ không tiến đúng.

- [ ] Viết test RED `test_scheduler_advances_after_multiple_rejections`: chạy liên tiếp nhiều iteration với Arena mock luôn reject; assert `last_epoch` tăng mỗi iteration, LR giảm đúng milestones cấu hình test ngắn (ví dụ `[2,4]`).
- [ ] Viết `test_scheduler_survives_accept_reject_accept` với chuỗi accept/reject.
- [ ] Tách **tiến độ LR toàn cục** khỏi **đời model best**. Chỉ rollback candidate weights + optimizer moments khi reject; **không rollback bộ đếm global scheduler**.
- [ ] Sau mỗi `step_scheduler()`, đồng bộ LR mới vào optimizer sẽ dùng ở iteration tiếp theo; không cho `optimizer.load_state_dict()` âm thầm khôi phục LR cũ.
- [ ] Lưu optimizer/scheduler state cuối iteration, **sau** khi bước scheduler đã hoàn tất; resume phải khớp LR đúng thời điểm.
- [ ] Nếu giữ `best_optimizer_state`, định nghĩa rõ nó là snapshot moments của last accepted candidate; các param group LR cần được cập nhật theo global schedule khi restore.

**Tiêu chí:** chạy 5 iteration test với các rejection liên tiếp vẫn giảm LR đúng tại milestones; resume giữa các mốc cho kết quả LR như chạy liên tục.

### P1.2. Thiết kế checkpoint không bị "trộn đời"

- [ ] Quy định **một nguồn sự thật** cho resume: `latest_manifest.json` trỏ tới **iteration hoàn chỉnh cuối cùng**.
- [ ] Lưu đủ: `iteration`, best weights/model ID, candidate/optimizer state phù hợp, global scheduler state, `config`, seed và RNG states (`random`, `numpy`, `torch`, CUDA), thông tin phiên bản schema và fingerprint kiến trúc.
- [ ] Giữ `best_model.pt` làm checkpoint model tốt nhất dùng để inference/GTP; **không** dùng nó làm nguồn duy nhất khôi phục trạng thái train.
- [ ] Buffer phải đi cùng mốc iteration: ví dụ `buffer_iter_000012.pkl` tương ứng `train_state_iter_000012.pt`.
- [ ] Chỉ ghi `latest_manifest.json` sau khi **tất cả** file của iteration đã ghi thành công. Ghi vào `.tmp` rồi `os.replace()` trên cùng ổ đĩa để tránh file checkpoint dở dang.
- [ ] Có hàm `validate_checkpoint_compatibility`: `board_size`, `num_filters`, `num_res_blocks`, `input_planes`, action size, schema version, komi và các luật cơ bản.
- [ ] Nếu thiếu/mất file phiên bản mới nhất, resume từ iteration hoàn chỉnh trước đó; không khôi phục buffer mới với optimizer cũ.
- [ ] Không âm thầm bỏ qua lỗi load và tiếp tục train model ngẫu nhiên. Báo lỗi rõ hoặc yêu cầu người dùng chọn `--fresh`.

**Test bắt buộc:**

- [ ] `test_resume_equals_uninterrupted_run` — với seeds cố định, CPU (hoặc kiểm soát RNG), so sánh model weights/LR/optimizer state của chạy liên tục và chạy ngắt rồi resume; các sai số do nondeterminism nếu có phải được ghi nhận.
- [ ] `test_crash_during_buffer_save_recovers_previous_commit`.
- [ ] `test_corrupt_checkpoint_fails_loudly`.
- [ ] `test_resume_rejects_architecture_mismatch`.
- [ ] `test_best_model_is_not_overwritten_on_candidate_rejection`.

### P1.3. Sửa `--no-resume` và quản lý run

**Hiện trạng:** Khi `resume=False`, code tạo model ngẫu nhiên trong RAM nhưng chỉ tạo `best_model.pt` mới nếu file cũ chưa tồn tại — rủi ro trạng thái không đồng nhất.

- [ ] Với `--no-resume`, **mặc định từ chối** chạy khi `checkpoint-dir` chứa artifact cũ. Chỉ cho ghi đè sau một cơ chế explicit `--force-new` + sao lưu rõ ràng (hoặc đơn giản bắt buộc thư mục mới).
- [ ] Khi fresh run, khởi tạo mới đồng bộ best model, candidate model, optimizer, scheduler, buffer, Elo và metadata.
- [ ] Bổ sung `run_id` và `config.json`/`manifest.json` với snapshot siêu tham số thực tế.
- [ ] Test: `test_no_resume_existing_dir_does_not_mix_state`.
- [ ] Test: chạy 5×5 rồi vô tình resume với `--board-size 9` phải báo lỗi có ích thay vì shape mismatch khó hiểu.

### P1.4. Xác thực CLI/config

- [ ] Bổ sung `--seed`, `--resign-threshold`, `--max-moves` nếu muốn điều khiển chúng qua CLI; hiện tại **chưa có những cờ này** trong `main.py`.
- [ ] Xác thực `games >= 1`, `sims >= 1`, `batch_size >= 1`, `train_steps >= 1`, `0 <= resign_disable_frac <= 1`, `arena_games >= 1`, `board_size` hợp lệ, `checkpoint_dir` hợp lệ.
- [ ] Chỉ bật AMP trên CUDA; log rõ thông tin AMP/dtype, LR và tham số mạng.
- [ ] Hạn chế train trên checkpoint dùng seed/config không tương thích; cảnh báo khi thay đổi thông số hợp lệ giữa lần resume.

**DoD P1:** tests accept/reject/resume/crash đều PASS, LR đạt đúng milestones, không ghi đè checkpoint cũ, CLI từ chối cấu hình bất hợp lệ.

---

## P2 — XÁC THỰC GAME RULES VÀ MCTS TRƯỚC KHI TIN DỮ LIỆU SELF-PLAY

**Mã nguồn:** `go/board.py`, `go/scoring.py`, `go/zobrist.py`, `mcts/node.py`, `mcts/search.py`, `nn/features.py`.

### P2.1. Luật cờ và bất biến bàn cờ

- [ ] Unit test: không đặt trên điểm đã có quân, không tự sát, bắt quân 1/nhiều quân, ko/superko, pass/pass kết thúc, max moves kết thúc, không đổi lượt khi hành động bất hợp lệ.
- [ ] Test sau mỗi nước đi hợp lệ: hash tăng dần == `compute_position_hash(grid)`, lịch sử positions nhất quán, current player đúng.
- [ ] Test `Board.copy()` độc lập, giữ toàn bộ lịch sử superko và tối thiểu 8 snapshot feature; không làm thay đổi bản gốc.
- [ ] Không tạo test bằng cách sửa `board.grid` trực tiếp rồi quên đồng bộ hash/lịch sử; nếu cần fixture chuyên dụng, phải có helper tạo trạng thái hợp lệ.
- [ ] Property test 100–1000 ván cờ nhỏ random seed cố định, kiểm tra invariant sau từng nước.
- [ ] Kiểm tra tính điểm ở seki, tranh chấp lãnh thổ và khi hết nước; đối chiếu các bộ thế cờ chuẩn độc lập hoặc engine khác nếu điều kiện cho phép.
- [ ] Xác định rõ việc `max_moves` là **adjudication kỹ thuật**, không phải ván kết thúc tự nhiên.

### P2.2. MCTS sign, terminal, noise và policy

- [ ] `test_puct_prefers_winning_move` và `test_puct_avoids_losing_move` với dummy NN/value deterministic.
- [ ] `test_value_backup_alternates_sign_by_ply`: child Q theo góc nhìn người sắp đi tại child; parent PUCT dùng `-Q_child`.
- [ ] `test_noise_applied_each_new_root_without_compounding`: priors gốc không bị cộng noise lũy kế sau `advance_root`.
- [ ] `test_illegal_actions_never_have_visits` trên cả trạng thái đầy và Ko.
- [ ] `test_policy_targets_sum_to_one_are_finite`; pass luôn giữ action `N²`, D4 không đổi xác suất pass.
- [ ] `test_search_after_two_passes`: không gọi NN/không sinh nước chơi mới nếu terminal.
- [ ] `test_subtree_reuse_same_position` xác minh cây sau `advance_root` tương ứng lịch sử game; nếu `board.play()` trả False phải raise, không lặng lẽ chạy tiếp.

### P2.3. Feature encoding và mô hình

- [ ] Test 17 planes nhất quán, chiều theo người đi, quay/lật policy tương ứng tọa độ board, giá trị chỉ 0/1.
- [ ] Test predict chỉ phân phối xác suất lên nước hợp lệ, output `value ∈ [-1,1]`, không NaN/Inf ở CPU/GPU.
- [ ] Giữ nguyên ResNet 6×96 trong pha này; không thay kiến trúc cùng lúc với sửa pipeline.

**DoD P2:** tất cả invariant hợp lệ, MCTS tactical fixtures đúng, các test rules/MCTS hiện hành không bị regress.

---

## P3 — SELF-PLAY, LABEL, RESIGN VÀ TRUNCATION

**Mã nguồn:** `selfplay/worker.py`, `main.py`, `config.py`, test selfplay mới.

### P3.1. Không resign ở giai đoạn model còn random

- [ ] Pilot khởi đầu sử dụng **`--resign-disable-frac 1.0`** (cờ đã có trong CLI); trong test Arena cũng tắt resign.
- [ ] Sau này mở resign khi có baseline đáng tin; đo false-resign bằng các ván holdout không được resign.
- [ ] Nếu bổ sung `--resign-threshold`, test disable hoàn toàn và ngưỡng thực tế; không dùng kết quả của model random để gán nhãn thua vì resign.

### P3.2. Kết quả ván đấu dạng có metadata

- [ ] Tạo `GameResult` chứa `samples`, `winner`, `black_score`, `white_score`, `moves`, `end_reason` (`two_passes`, `max_moves`, `resign`), `resigned_by`, `seed`, board size và komi.
- [ ] Để tránh làm hỏng code cũ, có thể giữ hàm `play_selfplay_game()` trả `List[Tuple]` và thêm hàm mới trả `GameResult`, hoặc nâng API cùng lúc cập nhật đầy đủ callers/tests.
- [ ] Sửa progress logging hiện tại: `winner_str = 'B' if game_samples[-1][2] > 0 else 'W'` là sai vì `z` đó được tính theo góc nhìn người chơi của **nước cuối**, không phải luôn là quân Đen.
- [ ] Test màu thắng chính xác cả khi nước cuối thuộc bên thua/thắng.
- [ ] Test tất cả `z` cùng một game khớp winner với `player_to_move` được lưu tại thời điểm lấy mẫu.
- [ ] Nếu `board.play(action)` thất bại, raise exception với board state/action để phát hiện lỗi MCTS ngay, không tiếp tục im lặng.

### P3.3. Các ván bị giới hạn `max_moves`

- [ ] Theo dõi **tỉ lệ** `end_reason == max_moves` và độ dài ván; tách khỏi kết thúc tự nhiên.
- [ ] Giữ rõ ràng chính sách hiện tại: dùng `compute_area_score` làm nhãn kỹ thuật khi đến cap nhưng **không coi tương đương hoàn toàn ván kết thúc bình thường**.
- [ ] Nếu nhiều ván chạm giới hạn, điều tra pass policy, scoring, max length và chiến lược adjudication; không tự động loại hết dữ liệu cap vì với model random có thể khiến không còn mẫu train.
- [ ] Test ván 2-pass, resign, max-moves đều có end_reason và nhãn nhất quán.

### P3.4. Kiểm định dữ liệu trước khi push buffer

- [ ] Shape `(17,N,N)`, `(N²+1,)`; dtype đúng; `π >= 0`, finite, `sum(pi)≈1`, `z ∈ {-1,+1}`.
- [ ] Kiểm tra pass index, policy/legal mask và z perspective bằng game fixture.
- [ ] Log `games`, `samples`, `resign_rate`, `max_move_rate`, `avg_game_length`, `mean_policy_entropy`, `black_win_rate`, `white_win_rate`.
- [ ] Với Arena benchmark: không resign mặc định trước khi được calibration.

**DoD P3:** 100% sample hợp lệ, log winner đúng, không false-resign giai đoạn random, biết bao nhiêu game chạm cap.

---

## P4 — REPLAY BUFFER NHANH, ÍT RAM VÀ PERSISTENCE AN TOÀN

**Mã nguồn:** `selfplay/replay_buffer.py`, `nn/features.py`, `tests/test_selfplay/test_replay_buffer.py`.

### P4.1. Sửa random access từ `deque`

**Lý do:** `ReplayBuffer.sample_batch()` dùng `self.buffer[idx]` trên `collections.deque`. Truy cập chỉ số ngẫu nhiên của deque có thể tốn O(n), không phù hợp buffer 200k mẫu.

- [ ] Viết benchmark sampling 2k, 20k, 200k mẫu (trong giới hạn RAM); ghi milliseconds/batch.
- [ ] Đổi sang **ring buffer có random index O(1)**; có thể dùng `list` + write pointer (nhẹ code), hoặc preallocated contiguous NumPy arrays (tốt hơn cho lớn).
- [ ] Với preallocated arrays: `states: (capacity, 17, N, N)` lưu `uint8` vì feature là 0/1; `pis` lưu `float16/float32` tùy kiểm tra sai số; `zs` lưu `int8`. Chỉ cast `states` thành float32 ngay trước train.
- [ ] Nếu tiếp tục lưu `float16` cho policy, test tổng xác suất, renormalize sau cast khi cần; giữ float32 nếu độ chính xác ảnh hưởng nhiều.
- [ ] `sample_batch` phải trả tensor đúng shape/dtype và áp D4 cho từng state-policy cặp theo **cùng** phép biến đổi.
- [ ] Duy trì phân phối lấy mẫu ngẫu nhiên đều, xử lý `batch_size > buffer_len` có replace.
- [ ] Test vòng đời ring buffer (đầy, overwrite, empty, wrap-around), bảo toàn mẫu mới nhất và không đọc vùng chưa ghi.

### P4.2. Schema và migration buffer

- [ ] Thiết kế `buffer_schema_version`, `board_size`, `capacity`, `size`, `write_index` và checksum/fingerprint cơ bản.
- [ ] Nếu muốn đọc `buffer.pkl` cũ, tạo migration **có test**; nếu không, fail rõ và yêu cầu run mới, tuyệt đối không âm thầm bỏ dữ liệu.
- [ ] Snapshot gắn mốc iteration P1; không cho resume buffer lệch mốc.
- [ ] Save atomic `.tmp` → `os.replace`; kiểm tra file và dung lượng trước khi commit manifest.
- [ ] Không nạp pickle từ nguồn không tin cậy.

### P4.3. Profiling sau tối ưu

- [ ] So sánh RAM/VRAM, thời gian sample 1 batch/100 batch với P0.
- [ ] Tất cả test D4, shape, replay persistence PASS.
- [ ] Không thay đổi training-target semantics chỉ để giảm RAM.

**DoD P4:** random-index O(1), buffer không leak RAM, save/resume nguyên vẹn, sampling nhanh hơn baseline và không làm sai dữ liệu.

---

## P5 — CHUẨN HÓA ARENA, GATING VÀ BENCHMARK

**Mã nguồn:** `arena/match.py`, `arena/elo.py`, `arena/baseline.py`, `main.py`.

### P5.1. Ghép cặp trận theo cùng khai cuộc

- [ ] Tách chức năng tạo `opening sequence` có seed cố định khỏi phần thi đấu.
- [ ] Mỗi cặp trận: candidate chơi Đen/baseline Trắng, sau đó đổi màu với **cùng khai cuộc**; xác minh khai cuộc hợp lệ dưới rules cùng komi.
- [ ] Trong phép đo strength chính, tắt resign và Dirichlet noise; sau khai cuộc dùng greedy policy từ visits.
- [ ] Giữ `opening_plies` đa dạng nhưng tránh đánh giá chênh lệch bằng các ván khởi đầu hoàn toàn không tương ứng.
- [ ] Test `test_color_swap_uses_same_opening`; `test_arena_score_is_symmetrical_on_model_swap` trong kiểm soát seed.

### P5.2. Số ván và thống kê

- [ ] Không dùng Arena 4–6 ván để **quyết định** model train tốt hơn; chúng chỉ dành cho smoke test.
- [ ] Với pilot 9×9, dùng 40–100 ván tùy throughput để xem tín hiệu ban đầu; khi chọn checkpoint quan trọng, tăng số trận và tính khoảng tin cậy của win rate.
- [ ] Ngưỡng `55%` là ngưỡng điểm, **không tự bảo đảm khác biệt có ý nghĩa thống kê**. Định nghĩa điều kiện promotion (ví dụ win rate > threshold và interval hoặc sequential test đạt tiêu chuẩn đã chọn).
- [ ] Ghi wins/losses theo màu, số nước, seeds, opponent version, sims, komi, kết quả từng cặp.
- [ ] Không suy diễn `EloTracker` hiện tại thành Elo tuyệt đối; các model/đối thủ không cùng một pool so sánh nên điểm chỉ mang tính tham khảo.

### P5.3. Đối thủ cố định

- [ ] Giữ `RandomModel` làm smoke baseline, không dùng làm thước đo cuối cùng: opponent random hiện chọn cả `pass`.
- [ ] Tạo `fixed_baselines`: random không pass quá sớm, capture heuristic, bản model cũ giữ cố định; nếu khả dụng, đấu GNU Go/KataGo với cấu hình/board size phù hợp.
- [ ] Đánh giá cùng sims/time control/seed, cân bằng màu trên 5×5/9×9.

**DoD P5:** arena đánh cặp màu công bằng, ghi đầy đủ kết quả; promotion có tiêu chí định lượng tin cậy và baseline cố định.

---

## P6 — HIỆU NĂNG GPU VÀ SELF-PLAY (CHỈ SAU KHI ĐÚNG THUẬT TOÁN)

**Mã nguồn:** `mcts/search.py`, `nn/model.py`, `selfplay/worker.py`, `main.py`, `config.py`.

### P6.1. Profiling trước khi tối ưu

- [ ] Ghi NN inference time, board clone time, legal move generation time, MCTS sims/s, games/h, GPU util và peak VRAM với cấu hình cố định.
- [ ] Ghi riêng thời gian self-play / training / arena (đừng chỉ xem total time).
- [ ] Không khai thác `NUM_WORKERS=5`, `INFER_BATCH_SIZE=64` như thể chúng đã hoạt động: hiện self-play tuần tự và NN được gọi theo từng leaf.

### P6.2. Batching inference an toàn

- [ ] Tách `model.predict_batch(states, legal_masks)` trả batch policy/value, giữ `predict` cũ như wrapper tương thích.
- [ ] Test với B=1, B=8, mask các nước bất hợp lệ; kiểm tra output batch xấp xỉ từng inference lẻ khi model ở eval mode.
- [ ] Triển khai inference queue cấp GPU gom leaf từ nhiều game; có timeout/batch cap để tránh treo khi ít yêu cầu.
- [ ] Thay đổi batching sẽ làm đổi thứ tự MCTS trong thực tế; test bằng tactical invariants và thống kê, không buộc visit counts bitwise-identical.
- [ ] Có feature flag `--inference-batching` để tắt nhanh nếu xảy ra regress.

### P6.3. Worker song song trên Windows (làm sau batching)

- [ ] Dùng `multiprocessing` phương thức `spawn` cho Windows và phân phối seed riêng theo worker/game.
- [ ] Không truyền CUDA model/contexts bừa bãi giữa process; ưu tiên worker CPU cho board/MCTS và **một GPU inference server** tập trung.
- [ ] Quy định start/stop, sentinel, timeouts, exception propagation, queue backpressure và graceful shutdown.
- [ ] Test 1 worker = baseline, 2 workers không deadlock, dừng/restart không mất trạng thái commit hoàn chỉnh.
- [ ] Benchmark throughput (games/h) và GPU memory trước/sau; chỉ merge nếu tốc độ tốt lên ổn định và pass hết tests.

**DoD P6:** hiệu năng cải thiện có đo đạc; không mất correctness; có thể quay lại chế độ tuần tự bằng một flag. Nếu batching/parallel quá rủi ro, được phép hoãn P6.2–P6.3 và pilot train tuần tự trước.

---

## P7 — THỬ TRAIN THEO CỔNG NGHIỆM THU

### P7.1. Smoke 5×5, checkpoint mới

```powershell
.\.venv\Scripts\python.exe -m pytest -q
.\.venv\Scripts\python.exe main.py --board-size 5 --iters 2 --games 2 --sims 16 --train-steps 10 --batch-size 32 --arena-games 4 --resign-disable-frac 1.0 --checkpoint-dir checkpoints/smoke5_v1 --no-resume
```

**Lưu ý:** `checkpoints/smoke5_v1` phải mới/tách biệt. `arena-games 4` chỉ test tích hợp, **không** đủ để đo độ mạnh. Nếu triển khai P1 thay đổi cờ `--no-resume`/`--fresh`, cập nhật câu lệnh phù hợp.

- [ ] Có `GameResult`, replay buffer, loss hữu hạn, checkpoint hoàn chỉnh, manifest đồng bộ.
- [ ] Không nước đi bất hợp lệ, crash, NaN/Inf.
- [ ] `python main.py --board-size 5 --iters 3 ... --checkpoint-dir checkpoints/smoke5_v1` (giữ toàn bộ tham số còn lại) phải **chỉ** chạy tiếp iteration 3.
- [ ] Ngắt chủ động giữa lần lưu file; resume từ iteration cuối hoàn chỉnh, dữ liệu không lẫn phiên bản.

### P7.2. Pilot 9×9 (đề xuất, không phải hyperparameter đã tối ưu)

```powershell
.\.venv\Scripts\python.exe main.py --board-size 9 --iters 20 --games 50 --sims 100 --train-steps 200 --batch-size 128 --arena-games 40 --resign-disable-frac 1.0 --checkpoint-dir checkpoints/pilot9_v1 --no-resume
```

- [ ] Kiểm tra sau iteration 1, 3, 5, 10; không mặc định để chạy hết 20 nếu có vấn đề.
- [ ] Log mỗi iteration: LR, policy loss, value loss, selfplay games/moves/time, % max-moves, % pass-pass, black/white win rate, sample entropy, Arena scores, peak VRAM, accepted/rejected.
- [ ] Benchmark với đối thủ cố định trước train và sau các mốc; cùng sims/komi/seed.
- [ ] Stop để điều tra nếu loss NaN/Inf, LR không đúng, nhiều illegal actions, checkpoint không resume, dữ liệu sai shape, tỉ lệ max-moves quá cao hoặc strength giảm liên tục.

### P7.3. Quyết định có train sâu chưa

- [ ] P0–P5 DoD PASS; P6.2–P6.3 không bắt buộc nếu chủ động chọn train tuần tự.
- [ ] Checkpoint có thể resume và đối chiếu được model/optimizer/LR từ đúng iteration.
- [ ] Benchmark trên đối thủ cố định cho thấy mô hình đang học (không dựa độc quyền vào loss).
- [ ] Tốc độ thực tế đủ cho ngân sách GPU/điện; theo dõi RAM/VRAM không tăng không kiểm soát.
- [ ] Tạo run ID và không đổi schema/model giữa các mốc.

---

## THỨ TỰ COMMIT / PR GỢI Ý

1. `test(baseline): capture environment, reproducible smoke and regression fixtures`
2. `fix(train): advance global LR scheduler across accept/reject/resume`
3. `fix(checkpoint): atomic iteration snapshots and strict resume compatibility`
4. `fix(cli): safe fresh runs, validation and explicit seed/resign controls`
5. `test(go-mcts): invariants, tactical PUCT and symmetry properties`
6. `fix(selfplay): reliable game outcome metadata, resign policy, cap metrics`
7. `perf(buffer): O(1) replay sampling and versioned persistence`
8. `test(arena): paired openings, no-resign benchmark and robust promotion`
9. `perf(inference): optional batch prediction and benchmark`
10. `perf(selfplay): optional Windows-safe multiworker inference`
11. `test(e2e): 5x5 smoke, 9x9 pilot, interrupted resume and acceptance report`

**Mỗi PR mô tả:** `Vấn đề`, `Test RED`, `Cách sửa`, `Test GREEN`, `Thay đổi API/schema`, `Benchmark nếu có`, `Rollback`, `Commit SHA`.

---

## BẢNG THEO DÕI NGHIỆM THU

| Pha | Điều kiện PASS tối thiểu | Trạng thái |
| --- | --- | --- |
| P0 | Reproducible env + pytest baseline + commit ghi lại | ☐ |
| P1 | Scheduler không reset; resume/crash/rollback đồng bộ | ☐ |
| P2 | Go rules, MCTS tactical, legal moves, D4 PASS | ☐ |
| P3 | Nhãn z, winner, end_reason chính xác; log chất lượng self-play | ☐ |
| P4 | O(1) random sampling; snapshot có schema; RAM/benchmark PASS | ☐ |
| P5 | Paired openings; color-swap; benchmark có đối thủ cố định | ☐ |
| P6 | Profiling trước/sau; batch/parallel an toàn hoặc ghi rõ defer | ☐ |
| P7 | Smoke 5×5, pilot 9×9 và resume kiểm chứng được | ☐ |

**Chỉ số thực tế sau mỗi pha:** `HEAD SHA`, tổng test PASS/FAIL, các test mới, thay đổi file, lỗi chưa xử lý, run_id và đường dẫn checkpoint thử nghiệm. Không chuyển pha khi có failure không được giải thích.

---

## PROMPT GIAO CHO OPENCODE / CODE AGENT — DÙNG MỖI LẦN 1 PHA

```text
Bạn là senior Python/PyTorch engineer chuyên AlphaZero và RL game engines.
Repo: luongviettienns/CoVayAI, dựa trên branch fix/train-readiness.
Hãy thực hiện DUY NHẤT task [MÃ PHA/TASK] trong KE_HOACH_SUA_COVAYAI_TRAIN_READY.md.

QUY TẮC:
1. Đọc code liên quan và chỉ ra nguyên nhân có bằng chứng.
2. Viết pytest regression tái hiện lỗi trước khi sửa; cho biết test RED thực sự fail vì đúng nguyên nhân.
3. Vá tối thiểu; giữ API/checkpoint tương thích nếu có thể, nếu phá tương thích phải có migration/fail rõ ràng.
4. Không refactor module khác; không tối ưu hiệu năng và đổi kiến trúc model ngoài phạm vi task.
5. Chạy targeted tests, rồi toàn bộ pytest; không được tự nhận PASS nếu chưa chạy.
6. Nếu không có CUDA, test CPU và ghi rõ giới hạn; không tuyên bố CUDA PASS.
7. Báo cáo: file thay đổi, diff quan trọng, lệnh test, kết quả thật, ảnh hưởng performance, điều kiện DoD, rủi ro còn lại.
8. Nếu test fail, dừng và giải thích trước khi triển khai pha tiếp theo.
9. Không xóa/ghi đè checkpoint cũ, không sửa main branch, không tự ý chạy huấn luyện dài.
10. Tạo commit riêng với thông điệp rõ ràng sau khi mọi test phạm vi task PASS.

Hãy bắt đầu từ [MÃ PHA/TASK] và hoàn thành có kiểm chứng.
```

## GHI CHÚ VỀ RỦI RO

- "Không bị lỗi" tuyệt đối là không thể bảo đảm; mục tiêu là **phát hiện lỗi sớm, cô lập thay đổi, chứng minh bằng test và có rollback**.
- Báo cáo 65/65 test là thông tin trong repository; chưa được chạy độc lập khi lập kế hoạch này.
- Nếu mã nguồn thay đổi sau commit `5f4898c1`, cần đọc diff trước và cập nhật các task liên quan.
