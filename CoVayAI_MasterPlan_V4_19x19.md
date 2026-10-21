# CoVayAI MASTER PLAN V4 — TRAIN 19×19 HIỆU NĂNG CAO, CHI PHÍ THẤP

**Ngày lập:** 21/10/2026  
**Repo:** https://github.com/luongviettienns/CoVayAI  
**Mốc khảo sát:** `fix/train-readiness` @ `578f21446e070d44b42ffa4e059239a7219a6a2a`  
**Mục tiêu:** chạy AlphaZero Go 19×19 có kiểm định, giảm *chi phí trên mỗi đơn vị tiến bộ sức chơi*, không chỉ giảm loss/thời gian.

> **CHÚ Ý**: Đây là **kế hoạch**, không phải thông báo các thay đổi đã được code. Tên file mới, các tham số CLI và cấu hình ghi **ĐỀ XUẤT** chưa tồn tại. Những số liệu mục tiêu về hiệu năng là ngưỡng dùng để ra quyết định sau benchmark, KHÔNG phải cam kết tốc độ. `main` không phải nhánh có các thay đổi train-readiness.

---

## A. Quy tắc thực hiện / định nghĩa DONE

1. **Không làm hỏng 9×9**: mọi bước phải chạy được toàn bộ bộ test hiện tại, test 5×5/9×9/19×19 và không sửa checkpoint lịch sử.
2. **Một PR/commit nhỏ cho mỗi hạng mục**. Nếu thay đổi chạy sai, quay lại commit ngay. Không cho AI Agent sửa đa pha trong một lần.
3. **Test tính đúng đắn trước benchmark.** Tăng tốc nhưng đổi nhãn `z`, hợp lệ nước đi, pass, superko hoặc PUCT = FAIL ngay.
4. **Đo cùng điều kiện**: commit, seed, số simulations, model, data positions, GPU, CPU, PyTorch, workers, batch, thời gian warmup phải ghi vào báo cáo.
5. **Không lấy số đo `predict_batch()` làm tốc độ self-play**. Phải đo `positions/hour`, `games/hour`, `MCTS simulations/second`, `NN evals/second`, CPU/GPU utilization và chi phí.
6. **Không dùng game cắt ngắn 20 nước để tạo training labels**. `--max-moves 20` chỉ dùng smoke test shape/runtime.
7. **Không dùng `--no-gating` win_rate = 100% làm bằng chứng model mạnh lên**. Đó là nhánh code gán `win_rate=1.0` khi tự động promote, không phải kết quả thi đấu.
8. **Checkpoint đủ để resume**: model, optimizer, global scheduler, AMP scaler, model/version/hash, Replay Buffer chỉ số ghi, RNG state, training progress, config. Thay đổi schema phải migration hoặc thông báo không tương thích.
9. **Không nhập thư viện ngoài không cần thiết**; ưu tiên Python/PyTorch hiện có. Native backend chỉ làm sau profiler chứng minh có lợi.
10. **GPU 4GB là profile ưu tiên**; máy 8GB và Cloud là profile mở rộng. Không mặc định đủ VRAM trước khi chạy.
11. **Ngân sách đặt trần cho từng thử nghiệm**; không để Cloud/Pipeline tự phát sinh compute không giới hạn.
12. **Nếu phụ thuộc chưa hoàn tất**, giữ nguyên triển khai tuần tự cũ để rollback thay vì chuyển toàn bộ đường thực thi.

### Các số liệu bắt buộc trong `benchmarks/results/*.json`

- `timestamp`, `git_sha`, `branch`, `host_os`, `cpu`, `ram_gb`, `gpu`, `vram_gb`, `torch`, `cuda`, `seed`
- `board_size`, `network_arch`, `num_blocks`, `num_filters`, `mcts_sims`, `num_actors`, `infer_batch`, `train_batch`, `precision`
- `positions_per_hour`, `completed_games_per_hour`, `mcts_sims_per_sec`, `nn_evals_per_sec`, `nn_batch_fill_rate`
- `avg_game_moves`, `pass_rate`, `resign_rate`, `illegal_moves`, `abnormal_end_rate`
- `selfplay_wall_s`, `train_wall_s`, `arena_wall_s`, `total_wall_s`, `gpu_peak_vram_mb`, `ram_peak_mb`, `disk_written_gb`
- `gpu_util_mean`, `cpu_util_mean`, `policy_loss`, `value_loss`, `training_nan_count`, `benchmark_winrate`, `benchmark_ci`
- `cloud_price_per_hour`, `local_power_kwh`, `local_kwh_price`, `cost_per_1000_positions`, `cost_per_1000_games` (nếu có)

**Công thức:** `cost_per_1000_positions = total_cost / valid_training_positions * 1000`; không dùng số position bị loại bởi bộ lọc dữ liệu làm mẫu số. So sánh sức chơi trên **cùng ngân sách compute**, cùng số mô phỏng khi đánh giá.

---

## P0 — ĐÓNG BĂNG, TÁCH NHÁNH, BASELINE 19×19

### Mục tiêu
Có bản gốc phục hồi và benchmark có thể lặp lại, không chỉnh train 9×9 đã có. Không sửa logic thuật toán.

### Thực hiện

- [ ] Kiểm tra `git status` sạch; nếu có code chưa commit, lưu hoặc commit riêng theo quy trình người dùng.
- [ ] `git fetch origin`; checkout `fix/train-readiness`; xác nhận SHA đang dùng.
- [ ] Tạo `feat/19x19-training-v4` từ commit đã kiểm định.
- [ ] Sao lưu `checkpoints/run9_prod/` ra ngoài repo, tính SHA256 các file quan trọng. Tuyệt đối không dùng `--force-new` trên thư mục 9×9.
- [ ] Ghi kết quả `python -m pytest -q`; lưu môi trường từ `pip freeze`.
- [ ] Tạo `benchmarks/profile_19.py` (MỚI) benchmark Board, Features, NN, MCTS, self-play có GPU warmup + synchronize.
- [ ] Benchmark 3 loại thế cờ 19×19 (đầu/giữa/cuối ván) với 3–5 seed; đo median và p95.
- [ ] Chạy shape smoke 19×19 bằng CLI hiện có, dùng thư mục riêng.
- [ ] Benchmark một số ván hoàn chỉnh 19×19 ở simulations thấp vừa đủ đo, riêng biệt với smoke giới hạn nước đi.
- [ ] Ghi CPU%/GPU%/VRAM/RAM và kích thước buffer ước tính.

### Lệnh hiện có (dùng PowerShell, ở thư mục dự án)

```powershell
git status
git fetch origin
git switch fix/train-readiness
git rev-parse HEAD
git switch -c feat/19x19-training-v4
python -m pytest -q
python -m pip freeze > benchmarks_env.txt
python main.py --board-size 19 --iters 1 --games 2 --sims 16 --train-steps 5 --batch-size 8 --arena-games 2 --max-moves 20 --resign-threshold -1.1 --resign-disable-frac 1.0 --checkpoint-dir checkpoints/smoke19_v4 --no-resume
```

> `benchmarks/profile_19.py` chưa có, cần viết sau khi tạo nhánh. Smoke `--max-moves 20` KHÔNG dùng để suy ra chất lượng học hoặc làm training set chính thức. Đối với benchmark full-game dùng `--max-moves` mặc định phù hợp 19×19 và batch/sims vừa sức GPU.

### Cửa nghiệm thu

- [ ] `pytest` PASS; smoke PASS, không illegal moves/NaN/OOM.
- [ ] Có `base19.json` với các trường trên, nguồn đo đúng 19×19, ít nhất 3 lần đo có warmup.
- [ ] Chưa ghi đè/bỏ mất checkpoint 9×9.

**Commit:** `chore(bench): establish reproducible 19x19 baseline`

---

## P1 — CHỨNG MINH LUẬT CỜ & LABEL 19×19 ĐÚNG

### Files
`go/board.py`, `go/scoring.py`, `go/zobrist.py`, `nn/features.py`, `mcts/search.py`, `tests/test_go/test_19x19_rules.py` (MỚI), `tests/test_19_invariants.py` (MỚI).

### Thực hiện

- [ ] Test `pass_action == 361`, `policy_size == 362`, tọa độ GTP đến T19 bỏ I.
- [ ] Kiểm tra bất biến hash Zobrist incremental == recompute sau mọi nước trong 100 ván random 19×19 với seed cố định.
- [ ] Test capture, ko/positional-superko (bao gồm pass), suicide, 2-pass, trạng thái terminal, giới hạn max moves.
- [ ] Cross-check score/komi/territory với bộ thế cờ đã biết; kiểm tra seki, chết quân, biên/territory và bài toán dead-stones. Quyết định rõ luật scoring cho terminal và ghi vào tài liệu, tránh label `z` sai.
- [ ] Đánh giá rủi ro `max_moves = 2*N*N`: game vượt giới hạn không phải kết thúc tự nhiên; log `end_reason=max_moves`, không tự coi label chất lượng như ván kết thúc chuẩn. Có chính sách chấm điểm/adjudication rõ ràng.
- [ ] MCTS: kiểm tra PUCT sign, root/child value perspective, prior mask, backup alternating, reuse, noise sau root reuse.
- [ ] Test kiểm tra `board.play()` luôn True ở mỗi bước MCTS, worker, arena; nếu sai phát hiện và FAIL ngay.
- [ ] Test `Board.copy()` độc lập đầy đủ `history_positions` dù `history_grids` chỉ lưu 8 state gần nhất.
- [ ] Test property/random nhiều board sizes 5, 9, 13, 19; regression snapshot.

### Cửa nghiệm thu
- [ ] Zero illegal moves/duplicate-position superko trong bộ property tests.
- [ ] Không có disagreement về rule/scoring với tập fixture đã duyệt. Nếu còn khác biệt, **không train 19×19 dài**.
- [ ] MCTS xử lý tất cả terminal tests, no NaN, correct perspective.

**Commit:** `test(go): validate 19x19 rules and terminal labels`

---

## P2 — MODEL V2 HỖ TRỢ 5×5 / 9×9 / 13×13 / 19×19

### Tại sao
`nn/model.py` hiện có `policy_fc(2*N*N -> N*N+1)` và `value_fc1(N*N -> 128)`: trọng số head phụ thuộc N, không thể dùng nguyên checkpoint 9×9 cho 19×19.

### Thiết kế ĐỀ XUẤT

- Giữ backbone Conv stem + ResBlocks; tạo model V2 riêng, KHÔNG sửa model legacy đang được 9×9 sử dụng.
- `policy_conv`: tạo *spatial logits* kích thước `[B,1,N,N]` -> flatten N*N; thêm **pass logit** từ global pooled features qua linear -> `N*N+1`.
- `value`: global pooling + MLP -> tanh `[B,1]`, không có dense layer phụ thuộc N.
- Tách `forward()` trả logits thô; legal mask áp dụng nhất quán tại inference/search, softmax rồi trả xác suất.
- Bổ sung board size / komi input hoặc feature encoding rõ ràng nếu đào tạo hỗn hợp nhiều kích thước; không nhầm plane 16 hiện là side-to-move, tránh đổi 17 planes âm thầm.
- Phiên bản model/checkpoint schema mới `arch = resnet_spatial_v2`, input-plane count ghi minh bạch.

### Files
`nn/model_v2.py` (MỚI), `nn/checkpoint_migrate.py` (MỚI), `tests/test_nn/test_model_v2.py` (MỚI), `main.py`, `config.py`.

### Thực hiện
- [ ] Chuẩn hóa interface `forward`, `predict`, `predict_batch`; shape/return type cụ thể.
- [ ] Unit test output shapes: 5×5→26, 9×9→82, 13×13→170, 19×19→362 và value `[B,1]`.
- [ ] Test D4 policy/pass logit mapping và legal mask cho 19×19.
- [ ] Benchmark V1/V2 với cùng backbone để loại bỏ nhiễu kích thước mạng.
- [ ] Migration script: chỉ copy key/shape khớp của conv trunk, BatchNorm trunk; head V2 khởi tạo mới và phải log rõ key copy/skipped.
- [ ] Chặn `strict=False` nuốt lỗi âm thầm; migration chỉ chấp nhận whitelist key đã định nghĩa.
- [ ] Checkpoint của V1/9×9 vẫn load theo V1. Không rewrite dữ liệu cũ.
- [ ] So sánh hai chiến lược: 19×19 từ random V2 vs V2 initialized-from-9×9 backbone dưới **cùng compute**.

### Gate
- [ ] Cross-size forward/backprop/infer & D4 invariants PASS.
- [ ] Migration báo cáo số tensors copied/skipped; không báo đã chuyển toàn bộ checkpoint.
- [ ] VRAM và inference benchmark đạt mức chấp nhận được trên GPU thực, không OOM.

**Commit:** `feat(nn): add board-size-independent policy and value heads`

---

## P3 — GO ENGINE / CPU MCTS LOW-LEVEL HOT PATH

### Mục tiêu
Giảm chi phí môi trường game mà không đổi luật, ưu tiên những điểm thực sự nóng trong flame graph 19×19.

### Files
`go/board.py`, `go/fast_board.py` (MỚI tùy chọn), `mcts/node.py`, `mcts/search.py`, `benchmarks/board_hotpaths.py` (MỚI), `tests/test_go/test_equivalence.py`.

### Thực hiện theo mức rủi ro
- [ ] Profile `_find_chain_and_liberties`, `_simulate_move`, `get_legal_moves`, `Board.copy`, zobrist, undo, `MCTSNode.best_child` bằng profiler; đo trên states thực.
- [ ] Tối ưu dễ kiểm chứng trước: đổi `queue.pop(0)` -> `collections.deque.popleft()`, tránh kiểm tra toàn bàn lặp không cần thiết, cache neighbors và empty counts.
- [ ] Tách đường `check_legality`, `apply_move`, `undo_move` để tránh copy board nhiều lần (nếu benchmark chứng minh có ích).
- [ ] Cân nhắc group/liberties incrementally cập nhật theo moves, song phải có test đúng với capture/ko/superko.
- [ ] Nếu Python vẫn là bottleneck lớn, triển khai backend **tùy chọn** C++/Rust hoặc Cython và duy trì Python reference backend; test 10.000 ván ngẫu nhiên so sánh move-by-move cả hai backend.
- [ ] MCTS node dùng array compact / lazy-children thay dict nếu profile cho thấy overhead đáng kể; không sửa Q sign.
- [ ] Benchmark tổng end-to-end sau mỗi tối ưu, không chỉ một microfunction.

### Gate
- [ ] Hai backend (nếu có) cho cùng legal moves, board, captured stones, hashes, score ở tất cả fixtures.
- [ ] Improvement measurable trên `MCTS sims/s`, không chỉ `Board.copy()`.
- [ ] Nếu tốc độ không cải thiện hoặc tỷ lệ lỗi khác 0, rollback thay đổi.

**Commit:** `perf(go): optimize verified board and MCTS hotspots`

---

## P4 — BATCHED GPU INFERENCE *THỰC SỰ* TRONG MCTS

### Hiện trạng quan trọng
`AlphaZeroNet.predict_batch()` có sẵn ở nhánh hiện tại, nhưng `MCTS._expand_node()` vẫn gọi `model.predict()` cho **một trạng thái**. Bench 15.92× ở NN batch chưa tác động trực tiếp vào self-play.

### Thiết kế
`actors (CPU) -> InferenceClient -> bounded request queue -> GPU InferenceServer -> matched responses -> actors`, server duy nhất sở hữu CUDA context/model. Bước đầu chạy **nhiều cây cờ độc lập**; không cần ngay tree-parallel + virtual loss.

### Files
`inference/server.py` (MỚI), `inference/client.py` (MỚI), `inference/types.py` (MỚI), `mcts/batched_search.py` (MỚI), `nn/model_v2.py`, `tests/test_inference_batch.py` (MỚI).

### Thực hiện
- [ ] Tạo protocol `InferenceRequest(request_id, game_id, model_version, state, legal_mask)`; response có `request_id`, `probs`, `value`, error.
- [ ] Bounded queue có backpressure, max batch và timeout gom batch, timeout phản hồi, worker cancellation, graceful shutdown.
- [ ] GPU Server tạo model `eval()` và dùng `torch.inference_mode()`; AMP FP16 chỉ nếu ổn định và được benchmark.
- [ ] Input nhóm theo `board_size` + `model_version` để không trộn shape/trọng số.
- [ ] Một request thất bại phải trả lỗi cho đúng actor, không treo toàn bộ job; nếu queue bị deadlock có watchdog.
- [ ] Bộ điều phối cấp lượt: mỗi actor đánh giá một leaf của ván hiện tại, server gom leaf từ các ván/actors khác nhau. Tránh deadlock do mỗi actor khóa chờ một tree.
- [ ] Khi có tree-parallel sau này, quản lý virtual loss và xử lý duplicate leaf, phải có test riêng; **chưa bật mặc định**.
- [ ] Không copy Torch CUDA tensors qua process bằng pickle; gửi arrays/metadata qua IPC hợp lệ, có checksum/bounds khi cần.
- [ ] Thử `max_batch ∈ {4,8,16,32}`, `batch_wait_ms ∈ {1,2,5}` trên GPU 4GB; theo dõi p50/p95 inference latency và actual batch fill rate.
- [ ] `model_version` cố định trọn vòng self-play hoặc sync tại ranh giới ván, tuyệt đối không đổi trọng số giữa một search.
- [ ] So sánh kết quả model batch vs single trong sai số FP32/FP16 và equality legal masks; không yêu cầu bitwise equality khi môi trường GPU không bảo đảm.

### Gate
- [ ] Không request bị bỏ/nhận nhầm, không treo khi worker crash, API có thể timeout an toàn.
- [ ] Self-play end-to-end nhanh hơn so với baseline P0 trên cùng số sims và model; nếu không, xem lại batch/wait/CPU thay vì tự nhận đã tối ưu.
- [ ] RAM/VRAM không vượt mức máy thực tế, p95 queue wait không tăng không kiểm soát.

**Commit:** `feat(inference): integrate batched GPU server with MCTS`

---

## P5 — SELF-PLAY MULTIPROCESSING & SCHEDULER HỢP LÝ

### Files
`selfplay/actor.py` (MỚI), `selfplay/coordinator.py` (MỚI), `selfplay/worker.py`, `main.py`, `tests/test_parallel_selfplay.py` (MỚI).

### Thiết kế Windows
- Dùng `spawn` (`multiprocessing.get_context('spawn')`) trên Windows; entrypoint có `if __name__ == '__main__'`.
- **Chỉ server/Trainer dùng CUDA**, actor CPU giữ Board/MCTS tree. Không instantiate `AlphaZeroNet(...).cuda()` riêng cho từng actor.
- Mỗi actor nhận `(run_id, model_version, game_id, seed)` riêng, ghi game result kèm legality metadata.

### Thực hiện
- [ ] Bật 1 actor qua API mới và xác nhận khớp pipeline tuần tự (theo seeded policy deterministic, với tolerance khi GPU).
- [ ] Thử 2 actors → 4 actors → 6 actors; đo CPU usage, batch fill, GPU, game throughput.
- [ ] Hàng đợi có capacity giới hạn để tránh ngốn RAM khi server/trainer chậm.
- [ ] Điều phối mỗi game một `model_version`; tuyệt đối không trộn nhãn từ network đang cập nhật trong cùng search.
- [ ] Actor exception/timeout báo về coordinator, ghi game lỗi và **không đưa mẫu nửa ván vào buffer**.
- [ ] Ghi tự động đủ `GameResult`, win reason, illegal moves, full-search positions, resign outcome.
- [ ] Khi dùng 1 GPU, mặc định **synchronous phases**: generate → train → eval; async chỉ sau khi đo GPU/VRAM và thread contention.
- [ ] Cấu hình workers bằng CLI mới, ghi rõ `--selfplay-workers` là ĐỀ XUẤT; phải có test CLI trước khi dùng.

### Gate
- [ ] Không duplicate game_id/seed, không data loss/deadlock/stray CUDA process.
- [ ] Shutdown/keyboard interrupt sạch, resume từ ván đã hoàn tất, buffer không chứa sample thiếu nhãn.
- [ ] Có lựa chọn số actors tốt nhất cho máy, dựa trên positions/hour, không mặc định nhiều hơn luôn tốt hơn.

**Commit:** `feat(selfplay): run parallel CPU actors with one GPU inference server`

---

## P6 — REPLAY DATA 19×19, RAM, I/O, CONSISTENT CHECKPOINT

### Dung lượng tham khảo, buffer 200.000 mẫu 19×19 hiện tại
- States `uint8`: `200000 × 17 × 19 × 19 = 1.2274 GB` (decimal).
- Pi `float32`: `200000 × 362 × 4 = 0.2896 GB`.
- Z `float32`: `0.0008 GB`.
- Tổng **~1.518 GB dữ liệu thô**, chưa tính metadata, extra arrays, working allocations. `ReplayBuffer.save()` đang `.copy()` từng mảng, có thể tạo **đỉnh RAM gấp đáng kể** khi snapshot.

### Files
`selfplay/replay_buffer.py`, `train/dataset.py` (MỚI), `storage/shards.py` (MỚI), `main.py`, `tests/test_storage_19.py` (MỚI).

### Thực hiện
- [ ] Giữ ring buffer uint8 đã có, tối ưu `sample_batch()` trên batch lớn/19×19, không cần rewrite data structure nếu profiler không thấy bottleneck.
- [ ] Tránh `.copy()` toàn bộ buffer mỗi iteration: sử dụng snapshot incremental / append-only immutable shards + atomic manifest; đo riêng RAM peak và bytes written.
- [ ] Shards có `schema_version, board_size, model_version, sample_count, checksum, dtype`; validate trước khi load.
- [ ] Data samples có `state, legal_mask? , pi, z, source_type, high_quality_policy, model_version, game_id`; chỉ lưu metadata cần thiết.
- [ ] V1/V2 migration nếu thay schema, có test round-trip, partial-written shard, corrupt file, disk full; rollback trả về snapshot mới nhất đã complete.
- [ ] Nếu giảm policy FP32→FP16, đo KL/MAE và ảnh hưởng policy loss, không tự chuyển mọi trường chỉ vì ít RAM.
- [ ] Có lựa chọn capacity theo RAM (50k/100k/200k), kiểm tra actual RSS và garbage collection.
- [ ] Keep cache dataset, tránh việc toàn bộ dữ liệu liên tục chuyển RAM↔GPU không cần thiết; có pinned memory khi có lợi đo được.
- [ ] RNG states (numpy/python/torch/cuda), AMP scaler và số mẫu mới/lịch sampling phải nằm trong checkpoint một cách nhất quán.
- [ ] `latest_manifest` chỉ commit sau khi *model checkpoint + shards + ratings* đều đã ghi xong. Crash bất kỳ lúc nào đều đọc được iteration trước đó.

### Gate
- [ ] Data round-trip chính xác với 5/9/13/19, không lệch `(s,pi,z)`.
- [ ] Resume qua power/crash simulation không mất checkpoint đã commit, không lặp mẫu sai.
- [ ] Peak memory, I/O và save duration có benchmark trên 19×19; giới hạn tự động khi gần thiếu RAM/disk.

**Commit:** `perf(storage): shard and snapshot 19x19 replay with crash consistency`

---

## P7 — TIẾT KIỆM MCTS COMPUTE MÀ KHÔNG LÀM HỎNG DATA

### Files
`selfplay/schedule.py` (MỚI), `mcts/search.py`, `selfplay/worker.py`, `train/dataset.py`, `config.py`, `tests/test_playout_schedule.py` (MỚI).

### Thực hiện
- [ ] Thiết lập riêng hyperparameters cho 19×19: Dirichlet alpha **thử nghiệm** 0.03, c_puct, temperature schedule, komi=7.5, resign disable early.
- [ ] Đo prior entropy/visit entropy/pass rate trước, tránh ép temperature theo 9×9 (`TEMP_MOVES=12`) mà không kiểm tra.
- [ ] Tắt resign trong giai đoạn model non-calibrated bằng threshold dưới -1; Arena cũng phải **tách biệt** cờ resign khỏi self-play.
- [ ] Thử **Playout Cap Randomization (PCR)** theo lưới thí nghiệm A: fixed 100 sims; B: 25% full 200 + 75% fast 60; C: 50% full 120 + 50% fast 40. Số liệu chỉ là cấu hình thử, không hứa tiết kiệm thực.
- [ ] Chỉ lấy policy targets từ *full-search* đáng tin cậy (hoặc gán trọng số thấp theo nghiên cứu/ablation rõ ràng). Fast-search moves để di chuyển ván; value target phải có nhãn cuối ván hợp lệ.
- [ ] So sánh dưới cùng **ngân sách compute wall-clock/GPU-hours**, không chỉ cùng số ván.
- [ ] Với early-pass, forced terminal và max_moves, gắn cờ dữ liệu kém, loại bỏ khỏi training theo chính sách.
- [ ] Chỉ thêm policy improvement (adaptive simulations theo độ bất định / early stop search) sau khi baseline PCR đã ổn định.

### Gate
- [ ] Policy soft labels hợp lệ, legal probabilities sum~1, target provenance có log.
- [ ] PCR tiết kiệm số eval/game **và** không làm suy giảm improvement-per-GPU-hour trên benchmark dài đủ tin cậy.
- [ ] Không đưa các lượt chỉ có ít search visit vào policy loss như sample full-search.

**Commit:** `feat(selfplay): add compute-aware MCTS scheduling with target-quality safeguards`

---

## P8 — TRAINER GPU / CHIẾN LƯỢC HỌC HIỆU QUẢ

### Files
`train/trainer.py`, `train/dataset.py` (MỚI), `main.py`, `config.py`, `tests/test_training_efficiency.py` (MỚI).

### Thực hiện
- [ ] Baseline batch size/throughput VRAM trên 19×19 với 6×96 V2, trước khi tăng model.
- [ ] Test AMP FP16/BF16 tùy GPU hỗ trợ; chọn mode chạy ổn định, không NaN/Inf, scaler chính xác.
- [ ] Thử `channels_last`, `torch.compile` **từng tính năng một**, ghi tốc độ, peak VRAM và cold-start compile. Rollback nếu VRAM tăng mạnh hoặc train chậm.
- [ ] Gradient accumulation nếu GPU 4GB không đủ effective batch, nhưng so sánh gradient/loss với batch tham chiếu; đồng bộ BN behavior.
- [ ] Split data train/hold-out **theo game_id và thời điểm**, tránh cùng một ván xuất hiện ở train và validation.
- [ ] Theo dõi KL(policy target vs predicted), value calibration, policy entropy, training sample reuse, `optimizer_steps / new_valid_samples`.
- [ ] Kiểm tra rule giới hạn training steps theo dữ liệu mới thay vì chỉ `buffer.size`, tránh học đi học lại quá nhiều.
- [ ] LR scheduler dựa trên global training steps hoặc iterations nhất quán; checkpoint đủ và test accept/reject/resume qua milestones.
- [ ] Train phases xen kẽ self-play hợp lý: tránh bỏ GPU chạy trainer quá lâu khi không có dữ liệu mới hữu ích.
- [ ] **Không mở rộng số blocks/filters trước khi benchmark chứng minh pipeline đủ compute.** So sánh 4×64, 6×96, 8×128 theo cost-to-strength, không chọn chỉ theo loss.

### Gate
- [ ] Train liên tục không NaN, validation không collapse, resume optimizer/LR/scaler ổn định.
- [ ] Improvement-per-hour tốt hơn baseline hoặc có lý do chất lượng rõ ràng để chấp nhận compute tăng.

**Commit:** `perf(train): optimize mixed-precision 19x19 training and data reuse`

---

## P9 — ARENA ĐÁNG TIN CẬY VÀ BENCHMARK CHẤT LƯỢNG

### Files
`arena/match.py`, `arena/evaluation.py` (MỚI), `arena/baseline.py`, `main.py`, `tests/test_arena_statistics.py` (MỚI).

### Thực hiện
- [ ] Dùng same paired openings, luân phiên màu, fixed seeds, fixed search budget và model eval mode.
- [ ] Thay Wilson tính trên tổng ván đơn lẻ (có tương quan trong cặp) bằng per-pair bootstrap CI / phương pháp thống kê có tính phụ thuộc theo cặp.
- [ ] Tách `win_rate_for_promotion` và `win_rate_vs_fixed_reference`; không kết luận progress chỉ vì thắng model gần nhất.
- [ ] Bộ đối thủ benchmark ở các cấp: Random, CaptureHeuristic, cố định model cũ, và nếu muốn GNU Go/KataGo bên ngoài với cấu hình cố định; không dùng level ngoài mà không ghi rõ sức mạnh và luật.
- [ ] Đo `win_rate` vs reference trên 19×19 đủ số cặp để phân biệt hiệu ứng; có sequential evaluation theo confidence bounds hoặc bracket rounds; không dùng 4/4 trận làm chứng cứ sức mạnh.
- [ ] Gating tùy chọn: phase đầu có thể `no-gating` để tiết kiệm compute nhưng vẫn phải kiểm tra reference định kỳ. Khi model mạnh hơn, dùng promotion có kiểm định đủ tin cậy.
- [ ] Phân tách và log riêng `selfplay resign`, `arena resign`, `max_moves` và tỷ lệ pass bất thường.
- [ ] Báo cáo `strength gain / 100 GPU hours` hoặc proxy trễ hơn từ match fixed reference.

### Gate
- [ ] Không lệch màu; opening pairs thật sự cùng sequence, metadata trận đầy đủ.
- [ ] CI/gating implementation được test với synthetic outcomes & paired correlation.
- [ ] Phát hiện regression sức chơi ở fixed reference, dù training loss vẫn giảm.

**Commit:** `feat(arena): implement paired confidence-aware 19x19 evaluation`

---

## P10 — RAMP-UP 19×19 THEO CỬA KIỂM ĐỊNH, KHÔNG ĐỐT NGÂN SÁCH

### Giai đoạn triển khai

| Gate | Dữ liệu / mục tiêu | Cấu hình gợi ý mở đầu | Điều kiện dừng / lên bước |
|---|---|---|---|
| G0 shape/runtime | smoke 1 iteration | 19, 2 ván, 16 sims, max 20 (chỉ smoke) | Test+checkpoint+memory PASS |
| G1 full game | 5–10 ván full | 19, 32–64 sims, 4×64 hoặc 6×96 V2, no resign | Không crash, pass/score hợp lệ |
| G2 local throughput | 3 lặp benchmark ở 2–4 actors | batch 4/8/16, mỗi profile 1 GPU | `games/hour`, cost/game tốt hơn P0 |
| G3 pilot learning | 20–50 iterations như thử nghiệm | 19, games 10–30/iter, sims 64–128, train steps theo fresh samples | Fixed baseline tiến bộ, data chất lượng |
| G4 ablation | Cố định compute, 2–3 configs | PCR vs fixed, V2 migrated vs scratch | Giữ cấu hình tăng sức chơi/chi phí tốt nhất |
| G5 long train | phân ngân sách định kỳ | CPU actors + GPU inference đã đo, checkpoint + alerts | Theo dõi regression, plateau, spend |

> Số vòng, ván và simulations ở G3 chỉ là **giá trị thăm dò**; GPU 4GB laptop có thể rất chậm. Cần điều chỉnh theo throughput G2, không xem đây là lệnh có thể chạy ngay: các flag kiến trúc/actors vẫn chưa tồn tại cho tới khi P2–P5 DONE.

### Dữ liệu mục tiêu
- Một run V2 `train19_v4` mới; không trộn trực tiếp 9×9 legacy buffer vào 19×19.
- Chỉ chuyển **trọng số trunk** tương thích từ 9×9 nếu test và ablation cho thấy có lợi.
- Báo cáo tối thiểu: số games, số valid positions, pass/resign/max_m_end, số eval, GPU hours, mỗi iteration loss, fixed arena winrate, checkpoint hash.
- Nếu sức chơi không tăng sau mốc ngân sách xác định, dừng, kiểm tra data target/arena/search trước khi tăng GPU.

### Gate
- [ ] Full games kết thúc hợp lệ, không dùng labels từ forced short smoke.
- [ ] Resume sau restart giữa hai iterations hoạt động, không làm mất accepted best model.
- [ ] Có bằng chứng model mạnh lên trên fixed baselines, không chỉ train loss giảm.

**Commit:** `chore(training): validate gated 19x19 pilot with reproducible metrics`

---

## P11 — HIỆU QUẢ CHI PHÍ, CLOUD GATE, RELEASE

### Tối ưu chi phí phải qua 3 lớp

1. **Dùng GPU đang có để debug/benchmark**: tránh thuê GPU khi `MCTS.search()` vẫn sync inference từng leaf.
2. **Điều chỉnh workload**: workers, batch fill, số sims, data reuse và checkpoint I/O trước khi đổi loại GPU.
3. **Thuê Cloud chỉ khi rẻ hơn theo đơn vị hiệu quả đã đo**: `$/1000 valid positions`, `$/1000 full-search targets`, và `$/fixed-reference Elo improvement` khi đủ trận.

### Chính sách budget
- [ ] Mỗi thí nghiệm có `budget_limit`, `max_hours`, `max_gpu_hours`, `max_storage_gb`; có kill-switch khi vượt giới hạn.
- [ ] Trước khi thuê 100 giờ phải chạy pilot 1–2 giờ trên đúng instance, ghi cost/hour thực tế + throughput + setup overhead.
- [ ] So sánh ba phương án: local điện; Cloud GPU nhỏ rẻ; Cloud GPU mạnh. Không mặc định GPU lớn luôn kinh tế nhất.
- [ ] Nhất quán CPU vCPU / RAM / disk và transfer khi benchmark Cloud.
- [ ] Snapshot/checkpoint thường xuyên theo dung lượng/chi phí ghi, không save 1.5GB full buffer mỗi iter nếu không cần.
- [ ] Khi validation plateau: giảm LR hoặc xem lại data mix, cân nhắc restart tốt hơn thay vì mở rộng GPU.

### Form báo cáo quyết định GPU

| Phương án | Giá/giờ | Full games/giờ | Valid targets/giờ | Tiền/1000 targets | Peak VRAM | Win gain cùng ngân sách |
|---|---:|---:|---:|---:|---:|---:|
| RTX 3050 Laptop local | đo điện | TBD | TBD | TBD | TBD | TBD |
| RTX 4060 local (nếu dùng) | đo điện | TBD | TBD | TBD | TBD | TBD |
| Cloud GPU A | lấy báo giá tại thời điểm thuê | TBD | TBD | TBD | TBD | TBD |
| Cloud GPU B | lấy báo giá tại thời điểm thuê | TBD | TBD | TBD | TBD | TBD |

### Gate cuối cùng (Go/No-Go)
- [ ] Bộ tests trên các board sizes 5/9/13/19 PASS, không regression luật.
- [ ] Batch GPU + multiprocessing chạy không deadlock và có số liệu faster end-to-end.
- [ ] Model V2 chuyển kích thước bàn và checkpoint reproducible theo giới hạn thực tế.
- [ ] Replay save/resume sau crash và full buffer luôn đọc được.
- [ ] So sánh benchmark fixed reference có CI theo paired openings.
- [ ] Không sử dụng nhãn giả từ forced max moves/smoke; không dùng `--no-gating` synthetic 100% như sức chơi.
- [ ] Budget cap và alert hiện hoạt; có số tiền/valid sample.
- [ ] Có `TRAINING_19X19_READINESS_REPORT.md` ghi sha, GPU, kết quả, giới hạn, quyết định train/không train.

**Commit:** `docs(release): certify 19x19 training readiness and cost efficiency`

---

## B. Bảng thứ tự ưu tiên theo hiệu quả thực tế

| Đầu việc | Lợi ích kỳ vọng | Rủi ro kỹ thuật | Nên làm khi |
|---|---|---|---|
| P0–P1 benchmarks & rules | Tránh chi phí train sai | Thấp | Ngay lập tức |
| P4 batched GPU + P5 actors | Tăng self-play throughput | Trung bình/cao | Sau P1, model API ổn |
| P3 Engine optimization | Giảm CPU overhead | Trung bình/cao | Sau profiler xác nhận |
| P6 Replay/I/O | Giảm RAM peak và disk traffic | Trung bình | Trước scale buffer 200k |
| P2 Model V2 | Transfer giữa board sizes; kiến trúc tốt cho 19 | Trung bình | Trước 19 long train |
| P7 PCR & search budget | Giảm eval/walltime | Cao (chất lượng) | Sau correctness + perf |
| P8 Training optimization | Giảm train GPU time | Trung bình | Khi selfplay đã đủ nhanh |
| P9 Evaluation | Tránh tăng tốc sai mục tiêu | Trung bình | Trước train pilot dài |
| Ownership/score auxiliary heads | Tăng data efficiency tiềm năng | Cao do labels/scoring | Sau P1 và có ablation |
| C++/Rust native engine | Giảm CPU overhead tiềm năng | Cao | Chỉ khi profiler 19 chỉ ra |
| Distributed/cloud | Scale khi kinh tế | Cao + chi phí | Sau P11 local baseline |

> P2–P5 là một chuỗi tích hợp; ưu tiên kiểm chứng interface và full-game chính xác trước tối ưu throughput. Nếu cần gấp, có thể benchmark batch scheduler với legacy network rồi mới chuyển V2, nhưng **không gộp hai thay đổi vào một commit**.

---

## C. Test matrix tối thiểu

| Hạng mục | 5×5 | 9×9 | 13×13 | 19×19 | Gate |
|---|:--:|:--:|:--:|:--:|---|
| Legal/pass/ko/score/hash | ✓ | ✓ | ✓ | ✓ | exact invariants |
| Model output shapes & masking | ✓ | ✓ | ✓ | ✓ | `N²+1`, softmax sum1 |
| PUCT sign/backup/Dirichlet | ✓ | ✓ | ✓ | ✓ | same policy logic |
| Batch vs single logits/probs | ✓ | ✓ | ✓ | ✓ | tolerance FP16/FP32 |
| Selfplay 1 actor vs multi | ✓ | ✓ | ✓ | ✓ | valid complete labels |
| Save/load buffer & migration | ✓ | ✓ | ✓ | ✓ | no corruption |
| Checkpoint/restart/rollback | ✓ | ✓ | ✓ | ✓ | state correctness |
| Arena paired openings | ✓ | ✓ | ✓ | ✓ | model-color symmetry |
| Performance/economics | smoke | regression | pilot | full | end-to-end improvement |

Tests không nhất thiết dùng nhiều simulations với tất cả kích thước trong CI nhanh; thêm một nightly / manual integration benchmark riêng cho 19×19 full games.

---

## D. Prompt mẫu giao việc cho OpenCode — dùng TỪNG PHA

> Dán mỗi phiên bản vào OpenCode; thay `P<X>` bằng pha cụ thể. Đây là prompt, không phải yêu cầu được phép chỉnh toàn bộ repo.

```text
Bạn là Senior ML Systems Engineer phụ trách CoVayAI (AlphaZero Go).
Repo: https://github.com/luongviettienns/CoVayAI
Branch đang làm: feat/19x19-training-v4 (xuất phát từ fix/train-readiness).
Tài liệu chuẩn: CoVayAI_MasterPlan_V4_19x19.md.

NHIỆM VỤ: CHỈ triển khai pha P<X> trong tài liệu. KHÔNG tự ý sửa các pha khác.
Trước khi sửa:
1. Đọc toàn bộ module liên quan và tests hiện có.
2. Liệt kê files sẽ sửa/thêm, lý do từng file, tác động checkpoint/data schema.
3. Xác nhận việc sửa không làm hỏng checkpoint 9x9 hiện có.

Trong khi sửa:
4. Làm test red-green-refactor: test fail trước, sửa để pass.
5. Giữ API cũ hoặc thêm adapter; mọi breaking change phải có migration.
6. Không huấn luyện dài, không thuê cloud, không ghi đè checkpoints/run9_prod.
7. Không thêm CLI flags mới nếu chưa implement + test parser.

Sau khi sửa:
8. Chạy pytest của pha + toàn repo và, nếu là pha performance, benchmark trước/sau cùng điều kiện.
9. Báo cáo metrics thật: times, throughput, GPU peak VRAM, exceptions, regressions.
10. Nếu không đạt gate: rollback thay đổi và nêu nguyên nhân; không tick DONE.
11. Nếu PASS: cập nhật checklist P<X> trong tài liệu, commit nhỏ tên như kế hoạch.

OUTPUT: danh sách files thay đổi, diff quan trọng, tests PASS/FAIL, benchmark before/after,
model compatibility, checkpoint safety, nguy cơ tồn đọng, commit hash, quyết định GO/NO-GO.
```

### Prompt đặc biệt P4 — GPU batching (khuyến nghị)

```text
Chỉ triển khai P4; giữ MCTS/NN legacy để so sánh. Có predict_batch() không đồng nghĩa
MCTS batching. Hãy thiết kế multi-game leaf scheduling, bounded GPU inference queue,
request_id mapping, batch per board_size/model_version, timeout/shutdown sạch.
Một server duy nhất giữ CUDA model. Không truyền tensors GPU sang subprocess.
Viết tests về legality, soft targets, response alignment, crash/no deadlock và
benchmark true selfplay positions/hour trước/sau. Không thêm tree-parallel virtual loss
trong PR này. Không sửa P2/P3/P5 nếu chưa được duyệt.
```

---

## E. Dashboard tối thiểu cho quyết định huấn luyện

**Daily run report:**

```text
Commit: __________    Profile: 19x19 / model V2 6x96 / GPU ______
Games: ____   Valid positions: ____   Full-search policy targets: ____
Time: selfplay ____h | train ____h | arena ____h | total ____h
Throughput: ____ games/h | ____ valid targets/h | ____ MCTS sims/s
GPU util: ____% | VRAM peak ____GB | RAM peak ____GB | Disk ____GB
Resign: ____% | pass anomalies: ____ | max-move ends: ____% | illegal moves: 0
Train loss: policy ____ value ____ | held-out loss: policy ____ value ____
Fixed benchmark winrate: ____% | paired CI: ____ | comparison model: ____
Cost: local electricity ____ or cloud ____ | cost/1000 valid targets ____
Best checkpoint: ______ | Resume verified: YES/NO | Next action: ______
```

**Các dấu hiệu phải dừng kiểm tra:** illegal moves >0; score disagreement ở golden cases; nan/inf; repeated OOM; GPU queue deadlock; model phiên bản không nhất quán; data shard corrupted; checkpoint resume sai; output pass hàng loạt không giải thích được; sức chơi giảm nhiều dù loss giảm.

---

## F. Quyết định cuối cùng

- **GO ngay:** P0 baseline + P1 correctness, vẫn giữ nguyên run 9×9 đang train nếu ổn định.
- **GO sau P1:** P2/P3/P4/P5 lần lượt, mỗi pha có gate; P6 snapshot và P7 MCTS efficiency.
- **NO-GO:** train dài 19×19 hoặc thuê GPU nhiều giờ trước khi có benchmark end-to-end, quality labels và cost/valid target.
- **Tối ưu thành công** không phải 'NN inference nhanh x lần' mà là **sức chơi tăng trên cùng số giờ GPU/chi phí** và resume ổn định.

## Nguồn tham khảo

- CoVayAI branch review: https://github.com/luongviettienns/CoVayAI/tree/fix/train-readiness
- KataGo training architecture / loops: https://github.com/lightvector/KataGo/blob/master/SelfplayTraining.md
- KataGo source: https://github.com/lightvector/KataGo
- PyTorch performance tuning guidance: https://pytorch.org/tutorials/recipes/recipes/tuning_guide.html

