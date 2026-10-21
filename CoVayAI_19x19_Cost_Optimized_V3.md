# CoVayAI — KẾ HOẠCH V3: NÂNG CẤP 19×19 + TỐI ƯU CHI PHÍ HUẤN LUYỆN

**Baseline đã đối chiếu:** `luongviettienns/CoVayAI`, nhánh `main`, commit `5f4898c1f256d38cc3c4a06960799ab980c66a5b` (20-10-2026).

**Mục tiêu:** AI cờ vây 19×19 học ổn định, sinh dữ liệu self-play nhanh, tiết kiệm GPU/RAM và **đạt sức chơi tốt nhất với một ngân sách xác định**, có bằng chứng cải thiện sức mạnh. Không tự nhận đã đạt hiệu năng cụ thể khi chưa benchmark trên máy thật.

## NGUYÊN TẮC KHÓA

1. Không xóa/sửa checkpoint và replay buffer cũ; mỗi thử nghiệm dùng một `checkpoint_dir` mới.
2. Mỗi PR chỉ đổi một nhóm chức năng. Test cũ phải PASS cùng test mới. Khi FAIL, rollback commit thay vì vá tùy tiện tiếp.
3. Không thay đổi luật, mạng, data format và optimizer trong cùng một PR.
4. Trước/sau mọi tối ưu phải có benchmark A/B trên **cùng phần cứng, cùng số sims, cùng model weights, cùng phân phối trạng thái**.
5. Không dùng kết quả ván bị cắt cưỡng bức để gán nhãn thắng/thua như một ván hoàn tất, trừ khi có bộ adjudication độc lập đã kiểm định.
6. Nếu dùng checkpoint 9×9 cũ: không giả định load thẳng vào policy/value heads 19×19 hiện tại; chỉ chuyển backbone tương thích, hoặc đổi kiến trúc size-independent trước.
7. Trong thử nghiệm tabula rasa: không dùng SGF người hoặc pretrained KataGo. Nếu dùng teacher/pretraining, báo rõ đó là nhánh huấn luyện khác.

## P0 — Cố định và đo baseline (ưu tiên P0, trước mọi sửa)

**File:** `benchmarks/bench_board.py`, `benchmarks/bench_inference.py`, `benchmarks/bench_mcts.py`, `benchmarks/bench_pipeline.py` (mới), `tests/`.

- [ ] `git switch -c feat/train-19x19`; ghi commit gốc, Python/PyTorch/CUDA/GPU/VRAM/CPU/RAM.
- [ ] `python -m pytest -q` và lưu `test_baseline.txt`.
- [ ] Ghi `board.copy/s`, `get_legal_moves/s`, `play/s`, `nodes/s`, `NN evaluations/s`, `positions/s`, `games/hour`, `GPU utilization`, peak VRAM/RAM.
- [ ] Tách benchmark infer batch = 1, 8, 16, 32, 64, 128 trên board 9, 13, 19; có warmup, `torch.cuda.synchronize()` trước và sau timing.
- [ ] Có ít nhất ba phân bố state: opening, middle-game và late-game; không chỉ benchmark bàn trống.
- [ ] Output CSV/JSON và log có cấu hình, git SHA, seed; P0 không đổi thuật toán.

**PASS:** có baseline tái chạy được; tất cả test cũ xanh; số liệu trước tối ưu đầy đủ.

## P1 — Sửa tính đúng đắn và khả năng train dài (blocker)

**File:** `main.py`, `train/trainer.py`, `go/board.py`, `go/scoring.py`, `selfplay/worker.py`, `tests/test_pipeline.py`, `tests/test_train/`.

- [ ] Scheduler global iteration/step: sau promote/reject đều tiến đúng một lần; state phải lưu **sau** scheduler step; resume từ accepted/rejected đều đúng LR, momentum, scheduler epoch.
- [ ] Checkpoint atomically: temp + `os.replace`, lưu `model`, `optimizer`, `scheduler`, `scaler`, `global_step`, `iteration`, RNG states, model metadata/config/version, source git SHA.
- [ ] Resume với `--no-resume` không giữ `best_model.pt` cũ; không đồng nhất `checkpoint_iter_i.pt` của candidate rejected với accepted best.
- [ ] Self-play: disable resign mặc định cho warm-up 19×19; riêng Arena tắt resign lúc benchmark chuẩn hoặc hiệu chuẩn độc lập.
- [ ] Luật cờ/score: kiểm định bằng bộ tình huống Ko/Superko/SeKi/2-pass; double-pass và dead stones cần đối chiếu engine độc lập; phân biệt full terminal với truncated max_moves.
- [ ] Nếu ván kết thúc bởi max_moves (19×19: 722 mặc định), lưu metadata `termination_reason=truncated`, không tạo z outcome từ score tạm một cách mặc định.
- [ ] Không đổi mẫu `(state, pi, z)` trực tiếp nếu chưa viết migration/backwards compatibility.

**PASS:** 3+ iteration accepted/rejected giả lập; dừng/khôi phục cho cùng model/optimizer/LR/buffer state; không có outcome label giả do truncation.

## P2 — Neural network dùng chung 9×9 / 13×13 / 19×19

**File:** `nn/model.py`, `nn/features.py`, `train/trainer.py`, `tests/test_nn/`.

- [ ] Giữ backbone conv + ResBlock; thay `policy_fc(2*N*N -> N*N+1)` bằng `policy_conv -> spatial logits [B,N,N]` cộng `pass_head(global pooled features) -> [B,1]`.
- [ ] Thay `value_fc1(N*N ->128)` bằng masked/global pooling `[B,C,N,N] -> [B,C]` + MLP -> tanh value.
- [ ] Nếu trộn kích thước trong cùng batch: board mask phải được xử lý trong pooling, loss, BN/conv theo thiết kế; phiên bản đầu nên group batches theo board-size để tránh complexity.
- [ ] Thêm metadata về komi và kích thước bàn (nếu các ván có komi khác nhau) vào đầu vào phù hợp; không dùng cùng trọng số value mà bỏ qua khác biệt komi.
- [ ] Maintain `policy logits = N*N+1`, pass cuối, value `[B,1]`, legal-mask chỉ áp dụng lên action hợp lệ; phép D4 giữ nguyên pass.
- [ ] Migration: chỉ copy backbone weights khi compatible; heads mới random init, lưu `architecture_version`; test checkpoint cũ ra cảnh báo rõ.
- [ ] Tạo test input board 9, 13, 19; backward finite, gradients nonzero; compare symmetry transformations.

**PASS:** 1 model forward/backward trên cả 3 cỡ bàn; xác suất hợp lệ; không phụ thuộc N ở state_dict của heads.

## P3 — CPU Go engine đủ nhanh cho 19×19

**File:** `go/board.py`, `go/scoring.py`, `go/zobrist.py`, `benchmarks/bench_board.py`.

- [ ] Thay BFS `list.pop(0)` bằng `collections.deque` hoặc stack; đo hiệu quả.
- [ ] Tách fast path legality khỏi `simulate_move`/`play`; hạn chế copy grid và copy history tại mỗi candidate.
- [ ] Dùng cấu trúc chain/liberty cập nhật incremental (union-find hoặc nhóm dây + tập khí) **chỉ sau khi differential tests đã đủ**.
- [ ] `Board.copy()` cần giữ lịch sử ko đầy đủ theo đúng luật trong khi chỉ giữ 8 board history gần nhất cho NN; test cả trường hợp mcts rollout dài.
- [ ] Nghiên cứu Numba/Cython/Rust/C++ khi profiling chứng minh Python rules chiếm đa số runtime.
- [ ] Differential / property-based tests: 10k+ random legal games, score/pass/captures; so sánh old/new trên cùng game seeds.

**PASS:** 0 sai lệch legality/score/hash so với oracle đã chọn; cải thiện `get_legal_moves/s` và `play/s` được đo, không đặt mức tăng tùy tiện.

## P4 — Inference batching và nhiều actor self-play

**File:** `mcts/search.py`, `selfplay/worker.py`, `selfplay/inference_server.py` (mới), `selfplay/actor_pool.py` (mới), `nn/model.py`.

- [ ] API `predict_batch(states, legal_masks)`: đầu vào [B,C,N,N], mask [B,N*N+1]; đầu ra policy [B,N*N+1], value [B].
- [ ] Mỗi actor CPU chạy game/MCTS độc lập; request NN tới inference server GPU qua queue; server batch theo max_batch và max_wait_ms, gom cùng cỡ bàn.
- [ ] Dùng Windows `multiprocessing.get_context('spawn')` và `if __name__ == '__main__'`; không gửi CUDA model qua queue; seed độc lập mỗi actor.
- [ ] Nếu dùng multiple MCTS in-flight trong **cùng** tree cần virtual loss/virtual visit và thread-safe backup; phiên bản đầu nên batch leaf giữa nhiều game để đơn giản.
- [ ] Queue bounded + timeout/retry/healthcheck; stop sạch khi `KeyboardInterrupt`/worker crash; không ghi nửa mẫu game khi đang lỗi.
- [ ] `torch.inference_mode`, AMP cho inference, hạn chế `.item()` và đồng bộ GPU per-sample, `model.eval()` một lần; CUDA metrics.
- [ ] Nếu 4GB VRAM: run selfplay và learner luân phiên, không đồng thời; RTX 4060 8GB có thể thử chia workload sau khi xác nhận peak VRAM.

**PASS:** batch=1 gần đồng nhất với API cũ; batch=8/16/32 outputs tương đương trong tolerance; không deadlock 1000 games; `positions/hour` tăng trong benchmark đúng.

## P5 — ReplayBuffer và data pipeline

**File:** `selfplay/replay_buffer.py`, `train/trainer.py`, `selfplay/storage.py` (mới).

- [ ] Tránh random index nhiều lần trên `collections.deque` lớn; chuyển sang ring buffer numpy/memmap hoặc array blocks cho O(1) truy cập.
- [ ] Các feature planes nhị phân lưu `uint8`, policy `float16` nếu xác nhận sai số cho phép, outcome `int8`; sang FP32/FP16 khi batch train.
- [ ] Metadata mỗi sample/game: `board_size`, `komi`, `termination_reason`, `model_version`, `sims`, `is_full_search`, `seed`.
- [ ] Lưu shards theo phiên bản với checksum + atomic rename; migrate dữ liệu pickle cũ **chỉ từ nguồn tin cậy**.
- [ ] D4 Augmentation dùng đúng policy map + pass; test roundtrip save/load và 200k random samples.
- [ ] Nếu mixed sizes: group batch theo size hoặc padding+mask đã test; sampling balance để tránh data 9×9 lấn át 19×19.

**PASS:** loss không sai lệch ngoài tolerance; giảm RAM đo được; buffer resume không đổi số mẫu, policy entropy và z.

## P6 — Hiệu quả thuật toán, sau khi correctness và batching đã ổn

**File:** `mcts/search.py`, `selfplay/worker.py`, `config.py`, `train/trainer.py`.

- [ ] Cấu hình Dirichlet alpha theo board size; **không mang mặc định 0.30 của bản 9×9 sang 19×19**. Có thể thử ~0.03 (giá trị AlphaZero Go 19×19), sweep 0.03, 0.05, 0.10.
- [ ] Introduce playout cap randomization: `full_sims=200`, `fast_sims=60`, `full_search_probability=0.25` là **điểm thử nghiệm**, không phải cấu hình tối ưu; chỉ dùng full-search positions làm high-quality policy targets; báo rõ giá mỗi sample.
- [ ] Tăng temperature exploration phase theo board size, nhưng đánh giá bằng diversity, entropy và winrate.
- [ ] Thêm auxiliary **ownership** / **score difference** khi terminal labels hợp lệ; cân loss coefficients, không thay đổi value/policy APIs ngoài backward-compatible extras.
- [ ] So sánh model 6×96 với 6×128, 8×128; dùng **strength per wall-clock GPU-hour** thay vì chỉ accuracy/loss.
- [ ] Tùy chọn curriculum 9 -> 13 -> 19. Nếu tabula-rasa, vẫn giữ tự đấu, không đưa human data vào.

**PASS:** 19×19 winrate vs frozen baseline cải thiện trên tập benchmark đủ mẫu, với cùng ngân sách GPU giờ.

## P7 — Triển khai vận hành 19×19 / checklists

- [ ] Profile sử dụng tài nguyên: GPU utilization, peak VRAM, CPU utilization per worker, process queue depth.
- [ ] Test Pytorch AMP và channels-last, cudnn benchmark, `torch.compile` **từng tính năng một**; `torch.compile` có thể tăng VRAM, đặc biệt mode reduce-overhead.
- [ ] Ghi training metrics: positions/hour, games/hour, expansions/s, evals/s, avg inference batch, policy/value/aux losses, rejected/accepted, winrate, strength@fixed search budget, model/board size, game-length percentiles, pass rate, false resign rate.
- [ ] Test fault injection: kill actor/GPU OOM/power interruption, atomic recovery, corrupt file handling.
- [ ] Train smoke test 5×5; train test 19×19 1 iteration với game length ngắn **chỉ test pipeline**, không dùng game truncated cho training chính thức.
- [ ] Chạy pilot 19×19 qua vài nghìn ván (khả năng tùy phần cứng), benchmark against random and immutable historic checkpoints; chỉ nâng ngân sách khi throughput và strength tốt lên.

**PASS:** qua end-to-end, load-resume không chệch config, A/B benchmark có lợi ích, 19×19 không OOM và winrate test đáng tin cậy.

## THỨ TỰ PR/KẾ HOẠCH COMMIT (không gộp)

1. `test(bench): add reproducible 19x19 profiling`
2. `fix(pipeline): synchronize checkpoint resume and lr scheduler`
3. `fix(go): distinguish terminal and truncated games`
4. `feat(nn): board-size independent spatial heads`
5. `perf(go): faster legality and board state implementation`
6. `feat(nn): batched masked inference API`
7. `feat(selfplay): multi-process actor inference queue`
8. `perf(buffer): compact random-access experience storage`
9. `feat(mcts): board-dependent exploration and playout caps`
10. `feat(train): ownership and score auxiliary supervision`
11. `test(arena): fixed-budget 19x19 evaluation and fault recovery`

## KPI/CHẤT LƯỢNG (đo trước/sau, không được tự bịa)

| KPI | Cách đo | Tiêu chí phát hành |
| --- | --- | --- |
| `NN eval/s` | CUDA warmup + synchronize, batch 1..128 | tăng hoặc có giải thích vì sao đổi model mạnh hơn |
| `MCTS nodes/s` | cùng model/sims/board fixture | không giảm do bug |
| `positions/hour` | 1 giờ wall-clock với full pipeline | tốt hơn baseline với cùng tài nguyên |
| `games/hour` | selfplay 19×19, game lengths thống kê | tăng và không ép kết thúc sai |
| Peak VRAM/RAM | `nvidia-smi`, process counters | không OOM trên target hardware |
| `resume equivalence` | split-run vs uninterrupted run | khớp về state nếu deterministic; khi multiprocessing đối chiếu invariant và lịch |
| Arena winrate | đổi màu, ghép khai cuộc, fixed sims | độ bất định rõ ràng, hơn baseline đủ ý nghĩa |
| Game label correctness | completed vs truncated + oracle | 0 nhãn tự gán cho ván truncated |

## DỰ TOÁN BỘ NHỚ BUFFER 19×19 (200k mẫu)

- Float32 17×19×19 features + float32 362 policy: khoảng **5.20 GB thô (~4.84 GiB)** chưa tính Python objects, fragmentation và learner batches.
- Uint8 features + float16 362 policy: khoảng **1.37 GB thô (~1.28 GiB)**, giảm xấp xỉ **3.79 lần**. Kết quả thực tế phụ thuộc metadata và layout.
- `np.packbits()` còn nén tốt hơn cho feature nhị phân nhưng cần benchmark chi phí decode.

## HARDWARE VÀ PROFILE KHUYẾN NGHỊ

- RTX 3050 **4 GB** (được nêu trong README repo): ưu tiên 4–6 blocks, 64–96 filters, minibatch 8–32, gradient accumulation, batch inference thận trọng. CPU có thể nghẽn trước GPU.
- RTX 4060 **8 GB** (nếu đó là GPU thực tế dùng train): bắt đầu 6×96, minibatch 32–64, AMP, actor 2–4 và inference batch 8–32. Không nên bật GPU learner + server song song trước khi test VRAM.
- Khởi điểm thử nghiệm `sims=64–200` tùy giai đoạn, `full_search=200/fast=60` chỉ sau khi chuẩn hóa target, `resign disabled` ở warm-up.
- Các profile chỉ là mức bắt đầu để benchmark, không đảm bảo phù hợp mọi CPU/RAM/GPU.

## MẪU YÊU CẦU OPENCode MỖI PR

```
Bạn đang chỉnh sửa CoVayAI theo CoVayAI_19x19_Upgrade_Roadmap.md.
Chỉ triển khai PR [X] và tuyệt đối không làm sang PR tiếp theo.
1) Phân tích current code + test, liệt kê invariants và rủi ro.
2) Viết regression test FAIL trước khi sửa nếu là bug; benchmark baseline nếu tối ưu.
3) Thực hiện thay đổi nhỏ, backward-compatible khi có thể.
4) Chạy unit/integration tests và benchmark liên quan.
5) Báo danh sách file thay đổi, expected vs observed, số liệu thực tế, lệnh test, limitations.
6) Nếu test FAIL hoặc không thể benchmark, dừng và báo nguyên nhân. Không tự gắn nhãn PASS.
7) Không sửa/chạm vào checkpoints và buffer hiện có. Không được giả định pretraining 9x9 heads dùng trực tiếp cho 19x19.
```

## THAM KHẢO

- CoVayAI: https://github.com/luongviettienns/CoVayAI
- KataGo Methods: https://github.com/lightvector/KataGo/blob/master/docs/KataGoMethods.md
- KataGo training configs: https://github.com/lightvector/KataGo/tree/master/cpp/configs/training
- KataGo paper: https://arxiv.org/abs/1902.10565
- PyTorch tuning: https://docs.pytorch.org/tutorials/recipes/recipes/tuning_guide.html


---

# PHẦN II — KIỂM SOÁT CHI PHÍ HUẤN LUYỆN (MỚI V3)

## C0 — Định nghĩa chi phí, đo trước khi tối ưu (BLOCKER)

Không dùng `seconds/step` hay `% GPU` làm KPI kinh tế duy nhất. Cần đo **chất lượng theo đồng tiền**:

- `total_compute_cost = gpu_hour_cost + incremental_cpu_and_electricity_cost + storage + transfer + other_fees`.
- `cost_per_1k_completed_games = total_compute_cost / completed_games * 1000`.
- `cost_per_1m_nn_evals = total_compute_cost / nn_evals * 1_000_000`.
- `cost_per_1k_full_search_positions = total_compute_cost / full_search_positions * 1000`.
- `cost_per_100_elo` chỉ dùng khi rating đo với **benchmark cố định, số game đủ và khoảng tin cậy**, không ép Elo tăng giả tạo.
- Với GPU tại nhà, so sánh theo **chi phí gia tăng** (điện, làm mát, hao mòn nếu tính) và không trừ toàn bộ giá phần cứng đã sở hữu; nếu quyết định MUA GPU mới thì phải tính thêm khấu hao/mua mới.
- Với cloud, luôn gồm chi phí ổ cứng khi máy dừng, snapshot/checkpoint, thời gian provision/download/warm-up, chi phí phát sinh theo chính sách nhà cung cấp.
- Đo `wall_hours_per_phase`, `energy_kWh` (nếu có đồng hồ đo), `GPU utilization`, `CPU utilization`, `games/hour`, `full_search_positions/hour`, `NN eval/s`, `arena strength@fixed search budget`.

**Việc giao OpenCode:** tạo `benchmarks/cost_report.py` và `benchmarks/cost_schema.json` cho input JSON gồm GPU price/h, total running hours, energy, electricity tariff, storage fees, evals/games/samples; tính công thức với input, không tự lấy số giá từ Internet và không tạo throughput giả. CSV/JSON báo cáo gắn với git SHA, board size, model architecture, search budget, seed, hardware.

**Test PASS:** các giá trị nhập vào 0, giá trị rất lớn, missing metrics; không chia cho 0; so chi phí hợp lý giữa 2 runs; báo `N/A` nếu thiếu games/evals thay vì tự suy luận.

## C1 — Không tiêu hao GPU khi CPU đang nghẽn (ưu tiên 1)

**Tệp:** `go/board.py`, `mcts/search.py`, `selfplay/actor_pool.py`, `selfplay/inference_server.py`.

- [ ] Benchmark phân tách thời gian `board.copy`, `get_legal_moves`, `_simulate_move`, NN batching, waiting queue, serialization.
- [ ] Chỉ tối ưu engine sau khi xác định tỷ trọng CPU lớn. `list.pop(0)` -> `deque.popleft()` là bước ít rủi ro; sau đó mới cân nhắc chain/liberty incremental.
- [ ] Gom leaf requests từ nhiều ván; dùng **một** GPU inference worker giữ model, CPU actors độc lập. Giới hạn queue để không tăng RAM vô hạn.
- [ ] Trên Windows dùng `spawn`; GPU model không copy sang mỗi CPU actor. Batch theo board size, dùng timeout để không chết cứng.
- [ ] Thử batch 1, 8, 16, 32, 64; giữ cấu hình nào cải thiện **throughput toàn pipeline** thay vì chỉ micro-benchmark.
- [ ] Cache NN inference (nếu dùng) phải hash theo **đủ 17 feature planes/8 trạng thái lịch sử + lượt đi**, không chỉ Zobrist của hình cờ hiện tại. Mask legal moves cần đúng superko/historical legality; cache raw logits/value và mask ngoài để tránh lấy nhầm action.

**PASS:** kết quả move legality / policy / value không đổi ngoài tolerance; `completed_games/hour` tăng; không OOM; GPU time per completed game giảm trên benchmark công bằng.

## C2 — Phân bổ MCTS budget thích nghi, không phá nhãn (ưu tiên 1)

**Tệp:** `config.py`, `mcts/search.py`, `selfplay/worker.py`, `train/trainer.py`.

- [ ] Thêm `search_budget_mode: fixed|randomized_cap` mặc định `fixed` để có A/B an toàn.
- [ ] Profile thử `p_full=0.25`, `full_sims=200`, `fast_sims=60`: trung bình lý thuyết `0.25*200+0.75*60=95 sims/move`, bớt 52.5% simulations so với 200 cố định. Không hứa bớt 52.5% wallclock.
- [ ] **Chỉ lưu chính sách π của lượt `full_search`** như policy target; `fast_search` có thể dùng để đi nước, nhưng không gán `π` độ tin cậy cao. Nếu mở rộng thành value-only records phải điều chỉnh loss và batch schema rõ ràng.
- [ ] Với fast-search, cân nhắc tắt exploration noise như phương pháp KataGo; đảm bảo game vẫn đủ đa dạng trong thống kê.
- [ ] `full-search percentage` có thể thay đổi theo phase và giữ audit log; khi không đủ target, tự rollback sang fixed budget.
- [ ] Không dùng kết quả resign sớm hoặc ván truncated làm z khi nhãn không đáng tin.

**PASS:** thống kê đúng sims thực tế, policy targets chỉ đến từ full-search; ít nhất 3 seeds/budget-matched A/B cho policy loss, outcomes và fixed-budget Arena. Nếu giảm chi phí nhưng sức chơi kém đáng kể, không bật mặc định.

## C3 — Giảm số optimizer steps thừa (ưu tiên 2)

**Tệp:** `main.py`, `train/trainer.py`, `selfplay/replay_buffer.py`.

- [ ] Record số mẫu mới `new_full_search_positions`, unique completed games, sample reuse count, effective epochs = `steps*batch_size/available_train_samples`.
- [ ] Giới hạn train steps theo số **mẫu đủ chất lượng mới** và dữ liệu trong replay window; không train 400 steps mặc định mù quáng ở các iteration đầu.
- [ ] Tách holdout theo **ván đấu** (không random theo positions của cùng game) để đo validation policy/value; D4 chỉ dùng train.
- [ ] Theo dõi val loss, KL(policy_learner || search_policy), mean value calibration; nếu overfit liên tiếp hoặc không tăng strength vs fixed benchmark, giảm train steps/learning rate và xem lại dữ liệu trước khi tiếp tục.
- [ ] Không dừng train chỉ vì một trận Arena ngắn thua, hoặc vì loss dao động nhỏ. Dùng rolling windows + giới hạn ngân sách, không gán tương đương loss với Elo.
- [ ] Thử optimizer groups + AMP hiện có, channels-last và `torch.compile` **từng tính năng riêng**; benchmark thời gian warm-up và peak VRAM. Có flag tắt nếu không tiết kiệm.

**PASS:** A/B với **cùng số ván hoàn tất**, tiết kiệm GPU-hour/đơn vị strength; không tăng overfit, không xảy ra NaN/OOM.

## C4 — Nén buffer, giảm I/O/chi phí ổ đĩa (ưu tiên 2)

**Tệp:** `selfplay/replay_buffer.py`, `selfplay/storage.py`, `train/trainer.py`.

- [ ] Lưu feature 17×N×N nhị phân bằng uint8 hoặc bitpack; π float16 chỉ sau test `abs/relative error`, sum-normalization, entropy, gradient parity; z int8.
- [ ] Trên 19×19, 200k samples: float32 features+policy ~5.20 GB thô; uint8 features+FP16 policy ~1.37 GB thô (chưa tính object/layout). Đo bằng RSS và bytes thực, không coi con số lý thuyết là RAM thực tế.
- [ ] Chuyển deque random-index sang numpy ring/chunk array/memmap; tránh pickling toàn bộ buffer thành một file lớn mỗi iteration.
- [ ] Shard dữ liệu có version, checksum, atomic writes; test kill process giữa save và resume; không tải pickle không tin cậy từ Internet.
- [ ] Giảm checkpoint kích thước nếu chiến lược snapshot cho phép, **nhưng luôn giữ model tốt nhất + metadata + state tối thiểu để phục hồi chính xác**.

**PASS:** load/save roundtrip + D4 parity + policy normalization đúng; memory và bytes lưu giảm; không mất dữ liệu đã xác nhận.

## C5 — Chỉ trả tiền thuê GPU khi nhanh hơn đủ nhiều (ưu tiên 3)

**Bảng giá tham khảo từ RunPod Pods công bố 08-10-2026 (USD/h), có thể đổi và tùy khu vực/tình trạng máy:**

| GPU | VRAM | USD/h | 100 giờ | 200 giờ |
| --- | ---: | ---: | ---: | ---: |
| RTX A5000 | 24 GB | 0.27 | 27 | 54 |
| RTX 3090 | 24 GB | 0.50 | 50 | 100 |
| RTX 4090 | 24 GB | 0.89 | 89 | 178 |
| RTX 5090 | 32 GB | 1.19 | 119 | 238 |

Nguồn: https://www.runpod.io/pricing ; không bao gồm lưu trữ, các khoản phí khác và thời gian GPU nhàn rỗi; pricing khác nhau giữa Pods và Serverless. Với Vast.ai, báo giá marketplace thay đổi liên tục; chỉ dùng giá tại thời điểm đặt và phải benchmark chính instance đó: https://vast.ai/pricing .

**Khuyến nghị thuê GPU**:
1. Chạy benchmark thật từ cùng một commit/weights và danh sách benchmark states trên GPU cá nhân.
2. Chỉ thuê GPU cloud chạy **benchmark ngắn** trước, tránh trả tiền nhiều ngày cho pipeline đang chậm.
3. Đo `cost_per_1k_full_search_positions` và `cost_per_1k_completed_games` theo *thông lượng thực tế*, không suy luận từ TFLOPS.
4. Ước lượng break-even dựa trên công thức `cloud_cost_per_game = price_hour / completed_games_per_hour` và cộng storage overhead; nếu cloud không giảm thời gian đủ để bù giá, trở về local.
5. Với interruptible/spot: checkpoint theo thời gian, upload thành công rồi mới xóa Pod; phải tính checkpoint/restore overhead và rủi ro preemption; không dùng nếu lưu chưa an toàn.

**Ví dụ điện ở nhà (CHỈ GIẢ ĐỊNH):** tải đo được trung bình `0.30 kW` của **cả hệ thống**, giá điện giả định `3,000 VND/kWh`, 200 giờ sẽ tiêu thụ `60 kWh` và chi phí điện khoảng `180,000 VND` chưa gồm làm mát, hao mòn, thuế và bậc giá thực. Không dùng giá bán lẻ điện bình quân quốc gia làm hóa đơn điện thực tế.

**PASS:** báo cáo giá/hiệu năng theo từng GPU trên cùng workload; không duyệt thuê GPU bằng cảm tính.

## C6 — Ngân sách pilot, gate trước khi mở rộng (ưu tiên 3)

1. **G0 - Baseline không thuê GPU:** chạy tests, profiler, benchmark 9×9/19×19; lập bảng bottleneck và dự đoán chi phí với số đo thật.
2. **G1 - Smoke 19×19:** 1–2 iteration để kiểm tra bộ nhớ, checkpoint, game termination/labels. Không đánh giá strength từ ván cắt ngắn.
3. **G2 - A/B 3 biến độc lập:** batching (on/off), fast/full playout (on/off), train steps adaptive (on/off). Mỗi thí nghiệm có giá thành và số game/target hữu ích; giữ các biến khác cố định.
4. **G3 - Pilot hạn mức:** khóa **ngân sách tuyệt đối trước khi chạy** (ví dụ cloud $30–$100; chỉ là mức mẫu), lưu audit, dừng khi chạm giới hạn hoặc validation/strength regression đáng kể.
5. **G4 - Scale:** tăng cloud spending **chỉ nếu** model thực sự tốt lên so với frozen baseline ở cùng search budget và chi phí/strength đủ tốt. Vượt ngân sách phải yêu cầu con người duyệt, không được tự tạo Pod/trừ tiền.

**Không đồng nhất:** full-search policy positions tăng không có nghĩa strength tăng; N GPU evals tăng không có nghĩa tiết kiệm; Elo có thể dao động do ít ván.

## C7 — Quy tắc chọn mô hình theo hiệu quả thay vì kích thước lớn

- So sánh `4×64`, `6×96`, `6×128` và (khi đủ nguồn lực) `8×128` bằng **same compute budget**, cùng test arena. Trên RTX 4060 8GB (nếu dùng máy này): 6×96 là điểm benchmark ban đầu, không cam kết cấu hình nào thắng.
- Chọn ứng viên dựa trên `strength improvement / GPU-hour`, không phải model loss thấp nhất.
- Nếu mục tiêu từ **tabula rasa**, curriculum 9→13→19 vẫn từ self-play, không dùng pretrained của bên ngoài; có thể dùng self-trained backbone với head mới.
- Nếu cần rẻ và mạnh nhanh: mở một **nhánh thử nghiệm riêng** sử dụng supervised/teacher distillation từ engine đã mạnh, với điều khoản dữ liệu/model hợp lệ và ghi rõ **không còn tabula rasa**. Chỉ chọn khi mục tiêu sản phẩm ưu tiên chất lượng/chi phí hơn nghiên cứu tự học từ đầu.

## C8 — Bộ test kiểm soát chi phí bắt buộc trước cloud

- [ ] Test scheduler/checkpoint/resume của V2 PASS.
- [ ] Test không bỏ lẫn policy targets từ fast-search.
- [ ] Test buffer compressed roundtrip và checksum.
- [ ] Test worker timeout/recovery, không mất game hoàn tất.
- [ ] Test cost report đầu vào 0/bị thiếu/âm; từ chối âm, `N/A` khi mẫu số 0.
- [ ] Test cloud interruption/kill job, resume không sai global step/best model.
- [ ] Test chi phí theo phase cộng đúng tổng; unit rõ USD, VND, kWh, hour.
- [ ] Test A/B fixed-seed, frozen weights, cùng board states và độ chính xác policy/value được chấp nhận.
- [ ] Nếu GPU load thấp nhưng CPU load cao: xử lý CPU trước khi đề xuất mua/thuê GPU mạnh.

## Thứ tự các PR tiết kiệm chi phí (sau các blocker P0-P1 của V2)

1. `feat(metrics): add cost and throughput audit, no optimization`.
2. `perf(cpu): profile and optimize board legal move generation`.
3. `feat(inference): batch GPU eval across CPU selfplay actors`.
4. `feat(mcts): randomized search budgets and tagged policy targets`.
5. `perf(buffer): compact storage and random access`.
6. `feat(train): compute-aware train steps and held-out games validation`.
7. `perf(torch): opt-in benchmarked memory format and compile`.
8. `feat(ops): cost gates, reproducible cloud benchmark and safe stop`.

## Prompt giao việc OpenCode cho MỘT PR về chi phí

```text
Bạn là kỹ sư hiệu năng ML làm việc trên CoVayAI.
Đọc toàn bộ CoVayAI_19x19_Cost_Optimized_V3.md.
Chỉ triển khai PR [TÊN PR], không sửa phần khác và không tự chạy training tốn tiền.
1. Ghi git SHA, GPU/CPU/RAM, toolchain, các test hiện có.
2. Viết regression test hoặc benchmark tái lập được trước khi sửa.
3. Đo baseline thực tế trên cùng dataset/seed/weights; ghi JSON/CSV.
4. Sửa nhỏ nhất có thể, giữ nguyên behavior và kết quả cờ vây.
5. Chạy đầy đủ tests + benchmark sau thay đổi; so các KPI workload và cost.
6. Báo lợi ích %, peak VRAM/RAM, CPU%, GPU%, throughput, lỗi/limitations.
7. Nếu chưa có benchmark hợp lệ, báo 'chưa xác minh', không tự ghi PASS.
8. Không thay đổi checkpoint/buffer cũ, không bật cloud, không chi tiền hay tự mở rộng quota.
9. Khi test FAIL, hoàn tác thay đổi trong PR thay vì sửa chồng nhiều module.
10. Kết thúc bằng checklist PASS/FAIL và commit message đề xuất.
```

## Nguồn

- CoVayAI: https://github.com/luongviettienns/CoVayAI
- KataGo self-play: https://arxiv.org/abs/1902.10565
- RunPod Pods (giá đối chiếu 08-10-2026): https://www.runpod.io/pricing
- Vast.ai động: https://vast.ai/pricing
- PyTorch performance: https://docs.pytorch.org/tutorials/recipes/recipes/tuning_guide.html
