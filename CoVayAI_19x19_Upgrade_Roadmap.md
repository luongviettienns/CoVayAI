# CoVayAI — Lộ trình nâng cấp training 19×19 (V2)

**Baseline đã đối chiếu:** `luongviettienns/CoVayAI`, nhánh `main`, commit `5f4898c1f256d38cc3c4a06960799ab980c66a5b` (20-10-2026).

**Mục tiêu:** AI cờ vây 19×19 học ổn định, sinh dữ liệu self-play nhanh, tiết kiệm GPU/RAM, có bằng chứng cải thiện sức mạnh. Không tự nhận đã đạt hiệu năng cụ thể khi chưa benchmark trên máy thật.

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

- [x] `git switch -c feat/train-19x19`; ghi commit gốc, Python/PyTorch/CUDA/GPU/VRAM/CPU/RAM.
- [x] `python -m pytest -q` và lưu `test_baseline.txt`.
- [x] Ghi `board.copy/s`, `get_legal_moves/s`, `play/s`, `nodes/s`, `NN evaluations/s`, `positions/s`, `games/hour`, `GPU utilization`, peak VRAM/RAM.
- [x] Tách benchmark infer batch = 1, 8, 16, 32, 64, 128 trên board 9, 13, 19; có warmup, `torch.cuda.synchronize()` trước và sau timing.
- [x] Có ít nhất ba phân bố state: opening, middle-game và late-game; không chỉ benchmark bàn trống.
- [x] Output CSV/JSON và log có cấu hình, git SHA, seed; P0 không đổi thuật toán.

**PASS:** có baseline tái chạy được; tất cả test cũ xanh; số liệu trước tối ưu đầy đủ.

## P1 — Sửa tính đúng đắn và khả năng train dài (blocker)

**File:** `main.py`, `train/trainer.py`, `go/board.py`, `go/scoring.py`, `selfplay/worker.py`, `tests/test_pipeline.py`, `tests/test_train/`.

- [x] Scheduler global iteration/step: sau promote/reject đều tiến đúng một lần; state phải lưu **sau** scheduler step; resume từ accepted/rejected đều đúng LR, momentum, scheduler epoch.
- [x] Checkpoint atomically: temp + `os.replace`, lưu `model`, `optimizer`, `scheduler`, `scaler`, `global_step`, `iteration`, RNG states, model metadata/config/version, source git SHA.
- [x] Resume với `--no-resume` không giữ `best_model.pt` cũ; không đồng nhất `checkpoint_iter_i.pt` của candidate rejected với accepted best.
- [x] Self-play: disable resign mặc định cho warm-up 19×19; riêng Arena tắt resign lúc benchmark chuẩn hoặc hiệu chuẩn độc lập.
- [x] Luật cờ/score: kiểm định bằng bộ tình huống Ko/Superko/SeKi/2-pass; double-pass và dead stones cần đối chiếu engine độc lập; phân biệt full terminal với truncated max_moves.
- [x] Nếu ván kết thúc bởi max_moves (19×19: 722 mặc định), lưu metadata `termination_reason=truncated`, không tạo z outcome từ score tạm một cách mặc định.
- [x] Không đổi mẫu `(state, pi, z)` trực tiếp nếu chưa viết migration/backwards compatibility.

**PASS:** 3+ iteration accepted/rejected giả lập; dừng/khôi phục cho cùng model/optimizer/LR/buffer state; không có outcome label giả do truncation.

## P2 — Neural network dùng chung 9×9 / 13×13 / 19×19

**File:** `nn/model.py`, `nn/features.py`, `train/trainer.py`, `tests/test_nn/`.

- [x] Giữ backbone conv + ResBlock; thay `policy_fc(2*N*N -> N*N+1)` bằng `policy_conv -> spatial logits [B,N,N]` cộng `pass_head(global pooled features) -> [B,1]`.
- [x] Thay `value_fc1(N*N ->128)` bằng masked/global pooling `[B,C,N,N] -> [B,C]` + MLP -> tanh value.
- [x] Nếu trộn kích thước trong cùng batch: board mask phải được xử lý trong pooling, loss, BN/conv theo thiết kế; phiên bản đầu nên group batches theo board-size để tránh complexity.
- [x] Thêm metadata về komi và kích thước bàn (nếu các ván có komi khác nhau) vào đầu vào phù hợp; không dùng cùng trọng số value mà bỏ qua khác biệt komi.
- [x] Maintain `policy logits = N*N+1`, pass cuối, value `[B,1]`, legal-mask chỉ áp dụng lên action hợp lệ; phép D4 giữ nguyên pass.
- [x] Migration: chỉ copy backbone weights khi compatible; heads mới random init, lưu `architecture_version`; test checkpoint cũ ra cảnh báo rõ.
- [x] Tạo test input board 9, 13, 19; backward finite, gradients nonzero; compare symmetry transformations.

**PASS:** 1 model forward/backward trên cả 3 cỡ bàn; xác suất hợp lệ; không phụ thuộc N ở state_dict của heads.

## P3 — CPU Go engine đủ nhanh cho 19×19

**File:** `go/board.py`, `go/scoring.py`, `go/zobrist.py`, `benchmarks/bench_board.py`.

- [x] Thay BFS `list.pop(0)` bằng `collections.deque` hoặc stack; đo hiệu quả.
- [x] Tách fast path legality khỏi `simulate_move`/`play`; hạn chế copy grid và copy history tại mỗi candidate.
- [x] Dùng cấu trúc chain/liberty cập nhật incremental (union-find hoặc nhóm dây + tập khí) **chỉ sau khi differential tests đã đủ**.
- [x] `Board.copy()` cần giữ lịch sử ko đầy đủ theo đúng luật trong khi chỉ giữ 8 board history gần nhất cho NN; test cả trường hợp mcts rollout dài.
- [x] Nghiên cứu Numba/Cython/Rust/C++ khi profiling chứng minh Python rules chiếm đa số runtime.
- [x] Differential / property-based tests: 10k+ random legal games, score/pass/captures; so sánh old/new trên cùng game seeds.

**PASS:** 0 sai lệch legality/score/hash so với oracle đã chọn; cải thiện `get_legal_moves/s` và `play/s` được đo, không đặt mức tăng tùy tiện.

## P4 — Inference batching và nhiều actor self-play

**File:** `mcts/search.py`, `selfplay/worker.py`, `selfplay/inference_server.py` (mới), `selfplay/actor_pool.py` (mới), `nn/model.py`.

- [x] API `predict_batch(states, legal_masks)`: đầu vào [B,C,N,N], mask [B,N*N+1]; đầu ra policy [B,N*N+1], value [B].
- [x] Mỗi actor CPU chạy game/MCTS độc lập; request NN tới inference server GPU qua queue; server batch theo max_batch và max_wait_ms, gom cùng cỡ bàn.
- [x] Dùng Windows `multiprocessing.get_context('spawn')` và `if __name__ == '__main__'`; không gửi CUDA model qua queue; seed độc lập mỗi actor.
- [x] Nếu dùng multiple MCTS in-flight trong **cùng** tree cần virtual loss/virtual visit và thread-safe backup; phiên bản đầu nên batch leaf giữa nhiều game để đơn giản.
- [x] Queue bounded + timeout/retry/healthcheck; stop sạch khi `KeyboardInterrupt`/worker crash; không ghi nửa mẫu game khi đang lỗi.
- [x] `torch.inference_mode`, AMP cho inference, hạn chế `.item()` và đồng bộ GPU per-sample, `model.eval()` một lần; CUDA metrics.
- [x] Nếu 4GB VRAM: run selfplay và learner luân phiên, không đồng thời; RTX 4060 8GB có thể thử chia workload sau khi xác nhận peak VRAM.

**PASS:** batch=1 gần đồng nhất với API cũ; batch=8/16/32 outputs tương đương trong tolerance; không deadlock 1000 games; `positions/hour` tăng trong benchmark đúng.

## P5 — ReplayBuffer và data pipeline

**File:** `selfplay/replay_buffer.py`, `train/trainer.py`, `selfplay/storage.py` (mới).

- [x] Tránh random index nhiều lần trên `collections.deque` lớn; chuyển sang ring buffer numpy/memmap hoặc array blocks cho O(1) truy cập.
- [x] Các feature planes nhị phân lưu `uint8`, policy `float16` nếu xác nhận sai số cho phép, outcome `int8`; sang FP32/FP16 khi batch train.
- [x] Metadata mỗi sample/game: `board_size`, `komi`, `termination_reason`, `model_version`, `sims`, `is_full_search`, `seed`.
- [x] Lưu shards theo phiên bản với checksum + atomic rename; migrate dữ liệu pickle cũ **chỉ từ nguồn tin cậy**.
- [x] D4 Augmentation dùng đúng policy map + pass; test roundtrip save/load và 200k random samples.
- [x] Nếu mixed sizes: group batch theo size hoặc padding+mask đã test; sampling balance để tránh data 9×9 lấn át 19×19.

**PASS:** loss không sai lệch ngoài tolerance; giảm RAM đo được; buffer resume không đổi số mẫu, policy entropy và z.

## P6 — Hiệu quả thuật toán, sau khi correctness và batching đã ổn

**File:** `mcts/search.py`, `selfplay/worker.py`, `config.py`, `train/trainer.py`.

- [x] Cấu hình Dirichlet alpha theo board size; **không mang mặc định 0.30 của bản 9×9 sang 19×19**. Có thể thử ~0.03 (giá trị AlphaZero Go 19×19), sweep 0.03, 0.05, 0.10.
- [x] Introduce playout cap randomization: `full_sims=200`, `fast_sims=60`, `full_search_probability=0.25` là **điểm thử nghiệm**, không phải cấu hình tối ưu; chỉ dùng full-search positions làm high-quality policy targets; báo rõ giá mỗi sample.
- [x] Tăng temperature exploration phase theo board size, nhưng đánh giá bằng diversity, entropy và winrate.
- [x] Thêm auxiliary **ownership** / **score difference** khi terminal labels hợp lệ; cân loss coefficients, không thay đổi value/policy APIs ngoài backward-compatible extras.
- [x] So sánh model 6×96 với 6×128, 8×128; dùng **strength per wall-clock GPU-hour** thay vì chỉ accuracy/loss.
- [x] Tùy chọn curriculum 9 -> 13 -> 19. Nếu tabula-rasa, vẫn giữ tự đấu, không đưa human data vào.

**PASS:** 19×19 winrate vs frozen baseline cải thiện trên tập benchmark đủ mẫu, với cùng ngân sách GPU giờ.

## P7 — Triển khai vận hành 19×19 / checklists

- [x] Profile sử dụng tài nguyên: GPU utilization, peak VRAM, CPU utilization per worker, process queue depth.
- [x] Test Pytorch AMP và channels-last, cudnn benchmark, `torch.compile` **từng tính năng một**; `torch.compile` có thể tăng VRAM, đặc biệt mode reduce-overhead.
- [x] Ghi training metrics: positions/hour, games/hour, expansions/s, evals/s, avg inference batch, policy/value/aux losses, rejected/accepted, winrate, strength@fixed search budget, model/board size, game-length percentiles, pass rate, false resign rate.
- [x] Test fault injection: kill actor/GPU OOM/power interruption, atomic recovery, corrupt file handling.
- [x] Train smoke test 5×5; train test 19×19 1 iteration với game length ngắn **chỉ test pipeline**, không dùng game truncated cho training chính thức.
- [x] Chạy pilot 19×19 qua vài nghìn ván (khả năng tùy phần cứng), benchmark against random and immutable historic checkpoints; chỉ nâng ngân sách khi throughput và strength tốt lên.

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
