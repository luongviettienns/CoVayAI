# 🥋 CoVayAI (GoAI) - AlphaZero for Weiqi / Go

> **Hệ thống AI chơi Cờ Vây tự học hoàn chỉnh theo thuật toán AlphaZero (DeepMind) từ tabula rasa.**

[![Python](https://img.shields.io/badge/Python-3.13-blue.svg)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.6.0%2Bcu124-red.svg)](https://pytorch.org/)
[![CUDA](https://img.shields.io/badge/CUDA-12.7%20%2F%20RTX%203050-green.svg)](https://developer.nvidia.com/cuda-zone)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

---

## 📌 1. Giới thiệu dự án

**CoVayAI** là dự án xây dựng toàn diện một engine AI chơi Cờ vây theo trường phái học tăng cường (Deep Reinforcement Learning) mô phỏng chính xác thuật toán **AlphaZero**:
- **Tabula Rasa**: Không sử dụng bất kỳ dữ liệu ván đấu nào của con người, AI tự học thông qua hàng triệu ván tự đấu (Self-play).
- **Luật cờ Trung Quốc (Chinese Rules)**: Áp dụng phương thức tính điểm diện tích (Area Scoring) và luật cấm lặp lại hình cờ toàn cục (Positional Superko).
- **Chuẩn bàn cờ**: Khởi đầu tối ưu trên bàn **$9 \times 9$** (huấn luyện nhanh, hội tụ tốt trên máy cá nhân với GPU RTX 3050 4GB) và mở rộng cho bàn **$19 \times 19$**.
- **Chuẩn giao tiếp GTP (Go Text Protocol v2)**: Sẵn sàng kết nối trực tiếp với các phần mềm đồ họa phổ biến như Sabaki, Lizzie, Katrain, hoặc thi đấu với GNU Go.

---

## 🏗️ 2. Kiến trúc Hệ thống

```
CoVayAI/
├── go/         # Trọng tài & Cơ chế bàn cờ (Luật Trung Quốc)
│   ├── board.py       # Biểu diễn bàn cờ, chuỗi quân (chains), khí (liberties), bắt quân, cấm tự sát
│   ├── zobrist.py     # Băm Zobrist hash 64-bit cho thế cờ, kiểm soát Positional Superko
│   └── scoring.py     # Tính điểm diện tích (Area scoring), áp dụng komi (7.5), xác định kết quả ván cờ
├── nn/         # Mạng nơ-ron sâu Dual-Head ResNet
│   ├── features.py    # Trích xuất đặc trưng Tensor N×N×C (chuẩn hóa góc nhìn canonical của người đi)
│   └── model.py       # Kiến trúc Dual-Head: Policy Head (N²+1 logits) & Value Head (tanh ∈ [-1, 1])
├── mcts/       # Thuật toán tìm kiếm cây Monte Carlo (MCTS)
│   ├── node.py        # Cấu trúc nút MCTS lưu N(s,a), W(s,a), Q(s,a), P(s,a)
│   └── search.py      # Thuật toán PUCT: Select -> Expand/Mask -> Backup, Dirichlet noise, Subtree reuse
├── selfplay/   # Sinh dữ liệu tự đấu
│   ├── worker.py      # Vòng lặp tự chơi sinh mẫu (s, π, z), xử lý resign threshold và max moves
│   └── replay_buffer.py # Bộ nhớ đệm trượt (sliding window), Augmentation 8 phép đối xứng nhóm D4
├── train/      # Tối ưu hóa mô hình
│   └── trainer.py     # Hàm mất mát tổng hợp L = (z - v)² - πᵀ log p + c||θ||², optimizer, lr scheduler
├── arena/      # Đấu trường đánh giá & Tuyển chọn
│   ├── match.py       # Loạt trận đối đầu giữa θ_new và θ_best (luân phiên quân, τ -> 0)
│   ├── elo.py         # Đếm winrate (ngưỡng 55%) và tính toán cập nhật điểm Elo qua các thế hệ
│   └── baseline.py    # Đối thủ chuẩn ngẫu nhiên (RandomModel) và hàm đánh giá khách quan
├── gtp/        # Go Text Protocol Server
│   └── gtp_server.py  # Chuẩn giao tiếp GTP v2 điều khiển engine từ GUI bên ngoài
└── tests/      # Bộ kiểm thử tự động toàn diện (Unit tests, Rule invariants, Regression tests)
```

---

## 🔁 3. Vòng lặp AlphaZero (Pipeline)

```
       ┌─────────────────────────────┐
       │   Khởi tạo f_θ ngẫu nhiên   │
       └──────────────┬──────────────┘
                      │
           ┌──────────▼──────────┐
    ┌─────►│      Self-Play      │◄── (Chạy MCTS + f_θ_best)
    │      │  Sinh N ván tự đấu  │
    │      └──────────┬──────────┘
    │                 │ Lưu (s, π, z) vào Replay Buffer (D4 Augmentation)
    │      ┌──────────▼──────────┐
    │      │       Training      │
    │      │  Tối ưu hóa loss L  │──► Sinh mô hình ứng viên θ_new
    │      └──────────┬──────────┘
    │                 │
    │      ┌──────────▼──────────┐
    │      │        Arena        │
    │      │   θ_new vs θ_best   │
    │      └──────────┬──────────┘
    │                 │
    └─────── [Thắng ≥ 55% ?] ── Thắng ──► θ_best = θ_new
                      │
                    Không ──► Giữ nguyên θ_best hiện tại
```

---

## 🚀 4. Cài đặt & Bắt đầu

### Yêu cầu hệ thống:
- Python 3.10+ (Đã kiểm thử trên Python 3.13)
- PyTorch 2.x (Hỗ trợ GPU CUDA hoặc CPU)
- NumPy, Pytest

### Khởi tạo môi trường:
```bash
# Tạo môi trường ảo
python -m venv .venv

# Kích hoạt trên Windows
.venv\Scripts\activate

# Cài đặt PyTorch với CUDA 12.4
pip install torch --index-url https://download.pytorch.org/whl/cu124
pip install numpy pytest
```

---

## 🎮 5. Hướng dẫn Sử dụng & Vận hành

### 1. Chạy toàn bộ Test Suite (58/58 Tests PASS):
```bash
pytest
```

### 2. Bắt đầu vòng lặp huấn luyện AlphaZero:
Hệ thống tự động hiển thị thanh tiến trình trực quan (`tqdm`) theo thời gian thực và tự động lưu `buffer.pkl` định kỳ (hỗ trợ tự động resume):
```bash
# Huấn luyện nhanh để theo dõi từng vòng (khuyên dùng trên máy cá nhân):
python main.py --iters 50 --games 8 --sims 60 --train-steps 100 --batch-size 64 --arena-games 6

# Hoặc huấn luyện chuẩn sâu:
python main.py --iters 50 --games 20 --sims 100 --train-steps 200 --batch-size 128 --arena-games 20

# Tự động tiếp tục từ checkpoint và buffer cũ:
python main.py --resume
```

### 3. Khởi động GTP Server (Kết nối Sabaki / Lizzie / Katrain):
```bash
# Chạy server GTP v2 với checkpoint đã huấn luyện
python -m gtp.gtp_server --checkpoint checkpoints/best_model.pt --size 9 --sims 200
```
Trong Sabaki hoặc Lizzie:
1. Vào `Engines` -> `Manage Engines` -> `Add`.
2. Điền path tới Python venv (`.venv\Scripts\python.exe`) và arguments `-m gtp.gtp_server --checkpoint checkpoints/best_model.pt`.
3. Bắt đầu trận đấu trực tiếp giữa bạn và CoVayAI!

---

## 📈 6. Tiến độ Phát triển & Kiểm thử
- **Trạng thái:** Toàn bộ các Phase cốt lõi và tối ưu hóa vận hành đã hoàn thành xuất sắc!
- **Kiểm thử tự động:** 65/65 bài test passing 100%.
- Xem chi tiết từng giai đoạn tại [PROJECT_PROGRESS.md](PROJECT_PROGRESS.md).
