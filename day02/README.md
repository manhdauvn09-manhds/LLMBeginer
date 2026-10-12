# Day 02 — Thực hành: Gradient Check & Huấn luyện MLP bằng Autograd

Nguồn bài tập: `day02/Day02_practice.html` trong repo LLMBeginer.

## Các file

| File | Bài | Mô tả |
|---|---|---|
| `gradient_check.py` | Bài 1 | Đối chiếu gradient của `torch.autograd` với sai phân hữu hạn trên mạng mini 2 lớp. Kỳ vọng: **4/4 PASS, max_error < 1e-4** |
| `train_mlp.py` | Bài 2 + 3 | Huấn luyện MLP dự đoán token kế tiếp (chuỗi `0 1 2 3…`), log gradient norm mỗi epoch, so sánh SGD vs AdamW, vẽ plot |

## Cách chạy

```bash
# Bài 1 — gradient check (chạy ~vài giây)
python gradient_check.py

# Bài 2 — train cả SGD và AdamW, vẽ plot, in bảng so sánh
python train_mlp.py

# Chỉ chạy 1 optimizer / đổi số epoch
python train_mlp.py --optim adamw --epochs 100

# Bài 3 (Green) — thử mixed precision nếu có GPU
python train_mlp.py --amp
```

## Kết quả mong đợi

- `gradient_check.py` → 4/4 PASS (thường max_error ~1e-9 ở float64)
- `train_mlp.py` → accuracy ≥ 95%, sinh ra `plot_sgd.png` và `plot_adamw.png`
  (2 panel: loss | gradient norm thang log), in bảng so sánh SGD vs AdamW

## Đọc plot gradient norm (3 "nhịp tim" cần thuộc)

- Norm giảm dần về 0 trong khi loss vẫn cao → vanishing gradient / học quá chậm
- Norm vọt lên rồi loss NaN → exploding gradient → giảm lr, bật clip
- Norm dao động đều, loss giảm mượt → khỏe
