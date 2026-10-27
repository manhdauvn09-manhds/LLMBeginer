# Day 17 — Ablation LoRA r = 4 / 8 / 16 trên GPT-2 (124M)

Chạy trên VM CPU-only, torch 2.14.1+cpu, transformers 5.20.2,
peft 0.21.2, GPT-2 float32. Quy tắc vàng giữ suốt lab: **chỉ rank
thay đổi** — cùng dataset, split, seed (42), epoch (2), LR (1e-4),
batch (4), tỉ lệ α/r = 2 cố định, decoding greedy 30 token.
(Số tham số train được KHÁC NHAU giữa các rank là đúng ý đồ —
rank chính là biến độc lập, không áp quy tắc "2 model bằng tham
số" của tuần trước vào đây.)

## Cách chạy

```bash
python lora_lab.py
```

Lưu ý môi trường: GPT-2 dùng lại thư mục ModelScope của Day 16
(HF CDN bị kẹt qua proxy VM). peft cài thêm vào venv Day 15.

## Bài 0 — Rà dataset TRƯỚC khi ablation (cả 3 kiểm chứng PASS)

- 32 mẫu, 8 mẫu mỗi kind (qa, summary, email, translate): 14 mẫu
  Day 16 + 18 mẫu mới. Dữ liệu viết có dấu đầy đủ (bài học Day 15).
- Stratified split 80/20 theo kind: **train 28 (7 mỗi kind), val 4
  (1 mỗi kind)** — tỉ lệ kind của val giống hệt train.
- Eval audit: **6 prompt đánh giá đều chưa từng thấy** trong train
  (không chứa nguyên văn instruction nào của train).
- Masking: **-100 trên tất cả 32 mẫu** (không chỉ mẫu 0).

## Bài 1 — Đếm tham số theo rank (khớp công thức)

c_attn của GPT-2: vào 768, ra 2304 (gộp Q/K/V), 12 lớp.
Công thức: r × (768 + 2304) × 12 = r × 36.864.

| Rank | Train được | Tỉ lệ | Công thức | Khớp? |
|---|---|---|---|---|
| r = 4 | 147.456 | 0,12% | 147.456 | ✅ đúng |
| r = 8 | 294.912 | 0,24% | 294.912 | ✅ đúng |
| r = 16 | 589.824 | 0,47% | 589.824 | ✅ đúng |

Assert đóng băng W gốc PASS ở cả 3 rank: chỉ tham số tên chứa
`lora_` mới có `requires_grad=True`.

## Bài 2 — Bảng ablation (chỉ đổi rank)

| Rank | Tham số train được | Train loss cuối | Val loss cuối | Thời gian |
|---|---|---|---|---|
| r = 4 | 147.456 | 3,6436 | 2,4515 | 114,9s |
| r = 8 | 294.912 | 3,6286 | 2,4352 | 91,4s |
| r = 16 | 589.824 | 3,5502 | 2,4002 | 94,2s |

Nhận xét:

- Val loss giảm đều theo rank: 2,4515 → 2,4352 → 2,4002.
  **Rank tốt nhất: r = 16.**
- Nhưng lợi ích nhỏ: gấp 4 lần tham số (r=4 → r=16) chỉ đổi lấy
  ~2% val loss. Trái với kỳ vọng "r=8 ≈ r=16" của đề bài, trong
  lần chạy này khoảng 8→16 (−0,035) còn lớn hơn 4→8 (−0,016).
- Train loss epoch 2 cao hơn epoch 1 ở CẢ 3 rank (vd r=16:
  3,3377 → 3,5502). Nguyên nhân khả dĩ nhất: cùng seed 42 nên cả
  3 rank gặp cùng thứ tự shuffle, epoch 2 rơi vào batch khó hơn
  (mẫu email dài). Val loss — không shuffle, không dropout — mới
  là thước đo đáng tin, và nó giảm đều theo rank.
- Đầu ra trên prompt chưa thấy: cả 3 rank đều còn echo/lan man
  ("ượn hỏi: Thủ đô của Hàn Qu...", "ượn lời: ượn lời:...") — 2
  epoch × 28 mẫu vẫn quá ít cho model tiếng Anh làm nhiệm vụ
  tiếng Việt, đúng giới hạn đã thấy ở Day 16. Rank cao hơn chưa
  đủ để biến echo thành đáp án đúng.

## Bài 3 — Merge adapter r = 16

- Dung lượng adapter: **2,37 MB** so với ~500 MB của GPT-2 đầy
  đủ → tiết kiệm ~99,5%.
- Merge check: **6/6 đầu ra giống hệt** trước/sau merge
  (`merge_and_unload`), không lệch do làm tròn trong lần chạy này.

## Mở rộng — Ablation target modules (c_attn + c_proj, r = 16)

- Train được: **1.622.016** tham số; train loss 3,4426; val loss
  **2,3722** (tốt hơn c_attn-only 2,4002, cải thiện 0,028).
- ⚠️ Chi tiết quan trọng: peft khớp `target_modules` theo hậu tố
  tên, nên `"c_proj"` bắt cả **attn.c_proj** (768→768) VÀ
  **mlp.c_proj** (3072→768). Tính lại: mỗi lớp 16×(3072+1536+
  3840) = 135.168, × 12 lớp = 1.622.016 ✅ khớp đúng số code in.
  Vậy ablation này thực chất là c_attn + attn.c_proj + mlp.c_proj
  (đã lan sang MLP), không chỉ attention như ý đồ ban đầu — cải
  thiện val loss đi kèm gấp 2,75 lần tham số so với c_attn-only.
- Đây là ablation riêng, không gộp vào bảng rank.

## Kết luận thật

- LoRA hoạt động đúng cơ chế: tham số khớp công thức, W gốc đóng
  băng, merge không đổi đầu ra, adapter chỉ 2,37 MB.
- Về câu hỏi "rank cao hơn tốn thêm bao nhiêu mà được thêm bao
  nhiêu": gấp 4 lần tham số chỉ mua được ~2% val loss, và không
  đổi chất lượng định tính trên prompt chưa thấy sau 2 epoch.
  Với tập nhỏ + lệch ngôn ngữ này, tăng rank không phải nút thắt
  — dữ liệu và số bước train mới là nút thắt (như Day 16 đã chỉ).
- Tốc độ CPU: mỗi rank ~90–115 giây (2 epoch, 28 mẫu).

## Checklist tự soát (7 điểm AGENTS.md)

- Edge case: instruction/output rỗng → ValueError; kind có < 5
  mẫu thì không lấy val (không xảy ra ở đây vì mỗi kind có 8).
- Docstring viết sau khi đọc lại code; tensor về `DEVICE`;
  thời gian đo bằng `perf_counter()`.
- Ablation chỉ đổi 1 biến (rank): cùng seed, dataset, split,
  epoch, LR, batch, α/r = 2, decoding; assert đóng băng W gốc.
- File I/O: model load từ thư mục local đã kiểm tra đủ file.
