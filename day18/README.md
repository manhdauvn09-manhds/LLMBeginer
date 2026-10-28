# Day 18 — QLoRA: đo cái giá của nén 4-bit bằng số liệu

Chạy trên VM CPU-only (không GPU), torch 2.14.1+cpu,
transformers 5.20.2, peft 0.21.2, bitsandbytes 0.50.2.
Quy tắc vàng: **chỉ cách giữ trọng số gốc thay đổi** (fp32 so
với NF4 4-bit) — cùng GPT-2, dataset Ngày 17, split phân tầng,
seed 42, rank r = 8, α = 16 (α/r = 2), target c_attn, LR 1e-4,
epoch 2, batch 4, decoding greedy 30 token.

## Lưu ý môi trường (ghi rõ, không giả vờ có GPU)

Đề bài nói Bài 1–3 "BẮT BUỘC có GPU" và dặn không có GPU thì
chỉ làm Bài 0 + bảng lý thuyết. Nhưng bitsandbytes 0.50.2 load
4-bit + forward + train **vẫn chạy được trên CPU** (qua fallback
kernel, có cảnh báo thiếu gói `kernels`), nên lab này chạy THẬT
cả hai cấu hình trên CPU thay vì chỉ điền lý thuyết. Hai điểm
khác biệt so với đề bài, ghi rõ để khỏi hiểu nhầm:

1. Bản tham chiếu dùng **fp32 thay vì fp16** (fp16 trên CPU vừa
   chậm vừa không ổn định; fp32 là dạng gốc của Day 16–17).
2. Peak VRAM thay bằng **peak RAM (RSS)**, đo riêng mỗi cấu hình
   trong process riêng (vì `ru_maxrss` là đỉnh process, không
   reset được). Số VRAM lý thuyết cho GPU vẫn đúng (xem cuối
   README); số RAM ở đây là bằng chứng hướng, không phải số VRAM.

## Bài 0 — Sờ tận tay NF4 (CPU, PyTorch thuần)

16 mức NF4 (trung vị của 16 khoảng đều xác suất N(0,1), chuẩn
hóa về [-1, 1]): [-1.0, -0.708, -0.542, -0.417, -0.311, -0.216,
-0.127, -0.042, 0.042, 0.127, 0.216, 0.311, 0.417, 0.542, 0.708,
1.0] — dày quanh 0, thưa ở đuôi.

| Dữ liệu (100.000 mẫu, block 64) | MSE NF4 | MSE int4 | Kết luận |
|---|---|---|---|
| w ~ N(0, 1) (dạng thật của trọng số) | 0,008473 | 0,011793 | NF4 thắng 1,39x ✅ |
| w ~ U(0, 1) (torch.rand như đề bài) | 0,002792 | 0,001625 | NF4 THUA (0,58x) ✅ |

Đúng như đề bài dự báo: NF4 chỉ thắng khi dữ liệu thật sự giống
phân phối chuẩn — đó là giả định của nó, không phải chân lý phổ
quát. Trên dữ liệu đều, mức dày quanh 0 của NF4 bị lãng phí,
chia đều (int4) mới tối ưu.

Double quantization (số học tay): 100.000 tham số → 1.563 scale
(block 64). Scale fp32: 6.252 byte; scale int8: 1.563 byte; tiết
kiệm 4.689 byte = **0,375 bit/tham số** (từ 4,5 xuống 4,125
bit/tham số, khớp con số bài sáng ~4,5 → ~4,13).

Áp cho GPT-2 (124.439.808 params): tiết kiệm 5,83 MB trên scale;
trọng số 4-bit ~62,2 MB + scale int8 ~1,9 MB = **~64,2 MB** (so
với fp16 ~249 MB, fp32 ~498 MB).
<!-- [TRAINER] Đã sửa số params GPT-2: bản đầu ghi 124.587.264 —
đó là tổng Day 17 khi đã gắn adapter LoRA r=4 (124.439.808 base +
147.456 adapter). Đếm trực tiếp từ model: base = 124.439.808.
Số scale kéo theo: 1.944.372 scale, tiết kiệm 5,83 MB (trước ghi
5,84). Kết luận ~64,2 MB không đổi. -->

## Bài 1 — Gắn LoRA, kiểm tra dtype tận tay

| | Base (c_attn) | Adapter | W gốc đóng băng? | Train được |
|---|---|---|---|---|
| LoRA fp32 | Parameter fp32 | Parameter fp32 | ✅ grad=False | 294.912 |
| QLoRA 4-bit | **Params4bit** (uint8) | Parameter fp32 | ✅ grad=False | 294.912 |

Hai bản train được **bằng nhau đúng 294.912** (cùng rank, cùng
target) — nếu khác nhau thì đã đổi thứ gì đó ngoài cách nén.
Lưu ý trung thực so với kỳ vọng của đề bài: adapter đo được là
fp32 chứ không phải bf16 (peft giữ adapter ở fp32 trên CPU với
base đã lượng tử); điều quan trọng vẫn giữ: adapter **không bị
nén theo base**, học ở độ chính xác cao. (Embedding wte ở bản
4-bit vẫn là bf16 vì embedding không phải tầng Linear nên không
bị lượng tử — chỉ các tầng Linear mới thành Params4bit.)

## Bài 2 — Bảng so sánh (cùng dữ liệu, cùng seed)

| Cấu hình | Train loss cuối | Val loss cuối | Thời gian | Peak RSS |
|---|---|---|---|---|
| LoRA fp32 (tham chiếu) | 3,6286 | 2,4352 | 108,8s | 2,91 GB |
| QLoRA 4-bit (NF4 + double quant) | 3,7882 | 2,4524 | 41,1s | 2,13 GB |
| Chênh lệch | +0,160 | +0,017 (+0,7%) | **−62% (nhanh hơn)** | −0,78 GB (−27%) |

Nhận xét:

- **Chất lượng: cái giá của nén rất nhỏ.** Val loss 4-bit chỉ
  kém fp32 0,017 (0,7%). Train loss kém hơn rõ hơn (+0,16) vì
  nhiễu lượng tử làm khớp train khó hơn, nhưng val — thước đo
  đáng tin — gần như không đổi.
- **Tốc độ: NGƯỢC với kỳ vọng của đề bài trên GPU.** Đề bài nói
  bản 4-bit "thường chậm hơn một chút vì dequant mỗi lần
  forward". Trên CPU này, 4-bit NHANH HƠN 62% (41,1s so với
  108,8s): dequant về bf16 rồi tính bằng bf16 (CPU này có lệnh
  AVX512-BF16) nhanh hơn tính fp32, và lưu lượng bộ nhớ thấp
  hơn nhiều. Kết luận "4-bit chậm hơn" là đúng cho GPU nhưng
  không đúng cho CPU này — phải đo mới biết.
- **Bộ nhớ: 4-bit tiết kiệm 0,78 GB RAM đỉnh** (−27%). Tỉ lệ này
  bị "pha loãng" vì RSS bao gồm cả Python + torch (~2 GB nền);
  riêng phần trọng số, lý thuyết là 498 MB → 64 MB (−87%).
- Tính nhất quán: bản fp32 cho train/val loss **giống hệt Day 17
  r=8** (3,4131 → 3,6286; val 2,4352) — cùng dataset, split, seed,
  rank, LR thì kết quả lặp lại đúng, pipeline đáng tin.
- Đầu ra định tính: cả hai bản train xong đều còn echo/lan man
  trên prompt chưa thấy (giới hạn đã biết từ Day 16–17: 2 epoch ×
  28 mẫu, tiếng Việt trên model tiếng Anh). Chênh lệch định tính
  giữa hai bản không có ý nghĩa — val loss mới là thước đo.

## Mở rộng — PagedAdamW8bit cho bản 4-bit

Chạy được nhưng **không có tác dụng trên CPU**: bitsandbytes
cảnh báo "Paged optimizers are not supported on CPU. Falling
back to non-paged optimizer behavior". Kết quả gần như đồng nhất
với AdamW thường: train loss 3,7925 (so với 3,7882), val loss
2,4619 (so với 2,4524), thời gian 41,6s (so với 41,1s), peak RSS
2,10 GB (so với 2,13 GB). Kết luận đúng như bài sáng: paging là
lưới an toàn cho VRAM GPU, không phải chế độ mặc định; ở đây
không cần nó (peak chỉ ~2,1 GB, RAM còn nhiều).

## Bài 3 — Tách ba nguồn lỗi

<!-- [TRAINER] Đã sửa nhãn (a)/(c) ở hai mục dưới cho khớp đề bài
((a) dữ liệu, (b) adapter, (c) nén) và khớp chính đoạn kết luận ở
cuối mục này — bản đầu gắn (a) cho nén và (c) cho dữ liệu trong
tiêu đề mục, nhưng kết luận lại dùng (c) cho nén, (a) cho dữ
liệu. Nội dung đo không đổi, chỉ sửa nhãn. -->

**(c) Lỗi do nén (thuần túy, chưa train):** base fp32 vs base
4-bit chưa train sinh trên 6 prompt đánh giá: **0/6 đầu ra giống
nhau**. Nhiễu lượng tử làm đầu ra khác hẳn ngay từ trước khi
train — nhưng cả hai đều là "tiếng Việt rác" (base model vốn
không biết nhiệm vụ), nên 0/6 này chỉ chứng minh nén CÓ làm
nhiễu, không chứng minh nén làm hỏng nhiệm vụ.

**(b) Lỗi do adapter (tương tác):** bản train xong chênh nhau
chỉ +0,017 val loss — tức adapter rank 8 gần như bù được hết
nhiễu nén trên thước đo val loss. Phần chênh lệch vượt quá sai
số nén thuần túy là rất nhỏ.

**(a) Lỗi do dữ liệu:** audit lại Ngày 17 trên đúng dataset đang
dùng: PASS hết (28 train / 4 val phân tầng theo kind, eval
không trùng train, masking −100 trên cả 32 mẫu). Dữ liệu KHÔNG
phải nguồn lỗi ở đây.

**Kết luận chẩn đoán:** "Trên nhiệm vụ này, chênh lệch QLoRA so
với LoRA chủ yếu đến từ **nhiễu nén (c)**, nhưng ở mức rất nhỏ
(+0,7% val loss) và adapter đã bù gần hết; **không phải do dữ
liệu (a)** (audit PASS) và **không phải do adapter thiếu sức
(b)** — bằng chứng là val loss hai bản gần như trùng nhau."

## Góc năng lượng (có số đo, không đoán)

Công thức bài sáng: năng lượng thay đổi theo tỉ lệ
(1 + X/100) × Y, với X là % chậm hơn, Y là tỉ lệ điện của card
nhỏ so với card lớn.

- Trên CPU này: 4-bit NHANH hơn 62% (X = −62, tỉ lệ thời gian
  0,38x) trên cùng máy (cùng công suất) → năng lượng cũng chỉ
  còn **0,38x** — QLoRA tiết kiệm năng lượng rõ rệt ở đây.
- Điều kiện đảo ngược: trên GPU, nếu 4-bit chậm hơn 30% (X = 30)
  thì cần Y < 1/1,3 ≈ 0,77 — tức card nhỏ phải tiêu thụ dưới 77%
  điện của card lớn thì mới hòa vốn; chậm hơn gấp đôi (X = 100)
  thì cần Y < 0,5. Nói cách khác: "4-bit = xanh hơn" chỉ đúng khi
  nó cho phép bạn xuống card đủ nhỏ để bù phần chậm lại do
  dequant — không phải chân lý mặc định.

## Checklist tự soát (7 điểm AGENTS.md)

- Edge case: tensor rỗng → ValueError; block toàn 0 không chia
  cho 0 (giữ nguyên 0).
- Docstring viết sau khi đọc lại code; thời gian đo bằng
  `perf_counter()` (đề bài dùng `time.time()` — ở đây dùng
  `perf_counter` theo checklist Day 13 vì là đo khoảng, không
  phải timestamp).
- Ablation chỉ đổi 1 biến (cách giữ trọng số gốc): cùng dataset,
  split, seed, rank, α/r, LR, epoch, batch, decoding; trainable
  bằng nhau đúng 294.912 cả hai bản.
- Không suy diễn "không có GPU = không làm được": thử chạy thật
  bitsandbytes trên CPU trước, chỉ ghi chú giới hạn sau khi đo.
