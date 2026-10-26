# Day 16 — Lab SFT model nhỏ GPT-2 (124M)

Chạy trên VM CPU-only, torch 2.14.1+cpu, transformers 5.20.2,
GPT-2 load ở float32. Template cố định cho cả train và sinh
(cùng hàm `format_prompt`, đổi template giữa chừng là đổi luôn
bài kiểm tra):

```
Câu hỏi: {instruction}
Trả lời: {output}<|endoftext|>
```

## Cách chạy

```bash
python sft_lab.py       # Bài 1-3 + ablation masking (2 epoch)
python train_longer.py  # Bổ sung: 10 epoch + val loss mỗi epoch
```

Lưu ý môi trường: tải GPT-2 qua Hugging Face bị kẹt như Day 15
(proxy VM chặn CDN). Trọng số lấy qua ModelScope
(`openai-community/gpt2`, chỉ lấy file `*.json`, `*.txt`,
`*.safetensors` — bản full kèm TF/Flax nặng nhiều GB, tải ở
~650kB/s quá chậm). Trên máy thường, đặt `MODEL_DIR` về
`"gpt2"` là chạy thẳng từ HF.

## Bài 1 — Masking: PASS

- Mẫu 0 có 54 token, trong đó 9 token tính loss (đáp án
  "Hà Nội.<|endoftext|>").
- Assert: mọi label ở vị trí prompt là -100, label ở vị trí đáp
  án bằng đúng token thật → PASS.
- Edge case: instruction/output rỗng → `ValueError` rõ ràng
  (tránh labels toàn -100 làm loss thành NaN âm thầm).

## Bài 2 — Padding

- 14 mẫu, mẫu dài nhất 175 token (email dài, GPT-2 BPE tiếng Anh
  chẻ tiếng Việt rất nhỏ).
- Tổng token kể cả padding: 2450; token thật 1301 (**53%**);
  token đệm lãng phí 1149 (**47%** — gần nửa compute đổ vào đệm).
- Ước lượng packing (sequence 128 token): 11 sequence, token đệm
  chỉ còn 107 (**7,6%**) thay vì 47%.

## Bài 3 — Train và đánh giá trước–sau

Split: **11 train / 3 val** (val = 3 mẫu dịch cuối). LR 5e-5,
batch 4, 2 epoch theo đúng đề bài (dừng ở epoch 2 theo quy định
của đề, chưa dùng early stopping — xem phần bổ sung).

### Trước khi SFT (base model)

Đúng như bài sáng cảnh báo: model không dừng, sinh lan man hoặc
lặp khuôn mẫu — toàn "tiếng Việt rác" do GPT-2 chỉ train tiếng
Anh ("ưệt hệt hệt hệt...", "ưọn lời lời lời...").

### Sau 2 epoch, masking BẬT (chuẩn SFT)

- Train loss: 3,3230 → 2,3886 (giảm ✓).
- Hành vi đổi khác nhưng CHƯA ra đáp án đúng: model chuyển từ
  "rác" sang **echo lại prompt** ("ủ đô của Việt Nam là gì?",
  "ột câu chào buổi sáng lịch sự."). Đây là hành vi dễ học nhất
  với 11 mẫu × 2 epoch: copy token từ prompt thay vì sinh đáp án.

### Ablation: masking TẮT (chỉ đổi 1 biến — cùng seed, dữ liệu,
epoch, LR)

- Train loss: 3,5344 → 2,8956 (cao hơn bản masking, vì tính loss
  cả prompt khó đoán).
- Đầu ra lộ rõ cái giá của việc không masking: "ả lời: ả lời:
  ả lời:" — lặp mảnh template, sinh tiếp khuôn mẫu thay vì đáp
  án. Kết luận: masking không phải chi tiết nhỏ, nó quyết định
  model học "trả lời" hay học "nhại lại khuôn mẫu".

### Bổ sung: 10 epoch, masking BẬT (train_longer.py)

Để kiểm chứng giả thuyết "2 epoch chỉ là chưa đủ", train thêm
10 epoch và đo val loss mỗi epoch:

| Epoch | Train loss | Val loss |
|---|---|---|
| 1 | 3,3230 | 4,1005 |
| 2 | 2,3623 | 3,5847 |
| 4 | 2,1654 | 3,4153 |
| 6 | 1,5809 | **3,3064 (thấp nhất)** |
| 8 | 1,2236 | 3,4221 |
| 10 | 1,1316 | 3,5066 |

Val loss chạm đáy ở epoch 6 rồi tăng trở lại → **overfitting
kinh điển, early stopping nên dừng ở epoch 6**.

Đầu ra sau 10 epoch: model ĐÃ học được định dạng (trả lời ngắn
rồi dừng), nhưng nội dung vẫn sai theo kiểu sụp đổ về đáp án
ngắn quen thuộc nhất: "Tokyo." cho cả câu hỏi thủ đô Việt Nam,
thủ đô Pháp và câu dịch! Câu tóm tắt mèo thì đúng nửa đầu
("Mèo là loài vật nuôi hiền lành và thích...").

## Kết luận thật (không tô hồng)

- Masking, template cố định, train/val split đều làm đúng — nhưng
  với 11 mẫu tiếng Việt trên model chỉ biết tiếng Anh, SFT 2 epoch
  chỉ đổi được hành vi (rác → echo), 10 epoch chỉ học được định
  dạng (ngắn + dừng) chứ chưa học được nội dung đúng.
- Giới hạn thật nằm ở **dữ liệu quá ít + lệch ngôn ngữ**, không
  phải ở masking hay template. Đây chính là lý do bài học dặn:
  đừng train 5-10 epoch trên tập nhỏ (val loss đã chứng minh
  overfit từ epoch 6), và SFT cần dữ liệu đa dạng, đủ lớn.
- Tốc độ CPU: 2 epoch (11 mẫu) ~60 giây; 10 epoch ~274 giây.

## Checklist tự soát (7 điểm AGENTS.md)

- Edge case: output rỗng bị chặn bằng ValueError (không NaN).
- Docstring viết sau khi đọc lại code; tensor về `DEVICE`
  (CPU); thời gian đo bằng `perf_counter()`.
- Ablation chỉ đổi 1 biến (masking): cùng seed, dữ liệu, epoch,
  LR, thứ tự shuffle; labels encode sẵn trước khi train.
- File I/O: model load từ thư mục local đã kiểm tra đủ file
  (config, safetensors, vocab, merges, tokenizer).
