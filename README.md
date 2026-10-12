# LLMBeginer — Hành trình 60 ngày trở thành AI Engineer

Repo lưu source code + tài liệu các bài thực hành trong khóa học
"Trở Thành AI Engineer" (60 ngày: 30 ngày học LLM + 30 ngày nghiên cứu).

Page Facebook: https://www.facebook.com/BecomeAIEngineer/

## Cấu trúc

Mỗi ngày một thư mục `dayDD/`:

```
day01/
├── bigram_lm.py               # Source code bài thực hành (PyTorch)
├── corpus.txt                 # Dữ liệu huấn luyện mẫu
└── bigram_lm_giai_thich.html  # Tài liệu giải thích chi tiết (tiếng Việt, mở bằng trình duyệt)
day02/
├── Day02_lesson.html     # Bài học: Tensor, autograd và tối ưu hoá (mở bằng trình duyệt)
├── Day02_quiz.html       # Quiz 5 câu trắc nghiệm (chấm điểm ngay trong file)
└── Day02_practice.html   # Bài thực hành: tự code backprop tay bằng numpy
```

## Day 01 — Bigram Language Model (char-level, PyTorch)

Tự code model ngôn ngữ bigram: nhìn 1 ký tự, đoán ký tự tiếp theo.

```bash
python day01/bigram_lm.py day01/corpus.txt --steps 5000 --batch-size 32 --block-size 16
```

Điểm đáng học trong code:
- **Checkpoint theo val loss tốt nhất** — model cuối chưa chắc đã tốt nhất (overfitting).
- **Early stopping với patience** — tự dừng khi val loss không cải thiện, tiết kiệm điện.
- **Green checkpoint** — đo thời gian train, số bước optimizer, peak RAM.

## Day 02 — Tensor, autograd và tối ưu hoá

Nhìn gradient bằng con số: tensor & broadcasting, `torch.autograd`, backprop/chain rule,
SGD/AdamW + gradient clipping, góc tiết kiệm năng lượng (mixed precision).

Bài thực hành tối: tự code 1 vòng backprop hoàn chỉnh bằng numpy rồi đối chiếu từng con số
với `torch.autograd` — xem chain rule "sống" trong code như thế nào.
