# LLMBeginer — Hành trình 60 ngày trở thành AI Engineer

Repo lưu source code + tài liệu các bài thực hành trong khóa học
"Trở Thành AI Engineer" (60 ngày: 30 ngày học LLM + 30 ngày nghiên cứu).

Page Facebook: https://www.facebook.com/BecomeAIEngineer/

## Cấu trúc

Mỗi ngày một thư mục `dayDD/`:

```
day01/
├── bigram_lm.py            # Source code bài thực hành
├── corpus.txt              # Dữ liệu huấn luyện mẫu
└── bigram_lm_giai_thich.html  # Tài liệu giải thích chi tiết (tiếng Việt, mở bằng trình duyệt)
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
