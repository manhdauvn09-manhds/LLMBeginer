# Day 15 — Lab load model thật: Qwen2.5-0.5B-Instruct

Chạy trên VM CPU-only (không GPU), Python 3.12, torch 2.14.1+cpu,
transformers 5.20.2. Model load ở float32, max_new_tokens = 30
(giảm từ 60 theo Bẫy 3 của đề bài cho CPU).

## Cách chạy

```bash
python load_model_lab.py        # Bài 1-4 đầy đủ
python gen_diacritics.py        # Bổ sung: Bài 4 với prompt có dấu
```

Lưu ý môi trường: trên máy này, tải trọng số qua Hugging Face bị
kẹt (proxy chặn CDN sau ~16 MB, thử cả hf_transfer và mirror đều
đứng). Trọng số lấy qua ModelScope (`Qwen/Qwen2.5-0.5B-Instruct`,
988 MB, tốc độ ~30 MB/s) + `config.json` lấy từ HF cache (file
nhỏ tải được bình thường). Trên máy thường (không qua proxy này),
chỉ cần đặt `MODEL = "Qwen/Qwen2.5-0.5B-Instruct"` là chạy thẳng
từ HF, không cần ModelScope.

## Bài 1 — Config (đọc trước khi load trọng số)

| Mục | Số liệu thật |
|---|---|
| Kiến trúc | Qwen2ForCausalLM |
| Số lớp | 24 |
| Hidden size | 896 |
| Attention head / KV head | 14 / 2 → **GQA** (KV cache nhẹ hơn 7x so với MHA) |
| Cửa sổ ngữ cảnh | 32.768 token (prompt + output cộng lại) |
| Dtype gốc | bfloat16 |
| Tổng params (đếm thật) | 494.032.768 (~0,494 tỉ) |
| Ước lượng VRAM bf16 | 0,99 GB (params × 2 byte, chưa KV cache) → khớp "nhẹ ~1 GB" |

## Bài 2 — Tokenizer

- "Thủ đô của Pháp là gì?" → 9 token; câu tiếng Anh tương đương
  "What is the capital of France?" → 7 token.
- "Giải thích transformer trong một câu." → 11 token, từ tiếng Việt
  bị chẻ nhỏ ("Giải" → 'G','ia','i'): tiếng Việt tốn token hơn
  tiếng Anh.
- eos = `<|im_end|>` (151645), pad = `<|endoftext|>` (151643).
- Vocab: `len(tok)` = 151.665 (config ghi vocab_size 151.936).
- `add_special_tokens=True` và `False` cho kết quả **giống hệt
  nhau**: Qwen không tự thêm BOS/EOS khi encode (khác BERT).
- Decode ngược câu có khoảng trắng đầu + dấu câu: khớp 100%.

## Bài 3 — Chat template

Prompt trần: `'Thu do cua Phap la gi?'`

Qua template (đúng định dạng model học khi SFT):

```
<|im_start|>system
Ban la tro ly huu ich, tra loi ngan gon.<|im_end|>
<|im_start|>user
Thu do cua Phap la gi?<|im_end|>
<|im_start|>assistant
```

Chuỗi kết thúc bằng `<|im_start|>assistant` = "đến lượt model nói".
Đổi system prompt thành giọng cướp biển: chỉ phần nội dung system
thay đổi, khung `<|im_start|>`/`<|im_end|>` giữ nguyên → system
prompt thực chất chỉ là một phần của prompt.

## Bài 4 — Greedy vs sampling (cùng prompt, đổi 1 biến)

### Lần 1: prompt KHÔNG dấu (script chính)

- Greedy 3 lần → **cả 3 giống hệt nhau**: "Xin lỗi, nhưng tôi
  không thể giúp bạn với câu hỏi này." (ổn định, lặp lại được ✓,
  nhưng nội dung là từ chối).
- Sampling temp=0.8, 3 lần → 3 câu khác nhau, đều lạc đề ("không
  phải là một từ", "trang web trực tuyến đáng tin cậy"...).
- Temp=1.5 → câu mở đầu xã giao, lan man.
- Prompt trần + greedy → "1. Cau Giay Thieu: Cau Giay Thieu là
  một trong những cau gioi thieu nổi tiếng nhất trên..." (hiểu sai
  cụm từ thành danh từ riêng).

### Lần 2: prompt CÓ dấu (gen_diacritics.py) — phát hiện quan trọng

- Template + greedy: "Trí tuệ nhân tạo (AI) là một lĩnh vực nghiên
  cứu và phát triển tiên tiến trong công nghệ hiện đại..." → trả lời
  đúng, gọn.
- Template + sampling: "Tất nhiên, tôi có thể giúp bạn viết một câu
  giới thiệu về trí tuệ nhân tạo..." → đúng chủ đề, khác greedy.
- Prompt trần + greedy: "Trí tuệ nhân tạo là một trong những công
  nghệ tiên tiến nhất hiện nay..." → vẫn đúng, nhưng mở đầu bằng
  khoảng trắng + dấu ngoặc kép (nối tiếp văn bản, không "vào vai"
  assistant như bản qua template).

**Kết luận:** với tiếng Việt, dấu là yếu tố quyết định chất lượng
hơn cả template trong thí nghiệm này — prompt không dấu làm model
hiểu sai ngay từ đầu, còn prompt có dấu thì cả template lẫn prompt
trần đều trả lời được. Template vẫn hơn ở chỗ định dạng sạch và
"vào vai" đúng.

## Tốc độ trên CPU

- Load trọng số (fp32): 39,0 giây.
- Sinh 30 token: greedy ~3,5 giây (sau warmup), sampling ~6 giây
  → khoảng 5-9 token/giây, đủ chạy lab nhưng chậm hơn GPU nhiều.

## Checklist tự soát (7 điểm AGENTS.md)

- Edge case: `add_special_tokens` True/False đều kiểm tra; decode
  ngược có khoảng trắng đầu câu.
- Docstring viết sau khi đọc lại code; device dùng `model.device`
  (CPU); thời gian đo bằng `time.perf_counter()`; không có
  tracker cần `finally` (không dùng CodeCarbon ở bài này).
- So sánh decoding giữ nguyên prompt, chỉ đổi 1 biến (nguyên tắc
  ablation 1 biến).
