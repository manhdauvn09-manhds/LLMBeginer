# Capstone GreenLLM — Research Plan (Ngày 29)

## Câu hỏi nghiên cứu
Làm sao cắt giảm năng lượng suy luận mà không mất nhiều chất lượng?

## Giả thuyết (viết TRƯỚC khi chạy)
Kết hợp quantization INT8 với pruning 30% trọng số nhỏ sẽ giảm
ít nhất 40% năng lượng mỗi token, trong khi điểm đánh giá giảm
tối đa 2 điểm phần trăm.

## Dự đoán đo được
- energy/token ↓ ≥ 40% so với baseline
- accuracy ↓ ≤ 2 điểm phần trăm so với baseline

## Thí nghiệm
- Baseline: mô hình gốc, chưa làm gì.
- Biến thể: CHỈ thay quantization INT8 + pruning 30%
  (thứ duy nhất thay đổi là cách nén trọng số).
- Cùng dữ liệu, cùng phần cứng, cùng bộ đánh giá, batch 32.
- Chạy lặp 3 lần với 3 seed khác nhau, báo mean ± std.

## Ngưỡng bác bỏ
- Nếu accuracy giảm > 2 điểm phần trăm → giả thuyết BỊ BÁC BỎ.
- Nếu energy/token giảm < 40% → giả thuyết BỊ BÁC BỎ.
- Kết quả âm vẫn là kết quả tốt: ghi lại, không giấu.

## Thí nghiệm thu nhỏ tối nay (Day 29)
- Mô phỏng bằng MLP nhỏ trên bài toán phân loại tổng hợp
  (make_classification, 4000 mẫu, 32 đặc trưng) thay cho LLM thật.
- Đo accuracy + thời gian suy luận (proxy năng lượng, chưa phải
  bằng chứng năng lượng — fake-quant trên CPU không giảm thời
  gian thật; muốn đo năng lượng thật cần GPU + nvidia-smi hoặc
  codecarbon).
- Nhánh accuracy của giả thuyết được kiểm chứng trực tiếp theo
  ngưỡng ≤ 2 điểm %; nhánh năng lượng (↓ ≥ 40%) CHƯA kiểm chứng
  được bằng thí nghiệm này — ghi rõ, không kết luận thay.

## Kết quả (điền SAU khi chạy, không sửa ngưỡng)
- Accuracy giảm trung bình: 0,37 ± 0,21 điểm % (C, 3 seed;
  A 0,33 ± 0,25; B -0,03 ± 0,06) → nhánh accuracy:
  ☑ được ủng hộ / ☐ bị bác bỏ.
- Nhánh năng lượng: ☑ chưa kiểm chứng được (thời gian
  suy luận trung vị x0,98–0,99 so với baseline — fake-quant
  trên CPU không giảm thời gian thật, chỉ là proxy).
- Ablation: pruning mất điểm nhiều hơn quant (0,33 vs
  -0,03 điểm %); chọn C (pruning + quant) cho capstone
  vì kích thước nhỏ nhất tính thật (x0,17 so với baseline;
  A x0,70, B x0,25 — đếm số trọng số khác 0 × byte/trọng số).
- Phạm vi độ lệch: ± điểm % chỉ phản ánh biến thiên KHỞI
  TẠO (dữ liệu cố định theo protocol); Phase 2 nên để seed
  chi phối cả chia dữ liệu (hoặc k-fold).
