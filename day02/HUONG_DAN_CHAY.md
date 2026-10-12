# Day 02 — Hướng dẫn chạy & test (Bài thực hành buổi tối)

> Nguồn bài tập: `Day02_practice.html` (mở bằng trình duyệt để đọc đề + lý thuyết).
> File thực hành: `gradient_check.py` (Bài 1), `train_mlp.py` (Bài 2+3).

---

## 0. Chuẩn bị (làm một lần duy nhất)

1. **Lấy code mới nhất** về máy:
   - Chạy file `git_pull.bat` ở thư mục gốc repo, hoặc `git pull`.
2. **Cài thư viện Python** (dùng bản CPU cho nhẹ — toàn bộ bài này chạy tốt trên CPU,
   không cần GPU):
   ```bash
   pip install torch --index-url https://download.pytorch.org/whl/cpu
   pip install matplotlib
   ```
   (Trên Windows nếu `pip` không nhận lệnh, thử `py -m pip install ...`.
   Gói torch ~200MB nên lần cài đầu sẽ mất vài phút — bình thường.)
3. **Kiểm tra môi trường** (theo mục "Chuẩn bị môi trường" trong đề bài):
   ```bash
   python -c "import torch, matplotlib; print(torch.__version__)"
   ```
   Kỳ vọng: in ra version torch (vd `2.x.x`), không báo lỗi.

---

## 1. Bài 1 — Gradient check: bắt autograd "khai gian"

```bash
cd day02
python gradient_check.py
```

**Đạt khi:**
- In ra **4/4 PASS**, `max_error` cỡ `1e-11` (ngưỡng yêu cầu `< 1e-4`).
- Phần bonus cuối file tự in so sánh `float32` vs `float64`: ở float32 sai số
  cỡ `3.6e-02` (trượt ngưỡng) — đó chính là câu trả lời bằng số liệu cho
  *"Vì sao không chạy gradient check ở float32?"*.

**Nếu FAIL:** đừng sửa threshold — hãy debug (gợi ý: kiểm tra công thức forward,
chỗ perturb có trả param về giá trị gốc chưa, có gọi `.backward()` trong vòng
lặp numeric không).

---

## 2. Bài 2 — Huấn luyện MLP dự đoán token kế tiếp + log gradient norm

```bash
python train_mlp.py
```

**Đạt khi:**
- `accuracy = 100%` (yêu cầu ≥ 95%).
- In **bảng so sánh SGD vs AdamW** (4 dòng như đề bài yêu cầu):
  - AdamW chạm accuracy 95% ngay epoch 0, loss cuối ~0.0005 (hội tụ nhanh nhờ adaptive lr).
  - SGD cần ~11 epoch, loss cuối ~0.52 (chậm hơn vì lr cố định).
- Sinh 2 file ảnh: `plot_sgd.png`, `plot_adamw.png` — mỗi ảnh 2 panel:
  **(a)** loss theo epoch, **(b)** gradient norm theo epoch (trục y thang log).

**Cách đọc plot gradient norm** (3 "nhịp tim" cần thuộc):
- Norm giảm dần về 0 trong khi loss vẫn cao → vanishing gradient / học quá chậm
  → tăng lr, kiểm tra khởi tạo.
- Norm vọt lên rồi loss NaN → exploding → giảm lr, bật `clip_grad_norm_`.
- Norm dao động đều, loss giảm mượt → **khỏe** (bài này phải ra dạng này).

**Các flag hữu ích:**
```bash
python train_mlp.py --optim adamw --epochs 100   # chỉ chạy 1 optimizer / đổi số epoch
python train_mlp.py --amp                         # thử mixed precision (cần GPU/CUDA;
                                                  # trên CPU sẽ tự bỏ qua và báo)
```

---

## 3. Bài 3 — Green checkpoint (Bài 3 trong đề)

- **Có GPU:** chạy `python train_mlp.py --amp`, so sánh float32 vs AMP theo mẫu
  trong đề (loss cuối, accuracy, ms/epoch, peak VRAM) rồi kết luận 1 câu.
- **Chỉ có CPU:** Bài 1 đã có sẵn demo `float32` vs `float64` (thời gian +
  max_error của gradient check) — lấy số liệu đó báo cáo, ghi rõ
  "chạy CPU, bỏ qua so sánh AMP". Nguyên tắc green giữ nguyên:
  *không tăng mô hình trước khi baseline đúng.*

---

## 4. Checklist nộp bài (đối chiếu trước khi paste vào chat lớp)

- [ ] `gradient_check.py` chạy ra **4/4 PASS, max_error < 1e-4**
- [ ] `train_mlp.py` chạy được, **accuracy ≥ 95%**
- [ ] Có plot 2 panel: loss + gradient norm (log-scale) theo epoch
- [ ] Có bảng so sánh SGD vs AdamW (4 dòng)
- [ ] Có số liệu Green checkpoint (float32 vs AMP, hoặc float32 vs float64 nếu CPU)
- [ ] Paste vào chat: code 2 file (hoặc link repo) + ảnh plot + bảng so sánh +
      2–3 dòng nhận xét *"gradient norm của em trông như thế nào, em chẩn đoán gì"*

---

## 5. Xử lý lỗi thường gặp

| Triệu chứng | Cách xử lý |
|---|---|
| `ModuleNotFoundError: No module named 'torch'` | Chưa cài torch — xem lại mục 0 |
| `pip` không nhận lệnh (Windows) | Dùng `py -m pip install ...` |
| Bài 1 FAIL | Đừng nới threshold — kiểm tra công thức forward và chỗ perturb ±eps |
| Bài 2 accuracy thấp / loss NaN | Đọc plot grad norm theo mục 2 (vanishing/exploding), thử đổi lr |
| Muốn xem tác dụng của clip | Comment dòng `clip_grad_norm_` trong `train_mlp.py` rồi chạy lại, so sánh plot |
