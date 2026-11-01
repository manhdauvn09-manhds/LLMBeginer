# Day 22 (buoi toi) — Lab Quantization: quantize that, do that

Chay tren CPU (torch 2.14.1+cpu), khong GPU, khong tai model
lon. Toan bo lab chay trong ~6,5s.

## Bai 1 — Affine quantization

Ham `quantize_dequantize` tu viet (symmetric + asymmetric),
tra ve (x_hat, scale, zero_point).

- Tensor [-1.2, -0.3, 0.0, 0.3, 0.6], INT8 asymmetric: scale =
  0.00706, zero-point = 170 (khop ky vong de bai). Sai so lon
  nhat 0.00353 = dung scale/2 (diem lam tron nua muc, de bai
  noi "< scale/2" - that ra bang dung o bien khi lam tron
  half-to-even).
- INT4 tren cung tensor: sai so lon nhat 0.06000, gap INT8
  17.0x - dung ti le scale (255/15 = 17).
- Tensor lech 1 phia [0.5, 2.5]: symmetric scale 0.01969, sai
  so 0.00945, chi dung ~103/255 muc (40%); asymmetric scale
  0.00784, sai so 0.00392, dung het 256 muc - tot hon 2.4x vi
  khong bo phi nua thang do am.
- Edge case: tensor hang so (max == min) va tensor toan 0
  duoc guard (scale 1.0, khoi phuc dung, khong chia cho 0);
  bits ngoai [2, 16] bi chan bang ValueError.

## Bai 2 — Outlier pha thang do

Ma tran 4x1000, channel 0 nhan 50. MSE theo channel:

| | INT8 per-tensor | INT8 per-channel | INT4 per-tensor | INT4 per-channel |
|---|---|---|---|---|
| Channel 0 (outlier) | 0,1645 | 0,1645 | 48,6712 | 48,6712 |
| Channel 1–3 (TB) | 0,1670 | 0,0001 | 0,9605 | 0,0170 |

Outlier lam hai NHAT cac channel binh thuong o per-tensor:
chung bi don vao vai muc, MSE 0.167 (gan bang ca phuong sai
cua du lieu, tuc thong tin gan nhu mat het). Per-channel cho
moi channel thang do rieng: MSE tro ve 0.0001, tot hon
2844x o INT8. O INT4, per-channel van tot hon per-tensor
56x nhung MSE tuyet doi 0.017 (gap INT8 170x) - per-channel
cung bat dau duoi, can them per-group/GPTQ.

## Bai 3 + 4 — Weight-only MLP + benchmark

MLP 1.218 tham so, train 200 epoch tren du lieu tong hop
(vong tron r^2 > 1). Accuracy FP32 = 0.995.

| Phien ban | Accuracy | Bo nho | Tre b1 | Tre b256 | Regression 8/8 |
|---|---|---|---|---|---|
| FP32 | 0,995 | 4,8 KB | 0,034 ms | 0,072 ms | 100% |
| INT8 | 0,995 | 1,2 KB | 0,033 ms | 0,071 ms | 100% |
| INT4 | 0,983 | 0,6 KB | 0,034 ms | 0,071 ms | 50% |
| INT4 per-channel | 0,988 | 0,6 KB | 0,033 ms | 0,072 ms | 38% |

Ghi chu trung thuc ve do tre: ban dau do bang trung binh
200 lan cho ra so nhay loan (INT4 0.378ms - nhieu do, khong
phai tin hieu). Do lai bang TRUNG VI 1000 lan chay rieng le:
ca 4 phien ban gan nhu GIONG HET nhau, dung nhu Cau 3 cua de
bai - gia lap (dequantize ve FP32 roi tinh FP32) khong the
nhanh hon; muon nhanh that can kernel INT8/INT4 that. Khong
duoc doc bang nay thanh "INT4 cham hon".

Nang luong (uoc luong, khong phai so do dien): batch 1
memory-bound, energy/token ti le voi byte trong so phai doc
-> INT8/FP32 = 0.25x, INT4/FP32 = 0.125x.

## Quyet dinh (suy tu so do)

1. Chatbot can giu chat luong: **INT8 weight-only** -
   accuracy bang FP32, regression 8/8, bo nho giam 4x.
2. May yeu offline: **INT4 CHUA duoc release** - ca hai ban
   INT4 deu lat lung tung gan bien quyet dinh (regression
   50% va 38%, du per-channel accuracy cao hon per-tensor).
   Dung INT8, hoac cai thien INT4 bang per-group/GPTQ roi
   do lai. Quyet dinh nay nguoc voi du dinh ban dau ("chon
   INT4 per-channel cho may yeu") - so do thang, ghi that.
3. Regression set (8 input, ke ca bien r^2 gan 1: (0.7,0.7)
   trong, (0.71,0.71) ngoai): (0,0), (1,0), (0,1), (0.7,0.7),
   (0.71,0.71), (2,2), (-0.7,-0.7), (0.5,0.86). Moi phien ban
   sau nay phai dong y 8/8 voi FP32 truoc khi release.

## Checklist tu soat (7 diem AGENTS.md)

- Edge case: tensor rong, bits ngoai [2,16], tensor hang so
  (chia cho 0) deu co guard o dau ham.
- Docstring viet sau khi doc lai code (kieu tra ve, Args).
- Khong tao tensor moi trong forward (lab dung toan tensor
  co san; quantize lam viec tren x truyen vao, khong dinh
  bug device).
- Khong doc file - khong dinh gia dinh file I/O.
- Thoi gian do bang perf_counter(), lay TRUNG VI 1000 lan
  (sau khi phat hien trung binh 200 lan bi nhieu keo lech).
- Ablation chi doi 1 bien: INT8/INT4 tren cung model da
  train, cung seed, cung tap test; per-tensor vs per-channel
  tren cung ma tran W.
- Khong co resource can dong trong finally.
