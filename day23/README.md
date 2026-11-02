# Day 23 (buoi toi) — Lab Distillation: student nho hoc tu teacher lon

Chay tren CPU (torch 2.14.1+cpu), khong GPU, khong tai model
lon. Toan bo lab chay trong ~11s (venv Day 15).

Quy tac thi nghiem, viet truoc khi chay Bai 3: thu duy nhat
thay doi la cach tinh loss (hard-label hay distillation);
du lieu, kien truc student, seed va so epoch giu nguyen.

## Du lieu va teacher

Du lieu 3 lop tong hop: lop 0 tai (0, 0), lop 1 tai
(1.2, 0.3) - gan lop 0, lop 2 tai (0, 4.0) - xa ca hai.
Nhieu 0.6, 400 mau/lop, chia 900 train / 300 test (seed 0).

Teacher MLP(128), train 150 epoch tren du 900 mau:
17.283 params, test acc = 0.843. Lop 2 dat 1.000 (tach
biet), lop 0/1 chong lap nhau nen teacher cung chi dat
0.789 / 0.758 o 2 lop nay - do la noi dark knowledge co
the giup, cung la noi loi cua teacher tap trung.

## Bai 1 — Soft target va temperature

Mau dau tien cua test (lop 0, logits [6.83, 4.28, -12.65]):

| T | p0 | p1 | p2 | p1/p2 |
|---|---|---|---|---|
| 1 | 0.9275 | 0.0725 | 0.0000 | 22.496.088 |
| 2 | 0.7815 | 0.2184 | 0.0000 | 4.743 |
| 5 | 0.6170 | 0.3705 | 0.0125 | 29.5 |
| 10 | 0.5215 | 0.4041 | 0.0744 | 5.4 |

O T=1, p2 ~ 3e-09, gan nhu bang 0 - quan he "1 gan, 2 xa"
bi chon vui. O T=5 quan he lo ro nhat: lop 1 noi len
0.3705, lop 2 van chi 0.0125. O T=10 phan phoi tien ve
deu (0.52/0.40/0.07), tuong phan mo dan.

Trung binh tren 114 mau lop 0 cua test cung noi len
hinh do: T=1 [0.7629, 0.2301, 0.0070], T=5 [0.6183,
0.3577, 0.0240], T=10 [0.5383, 0.3787, 0.0830].

## Bai 2 — Loss distillation

Ham `distill_loss` tu viet: alpha*CE + (1-alpha)*T^2*KL,
teacher detach, F.kl_div nhan log_softmax student va
softmax teacher (dung chieu KL(teacher || student)).

Batch gia (seed 123, 8 mau, alpha=0.3):

| T | Loss tong | CE | KL goc | T^2 * KL |
|---|---|---|---|---|
| 1.0 | 0.6530 | 1.2987 | 0.3762 | 0.3762 |
| 4.0 | 0.7329 | 1.2987 | 0.0306 | 0.4904 |

KL goc giam 12x khi T tang 1 -> 4, nhung loss tong lai
TANG vi nhan T^2 - do la chu dich cua he so T^2, khong
phai loi. Tai tao cong thuc dung o ca 2 T; detach that:
backward xong grad teacher la None, grad student co gia
tri. 5 guard (T=0, alpha=1.5, shape khac, batch rong,
nhan ngoai [0, 2]) deu chan dung bang ValueError.

## Bai 3 — Hard-label vs distillation (co kiem soat)

Student MLP(32), chi thay 300 mau train dau, 100 epoch,
seed 0. A vs B chi khac cach tinh loss; B vs C chi khac
T (4.0 -> 2.0), alpha/seed/epoch giong het.

| Cau hinh | Test acc | Dong y teacher | Lop 0 | Lop 1 | Lop 2 | Sao chep cho teacher sai |
|---|---|---|---|---|---|---|
| Teacher (900 mau) | 0.843 | — | 0.789 | 0.758 | 1.000 | — |
| A hard-only | 0.853 | 0.943 | 0.798 | 0.779 | 1.000 | 0.787 |
| B distill T=4, a=0.3 | 0.857 | 0.953 | 0.833 | 0.747 | 1.000 | 0.809 |
| C distill T=2, a=0.3 | 0.857 | 0.953 | 0.825 | 0.758 | 1.000 | 0.809 |

Sao chep cho teacher sai: tinh tren 47 mau test ma
teacher du doan sai. B sao chep 38/47, A sao chep 37/47.

Kiem tra them seed 1..4 (trung binh 5 seed, gom seed 0):
A = 0.848, B = 0.859, C = 0.855. B hon A trung binh
0.011 (~3 mau test) va hon o ca 5 seed rieng le - chenh
lech nho nhung on dinh, khong phai may man cua 1 seed.
Test chi co 300 mau nen 1% = 3 mau: khong duoc doc
0.857 vs 0.853 thanh "thang lon".

Green LLM (trung vi 1000 lan, perf_counter, 300 mau):

| | Params | ms/lan |
|---|---|---|
| Teacher | 17.283 | 0.295 |
| Student B | 1.251 | 0.084 |

Ti le params 0.072 (giam 92.8%), toc do nhanh hon 3.51x
trong lan chay nay. So ms tuyet doi phu thuoc tai may:
kiem tra doc lap ngay sau do cho 0.168 / 0.060 ms
(2.80x). Du 2.8x hay 3.5x, ket luan khong doi: it params
khong dong nghia toc do tang ty le - overhead moi lan
chay moi la phan chi phoi o model nho the nay, chu khong
phai so params.

Ket luan 2 cau: distillation giup it nhung on dinh, chu
yeu nho cai thien lop 0 (+0.035), doi lai lop 1 giam
0.032 va sao chep loi teacher nhieu hon 0.021. Doi
0.011 accuracy lay 92.8% it params va ~3x toc do la
dang o workload nay, vi student da vuot ca teacher -
nhung phai nho student cung ke thua loi cua teacher,
nen tap test that, doc lap la bat buoc, khong danh gia
tren du lieu do chinh teacher sinh ra.

## Checklist tu soat (12 diem AGENTS.md)

- Edge case: distill_loss guard ngay dau ham - batch
  rong, shape khac, nhan ngoai [0, 2], T <= 0/khong
  huu han, alpha ngoai [0, 1], khac device; ca 5
  truong hop deu co test chan dung trong Bai 2.
- Docstring viet sau khi doc lai code (kieu tra ve,
  Args, Raises cua distill_loss khop code that).
- Khong tao tensor moi trong forward (MLP chi dung
  nn.Linear/ReLU co san); du lieu tao 1 lan o
  build_data, khong dinh bug device.
- Khong doc file - khong dinh gia dinh file I/O.
- Thoi gian do bang perf_counter(), lay TRUNG VI 1000
  lan (sau warmup 50), theo bai hoc Day 22 ve trung
  binh bi nhieu keo lech.
- Ablation chi doi 1 bien: A vs B chi doi mode, B vs
  C chi doi T; diff loi goi duoc ghi trong docstring
  bai3, cung seed/du lieu/epoch/kien truc.
- Khong co resource can dong trong finally (khong mo
  file, khong tracker).
- Tham so tra ve tu nhat quan: distill_loss tra loss
  vo huong, Bai 2 tai tao lai tu CE/KL goc dung ca
  2 T; khong tra cap (gia tri, tham so) nao.
- Khong co tham chieu chet: moi ham phu (count_params,
  per_class_accuracy, copy_wrong_rate,
  _expect_value_error) deu duoc goi that; khong co
  hang/nhan nao khai bao roi bo khong.
- Ablation diff tung doi so: B vs C chi T khac nhau,
  ke ca doi so co mac dinh (alpha=0.3 giong het).
- Khong co bang tra cuu can audit sum() - khong mo
  rong bang nao trong lab nay.
