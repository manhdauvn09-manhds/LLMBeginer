# Day 19 (buoi toi) — Tu code loss DPO tren logits gia

Chay tren CPU, torch 2.14.1+cpu, khong can model that, khong
can GPU. Toan bo lab chay trong ~5s (logits gia, compute gan
nhu bang 0 - dung nguyen tac tiet kiem cua khoa hoc: chung minh
cong thuc dung tren do choi truoc, roi moi dot GPU cho model
that).

## Bai 1 — Log-prob cua mot cau tra loi

`sequence_logprob`: log_softmax theo chieu vocab -> gather
log-prob cua dung token tai moi vi tri -> cong tat ca.

- Cau 4 token, vocab 10 (seed 0): log-prob = **-9,1231** (am,
  dung ban chat log xac suat < 1).
- Doi token 0 thanh argmax tai vi tri do: **-7,8405** (tang
  len) - gather dung chieu.
- Edge case: cau rong -> ValueError (tong tren 0 token = 0.0,
  khong con la log-prob am, de gay hieu nham); token vuot
  vocab -> ValueError; beta <= 0 -> ValueError.

## Bai 2 — Loss DPO va gradient

Loss = -log sigmoid(beta * ((logp_w - logp_ref_w) -
(logp_l - logp_ref_l))), ref detach trong ham (phong thu: ke
ca caller dua ref co grad thi ham van detach).

- Policy == ref: loss = **0,6931** = -log(0.5) dung (ca hai
  hieu log-ratio bang 0, sigmoid(0) = 0.5) - diem xuat phat
  "chua phan biet duoc chosen va rejected".
- Gradient: chosen norm 0,1150; rejected norm 0,1157.
- Buoc nho theo -grad (lr=0.1): loss 0,6931 -> 0,6905 (giam);
  chosen logp -19,3522 -> -19,3258 (tang); rejected logp
  -18,0526 -> -18,0794 (giam). Gradient hai phia nguoc dau,
  dung truc giac DPO: day chosen len, keo rejected xuong.
- Thi nghiem quen detach: khong detach thi gradient chay vao
  ref (norm 0,1150 / 0,1157 - moc tu di chuyen, loss co the
  giam ma policy chang hoc duoc gi). Voi detach dung, ke ca
  ref co requires_grad thi grad van None.

Tra loi 3 cau cua de bai:
1. Loss ~0.693 vi policy = ref -> ca hai hieu bang 0 ->
   sigmoid(0) = 0.5 -> -log(0.5) = 0.693.
2. Gradient chosen/rejected nguoc dau vi loss phu thuoc vao
   HIEU (logp_w - logp_l): tang chosen hoac giam rejected deu
   lam tang hieu, giam loss.
3. Quen detach: gradient chay vao ref, moc di chuyen theo
   huong lam loss giam - policy co the khong hoc gi ma loss
   van giam. Do la loi nghiem trong, khong phai loi nho.

## Bai 3 — Quet beta (PHAT HIEN QUAN TRONG)

Cung cap logits gia, cung SGD lr=0.5, chi beta thay doi.

| beta | Δchosen 50 buoc | Δchosen 200 buoc | Do lech rejected (50b) | Loss 50b |
|---|---|---|---|---|
| 0.05 | +2,9656 | +8,7366 | -3,1592 | 0,5541 |
| 0.1 | +4,6831 | +9,7818 | -5,1517 | 0,3217 |
| 0.5 | +4,0078 | +5,2658 | -4,3459 | 0,0155 |
| 1.0 | +2,7389 | +3,3724 | -2,8914 | 0,0037 |

**Ket qua that KHONG don dieu nhu de bai noi.** De bai bao
"beta cang nho, policy cang troi xa ref va hoc preference cang
nhanh". So do cho thay voi 50 buoc co dinh, do troi co dang
chu U nguoc, dinh o beta=0.1.

Giai thich bang co che gradient: gradient ti le voi
beta*sigmoid(-beta*Δ). Beta nho khoi dong cham (gradient nho)
nhung khong bao hoa; beta lon khoi dong nhanh nhung bao hoa
som o Δ nho. Chay them 200 buoc kiem chung: beta=0.05 van tiep
tuc troi (+2,97 -> +8,74) trong khi beta=1.0 gan nhu dung yen
(+2,74 -> +3,37). Vay:

- "Beta nho troi xa hon" chi dung o gioi han TIEM CAN (du
  buoc), khong dung trong ngan sach buoc co dinh.
- "Hoc nhanh" neu do bang loss thi NGUOC LAI: beta lon giam
  loss nhanh hon (0.55 -> 0.004) vi beta nhan truc tiep vao
  margin trong sigmoid.

Ghi chu: vi policy khoi tao = ref nen (a) logp chosen tang va
(b) do lech khoi ref TRUNG NHAU dung - khong phai 2 so doc lap
nhu de bai liet ke; neu khoi tao khac ref thi chung moi khac
nhau.

## Bai 4 — Mo phong reward hacking

Reward = logp_policy(cau) + 0.5 x do dai, toi uu tran 30 buoc
(khong KL, khong ref), moi buoc noi them 1 token dem vo nghia.

| Buoc | Do dai | Reward | Token thong tin |
|---|---|---|---|
| 0 | 4 | -14,410 | 4 |
| 10 | 14 | -9,982 | 4 (dung yen) |
| 20 | 24 | -6,630 | 4 (dung yen) |
| 30 | 34 | +0,096 | 4 (dung yen) |

Reward tang deu (tu -14,4 len +0,1) nho cong do dai (+0.5 moi
token dem) va logp token dem cung tang theo ascent, nhung cau
khong them thong tin gi - so token co thong tin dung yen o 4.
Do la reward hacking thu nho: toi uu reward tran ma khong co
KL neo ve ref, model hoc "do dai" thay vi "chat luong". Ly do
bai sang yeu cau danh gia mu + dung som, khong chi nhin reward.

## Checklist tu soat (7 diem AGENTS.md)

- Edge case: cau rong, token vuot vocab, beta <= 0 deu co
  ValueError ro rang o dau ham.
- Docstring viet sau khi doc lai code (kieu tra ve, Args).
- Khong co tensor moi trong forward kieu arange/zeros (lab
  logits gia, CPU) - khong dinh bug device.
- Khong doc file - khong dinh gia dinh file I/O.
- Thoi gian do bang perf_counter().
- Ablation chi doi 1 bien: quet beta giu nguyen cap logits,
  seed, buoc, lr; them ban 200 buoc de kiem chung co che thay
  vi suy dien tu 1 bang so.
- Khong co resource can dong trong finally (khong tracker,
  khong file).
