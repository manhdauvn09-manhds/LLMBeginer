"""Lab Day 29 (buoi toi) - Gia thuyet bac bo duoc
va thi nghiem kiem chung (capstone GreenLLM).

Thu tu dung (bai hoc sang): viet gia thuyet +
nguong bac bo TRUOC, chay thi nghiem SAU - xem
research_plan.md (da viet truoc khi chay lab
nay). Gia thuyet: quantization INT8 + pruning
30% trong so nho giam it nhat 40% nang luong
moi token, accuracy giam toi da 2 diem %.

Thi nghiem thu nho toi nay: MLP nho tren bai
toan phan loai tong hop (mo phong dung "nghi
thuc" cua thi nghiem that). Baseline -> bien
the (chi doi 1 thu: cach nen) -> lap 3 seed.
Nhanh accuracy kiem chung truc tiep theo
nguong 2 diem %. Nhanh nang luong KHONG do
duoc o day: fake-quant tren CPU khong giam
thoi gian that, thoi gian suy luan chi la
proxy - ghi nhan, khong ket luan thay.

Bai 3 (them): ma tran ablation mini - A chi
pruning, B chi quant, C ca hai; moi bien the
chi khac baseline dung 1 thu.

Chay: python hypothesis_lab.py
Ket qua: in bang 3 seed + bang ablation +
ket luan theo nguong da dang ky truoc.
"""

# ============================================================
# [TRAINER] REVIEW Day 29 (ban hypothesis_lab_review.py)
# Kiem chung doc lap: chay lai toan bo lab - tu kiem tra
# PASS, so accuracy tai lap CHINH XAC tung so voi
# research_plan.md (A 0,33 +- 0,25; B -0,03 +- 0,06;
# C 0,37 +- 0,21 diem %; ket luan nhanh accuracy DUOC
# UNG HO dung nguong dang ky truoc 2 diem %). Protocol
# dung thu tu (gia thuyet + nguong viet truoc), nhanh
# nang luong ghi ro CHUA kiem chung duoc - trung thuc.
# KHONG co loi logic lam sai ket luan. 3 diem nhe:
#  (1) evaluate khong warmup + thu tu bien the co dinh
#      lam proxy thoi gian them nhieu (comment o evaluate).
#  (2) 3 seed chi doi khoi tao, du lieu co dinh - do lech
#      chua gom bien thien du lieu (comment o
#      chuan_bi_du_lieu).
#  (3) Thu tu uu tien chon capstone C > A > B la thu tu
#      cung, chua phai "nen manh nhat" theo kich thuoc
#      that (comment o ket_luan).
# ============================================================

import copy
import statistics
import time

import torch
import torch.nn as nn
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split

# --- Thong so thi nghiem (theo de bai) ---
N_MAU = 4000
N_DAC_TRUNG = 32
N_CO_TINH = 20
N_LOP = 2
TI_LE_TEST = 0.25
KICH_THUOC_AN = (128, 64)
N_EPOCHS = 15
LR = 1e-3
N_LAP_DO = 20
SEEDS = (0, 1, 2)
TI_LE_CAT_TIA = 0.30
# Nguong da dang ky TRUOC khi chay (xem
# research_plan.md): accuracy giam toi da 2
# diem phan tram, neu vuot thi gia thuyet bi
# bac bo. Khong duoc sua nguong sau khi thay
# ket qua.
NGUONG_ACC_GIAM_PP = 2.0


def _kiem_tra_so_nguyen(ten, gia_tri, toi_thieu=1):
    """Guard: so nguyen >= toi_thieu."""
    if not isinstance(gia_tri, int) \
            or isinstance(gia_tri, bool) \
            or gia_tri < toi_thieu:
        raise ValueError(
            f"{ten} phai la so nguyen >= "
            f"{toi_thieu}.")
    return gia_tri


def _kiem_tra_so_thuc(ten, gia_tri, nho_hon=None,
                      lon_hon_bang=None, lon_hon=None):
    """Guard: so thuc huu han, trong khoang.

    nho_hon: <; lon_hon_bang: >=; lon_hon: >
    (can rieng cho ti le cat tia: 0 < ti_le
    < 1, vi ti_le = 0 thi khong cat gi ca).
    """
    if isinstance(gia_tri, bool) \
            or not isinstance(gia_tri, (int, float)):
        raise ValueError(f"{ten} phai la so.")
    gia_tri = float(gia_tri)
    if gia_tri != gia_tri or gia_tri in (
            float("inf"), float("-inf")):
        raise ValueError(f"{ten} phai huu han.")
    if nho_hon is not None and not gia_tri < nho_hon:
        raise ValueError(f"{ten} phai < {nho_hon}.")
    if lon_hon_bang is not None \
            and not gia_tri >= lon_hon_bang:
        raise ValueError(
            f"{ten} phai >= {lon_hon_bang}.")
    if lon_hon is not None \
            and not gia_tri > lon_hon:
        raise ValueError(f"{ten} phai > {lon_hon}.")
    return gia_tri


# [TRAINER] Diem nhe 2 - ca 3 seed dung CHUNG 1 bo du
# lieu va 1 lan chia train/test co dinh (random_state=0
# o ca hai buoc duoi day); seed chi thay doi khoi tao
# mo hinh khi train baseline. Vi vay do lech +- 0,21
# diem % cua C chi phan anh bien thien KHOI TAO, chua
# gom bien thien DU LIEU. Ket luan "duoc ung ho" dung
# trong pham vi protocol da dang ky, nhung khi len
# Phase 2 voi du lieu that, nen de seed chi phoi ca
# viec chia du lieu (hoac dung k-fold) thi std moi co
# y nghia day du de so voi nguong.

def chuan_bi_du_lieu():
    """Du lieu tong hop co dinh cho ca 3 seed.

    make_classification (random_state=0) + chia
    train/test (random_state=0): du lieu GIONG
    nhau o moi seed, chi khoi tao mo hinh khac
    nhau theo seed - dung nhu de bai (Buoc 3).
    Tra (Xtr, Xte, ytr, yte) dang tensor.
    """
    X, y = make_classification(
        n_samples=N_MAU, n_features=N_DAC_TRUNG,
        n_informative=N_CO_TINH, n_classes=N_LOP,
        random_state=0)
    Xtr, Xte, ytr, yte = train_test_split(
        X, y, test_size=TI_LE_TEST, random_state=0)
    Xtr = torch.tensor(Xtr, dtype=torch.float32)
    Xte = torch.tensor(Xte, dtype=torch.float32)
    ytr = torch.tensor(ytr)
    yte = torch.tensor(yte)
    return Xtr, Xte, ytr, yte


class MLP(nn.Module):
    """MLP nho: Linear -> ReLU -> Linear -> ReLU
    -> Linear (theo de bai: 32-128-64-2)."""

    def __init__(self, n_vao=N_DAC_TRUNG,
                 n_an1=KICH_THUOC_AN[0],
                 n_an2=KICH_THUOC_AN[1],
                 n_ra=N_LOP):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_vao, n_an1), nn.ReLU(),
            nn.Linear(n_an1, n_an2), nn.ReLU(),
            nn.Linear(n_an2, n_ra))

    def forward(self, x):
        return self.net(x)


def train(model, Xtr, ytr, epochs=N_EPOCHS, lr=LR):
    """Huan luyen full-batch: AdamW + CE.

    Full-batch (khong mini-batch, khong xao
    tron) - dung nhu de bai. Tra loss cuoi.
    """
    if not isinstance(model, nn.Module):
        raise ValueError("model phai la nn.Module.")
    _kiem_tra_so_nguyen("epochs", epochs)
    _kiem_tra_so_thuc("lr", lr, lon_hon_bang=0)
    opt = torch.optim.AdamW(model.parameters(),
                            lr=lr)
    loss_fn = nn.CrossEntropyLoss()
    loss_cuoi = None
    for _ in range(epochs):
        model.train()
        opt.zero_grad()
        loss = loss_fn(model(Xtr), ytr)
        loss.backward()
        opt.step()
        loss_cuoi = float(loss.detach())
    return loss_cuoi


@torch.no_grad()
def evaluate(model, Xte, yte, n_lap=N_LAP_DO):
    """Do accuracy + thoi gian suy luan.

    Chay n_lap luot forward tren tap test, lay
    thoi gian trung binh / luot (proxy nang
    luong - xem docstring module). Dung
    time.perf_counter() de do thoi gian.
    Tra (acc, giay_trung_binh_1_luot).
    """
    if not isinstance(model, nn.Module):
        raise ValueError("model phai la nn.Module.")
    _kiem_tra_so_nguyen("n_lap", n_lap)
    model.eval()
    # [TRAINER] Diem nhe 1 - do thoi gian o day khong co
    # luot warmup, va chay_thi_nghiem luon do baseline
    # truoc roi cac bien the theo thu tu co dinh A, B, C:
    # ben nao chay dau trong dot thuong "an" chi phi khoi
    # dong (thread spin-up cua torch). Bang chung khi
    # Trainer chay lai: seed 0 baseline 0,62 ms nhung A
    # (bien the dau tien) 4,80 ms; seed 1 nguoc lai -
    # baseline 4,50 ms con cac bien the chi ~0,45 ms.
    # Thoi gian o day chi la proxy nen ket luan khong
    # doi, nhung nen them 1-2 luot forward warmup truoc
    # t0 (dung chuan warmup cua Day 28) de proxy bot
    # nhieu, va neu sau nay thoi gian duoc dung lam bang
    # chung thi phai xao thu tu bien the theo seed.
    model.eval()
    t0 = time.perf_counter()
    for _ in range(n_lap):
        _ = model(Xte)
    dt = (time.perf_counter() - t0) / n_lap
    acc = (model(Xte).argmax(1) == yte).float() \
        .mean().item()
    return acc, dt

def _kiem_tra_ti_le(ti_le):
    """Guard: ti le cat tia trong (0, 1)."""
    return _kiem_tra_so_thuc(
        "ti_le", ti_le, nho_hon=1, lon_hon=0)


def cat_tia(model, ti_le=TI_LE_CAT_TIA):
    """Pruning: dua 30% trong so nho nhat ve 0.

    Voi moi nn.Linear: nguong = phan vi 30%
    cua |trong so| (torch.quantile); trong so
    nao |gia tri| <= nguong thi ve 0. Chi sua
    weight, khong sua bias. Lam TAI CHO tren
    model truyen vao (no_grad). Tra ti le
    trong so thuc te bi cat (tinh tu du lieu,
    khoang 0.30 - quantile noi suy co the
    lech 1-2 trong so).
    """
    if not isinstance(model, nn.Module):
        raise ValueError("model phai la nn.Module.")
    ti_le = _kiem_tra_ti_le(ti_le)
    tong = 0
    da_cat = 0
    with torch.no_grad():
        for module in model.modules():
            if not isinstance(module, nn.Linear):
                continue
            w = module.weight.data
            nguong = torch.quantile(
                w.abs().flatten(), ti_le)
            mat_na = w.abs() <= nguong
            da_cat += int(mat_na.sum())
            tong += w.numel()
            w[mat_na] = 0.0
    if tong == 0:
        raise ValueError(
            "model khong co trong so Linear.")
    return da_cat / tong


def luong_hoa_int8(model):
    """Fake-quant INT8: lam tron ve 256 muc.

    w_q = round(w / scale) * scale voi scale =
    max|w| / 127 (theo tung lop Linear). Trong
    so 0 (da cat tia) van la 0. Day la fake-
    quant - du de thay hieu ung accuracy, con
    toc do that can kernel INT8. Lop nao
    max|w| == 0 thi bo qua (tranh chia 0).
    Lam TAI CHO, khong sua bias.
    """
    if not isinstance(model, nn.Module):
        raise ValueError("model phai la nn.Module.")
    with torch.no_grad():
        for module in model.modules():
            if not isinstance(module, nn.Linear):
                continue
            w = module.weight.data
            lon_nhat = float(w.abs().max())
            if lon_nhat == 0.0:
                continue
            scale = lon_nhat / 127.0
            w.copy_(torch.round(w / scale) * scale)


def prune_and_quantize(model, ti_le=TI_LE_CAT_TIA):
    """Nen: pruning 30% roi fake-quant INT8.

    Thu tu: cat tia truoc (trong so nho ve 0),
    roi luong hoa phan con lai. KHONG train
    lai sau khi nen - do la dieu thi nghiem
    muon do: nen "tran" mat bao nhieu diem.
    Tra ti le trong so thuc te bi cat tia.
    """
    ti_le_cat = cat_tia(model, ti_le)
    luong_hoa_int8(model)
    return ti_le_cat


def _tb_lc(ds):
    """Trung binh +- do lech chuan mau (n - 1)."""
    if len(ds) == 1:
        return float(ds[0]), 0.0
    return (float(statistics.mean(ds)),
            float(statistics.stdev(ds)))


def chay_thi_nghiem(Xtr, Xte, ytr, yte):
    """Chay thi nghiem co doi chung, lap 3 seed.

    Moi seed: train baseline MOI (torch.manual_
    seed(seed)) -> do baseline -> deepcopy roi
    nen tung bien the (A chi pruning, B chi
    quant, C ca hai) -> do lai. Moi bien the
    chi khac baseline dung 1 thu (Bai 3). Tra
    dict: moi bien the co list giam accuracy
    (diem %) va ti le thoi gian (t_bien / t_co
    so) qua 3 seed, kem baseline tung seed.
    """
    bien_the = {
        "A (chi pruning)": lambda m: cat_tia(m),
        "B (chi quant)": lambda m: luong_hoa_int8(m),
        "C (pruning + quant)":
            lambda m: prune_and_quantize(m),
    }
    ket_qua = {ten: {"giam_acc": [],
                     "ti_le_tgian": []}
               for ten in bien_the}
    co_so = []
    for seed in SEEDS:
        torch.manual_seed(seed)
        base = MLP()
        train(base, Xtr, ytr)
        acc_b, t_b = evaluate(base, Xte, yte)
        co_so.append((acc_b, t_b))
        print(f"  seed {seed}: baseline acc="
              f"{acc_b:.4f} t={t_b * 1000:.2f}ms")
        for ten, ham_nen in bien_the.items():
            var = copy.deepcopy(base)
            ham_nen(var)
            acc_v, t_v = evaluate(var, Xte, yte)
            giam = (acc_b - acc_v) * 100
            ket_qua[ten]["giam_acc"].append(giam)
            ket_qua[ten]["ti_le_tgian"].append(
                t_v / t_b)
            print(f"    {ten}: acc={acc_v:.4f} "
                  f"t={t_v * 1000:.2f}ms | giam "
                  f"{giam:.2f} diem %")
    return ket_qua, co_so


def ket_luan(ket_qua):
    """Ket luan theo nguong da dang ky TRUOC.

    Nhanh accuracy: giam trung binh <= 2 diem
    % thi "duoc ung ho", nguoc lai "bi bac
    bo". Nhanh nang luong: CHUA KIEM CHUNG
    DUOC - thi nghiem nay chi do thoi gian CPU
    lam proxy. Chon bien the cho capstone:
    trong cac bien the co giam trung binh <=
    2 diem %, chon bien the nen manh nhat (C
    > A, B); khong co bien the nao dat thi
    khong chon, de xuat gia thuyet moi (vd
    pruning 15%). Moi lua chon tinh tu so do.
    """
    print("\nBANG ABLATION (giam accuracy, diem %):")
    tb = {}
    for ten, ds in ket_qua.items():
        tb_giam, lc_giam = _tb_lc(ds["giam_acc"])
        # Thoi gian: lay TRUNG VI ti le (khong lay
        # trung binh) vi 1 lan do bi tre tren VM
        # dung chung co the lech 10x va keo ca
        # trung binh theo - thoi gian o day chi
        # la proxy, khong phai bang chung nang
        # luong.
        tgian_tv = float(
            statistics.median(ds["ti_le_tgian"]))
        tb[ten] = tb_giam
        print(f"  {ten}: giam {tb_giam:.2f} +- "
              f"{lc_giam:.2f} diem % | thoi gian "
              f"trung vi x{tgian_tv:.2f} so voi "
              f"baseline")
    ten_c = "C (pruning + quant)"
    giam_c = tb[ten_c]
    if giam_c <= NGUONG_ACC_GIAM_PP:
        print(f"\nKet luan nhanh accuracy: gia "
              f"thuyet DUOC UNG HO (C giam "
              f"{giam_c:.2f} diem %, trong nguong "
              f"<= {NGUONG_ACC_GIAM_PP:.0f} diem "
              f"%).")
        duoc_ung_ho = True
    else:
        print(f"\nKet luan nhanh accuracy: gia "
              f"thuyet BI BAC BO (C giam "
              f"{giam_c:.2f} diem %, vuot nguong "
              f"<= {NGUONG_ACC_GIAM_PP:.0f} diem "
              f"%). Ket qua am van la ket qua tot "
              f"- ghi lai, khong giau.")
        duoc_ung_ho = False
    print("Ket luan nhanh nang luong: CHUA KIEM "
          "CHUNG DUOC (fake-quant tren CPU khong "
          "giam thoi gian that; can GPU + "
          "nvidia-smi / codecarbon).")
    ten_a = "A (chi pruning)"
    ten_b = "B (chi quant)"
    if tb[ten_a] > tb[ten_b]:
        print(f"Pruning mat diem nhieu hon quant "
              f"({tb[ten_a]:.2f} vs {tb[ten_b]:.2f} "
              f"diem %).")
    elif tb[ten_b] > tb[ten_a]:
        print(f"Quant mat diem nhieu hon pruning "
              f"({tb[ten_b]:.2f} vs {tb[ten_a]:.2f} "
              f"diem %).")
    else:
        print(f"Pruning va quant mat diem nhu nhau "
              f"({tb[ten_a]:.2f} diem %).")
    # [TRAINER] Diem nhe 3 - thu tu (C, A, B) o day la
    # thu tu CO DINH trong code, duoc dien giai la "nen
    # manh nhat", nhung A (pruning 30%) va B (INT8) nen
    # theo hai chieu khac nhau: INT8 giam kich thuoc luu
    # tru ~4 lan, con pruning 30% chi tao 30% so 0 - neu
    # luu dang dense thi kich thuoc khong giam chut nao.
    # Lan nay C dat nguong nen khong anh huong; neu C
    # truot va A, B cung dat, tieu chi chon nen tinh
    # bang ti le kich thuoc / chi phi that cua tung bien
    # the thay vi thu tu cung nay.
    dat = [ten for ten in
           (ten_c, ten_a, ten_b)
           if tb[ten] <= NGUONG_ACC_GIAM_PP]
    if dat:
        print(f"Chon cho capstone: {dat[0]} (nen "
              f"manh nhat trong so cac bien the "
              f"dat nguong).")
        chon = dat[0]
    else:
        print("Khong chon bien the nao - tat ca "
              "vuot nguong 2 diem %. De xuat gia "
              "thuyet moi (vd: pruning 15%) cho "
              "lan chay sau.")
        chon = None
    return {"duoc_ung_ho": duoc_ung_ho,
            "giam_tb": tb, "chon": chon}


def _expect_value_error(fn, mo_ta):
    """True neu fn nem ValueError (guard chan)."""
    try:
        fn()
    except ValueError as exc:
        print(f"  guard {mo_ta}: chan dung ({exc})")
        return True
    print(f"  guard {mo_ta}: KHONG chan!")
    return False


def tu_kiem_tra():
    """Tu kiem tra bang assert (fail la dung).

    Kiem: guard chan sai dau vao, cat tia dung
    ~30%, quant lam tron dung muc + giu 0,
    bias khong doi, deepcopy cach ly baseline.
    """
    ket_qua = [
        _expect_value_error(
            lambda: _kiem_tra_ti_le(0),
            "ti le cat tia = 0"),
        _expect_value_error(
            lambda: _kiem_tra_ti_le(1.5),
            "ti le cat tia = 1.5"),
        _expect_value_error(
            lambda: cat_tia(None),
            "cat tia model None"),
        _expect_value_error(
            lambda: luong_hoa_int8("khong phai "
                                   "model"),
            "quant model sai kieu"),
        _expect_value_error(
            lambda: evaluate(MLP(),
                             torch.zeros(2, 32),
                             torch.zeros(2,
                                         dtype=torch.long),
                             n_lap=0),
            "evaluate n_lap=0"),
    ]
    assert all(ket_qua), "Co guard khong chan dung!"
    tb, lc = _tb_lc([1.0, 2.0, 3.0])
    assert abs(tb - 2.0) <= 1e-9
    assert abs(lc - 1.0) <= 1e-9

    # Cat tia ~30% tren model nho (seed co dinh).
    torch.manual_seed(0)
    nho = MLP(64, 32, 16, 2)
    ti_le_cat = cat_tia(nho)
    assert 0.28 <= ti_le_cat <= 0.32, \
        f"cat tia {ti_le_cat:.3f}, muon ~0.30"
    n_0_sau_cat = sum(
        int((m.weight.data == 0).sum())
        for m in nho.modules()
        if isinstance(m, nn.Linear))

    # Quant: giu nguyen so 0, gia tri la boi
    # cua scale, max|w| khong doi.
    torch.manual_seed(1)
    nho2 = MLP(64, 32, 16, 2)
    goc = {id(m): m.weight.data.clone()
           for m in nho2.modules()
           if isinstance(m, nn.Linear)}
    luong_hoa_int8(nho2)
    for m in nho2.modules():
        if not isinstance(m, nn.Linear):
            continue
        w = m.weight.data
        w_goc = goc[id(m)]
        scale = float(w_goc.abs().max()) / 127.0
        lech = (w / scale
                - torch.round(w / scale)).abs().max()
        assert float(lech) <= 1e-4, \
            "quant phai lam tron dung muc"
        assert abs(float(w.abs().max())
                   - float(w_goc.abs().max())) <= 1e-6

    # Prune + quant: bias khong doi, so 0 giu
    # nguyen sau quant, baseline cach ly khi
    # deepcopy roi nen ban sao.
    torch.manual_seed(2)
    base = MLP(64, 32, 16, 2)
    bias_truoc = [m.bias.data.clone()
                  for m in base.modules()
                  if isinstance(m, nn.Linear)]
    w_truoc = [m.weight.data.clone()
               for m in base.modules()
               if isinstance(m, nn.Linear)]
    var = copy.deepcopy(base)
    cat_tia(var)
    n_0 = sum(int((m.weight.data == 0).sum())
              for m in var.modules()
              if isinstance(m, nn.Linear))
    luong_hoa_int8(var)
    n_0_sau = sum(int((m.weight.data == 0).sum())
                  for m in var.modules()
                  if isinstance(m, nn.Linear))
    assert n_0 == n_0_sau, "quant khong duoc doi 0"
    for truoc, m in zip(bias_truoc,
                        [x for x in var.modules()
                         if isinstance(x, nn.Linear)]):
        assert torch.equal(truoc, m.bias.data), \
            "bias phai giu nguyen khi nen"
    for truoc, m in zip(w_truoc,
                        [x for x in base.modules()
                         if isinstance(x, nn.Linear)]):
        assert torch.equal(truoc, m.weight.data), \
            "nen ban sao khong duoc doi baseline"
    print("Tu kiem tra: PASS (tat ca assert deu qua).")


def tra_loi_tu_kiem_tra():
    """Tra loi ngan 3 cau tu kiem tra trong de."""
    print("\nTRA LOI TU KIEM TRA (3 cau trong de):")
    print("  1. Accuracy giam 3,5 diem % (vuot "
          "nguong 2 diem %): ket luan gia thuyet "
          "BI BAC BO, ghi ket qua am vao "
          "research_plan.md. Tuyet doi KHONG sua "
          "nguong thanh 4% cho vua ket qua.")
    print("  2. Train lai baseline moi moi seed vi "
          "khoi tao ngau nhien khac nhau lam "
          "baseline luc manh luc yeu; dung chung "
          "1 baseline thi 3 lan do khong doc lap, "
          "do lech chuan tinh ra la gia.")
    print("  3. Khong dung duoc: 2 thu cung doi "
          "(cach nen + learning rate) nen khong "
          "biet chenh lech den tu dau. Chay lai "
          "voi cung lr=1e-3 cho ca hai ben.")


def main():
    """Chay ca lab: tu kiem tra, thi nghiem,
    ket luan theo nguong da dang ky."""
    print("Lab Day 29 - Gia thuyet bac bo duoc & "
          "thi nghiem kiem chung")
    print("Gia thuyet (da viet truoc trong "
          "research_plan.md): INT8 + pruning 30% "
          "-> nang luong/token giam >= 40%, "
          "accuracy giam <= 2 diem %.")
    print(f"torch {torch.__version__}, thiet bi: "
          f"{'cuda' if torch.cuda.is_available() else 'cpu'}")
    tu_kiem_tra()
    Xtr, Xte, ytr, yte = chuan_bi_du_lieu()
    print(f"\nDu lieu: {len(Xtr)} train / "
          f"{len(Xte)} test, {N_DAC_TRUNG} dac "
          f"trung; MLP {N_DAC_TRUNG}-"
          f"{KICH_THUOC_AN[0]}-{KICH_THUOC_AN[1]}-"
          f"{N_LOP}, {N_EPOCHS} epochs, lr={LR}.")
    ket_qua, _ = chay_thi_nghiem(Xtr, Xte, ytr, yte)
    ket_luan(ket_qua)
    tra_loi_tu_kiem_tra()


if __name__ == "__main__":
    main()
