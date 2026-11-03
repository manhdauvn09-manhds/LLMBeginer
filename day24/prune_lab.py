#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Lab Pruning Day 24 (tuan 4 - Green LLM): cat tia MLP.

Bai toan: phan loai 2 lop tren du lieu tong hop 20 chieu.
Mo hinh: MLP 20 -> 64 -> 64 -> 2, train tren CPU.

Bai 1: magnitude pruning unstructured o 10% / 30% / 50%,
do accuracy + latency (khong retrain).
Bai 2: cung toi 50% bang one-shot + retrain va bang
iterative + retrain, cung tong 15 epoch retrain.
Bai 3: structured pruning - xoa 25% neuron fc1, xay
lai mo hinh nho that (64 -> 48 neuron), do latency
o batch=1 va batch=64.

Quy tac thi nghiem (viet truoc khi chay): thu duy nhat
thay doi la ti le / kieu pruning; con lai giu nguyen:
cung seed, cung du lieu, cung kien truc, cung learning
rate va so epoch retrain.

Cach chay (venv Day 15, can scikit-learn - neu thieu
thi cai bang: pip install scikit-learn):
    ~/workspace/venvs/day15/bin/python prune_lab.py
"""

import copy
import math
import statistics
import time

import torch
import torch.nn as nn
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split


# Hang so cho ca lab: doi 1 hang so la doi luon thi
# nghiem, nen moi con so deu co ten ro rang o day.
DATA_SEED = 0
N_SAMPLES = 4000
N_FEATURES = 20
N_INFORMATIVE = 12
N_REDUNDANT = 4
TEST_SIZE = 0.25
HIDDEN = 64
BASE_EPOCHS = 15
BASE_LR = 0.01
RETRAIN_EPOCHS = 15
RETRAIN_LR = 0.001


def build_data():
    """Tao du lieu phan loai 2 lop va chia train/test.

    Dung dung tham so cua de bai: 4000 mau, 20 chieu,
    12 chieu mang tin, 4 chieu thua, test chiem 25%.
    Ket qua xac dinh (random_state=0) nen lan chay
    nao cung ra cung bo du lieu.

    Returns:
        (X_train, y_train, X_test, y_test): X la
        tensor (N, 20) float32, y la tensor (N,)
        kieu long, gia tri 0/1.

    Raises:
        ValueError: du lieu rong, X chua NaN/Inf,
            y khong phai long, hoac 1 lop vang mat
            o train/test (mo hinh se hoc lech).
    """
    x_all, y_all = make_classification(
        n_samples=N_SAMPLES, n_features=N_FEATURES,
        n_informative=N_INFORMATIVE,
        n_redundant=N_REDUNDANT, random_state=DATA_SEED)
    x_tr, x_te, y_tr, y_te = train_test_split(
        x_all, y_all, test_size=TEST_SIZE,
        random_state=DATA_SEED)
    x_train = torch.tensor(x_tr, dtype=torch.float32)
    y_train = torch.tensor(y_tr)
    x_test = torch.tensor(x_te, dtype=torch.float32)
    y_test = torch.tensor(y_te)
    # Guard ngay khi co du lieu: loi o day ma lot
    # xuong duoi thi train xong moi phat hien, ton
    # cong do lai tu dau.
    if x_train.shape[0] == 0 or x_test.shape[0] == 0:
        raise ValueError("tap train/test khong duoc rong.")
    if y_train.dtype != torch.long \
            or y_test.dtype != torch.long:
        raise ValueError("nhan y phai co dtype long.")
    if not bool(torch.isfinite(x_train).all()) \
            or not bool(torch.isfinite(x_test).all()):
        raise ValueError("du lieu X phai huu han.")
    for y_tap, ten in ((y_train, "train"),
                       (y_test, "test")):
        if len(torch.unique(y_tap)) < 2:
            raise ValueError(f"tap {ten} thieu 1 lop.")
    return x_train, y_train, x_test, y_test


class MLP(nn.Module):
    """MLP 2 lop an, dung chung cho ca lab.

    Kien truc dung de bai: 20 -> hidden1 -> hidden2
    -> 2, mac dinh 64/64. Hai lop an co so neuron
    rieng biet vi structured pruning chi xoa neuron
    lop 1 (fc1), lop 2 giu nguyen 64 - neu 2 lop
    dung chung 1 so hidden thi mo hinh nho xay
    lai se sai shape fc2 (64, 48) thanh (48, 48).
    Khong tao tensor moi trong forward, nen khong
    dinh bug device khi copy-paste (checklist 3).
    """

    def __init__(self, hidden1=HIDDEN, hidden2=HIDDEN):
        """Khoi tao 3 lop Linear: 20 -> h1 -> h2 -> 2.

        Raises:
            ValueError: hidden1/hidden2 khong phai
                so nguyen duong (vi du hidden=0 thi
                khong co neuron an nao de train).
        """
        super().__init__()
        for ten, gia_tri in (("hidden1", hidden1),
                             ("hidden2", hidden2)):
            if isinstance(gia_tri, bool) \
                    or not isinstance(gia_tri, int) \
                    or gia_tri < 1:
                raise ValueError(
                    f"{ten} phai la so nguyen >= 1.")
        self.fc1 = nn.Linear(N_FEATURES, hidden1)
        self.fc2 = nn.Linear(hidden1, hidden2)
        self.fc3 = nn.Linear(hidden2, 2)

    def forward(self, x):
        """Tra ve logits (chua qua softmax) cho moi mau."""
        return self.fc3(torch.relu(self.fc2(
            torch.relu(self.fc1(x)))))


def count_params(model):
    """Dem tong so tham so (ke ca bias) cua model.

    Dung chung 1 ham cho baseline va mo hinh nho de
    2 con so dem theo cung 1 cach, khong cong thieu
    hay cong thua khi so sanh (checklist 6).
    """
    return sum(p.numel() for p in model.parameters())


def count_weights(model):
    """Dem so trong so (chi ma tran weight, khong bias).

    Day la mau so cua sparsity(): de bai chi cat
    weight, khong cat bias, nen bias khong duoc
    tinh vao ti le so 0.
    """
    total = 0
    for p in model.parameters():
        if p.dim() > 1:
            total += p.numel()
    return total


def _kiem_tra_mask(model, mask):
    """Kiem tra mask hop le truoc khi train.

    Mask la dict {ten_param: tensor bool} tra ve
    boi prune_unstructured / prune_structured.
    Kiem tra som o day de train khong chay nua
    chung roi moi chet vi mask sai shape.
    """
    ten_param = {n for n, _ in model.named_parameters()}
    for ten, mat in mask.items():
        if ten not in ten_param:
            raise ValueError(f"mask co ten la: {ten}.")
        tham_so = dict(model.named_parameters())[ten]
        if mat.dtype != torch.bool:
            raise ValueError(f"mask {ten} phai la bool.")
        if tuple(mat.shape) != tuple(tham_so.shape):
            raise ValueError(f"mask {ten} sai shape.")
        if mat.device != tham_so.device:
            raise ValueError(f"mask {ten} sai device.")


def _apply_mask(model, mask):
    """Ap mask len model: trong so bi cat ve dung 0.

    Goi sau moi buoc optimizer trong train() khi
    co mask, de trong so bi cat khong moc lai
    trong luc retrain (xem ghi chu o train).
    """
    with torch.no_grad():
        tham_so = dict(model.named_parameters())
        for ten, mat in mask.items():
            tham_so[ten].mul_(mat)


def train(model, x_train, y_train, epochs=BASE_EPOCHS,
          lr=BASE_LR, mask=None):
    """Train model bang CE nhan cung, full-batch.

    Neu co mask: sau moi buoc optimizer, ap lai
    mask - BAT BUOC. Khong ap lai thi gradient
    chay vao ca trong so da cat, Adam day chung
    khoi 0, retrain xong sparsity gan nhu ve 0
    va thi nghiem "toi 50%" thanh vo nghia.
    Neu mask=None: train thuong, khong cat gi.

    Args:
        model: MLP can train.
        x_train: tensor (N, 20) float32, huu han.
        y_train: tensor (N,) long, gia tri 0/1.
        epochs: so epoch, so nguyen >= 1.
        lr: learning rate, so thuc > 0, huu han.
        mask: dict mask cua prune_* hoac None.

    Returns:
        Model da train xong (chua chuyen eval).

    Raises:
        ValueError: du lieu rong/sai dtype/chua
            NaN, epochs/lr sai, mask sai ten,
            shape, dtype bool hoac device.
    """
    # Guard ngay dau ham (checklist 1, 12): hoi
    # "rong? dtype? huu han? kieu so?" truoc
    # khi tinh, giu 1 loai ValueError.
    if x_train.dim() != 2 or x_train.shape[0] == 0:
        raise ValueError("x_train phai la tensor 2 chieu"
                         ", khong rong.")
    if y_train.shape != (x_train.shape[0],):
        raise ValueError("y_train phai co shape (N,).")
    if y_train.dtype != torch.long:
        raise ValueError("y_train phai co dtype long.")
    if not bool(torch.isfinite(x_train).all()):
        raise ValueError("x_train phai huu han.")
    # [TRAINER] x phai la float32: truoc day guard
    # chi chat o dtype cua y (bai hoc Day 23), con
    # x float64/long lot qua het roi chet
    # RuntimeError "mat1 and mat2 must have the
    # same dtype" o trong Linear - loi vao sai
    # thi phai bao o guard, khong de crash o
    # lop tinh toan.
    if x_train.dtype != torch.float32:
        raise ValueError("x_train phai co dtype "
                         "float32.")
    if isinstance(epochs, bool) \
            or not isinstance(epochs, int) or epochs < 1:
        raise ValueError("epochs phai la so nguyen >= 1.")
    if isinstance(lr, bool) \
            or not isinstance(lr, (int, float)) \
            or not math.isfinite(lr) or lr <= 0:
        raise ValueError("lr phai la so thuc > 0, huu han.")
    if mask is not None:
        _kiem_tra_mask(model, mask)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    tieu_chi = nn.CrossEntropyLoss()
    for _ in range(epochs):
        opt.zero_grad()
        loss = tieu_chi(model(x_train), y_train)
        loss.backward()
        opt.step()
        if mask is not None:
            _apply_mask(model, mask)
    return model

@torch.no_grad()
def accuracy(model, x_test, y_test):
    """Do chinh xac cua model tren tap test.

    Returns:
        Ti le du doan dung, float trong [0, 1].

    Raises:
        ValueError: tap test rong, y khong phai
            long, X sai dtype float32 hoac chua
            NaN/Inf.
    """
    if x_test.shape[0] == 0:
        raise ValueError("tap test khong duoc rong.")
    if y_test.shape != (x_test.shape[0],):
        raise ValueError("y_test phai co shape (N,).")
    if y_test.dtype != torch.long:
        raise ValueError("y_test phai co dtype long.")
    if not bool(torch.isfinite(x_test).all()):
        raise ValueError("x_test phai huu han.")
    # [TRAINER] Tuong tu train(): x_test sai dtype
    # (float64/long) truoc day lot qua guard roi
    # crash trong Linear khi do accuracy.
    if x_test.dtype != torch.float32:
        raise ValueError("x_test phai co dtype "
                         "float32.")
    model.eval()
    pred = model(x_test).argmax(1)
    return float((pred == y_test).float().mean())


def latency_ms(model, x, batch=1, reps=500, warmup=50):
    """Do toc do suy luan: TRUNG VI ms / 1 lan chay.

    Lay trung vi cua tung lan chay rieng le (sau
    warmup), khong lay trung binh cua ca khoi:
    trung binh bi nhieu keo lech khi model nho,
    1 lan chay cham bat thuong cung keo ca so
    len - bai hoc tu Day 22. Dung
    time.perf_counter() (checklist 5), model o
    che do eval, khong tinh gradient.

    Args:
        model: model can do, se chuyen ve eval.
        x: tensor dau vao, lay batch dau tien.
        batch: so mau / lan chay, >= 1 va <= len(x).
        reps: so lan chay lay trung vi.
        warmup: so lan chay khoi dong bo qua.

    Returns:
        Trung vi thoi gian, don vi mili-giay / lan.

    Raises:
        ValueError: batch/reps/warmup sai kieu
            hoac ngoai pham vi, x rong hoac sai
            dtype float32.
    """
    if x.shape[0] == 0:
        raise ValueError("x khong duoc rong.")
    # [TRAINER] Ca ham do toc do cung phai chan
    # dtype x: x long/float64 truoc day chi chet
    # o lan chay warmup dau tien, mat cong do.
    if x.dtype != torch.float32:
        raise ValueError("x phai co dtype float32.")
    for ten, gia_tri, toi_thieu in (
            ("batch", batch, 1), ("reps", reps, 1),
            ("warmup", warmup, 0)):
        if isinstance(gia_tri, bool) \
                or not isinstance(gia_tri, int) \
                or gia_tri < toi_thieu:
            raise ValueError(
                f"{ten} phai la so nguyen >= {toi_thieu}.")
    if batch > x.shape[0]:
        raise ValueError("batch khong vuot qua so mau cua x.")
    model.eval()
    with torch.no_grad():
        x_batch = x[:batch]
        for _ in range(warmup):
            model(x_batch)
        thoi_gian = []
        for _ in range(reps):
            t0 = time.perf_counter()
            model(x_batch)
            thoi_gian.append(time.perf_counter() - t0)
    return float(statistics.median(thoi_gian) * 1000.0)


def sparsity(model):
    """Ti le trong so bang 0 / tong so trong so.

    Chi tinh ma tran weight (dim > 1), khong tinh
    bias - giong khung code de bai. Vi du: 5504
    trong so, 2752 so 0 thi sparsity = 0.5.

    Raises:
        ValueError: model khong co ma tran weight.
    """
    tong = count_weights(model)
    if tong == 0:
        raise ValueError("model khong co ma tran weight.")
    so_0 = 0
    for p in model.parameters():
        if p.dim() > 1:
            so_0 += int((p == 0).sum())
    return so_0 / tong


def _kiem_tra_ratio(ratio):
    """Kiem tra ratio la so thuc huu han trong [0, 1].

    Dung chung cho ca 2 ham prune: loi vao sai
    (chuoi, bool, NaN, am, > 1) deu bao ro bang
    ValueError ngay tu dau, khong de crash kho
    hieu o torch.quantile (checklist 12).
    """
    if isinstance(ratio, bool) \
            or not isinstance(ratio, (int, float)):
        raise ValueError("ratio phai la so thuc.")
    if not math.isfinite(ratio) or ratio < 0 or ratio > 1:
        raise ValueError("ratio phai trong [0, 1].")


def prune_unstructured(model, ratio):
    """Cat unstructured bang magnitude, toan cuc.

    Gom moi trong so (chi weight, khong cat bias)
    vao 1 tensor, lay nguong la phan vi `ratio`
    cua |w|, dat trong so duoi nguong ve 0.
    Vi du: 5504 trong so, ratio=0.5 thi nguong
    la trung vi cua |w|, khoang 2752 trong so
    nho nhat ve 0. Mask giu lai trong so co
    |w| >= nguong (dau >= nen trong so dung
    bang nguong duoc giu, sparsity do duoc co
    the lech chut it so voi ratio khi co
    nhieu gia tri trung nhau - luon do lai
    bang sparsity(), dung tin ratio tren giay).

    Ham nay sua model TAI CHO va tra ve mask de
    retrain giu nguyen sparsity (xem train).

    Args:
        model: MLP can cat.
        ratio: ti le muon cat, so thuc trong [0, 1].

    Returns:
        Dict {ten_param: tensor bool} - True la
        giu, False la cat - cho moi ma tran weight.

    Raises:
        ValueError: ratio sai kieu/gia tri, trong
            so chua NaN/Inf, model khong co weight.
    """
    _kiem_tra_ratio(ratio)
    cac_weight = [p for p in model.parameters()
                  if p.dim() > 1]
    if not cac_weight:
        raise ValueError("model khong co ma tran weight.")
    tat_ca = torch.cat([p.detach().flatten()
                        for p in cac_weight])
    if not bool(torch.isfinite(tat_ca).all()):
        raise ValueError("trong so phai huu han.")
    nguong = torch.quantile(tat_ca.abs(), float(ratio))
    mask = {}
    with torch.no_grad():
        for ten, p in model.named_parameters():
            if p.dim() > 1:
                giu = p.abs() >= nguong
                mask[ten] = giu
                p.mul_(giu)
    return mask


def prune_structured(model, ratio):
    """Cat structured: xoa ca neuron an cua fc1.

    Diem quan trong cua moi neuron = chuan L1
    cua hang weight tuong ung o fc1 (tong |w|
    tren hang). Xoa `ratio` neuron thap diem
    nhat: dat hang do o fc1 ve 0, bias tuong
    ung ve 0, VA cat cot tuong ung o fc2 -
    neu chi cat 1 phia, mo hinh van giu kich
    thuoc cu hoac loi kich thuoc (bay so 1).
    Vi du: hidden=64, ratio=0.25 thi xoa 16
    neuron, giu 48 neuron diem cao nhat.

    Ham nay sua model TAI CHO va tra ve mask
    + chi so neuron giu lai de xay mo hinh nho.

    Args:
        model: MLP can cat.
        ratio: ti le neuron muon xoa, so thuc
            trong [0, 1), phai giu it nhat 1 neuron.

    Returns:
        (mask, keep_idx): mask nhu prune_unstructured
        (them ca mask bias fc1), keep_idx la tensor
        long 1 chieu, cac chi so neuron giu lai,
        xep tang dan.

    Raises:
        ValueError: model khong phai MLP, ratio
            sai kieu/gia tri, ratio xoa het neuron,
            trong so chua NaN/Inf.
    """
    if not isinstance(model, MLP):
        raise ValueError("model phai la MLP cua lab nay.")
    _kiem_tra_ratio(ratio)
    hidden = model.fc1.out_features
    so_xoa = int(round(float(ratio) * hidden))
    so_giu = hidden - so_xoa
    if so_giu < 1:
        raise ValueError("ratio phai giu it nhat 1 neuron.")
    w1 = model.fc1.weight.detach()
    if not bool(torch.isfinite(w1).all()) \
            or not bool(torch.isfinite(
                model.fc2.weight.detach()).all()):
        raise ValueError("trong so phai huu han.")
    # L1 tung hang: neuron nao tong |w| nho thi
    # coi la it quan trong nhat, xoa truoc.
    diem = w1.abs().sum(dim=1)
    thu_tu = torch.argsort(diem, descending=True)
    keep_idx = torch.sort(thu_tu[:so_giu]).values
    mask = {}
    with torch.no_grad():
        mat_fc1 = torch.zeros_like(
            model.fc1.weight, dtype=torch.bool)
        mat_fc1[keep_idx] = True
        mask["fc1.weight"] = mat_fc1
        model.fc1.weight.mul_(mat_fc1)
        mat_bias = torch.zeros_like(
            model.fc1.bias, dtype=torch.bool)
        mat_bias[keep_idx] = True
        mask["fc1.bias"] = mat_bias
        model.fc1.bias.mul_(mat_bias)
        mat_fc2 = torch.zeros_like(
            model.fc2.weight, dtype=torch.bool)
        mat_fc2[:, keep_idx] = True
        mask["fc2.weight"] = mat_fc2
        model.fc2.weight.mul_(mat_fc2)
        mask["fc3.weight"] = torch.ones_like(
            model.fc3.weight, dtype=torch.bool)
    return mask, keep_idx


def rebuild_small(model, keep_idx):
    """Xay MLP nho that tu neuron giu lai.

    Copy weight cua neuron giu lai sang 1 MLP
    moi co hidden = so neuron giu: fc1 lay hang
    giu + bias giu, fc2 lay cot giu (bias fc2
    giu nguyen vi so neuron ra khong doi),
    fc3 giu nguyen. Mo hinh moi khong chia se
    bo nho voi model goc (copy, khong phai
    view) nen train mo hinh nho khong lam
    hong model goc.

    Args:
        model: MLP goc (thuong la baseline).
        keep_idx: tensor long 1 chieu, chi so
            neuron giu lai, khong trung, trong
            [0, hidden), it nhat 1 phan tu.

    Returns:
        MLP moi: lop an 1 co hidden1 = len(keep_idx),
        lop an 2 giu nguyen nhu model goc.

    Raises:
        ValueError: model khong phai MLP, keep_idx
            sai kieu/shape/dtype, rong, trung lap,
            hoac vuot pham vi.
    """
    if not isinstance(model, MLP):
        raise ValueError("model phai la MLP cua lab nay.")
    if not isinstance(keep_idx, torch.Tensor) \
            or keep_idx.dim() != 1 \
            or keep_idx.dtype != torch.long:
        raise ValueError("keep_idx phai la tensor long 1D.")
    if keep_idx.numel() == 0:
        raise ValueError("keep_idx khong duoc rong.")
    hidden = model.fc1.out_features
    if int(keep_idx.min()) < 0 \
            or int(keep_idx.max()) >= hidden:
        raise ValueError("keep_idx vuot pham vi neuron.")
    if len(torch.unique(keep_idx)) != keep_idx.numel():
        raise ValueError("keep_idx khong duoc trung lap.")
    nho = MLP(hidden1=int(keep_idx.numel()),
              hidden2=model.fc2.out_features)
    with torch.no_grad():
        nho.fc1.weight.copy_(model.fc1.weight[keep_idx])
        nho.fc1.bias.copy_(model.fc1.bias[keep_idx])
        nho.fc2.weight.copy_(model.fc2.weight[:, keep_idx])
        nho.fc2.bias.copy_(model.fc2.bias)
        nho.fc3.weight.copy_(model.fc3.weight)
        nho.fc3.bias.copy_(model.fc3.bias)
    return nho


def _expect_value_error(fn, mo_ta):
    """Chay fn, doi no nem ValueError, in ket qua.

    Ham phu nay duoc dung that trong tu_kiem_tra():
    moi guard deu phai chan dung truong hop sai
    cua no, khong chi nam im trong code.
    """
    try:
        fn()
    except ValueError as exc:
        print(f"  guard {mo_ta}: chan dung ({exc})")
        return True
    print(f"  CANH BAO: guard {mo_ta} KHONG chan!")
    return False


def tu_kiem_tra():
    """Tu kiem tra bang assert, chay truoc lab chinh.

    Kiem tra guard (moi truong hop sai phai bi
    chan) va tinh dung cua prune/retrain/rebuild
    tren 1 MLP nho (hidden=8) de loi hien ngay
    truoc khi train baseline. Chi IN khi assert
    da qua - in truoc roi moi assert thi loi
    van troi qua (checklist 13).
    """
    torch.manual_seed(123)
    x_nho = torch.randn(50, N_FEATURES)
    y_nho = torch.randint(0, 2, (50,))

    def mo_hinh_moi():
        torch.manual_seed(123)
        return MLP(hidden1=8, hidden2=8)

    ket_qua = [
        _expect_value_error(
            lambda: prune_unstructured(
                mo_hinh_moi(), "0.3"),
            "ratio dang chuoi"),
        _expect_value_error(
            lambda: prune_unstructured(
                mo_hinh_moi(), True),
            "ratio dang bool"),
        _expect_value_error(
            lambda: prune_unstructured(
                mo_hinh_moi(), float("nan")),
            "ratio NaN"),
        _expect_value_error(
            lambda: prune_unstructured(
                mo_hinh_moi(), -0.1),
            "ratio am"),
        _expect_value_error(
            lambda: prune_unstructured(
                mo_hinh_moi(), 1.5),
            "ratio > 1"),
        _expect_value_error(
            lambda: prune_structured(
                mo_hinh_moi(), 1.0),
            "structured xoa het neuron"),
        _expect_value_error(
            lambda: train(mo_hinh_moi(), x_nho, y_nho,
                          epochs=0),
            "epochs=0"),
        _expect_value_error(
            lambda: train(mo_hinh_moi(), x_nho, y_nho,
                          lr="0.01"),
            "lr dang chuoi"),
        # [TRAINER] 2 test cho guard dtype x moi:
        # truoc ban review, ca 2 deu nem RuntimeError
        # thay vi ValueError.
        _expect_value_error(
            lambda: train(mo_hinh_moi(), x_nho.double(),
                          y_nho, epochs=1),
            "x_train float64"),
        _expect_value_error(
            lambda: accuracy(mo_hinh_moi(), x_nho.long(),
                             y_nho),
            "x_test dang long"),
        _expect_value_error(
            lambda: latency_ms(mo_hinh_moi(), x_nho,
                               batch=0),
            "batch=0"),
        _expect_value_error(
            lambda: latency_ms(mo_hinh_moi(), x_nho,
                               batch=9999),
            "batch vuot so mau"),
        _expect_value_error(
            lambda: rebuild_small(
                mo_hinh_moi(),
                torch.tensor([], dtype=torch.long)),
            "keep_idx rong"),
        _expect_value_error(
            lambda: rebuild_small(
                mo_hinh_moi(), torch.tensor([0, 0])),
            "keep_idx trung lap"),
        _expect_value_error(
            lambda: rebuild_small(
                mo_hinh_moi(), torch.tensor([0, 8])),
            "keep_idx vuot pham vi"),
    ]
    assert all(ket_qua), "Co guard khong chan dung!"

    # ratio=0: khong cat gi, trong so nguyen ven.
    mo_hinh = mo_hinh_moi()
    w_goc = [p.detach().clone()
             for p in mo_hinh.parameters()]
    prune_unstructured(mo_hinh, 0.0)
    assert sparsity(mo_hinh) == 0.0, "ratio=0 phai 0%"
    for p_cu, p_moi in zip(w_goc, mo_hinh.parameters()):
        assert torch.equal(p_cu, p_moi), \
            "ratio=0 khong duoc doi trong so."

    # ratio=0.5: sparsity dung, cho cat ve 0, cho
    # giu nguyen ven, bias khong bi cat.
    mo_hinh = mo_hinh_moi()
    w_goc = [p.detach().clone()
             for p in mo_hinh.parameters()]
    mask = prune_unstructured(mo_hinh, 0.5)
    assert abs(sparsity(mo_hinh) - 0.5) <= 0.02, \
        "sparsity do duoc phai gan 0.5."
    for (ten, p_cu), p_moi in zip(
            zip([n for n, _ in mo_hinh.named_parameters()],
                w_goc),
            mo_hinh.parameters()):
        if p_cu.dim() > 1:
            mat = mask[ten]
            assert torch.equal(p_moi[~mat],
                               torch.zeros_like(p_moi[~mat])), \
                "cho cat phai ve 0."
            assert torch.equal(p_moi[mat], p_cu[mat]), \
                "cho giu phai nguyen ven."
        else:
            assert torch.equal(p_cu, p_moi), \
                "unstructured khong duoc cat bias."

    # Round-trip mask (checklist 8): ap mask len
    # trong so goc phai tai tao dung trong so
    # sau khi cat, khong thua khong thieu.
    for (ten, _), p_cu, p_moi in zip(
            mo_hinh.named_parameters(), w_goc,
            mo_hinh.parameters()):
        if ten in mask:
            assert torch.equal(p_cu * mask[ten], p_moi), \
                "mask ap lai phai tai tao dung weight."

    # Retrain giu mask: sparsity khong doi sau
    # khi train, vi _apply_mask chay sau moi buoc.
    so_0_truoc = [(p.detach() == 0)
                  for p in mo_hinh.parameters()]
    train(mo_hinh, x_nho, y_nho, epochs=2,
          lr=RETRAIN_LR, mask=mask)
    assert abs(sparsity(mo_hinh) - 0.5) <= 0.02, \
        "retrain co mask phai giu sparsity."
    for mat_cu, p_moi in zip(so_0_truoc,
                             mo_hinh.parameters()):
        assert torch.equal(mat_cu, p_moi == 0), \
            "vi tri so 0 phai dung yen khi retrain."

    # Structured: giu dung 6/8 neuron, mask 2
    # phia khop nhau, mo hinh nho chay giong
    # het mo hinh da cat (truoc retrain).
    mo_hinh = mo_hinh_moi()
    w_goc = copy.deepcopy(mo_hinh)
    mask_s, giu_idx = prune_structured(mo_hinh, 0.25)
    assert giu_idx.numel() == 6, "giu 6/8 neuron."
    assert torch.equal(
        mask_s["fc1.weight"].any(dim=1),
        torch.isin(torch.arange(8), giu_idx)), \
        "hang fc1 giu phai dung keep_idx."
    assert torch.equal(
        mask_s["fc2.weight"].any(dim=0),
        torch.isin(torch.arange(8), giu_idx)), \
        "cot fc2 giu phai dung keep_idx."
    nho = rebuild_small(w_goc, giu_idx)
    assert nho.fc1.weight.shape == (6, N_FEATURES)
    assert nho.fc2.weight.shape == (8, 6)
    assert count_params(nho) < count_params(w_goc)
    with torch.no_grad():
        chenh = (mo_hinh(x_nho) - nho(x_nho)).abs().max()
    assert float(chenh) < 1e-6, \
        "mo hinh nho phai chay giong mo hinh da cat."

    # Latency tra ve so huu han, duong.
    lat = latency_ms(mo_hinh_moi(), x_nho, reps=20,
                     warmup=5)
    assert math.isfinite(lat) and lat > 0, \
        "latency phai huu han va duong."
    print("Tu kiem tra: PASS (tat ca assert deu qua).")


def bai1(baseline, x_test, y_test):
    """Bai 1: duong cong quality-sparsity, 10/30/50%.

    Moi muc cat xuat phat tu CUNG baseline (copy),
    khong retrain - chi do tac dong cua viec cat.
    """
    print("\n" + "=" * 72)
    print("BAI 1 - Unstructured 10% / 30% / 50%")
    print("=" * 72)
    print("Thu duy nhat thay doi la ti le pruning; "
          "seed, du lieu, kien truc giu nguyen.")
    acc_goc = accuracy(baseline, x_test, y_test)
    lat_goc = latency_ms(baseline, x_test, batch=1)
    print(f"Baseline: acc={acc_goc:.3f} "
          f"latency(batch=1)={lat_goc:.3f} ms "
          f"sparsity={sparsity(baseline):.3f}")
    ket_qua = {}
    for ratio in (0.1, 0.3, 0.5):
        mo_hinh = copy.deepcopy(baseline)
        prune_unstructured(mo_hinh, ratio)
        sp = sparsity(mo_hinh)
        assert abs(sp - ratio) <= 0.02, \
            "sparsity do duoc phai gan ratio."
        acc = accuracy(mo_hinh, x_test, y_test)
        lat = latency_ms(mo_hinh, x_test, batch=1)
        ket_qua[ratio] = (acc, lat, sp)
        print(f"  ratio={ratio:.1f}: acc={acc:.3f} "
              f"latency(batch=1)={lat:.3f} ms "
              f"sparsity do duoc={sp:.3f}")
    acc_50, lat_50, _ = ket_qua[0.5]
    roi = (acc_goc - acc_50) * 100
    doi_lat = (lat_50 - lat_goc) / lat_goc * 100
    # [TRAINER] Truoc day nhanh <= 1.1x gom ca
    # truong hop latency GIAM MANH vao nhan
    # "gan nhu khong doi" - nhan xet khong phan
    # biet giam / hoa / tang. Nay tach 3 nhanh
    # theo dung so do (hom nay lat bang nhau
    # nen van roi vao nhanh giua nhu cu).
    if lat_50 < lat_goc * 0.9:
        nhan_xet = f"giam {-doi_lat:.1f}%"
    elif lat_50 <= lat_goc * 1.1:
        nhan_xet = "gan nhu khong doi"
    else:
        nhan_xet = (f"khong giam (do duoc cao hon "
                    f"{doi_lat:+.1f}%, nam trong bien "
                    f"dong tai may o muc 0.0x ms, "
                    f"khong ket luan pruning lam cham)")
    # Giai thich cung phai re nhanh theo so do:
    # cau "Latency khong giam" ghi cung se mau
    # thuan voi nhanh "giam 31.7%" o lan chay
    # ma tai may ha xuong giua 2 lan do. Muc
    # giam o quy mo 0.0x ms la bien dong tai
    # may, khong phai tac dung he thong.
    if lat_50 < lat_goc * 0.9:
        giai_thich = ("Muc giam nay nam trong bien "
                      "dong tai may o muc 0.0x ms, "
                      "khong phai tac dung he thong "
                      "cua pruning: kernel dense van "
                      "duyet toan bo ma tran, ke ca "
                      "so 0.")
    else:
        giai_thich = ("Latency khong giam vi kernel "
                      "dense van duyet toan bo ma "
                      "tran, ke ca so 0 - sparsity "
                      "chi la con so tren giay.")
    print(f"Ket luan: cat 50% khong retrain lam acc "
          f"roi {roi:.1f} diem, latency {nhan_xet} "
          f"({lat_goc:.3f} -> {lat_50:.3f} ms). "
          f"{giai_thich}")
    return {"acc_goc": acc_goc, "lat_goc": lat_goc,
            "acc_50": acc_50, "lat_50": lat_50}


def bai2(baseline, x_train, y_train, x_test, y_test):
    """Bai 2: one-shot vs iterative, cung toi 50%.

    Cung tong 15 epoch retrain, cung LR nho
    0.001: one-shot cat 1 phat 50% roi retrain
    15 epoch; iterative cat 10% -> 30% -> 50%
    (ti le tich luy, tinh tren toan bo weight
    ke ca so 0 da cat), moi vong retrain 5 epoch.
    """
    print("\n" + "=" * 72)
    print("BAI 2 - One-shot vs iterative (toi 50%)")
    print("=" * 72)
    print("Thu duy nhat thay doi la lich cat; tong "
          "epoch retrain (15) va LR (0.001) giong nhau.")
    acc_goc = accuracy(baseline, x_test, y_test)
    # One-shot: cat 1 phat, retrain 1 lan.
    mot_phat = copy.deepcopy(baseline)
    mask_1 = prune_unstructured(mot_phat, 0.5)
    train(mot_phat, x_train, y_train,
          epochs=RETRAIN_EPOCHS, lr=RETRAIN_LR,
          mask=mask_1)
    acc_1 = accuracy(mot_phat, x_test, y_test)
    sp_1 = sparsity(mot_phat)
    assert abs(sp_1 - 0.5) <= 0.01, \
        "one-shot sau retrain phai giu 50%."
    print(f"  One-shot + retrain: acc={acc_1:.3f} "
          f"sparsity={sp_1:.3f}")
    # Iterative: cat dan, retrain sau moi vong.
    dan = copy.deepcopy(baseline)
    for ratio, epoch_vong in ((0.1, 5), (0.3, 5),
                              (0.5, 5)):
        mask_v = prune_unstructured(dan, ratio)
        train(dan, x_train, y_train, epochs=epoch_vong,
              lr=RETRAIN_LR, mask=mask_v)
        print(f"    vong toi {ratio:.0%}: "
              f"sparsity={sparsity(dan):.3f} "
              f"acc={accuracy(dan, x_test, y_test):.3f}")
    acc_2 = accuracy(dan, x_test, y_test)
    sp_2 = sparsity(dan)
    assert abs(sp_2 - 0.5) <= 0.01, \
        "iterative sau retrain phai giu 50%."
    print(f"  Iterative + retrain: acc={acc_2:.3f} "
          f"sparsity={sp_2:.3f}")
    chenh = (acc_2 - acc_1) * 100
    # [TRAINER] Truoc day chuoi ket luan ghi cung
    # "= 1 mau test": hom nay dung vi chenh dung
    # 0.1 diem, nhung neu chenh = 0.0 thi cau chu
    # thanh "0.0 diem = 1 mau" - sai ngay tren so
    # cua minh. Nay tinh so mau that tu chenh va
    # kich thuoc test.
    so_mau_chenh = int(round(abs(acc_2 - acc_1)
                             * x_test.shape[0]))
    if abs(chenh) < 0.2:
        hon = (f"gan nhu hoa nhau, chenh {chenh:.1f} "
               f"diem = {so_mau_chenh} mau test tren "
               f"{x_test.shape[0]} mau")
    elif chenh > 0:
        hon = f"iterative cao hon {chenh:.1f} diem"
    else:
        hon = f"one-shot cao hon {-chenh:.1f} diem"
    print(f"Ket luan: {hon} ({acc_1:.3f} vs "
          f"{acc_2:.3f}). Ca 2 ban retrain deu vuot "
          f"baseline ({acc_goc:.3f}) vi duoc them 15 "
          f"epoch - so sanh chi co nghia giua 2 cach "
          f"cat voi nhau, khong phai voi baseline.")
    return {"acc_1": acc_1, "acc_2": acc_2,
            "sp_1": sp_1, "sp_2": sp_2}


def bai3(baseline, x_train, y_train, x_test, y_test):
    """Bai 3: structured 25% + mo hinh nho that.

    Xoa 25% neuron fc1 (64 -> 48), xay lai MLP
    nho that, do latency o batch=1 va batch=64,
    them batch=1000 de kiem tra che do tinh toan.
    """
    print("\n" + "=" * 72)
    print("BAI 3 - Structured: xoa neuron, mo hinh nho")
    print("=" * 72)
    print("Thu duy nhat thay doi la kieu pruning "
          "(structured thay cho unstructured); du lieu, "
          "seed, LR retrain giu nguyen.")
    acc_goc = accuracy(baseline, x_test, y_test)
    lat1_goc = latency_ms(baseline, x_test, batch=1)
    lat64_goc = latency_ms(baseline, x_test, batch=64)
    print(f"Baseline (64 neuron): acc={acc_goc:.3f} "
          f"lat1={lat1_goc:.3f} ms "
          f"lat64={lat64_goc:.3f} ms "
          f"params={count_params(baseline)}")
    # Cat tren ban copy, giu baseline nguyen ven.
    da_cat = copy.deepcopy(baseline)
    _, giu_idx = prune_structured(da_cat, 0.25)
    diem_l1 = baseline.fc1.weight.detach().abs().sum(1)
    giu_thap = float(diem_l1[giu_idx].min())
    xoa_cao = float(diem_l1[
        torch.isin(torch.arange(HIDDEN), giu_idx,
                   invert=True)].max())
    print(f"  Giu {giu_idx.numel()} neuron: L1 thap "
          f"nhat giu lai={giu_thap:.2f}, L1 cao nhat "
          f"bi xoa={xoa_cao:.2f}.")
    print(f"  Masked (van 64 neuron): "
          f"acc={accuracy(da_cat, x_test, y_test):.3f} "
          f"lat1={latency_ms(da_cat, x_test, batch=1):.3f} "
          f"ms sparsity={sparsity(da_cat):.3f}")
    # Xay mo hinh nho tu baseline + keep_idx, roi
    # kiem chung: no phai chay giong het ban masked.
    nho = rebuild_small(baseline, giu_idx)
    with torch.no_grad():
        chenh = (da_cat(x_test) - nho(x_test)).abs().max()
    assert float(chenh) < 1e-6, \
        "mo hinh nho phai giong ban masked."
    acc_nho = accuracy(nho, x_test, y_test)
    lat1_nho = latency_ms(nho, x_test, batch=1)
    lat64_nho = latency_ms(nho, x_test, batch=64)
    print(f"  Nho that (48 neuron, chua retrain): "
          f"acc={acc_nho:.3f} lat1={lat1_nho:.3f} ms "
          f"lat64={lat64_nho:.3f} ms "
          f"params={count_params(nho)}")
    # Retrain mo hinh nho 15 epoch, cung LR nhu Bai 2.
    nho_hoc = copy.deepcopy(nho)
    train(nho_hoc, x_train, y_train,
          epochs=RETRAIN_EPOCHS, lr=RETRAIN_LR)
    acc_hoc = accuracy(nho_hoc, x_test, y_test)
    lat1_hoc = latency_ms(nho_hoc, x_test, batch=1)
    lat64_hoc = latency_ms(nho_hoc, x_test, batch=64)
    print(f"  Nho that + retrain 15 epoch: "
          f"acc={acc_hoc:.3f} lat1={lat1_hoc:.3f} ms "
          f"lat64={lat64_hoc:.3f} ms")
    giam_param = (count_params(baseline)
                  - count_params(nho)) \
        / count_params(baseline) * 100
    nhanh_1 = lat1_goc / lat1_nho
    nhanh_64 = lat64_goc / lat64_nho
    # Kiem tra them ngoai 2 batch de bai yeu cau:
    # batch=1000 (toan bo test 1 lan) de xem toc
    # do co hien ra khi tinh toan at chi phi
    # khoi dong kernel hay khong.
    lat1000_goc = latency_ms(baseline, x_test,
                             batch=x_test.shape[0])
    lat1000_nho = latency_ms(nho, x_test,
                             batch=x_test.shape[0])
    nhanh_1000 = lat1000_goc / lat1000_nho
    print(f"  Kiem tra them batch=1000: baseline "
          f"{lat1000_goc:.3f} ms, nho "
          f"{lat1000_nho:.3f} ms, "
          f"nhanh {nhanh_1000:.2f}x.")
    def mo_ta_toc_do(nhanh, batch):
        if nhanh <= 1.05:
            return (f"batch={batch}: khong do duoc "
                    f"nhanh hon ({nhanh:.2f}x)")
        return f"batch={batch}: nhanh hon {nhanh:.2f}x"
    # [TRAINER] LOI KET LUAN (muc chinh cua ban
    # review nay): cau batch=1000 truoc day GHI
    # CUNG "toc do moi hien ra ... gan ti le giam
    # phep tinh" bat ke so do the nao. Lan chay
    # 03/11 cho nhanh_1000 = 1.00x (0.297 ms ca 2
    # ben, khong nhanh hon chut nao) trong khi ti
    # le giam phep tinh la 5504/4160 = 1.32x - ket
    # luan cu sai ngay tren so cua chinh no, dung
    # benh ghi cung da bi neu o Day 23. Nay dung
    # mo_ta_toc_do cho ca batch=1000 va chi noi
    # toc do "hien ra" khi do duoc > 1.05x.
    if nhanh_1000 <= 1.05:
        cau_1000 = (
            f"{mo_ta_toc_do(nhanh_1000, 1000)} - ke "
            f"ca o batch lon nhat da do, toc do van "
            f"khong hien ra, du phep tinh da giam "
            f"con 4160/5504 (ti le 1.32x): overhead "
            f"cua moi truong do nay con lon hon ca "
            f"phan tinh toan o quy mo nay")
    else:
        cau_1000 = (
            f"{mo_ta_toc_do(nhanh_1000, 1000)} - "
            f"toc do chi hien ra khi batch du lon "
            f"de tinh toan at duoc overhead")
    print(f"Ket luan: mo hinh nho giam {giam_param:.1f}% "
          f"tham so; {mo_ta_toc_do(nhanh_1, 1)}, "
          f"{mo_ta_toc_do(nhanh_64, 64)} - mo hinh "
          f"qua nho nen chi phi khoi dong kernel at "
          f"het phan tinh toan tiet kiem duoc. "
          f"{cau_1000}. Retrain giup acc hoi "
          f"{(acc_hoc - acc_nho) * 100:.1f} diem.")
    return {"lat1_goc": lat1_goc, "lat1_nho": lat1_nho,
            "lat64_goc": lat64_goc,
            "lat64_nho": lat64_nho,
            "nhanh_1": nhanh_1, "nhanh_64": nhanh_64,
            "nhanh_1000": nhanh_1000,
            "acc_nho": acc_nho, "acc_hoc": acc_hoc,
            "giam_param": giam_param}


def main():
    """Chay ca lab: tu kiem tra, baseline, 3 bai."""
    print("Lab Pruning Day 24 - CPU, "
          f"torch {torch.__version__}")
    x_train, y_train, x_test, y_test = build_data()
    print(f"train: {tuple(x_train.shape)} | "
          f"test: {tuple(x_test.shape)}")
    tu_kiem_tra()
    # Dat lai seed SAU tu kiem tra (no dung seed
    # rieng), de khoi tao baseline xac dinh.
    torch.manual_seed(DATA_SEED)
    baseline = train(MLP(), x_train, y_train)
    print(f"Baseline: acc="
          f"{accuracy(baseline, x_test, y_test):.3f} "
          f"latency(batch=1)="
          f"{latency_ms(baseline, x_test, batch=1):.3f} "
          f"ms params={count_params(baseline)} "
          f"weights={count_weights(baseline)}")
    kq1 = bai1(baseline, x_test, y_test)
    kq2 = bai2(baseline, x_train, y_train, x_test, y_test)
    kq3 = bai3(baseline, x_train, y_train, x_test, y_test)
    print("\nBANG KET QUA: baseline "
          f"acc={kq1['acc_goc']:.3f} | unstructured 50% "
          f"(khong retrain) acc={kq1['acc_50']:.3f} | "
          f"one-shot acc={kq2['acc_1']:.3f} vs iterative "
          f"acc={kq2['acc_2']:.3f} | mo hinh nho 48 neuron "
          f"acc={kq3['acc_nho']:.3f} (retrain "
          f"{kq3['acc_hoc']:.3f}), toc do batch=1 "
          f"{kq3['nhanh_1']:.2f}x / batch=64 "
          f"{kq3['nhanh_64']:.2f}x / batch=1000 "
          f"{kq3['nhanh_1000']:.2f}x.")
    # [TRAINER] Ket luan chung truoc day cung ghi
    # cung rang mo hinh nho "chi nhanh that khi
    # tinh toan at chi phi khoi dong" - nghe nhu
    # o batch=1000 no DA nhanh that, trong khi so
    # do hom nay la 1.00x. Nay re nhanh theo dung
    # nhanh_1000 vua do.
    if kq3["nhanh_1000"] > 1.05:
        print("Ket luan chung: unstructured chi nhe "
              "tren giay - ngay ca mo hinh nho that "
              "cung chi nhanh that khi tinh toan at "
              "chi phi khoi dong, nen phai do dung "
              "batch size trien khai. Nang luong "
              "(kWh) khong do o lab nay, nen khong "
              "tuyen bo tiet kiem dien.")
    else:
        print("Ket luan chung: unstructured chi nhe "
              "tren giay; mo hinh nho that 48 neuron "
              "o MOI batch da do (1, 64, 1000) deu "
              "khong nhanh hon do duoc - quy mo nay "
              "qua nho so voi overhead cua moi "
              "truong do, nen muon thay toc do phai "
              "do o mo hinh hoac batch lon hon nhieu, "
              "dung suy ra tu ti le tham so. Nang "
              "luong (kWh) khong do o lab nay, nen "
              "khong tuyen bo tiet kiem dien.")


if __name__ == "__main__":
    main()
