#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Lab Distillation Day 23 (tuan 4 - Green LLM): student nho hoc
tu teacher lon qua soft target.

Chay tren CPU, PyTorch, khong can GPU, khong can tai model
lon. Du lieu la bai toan phan loai 3 lop tong hop, co tinh
thiet ke de co dark knowledge: lop 0 gan lop 1, xa lop 2.

Bai 1: soft target o T = 1, 2, 5, 10 cho 1 mau lop 0, xem
    temperature nao lo ro quan he "0 gan 1, xa 2".
Bai 2: tu viet distill_loss, kiem tra huu han, detach
    teacher, phan tach CE / KL goc / T^2 * KL.
Bai 3: thi nghiem co kiem soat - cung 300 mau, cung student
    MLP(32), cung seed, cung 100 epoch; chi doi cach tinh
    loss: A hard-only, B distill T=4 alpha=0.3, C distill
    T=2 alpha=0.3 (doi 1 bien T so voi B). Do them params
    va toc do (trung vi, perf_counter) theo tinh than
    Green LLM.

Quy tac thi nghiem (viet truoc khi chay Bai 3):
    Thu duy nhat thay doi la cach tinh loss (hard-label
    hay distillation); du lieu, kien truc student, seed
    va so epoch giu nguyen.

Cach chay (venv Day 15):
    ~/workspace/venvs/day15/bin/python distill_lab.py
"""

import math
import statistics
import time

import torch
import torch.nn as nn
import torch.nn.functional as F

# Seed duy nhat cho ca du lieu lan khoi tao teacher. Moi lan
# train_student() dat lai seed nay truoc khi tao student, nen
# 3 cau hinh A/B/C khoi tao giong het nhau.
DATA_SEED = 0
N_PER_CLASS = 400
N_TRAIN = 900
N_STUDENT = 300
TEACHER_EPOCHS = 150
STUDENT_EPOCHS = 100


def build_data(seed=DATA_SEED):
    """Tao du lieu 3 cum va chia train/test theo de bai.

    Thu tu goi ham sinh so ngau nhien phai giu nguyen nhu
    de bai: tao cum theo thu tu lop 0, 1, 2 roi moi xao
    tron. Doi thu tu la doi luon bo du lieu, ket qua
    khong con so sanh duoc voi lan chay truoc.

    Args:
        seed: seed cho torch.manual_seed truoc khi sinh.

    Returns:
        (X_train, y_train, X_test, y_test): X la tensor
        (N, 2), y la tensor (N,) kieu long, gia tri 0..2.
    """
    torch.manual_seed(seed)
    # 3 cum: lop 0 tai (0, 0), lop 1 tai (1.2, 0.3) - gan
    # lop 0, lop 2 tai (0, 4.0) - xa ca hai lop kia. Nhieu
    # do lech chuan 0.6 nen lop 0 va 1 chong lap nhau
    # dang ke, lop 2 gan nhu tach biet.
    centers = torch.tensor([[0.0, 0.0], [1.2, 0.3],
                            [0.0, 4.0]])
    xs, ys = [], []
    for c in range(3):
        xs.append(centers[c] + 0.6 * torch.randn(N_PER_CLASS, 2))
        ys.append(torch.full((N_PER_CLASS,), c))
    x_all = torch.cat(xs)
    y_all = torch.cat(ys)
    perm = torch.randperm(len(x_all))
    x_all, y_all = x_all[perm], y_all[perm]
    return (x_all[:N_TRAIN], y_all[:N_TRAIN],
            x_all[N_TRAIN:], y_all[N_TRAIN:])


class MLP(nn.Module):
    """MLP 2 lop an, dung chung cho teacher va student.

    Kien truc giong het de bai, chi khac hidden: teacher
    hidden=128, student hidden=32. Khong tao tensor moi
    trong forward, nen khong dinh bug device.
    """

    def __init__(self, hidden):
        """Khoi tao 3 lop Linear: 2 -> hidden -> hidden -> 3."""
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(2, hidden), nn.ReLU(),
            nn.Linear(hidden, hidden), nn.ReLU(),
            nn.Linear(hidden, 3))

    def forward(self, x):
        """Tra ve logits (chua qua softmax) cho moi mau."""
        return self.net(x)


def count_params(model):
    """Dem tong so tham so (ke ca bias) cua model.

    Dung chung 1 ham cho teacher va student de 2 con so
    dem theo cung 1 cach, khong cong thieu/cong thua.
    """
    return sum(p.numel() for p in model.parameters())


def train_teacher(model, x_train, y_train, epochs=TEACHER_EPOCHS):
    """Train teacher tren du 900 mau bang CE nhan cung.

    Args:
        model: MLP(128) vua khoi tao.
        x_train: tensor (900, 2).
        y_train: tensor (900,) nhan 0..2.
        epochs: so epoch, mac dinh 150 theo de bai.

    Returns:
        Model teacher da train xong (chua chuyen eval).
    """
    opt = torch.optim.Adam(model.parameters(), lr=0.01)
    for _ in range(epochs):
        opt.zero_grad()
        loss = F.cross_entropy(model(x_train), y_train)
        loss.backward()
        opt.step()
    return model


def accuracy(model, x_test, y_test):
    """Do chinh xac cua model tren tap test.

    Returns:
        Ti le du doan dung, float trong [0, 1].
    """
    model.eval()
    with torch.no_grad():
        pred = model(x_test).argmax(1)
    return float((pred == y_test).float().mean())


def distill_loss(student_logits, teacher_logits, y_true,
                 T=4.0, alpha=0.3):
    """Loss distillation tong hop cua Hinton.

    Cong thuc:
        L = alpha * CE + (1 - alpha) * T^2 * KL(
            teacher_T || student_T)
    Trong do CE la cross-entropy voi nhan that, KL tinh
    tren phan phoi mem o temperature T. He so T^2 bu lai
    gradient cua phan KL bi thu nho khoang T^2 lan khi
    chia logits cho T; quen T^2 thi phan hoc tu teacher
    bi phan CE lan at.

    Vi du so: cung 1 batch gia, T=1 cho KL goc 0.3762,
    T=4 cho KL goc 0.0306 (nho hon 12x) nhung T^2*KL
    lai la 0.4904 - loss tong co the TANG theo T, do la
    chu dich cua he so T^2, khong phai loi.

    Args:
        student_logits: tensor (N, 3), logits student.
        teacher_logits: tensor (N, 3), logits teacher.
        y_true: tensor (N,) dtype long, nhan that 0..2.
        T: temperature, so thuc, phai > 0 va huu han.
        alpha: so thuc, trong so cho CE, trong [0, 1].

    Returns:
        Tensor vo huong: loss tong hop.

    Raises:
        ValueError: batch rong, shape 2 tensor khong
            khop nhau, y_true khong phai dtype long,
            logits chua NaN/Inf, nhan ngoai [0, 2],
            T/alpha khong phai so thuc, T <= 0 hoac
            khong huu han, alpha ngoai [0, 1], 2 logits
            khac device nhau.
    """
    # Guard ngay dau ham, truoc moi phep tinh (checklist
    # 1): loi vao sai phai bao ro, khong de crash kho
    # hieu o sau trong F.cross_entropy / F.kl_div.
    if student_logits.dim() != 2 or teacher_logits.dim() != 2:
        raise ValueError("logits phai la tensor 2 chieu "
                         "(N, so_lop).")
    if student_logits.shape != teacher_logits.shape:
        raise ValueError("student va teacher logits phai "
                         "cung shape, nhan duoc "
                         f"{tuple(student_logits.shape)} va "
                         f"{tuple(teacher_logits.shape)}.")
    if student_logits.numel() == 0:
        raise ValueError("batch khong duoc rong.")
    if student_logits.device != teacher_logits.device:
        raise ValueError("student va teacher logits phai "
                         "cung device.")
    # [TRAINER] y_true phai la Long: truoc day nhan float
    # (0.0/1.0/2.0) lot qua ca guard shape lan range, roi
    # chet RuntimeError kho hieu trong F.cross_entropy -
    # trai dung triet ly guard cua ham nay (loi vao sai
    # phai bao ro ngay tu dau).
    if y_true.dtype != torch.long:
        raise ValueError("y_true phai co dtype long "
                         "(nhan nguyen 0..n_class-1).")
    # [TRAINER] Logits phai huu han: truoc day logits
    # NaN/Inf khong bi chan, loss tra ra NaN mot cach
    # lang le - dung loai loi "code chay binh thuong,
    # ket qua lang le te" ma Bai 2 tu canh bao.
    if not bool(torch.isfinite(student_logits).all()) \
            or not bool(torch.isfinite(teacher_logits).all()):
        raise ValueError("logits phai huu han "
                         "(khong chua NaN/Inf).")
    # [TRAINER] T/alpha phai la so thuc: truoc day T="4"
    # nem TypeError tu math.isfinite, khong nhat quan
    # voi cac guard con lai (deu la ValueError).
    if not isinstance(T, (int, float)) \
            or not isinstance(alpha, (int, float)):
        raise ValueError("T va alpha phai la so thuc.")
    if y_true.shape != (student_logits.shape[0],):
        raise ValueError("y_true phai co shape "
                         f"({student_logits.shape[0]},).")
    n_class = student_logits.shape[1]
    if int(y_true.min()) < 0 or int(y_true.max()) >= n_class:
        raise ValueError(f"nhan phai trong [0, {n_class - 1}].")
    if not math.isfinite(T) or T <= 0:
        raise ValueError("T phai > 0 va huu han.")
    if not 0.0 <= alpha <= 1.0:
        raise ValueError("alpha phai trong [0, 1].")
    # 1. CE voi nhan that.
    ce_loss = F.cross_entropy(student_logits, y_true)
    # 2. Soft target teacher: detach BAT BUOC. Khong detach
    # thi gradient chay nguoc vao teacher, teacher cung bi
    # cap nhat trong luc train student - thi nghiem mat
    # kiem soat va ton bo nho vo ich (bay so 1 cua de bai).
    soft_teacher = F.softmax(teacher_logits.detach() / T, dim=1)
    # 3. Soft student dang log-prob: F.kl_div doi dau vao
    # thu nhat la log-prob, thu hai la prob. Dao 2 dau
    # vao hoac dua ca 2 deu la prob deu cho loss sai ma
    # code van chay "binh thuong" (bay so 2 cua de bai).
    log_student = F.log_softmax(student_logits / T, dim=1)
    # 4. KL(teacher || student): phat nang khi student gan
    # xac suat thap cho dieu teacher tin la dung, ep
    # student bao phu het nhung gi teacher biet. Chieu
    # nguoc lai cho phep student bo qua mode cua teacher.
    kl_loss = F.kl_div(log_student, soft_teacher,
                       reduction="batchmean")
    # 5. Tong hop, nhan T^2 cho phan KL.
    return alpha * ce_loss + (1.0 - alpha) * (T ** 2) * kl_loss


def _expect_value_error(fn, mo_ta):
    """Chay fn, doi no nem ValueError, in ket qua kiem tra.

    Ham phu nay duoc dung that trong bai2(): moi guard
    cua distill_loss deu phai chan dung truong hop sai
    cua no, khong chi nam im trong code.
    """
    try:
        fn()
    except ValueError as exc:
        print(f"  guard {mo_ta}: chan dung ({exc})")
        return True
    print(f"  CANH BAO: guard {mo_ta} KHONG chan!")
    return False


def bai1(teacher, x_test, y_test):
    """Bai 1: nhin dark knowledge qua temperature.

    Lay mau dau tien cua test (thuoc lop 0, logits
    [6.83, 4.28, -12.65]) va in softmax o T = 1, 2, 5,
    10. Them bang trung binh tren toan bo 114 mau lop 0
    cua test de ket luan khong dua vao 1 mau duy nhat.
    """
    print("=" * 72)
    print("BAI 1 - Soft target va temperature")
    print("=" * 72)
    teacher.eval()
    with torch.no_grad():
        logits_all = teacher(x_test)
    # Mau dau tien: de bai dung X_test[:5] va in p[0];
    # kiem tra nhan that cua no la 0 roi moi phan tich.
    first_label = int(y_test[0])
    print(f"Mau dau tien cua test: nhan that = {first_label}, "
          f"logits = {[round(v, 2) for v in logits_all[0].tolist()]}")
    print("Soft target cua mau nay:")
    for temp in [1, 2, 5, 10]:
        probs = F.softmax(logits_all[0] / temp, dim=0)
        p0, p1, p2 = probs.tolist()
        # Ti le p1/p2 noi len quan he "1 gan, 2 xa": o T=1
        # ti le nay ~2.2e7 vi p2 ~3e-09 gan nhu bang 0.
        ratio = p1 / p2 if p2 > 0 else float("inf")
        print(f"  T={temp:>2}: [{p0:.4f}, {p1:.4f}, {p2:.4f}] "
              f"| p1/p2 = {ratio:.1f}")
    # Trung binh tren tat ca mau lop 0 cua test.
    mask0 = y_test == 0
    print(f"\nTrung binh tren {int(mask0.sum())} mau lop 0 "
          "cua test (p0, p1, p2):")
    for temp in [1, 2, 5, 10]:
        mean_p = F.softmax(logits_all[mask0] / temp,
                           dim=1).mean(0)
        print(f"  T={temp:>2}: "
              f"[{mean_p[0]:.4f}, {mean_p[1]:.4f}, "
              f"{mean_p[2]:.4f}]")
    print("Nhan xet: o mau nay, T=1 lop thang chiem gan het, "
          "lop 2 gan nhu 0 - dark knowledge bi chon vui. O "
          "T=2-5 xac suat lop 1 noi len ro, quan he 0 gan 1, "
          "xa 2 lo ro nhat o T=5. O T=10 phan phoi tien ve "
          "deu (0.52/0.40/0.07 o mau nay), tuong phan mo dan.")


def bai2():
    """Bai 2: kiem tra distill_loss tren batch gia.

    Kiem tra 3 thu: loss huu han, teacher detach that
    (gradient khong chay vao teacher), va cong thuc
    tong hop dung (loss = alpha*CE + (1-alpha)*T^2*KL).
    """
    print("\n" + "=" * 72)
    print("BAI 2 - Kiem tra distill_loss")
    print("=" * 72)
    torch.manual_seed(123)
    student_logits = torch.randn(8, 3, requires_grad=True)
    teacher_logits = torch.randn(8, 3, requires_grad=True)
    y_true = torch.randint(0, 3, (8,))
    for temp in [1.0, 4.0]:
        loss = distill_loss(student_logits, teacher_logits,
                            y_true, T=temp, alpha=0.3)
        # Phan tach lai de thay KL goc giam khi T tang,
        # trong khi loss tong co the tang vi nhan T^2.
        with torch.no_grad():
            ce_part = F.cross_entropy(student_logits, y_true)
            kl_raw = F.kl_div(
                F.log_softmax(student_logits / temp, dim=1),
                F.softmax(teacher_logits / temp, dim=1),
                reduction="batchmean")
            rebuilt = 0.3 * ce_part + 0.7 * (temp ** 2) * kl_raw
        print(f"T={temp}: loss={loss.item():.4f} "
              f"(huu han: {bool(torch.isfinite(loss))}) | "
              f"CE={float(ce_part):.4f} | KL goc="
              f"{float(kl_raw):.4f} | T^2*KL="
              f"{float((temp ** 2) * kl_raw):.4f} | "
              f"tai tao dung: "
              f"{bool(torch.allclose(loss.detach(), rebuilt))}")
        # [TRAINER] Truoc day 2 kiem tra nay chi IN
        # True/False: neu mot ngay loss thanh NaN hay
        # cong thuc lech, script van chay "thanh cong".
        # Assert de lan chay FAIL ngay tai cho.
        assert bool(torch.isfinite(loss)), \
            "loss phai huu han"
        assert bool(torch.allclose(loss.detach(), rebuilt)), \
            "loss phai bang alpha*CE + (1-alpha)*T^2*KL"
    # Detach: cho loss chay backward, teacher phai co
    # grad None vi logits da detach ben trong ham.
    student_logits.grad = None
    teacher_logits.grad = None
    loss = distill_loss(student_logits, teacher_logits,
                        y_true, T=4.0, alpha=0.3)
    loss.backward()
    print(f"Detach teacher: grad teacher la None = "
          f"{teacher_logits.grad is None} | grad student "
          f"co gia tri = {student_logits.grad is not None}")
    # [TRAINER] Tuong tu: chi in thi detach hong van di
    # qua. Assert ca 2 phia (teacher None VA student co).
    assert teacher_logits.grad is None, \
        "teacher khong duoc nhan gradient"
    assert student_logits.grad is not None, \
        "student phai nhan gradient"
    # Guard: moi truong hop sai phai bi chan ro rang.
    print("Kiem tra guard (loi vao sai):")
    good = torch.randn(4, 3)
    # [TRAINER] Truoc day gia tri tra ve cua
    # _expect_value_error bi BO QUA: guard nao hong,
    # script van ket thuc binh thuong (exit 0). Nay gom
    # ket qua lai va assert tong o cuoi, kem 3 guard moi
    # cua ban review (dtype y, logits NaN, T dang chuoi).
    guard_results = [
        _expect_value_error(
            lambda: distill_loss(good, good, y_true[:4], T=0.0),
            "T=0"),
        _expect_value_error(
            lambda: distill_loss(good, good, y_true[:4],
                                 alpha=1.5),
            "alpha=1.5"),
        _expect_value_error(
            lambda: distill_loss(good, torch.randn(4, 2),
                                 y_true[:4]),
            "shape teacher khac student"),
        _expect_value_error(
            lambda: distill_loss(torch.empty(0, 3),
                                 torch.empty(0, 3),
                                 torch.empty(0, dtype=torch.long)),
            "batch rong"),
        _expect_value_error(
            lambda: distill_loss(good, good,
                                 torch.tensor([0, 1, 2, 3])),
            "nhan 3 ngoai [0, 2]"),
        _expect_value_error(
            lambda: distill_loss(good, good,
                                 y_true[:4].float()),
            "y_true dtype float"),
        _expect_value_error(
            lambda: distill_loss(
                torch.tensor([[float("nan"), 0.0, 0.0]]),
                torch.zeros(1, 3), torch.tensor([0])),
            "logits chua NaN"),
        _expect_value_error(
            lambda: distill_loss(good, good, y_true[:4],
                                 T="4"),
            "T la chuoi, khong phai so"),
    ]
    assert all(guard_results), \
        "Co guard khong chan dung truong hop sai!"
    print("  Tat ca guard deu chan dung (da assert).")


def train_student(mode, teacher, x_train, y_train, x_test,
                  y_test, T=4.0, alpha=0.3,
                  epochs=STUDENT_EPOCHS, seed=DATA_SEED):
    """Train 1 student theo dung 1 cau hinh, tra ve ket qua.

    Dat lai seed truoc khi tao student nen moi cau hinh
    khoi tao giong het nhau. Teacher logits tinh 1 lan
    trong no_grad truoc vong lap, dung chung cho moi
    epoch - vua nhanh, vua dam bao teacher khong doi.

    Args:
        mode: "hard" (chi CE) hoac "distill" (ham Bai 2).
        teacher: model teacher da train, o che do eval.
        x_train/y_train: tap train day du (900 mau);
            ham tu lay 300 mau dau lam tap cua student.
        x_test/y_test: tap test de danh gia.
        T/alpha: chi co y nghia khi mode="distill".
        epochs: so epoch, mac dinh 100 theo de bai.
        seed: seed khoi tao student.

    Returns:
        (student, acc, agree): student da train, chinh
        xac tren test, ti le dong y voi teacher tren test.
    """
    torch.manual_seed(seed)
    student = MLP(32)
    opt = torch.optim.Adam(student.parameters(), lr=0.01)
    xs, ys = x_train[:N_STUDENT], y_train[:N_STUDENT]
    with torch.no_grad():
        teacher_logits = teacher(xs)
    for _ in range(epochs):
        opt.zero_grad()
        s_logits = student(xs)
        if mode == "hard":
            loss = F.cross_entropy(s_logits, ys)
        elif mode == "distill":
            loss = distill_loss(s_logits, teacher_logits, ys,
                                T=T, alpha=alpha)
        else:
            raise ValueError("mode phai la 'hard' hoac "
                             "'distill'.")
        loss.backward()
        opt.step()
    student.eval()
    with torch.no_grad():
        pred = student(x_test).argmax(1)
        teacher_pred = teacher(x_test).argmax(1)
    acc = float((pred == y_test).float().mean())
    agree = float((pred == teacher_pred).float().mean())
    return student, acc, agree


def per_class_accuracy(model, x_test, y_test):
    """Chinh xac tung lop, de thay lop nao la chien truong.

    Lop 2 xa nen de dat gan 1.0; lop 0/1 chong lap nhau
    nen moi la noi distillation co the doi duoc gi.

    Returns:
        List 3 float: chinh xac lop 0, 1, 2; NaN cho
        lop vang mat trong test (khong co mau).
    """
    model.eval()
    with torch.no_grad():
        pred = model(x_test).argmax(1)
    result = []
    for c in range(3):
        mask = y_test == c
        # [TRAINER] Lop vang mat trong test: truoc day
        # mean() cua tap rong tu sinh NaN lang le, doc
        # bang ket qua khong biet la "khong co mau" hay
        # "do chua". Nay tra NaN CO CHU DICH kem dieu
        # kien ro rang; muon biet lop nao thieu thi dem
        # int(mask.sum()).
        if int(mask.sum()) == 0:
            result.append(float("nan"))
            continue
        result.append(float((pred[mask] == y_test[mask]).
                            float().mean()))
    return result


def copy_wrong_rate(student, teacher, x_test, y_test):
    """Ti le student sao chep dung cho teacher SAI.

    Chi tinh tren cac mau test ma teacher du doan sai.
    Ti le cao nghia la student dang sao chep
    ca loi cua teacher, dung nhu canh bao buoi sang.

    Returns:
        (ti_le, so_mau_teacher_sai); ti_le la NaN khi
        teacher dung het (0 mau sai, khong dinh nghia).
    """
    student.eval()
    teacher.eval()
    with torch.no_grad():
        s_pred = student(x_test).argmax(1)
        t_pred = teacher(x_test).argmax(1)
    wrong = t_pred != y_test
    # [TRAINER] Teacher dung het (0 mau sai): ti le sao
    # chep KHONG dinh nghia. Truoc day mean() cua tap
    # rong tra NaN lang le; nay tra (NaN, 0) ro rang de
    # noi goi ham tu nhan biet n=0 ma dien giai.
    if int(wrong.sum()) == 0:
        return float("nan"), 0
    rate = float((s_pred[wrong] == t_pred[wrong]).
                 float().mean())
    return rate, int(wrong.sum())


def bench(model, x, warmup=50, repeats=1000):
    """Do toc do suy luan: trung vi ms / 1 lan chay.

    Dung time.perf_counter() (dong ho don dieu, phan giai
    cao) va lay TRUNG VI sau warmup, khong lay trung
    binh: trung binh 200 lan o Day 22 bi nhieu keo lech
    toi 0.378 ms. Khong co resource can dong trong
    finally o day (khong mo file, khong tracker).

    Args:
        model: model can do, se chuyen ve eval.
        x: batch dau vao, o day la toan bo 300 mau test.
        warmup: so lan chay khoi dong bo qua.
        repeats: so lan chay lay trung vi.

    Returns:
        Trung vi thoi gian, don vi mili-giay / lan.
    """
    model.eval()
    times = []
    with torch.no_grad():
        for _ in range(warmup):
            _ = model(x)
        for _ in range(repeats):
            start = time.perf_counter()
            _ = model(x)
            times.append(time.perf_counter() - start)
    return statistics.median(times) * 1000.0


def bai3(teacher, x_train, y_train, x_test, y_test):
    """Bai 3: thi nghiem co kiem soat A/B/C + Green LLM.

    Diff loi goi (checklist 10): A vs B chi khac mode
    (hard -> distill), moi doi so khac giong het; B vs C
    chi khac T (4.0 -> 2.0), alpha/seed/epoch giong het.
    """
    print("\n" + "=" * 72)
    print("BAI 3 - Hard-label vs distillation (co kiem soat)")
    print("=" * 72)
    print("Thu duy nhat thay doi la cach tinh loss "
          "(hard-label hay distillation); du lieu, kien "
          "truc student, seed va so epoch giu nguyen.")
    configs = [
        ("A hard-only", {"mode": "hard"}),
        ("B distill T=4 a=0.3", {"mode": "distill"}),
        ("C distill T=2 a=0.3",
         {"mode": "distill", "T": 2.0}),
    ]
    results = {}
    details = {}
    for name, kwargs in configs:
        student, acc, agree = train_student(
            teacher=teacher, x_train=x_train, y_train=y_train,
            x_test=x_test, y_test=y_test, **kwargs)
        results[name] = (student, acc, agree)
        per_class = per_class_accuracy(student, x_test, y_test)
        copy_rate, n_wrong = copy_wrong_rate(
            student, teacher, x_test, y_test)
        details[name] = (per_class, copy_rate)
        print(f"{name}: test acc={acc:.3f} | dong y voi "
              f"teacher={agree:.3f} | tung lop="
              f"{[round(v, 3) for v in per_class]} | sao "
              f"chep cho teacher sai={copy_rate:.3f} "
              f"({n_wrong} mau)")
    # Kiem tra them: lap lai A/B/C o seed 1..4. Day KHONG
    # phai bang chinh (bang chinh dung seed 0 theo de
    # bai); no chi cho biet chenh lech o seed 0 co on
    # dinh khong, vi test chi co 300 mau (1% = 3 mau).
    # Trung binh 5 seed (seed 0 + seed 1..4) duoc tinh
    # that tu cac lan chay nay, khong ghi tay.
    print("\nKiem tra them - seed 1..4 (trung binh):")
    mean5 = {}
    for name, kwargs in configs:
        accs = [results[name][1]]
        for seed in [1, 2, 3, 4]:
            _, acc, _ = train_student(
                teacher=teacher, x_train=x_train,
                y_train=y_train, x_test=x_test, y_test=y_test,
                seed=seed, **kwargs)
            accs.append(acc)
        mean5[name] = sum(accs) / len(accs)
        print(f"  {name}: {[round(v, 3) for v in accs[1:]]} "
              f"| trung binh 5 seed={mean5[name]:.3f}")
    # Green LLM: cung ham dem params, cung ham bench.
    teacher_params = count_params(teacher)
    student_b = results["B distill T=4 a=0.3"][0]
    student_params = count_params(student_b)
    teacher_ms = bench(teacher, x_test)
    student_ms = bench(student_b, x_test)
    print(f"\nTeacher: {teacher_params} params, "
          f"{teacher_ms:.3f} ms/lan (trung vi, 300 mau).")
    print(f"Student B: {student_params} params, "
          f"{student_ms:.3f} ms/lan (trung vi, 300 mau).")
    print(f"Ti le params student/teacher = "
          f"{student_params / teacher_params:.3f} "
          f"(giam {100 * (1 - student_params / teacher_params):.1f}%).")
    print(f"Toc do nhanh hon = {teacher_ms / student_ms:.2f}x.")
    teacher_acc = accuracy(teacher, x_test, y_test)
    print(f"\nBANG KET QUA: teacher acc={teacher_acc:.3f} | "
          f"A={results['A hard-only'][1]:.3f} | "
          f"B={results['B distill T=4 a=0.3'][1]:.3f} | "
          f"C={results['C distill T=2 a=0.3'][1]:.3f} | "
          f"params {student_params}/{teacher_params} | "
          f"toc do {teacher_ms / student_ms:.2f}x.")
    # Ket luan viet bang so vua do, khong ghi tay: chenh
    # lech trung binh 5 seed, cai thien lop 0 / giam lop
    # 1, ti le sao chep loi, ti le params, toc do.
    diff_mean = mean5["B distill T=4 a=0.3"] - mean5["A hard-only"]
    gain0 = (details["B distill T=4 a=0.3"][0][0] -
             details["A hard-only"][0][0])
    drop1 = (details["A hard-only"][0][1] -
             details["B distill T=4 a=0.3"][0][1])
    copy_diff = (details["B distill T=4 a=0.3"][1] -
                 details["A hard-only"][1])
    speedup = teacher_ms / student_ms
    param_ratio = teacher_params / student_params
    print(f"Ket luan: distillation giup it nhung on dinh - "
          f"B hon A trung binh {diff_mean:.3f} qua 5 seed "
          f"(~{diff_mean * 300:.0f} mau test), chu yeu nho "
          f"cai thien lop 0 (+{gain0:.3f}), doi lai lop 1 "
          f"giam {drop1:.3f} va sao chep loi teacher nhieu "
          f"hon {copy_diff:.3f}. Voi workload nay, doi "
          f"{diff_mean:.3f} accuracy lay "
          f"{100 * (1 - student_params / teacher_params):.1f}% "
          f"it params va {speedup:.2f}x toc do la dang, vi "
          f"student da vuot ca teacher; nhung {speedup:.2f}x "
          f"chu khong phai {param_ratio:.1f}x cho thay "
          f"params it khong dong nghia toc do tang ty le - "
          f"overhead moi lan chay moi la phan chi phoi o "
          f"model nho the nay.")


def main():
    """Chay ca lab: du lieu, teacher, Bai 1, Bai 2, Bai 3."""
    x_train, y_train, x_test, y_test = build_data()
    print(f"train: {tuple(x_train.shape)} | "
          f"test: {tuple(x_test.shape)}")
    teacher = train_teacher(MLP(128), x_train, y_train)
    teacher.eval()
    teacher_acc = accuracy(teacher, x_test, y_test)
    print(f"teacher test acc: {teacher_acc:.3f} | params: "
          f"{count_params(teacher)}")
    bai1(teacher, x_test, y_test)
    bai2()
    bai3(teacher, x_train, y_train, x_test, y_test)


if __name__ == "__main__":
    main()
