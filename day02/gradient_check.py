#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Day 02 - Bai 1: GRADIENT CHECK - "bat" autograd khai gian
================================================================
Muc dich:
    Kiem chung rang gradient do torch.autograd tinh ra (giai tich)
    KHOP voi gradient tinh bang SAI PHAN HUU HAN (finite differences).

Vi sao phai lam viec nay?
    - Backprop chi la chain rule duoc tu dong hoa. Neu gradient sai
      o dau do (bug mask, sai shape, custom op viet sai cong thuc...),
      model van chay, loss van giam "binh thuong", nhung hoc ra rac.
    - Sai phan huu han la cach DOC LAP de xac thuc, khong dung autograd:
          df/dx  ~  [ f(x + eps) - f(x - eps) ] / (2 * eps)
      Cong thuc nay chi dung phep cong/tru/nhan/chia thuan tuy -
      khong he dung den dao ham - nen neu 2 cach cho cung mot ket qua,
      ta tin duoc autograd.

Cach chay:
    python gradient_check.py
    # Ky vong: 4/4 param PASS, max_error < 1e-4 (thuong ~1e-9 o float64)

Thuc te su dung:
    - Chi chay gradient check tren model MINI (vua du de check bang tay).
    - Check xong, train binh thuong bang float32/bf16 (nhanh hon nhieu).
"""

import time

import torch

# ---------------------------------------------------------------------------
# 0. THAM SO KIEM TRA
# ---------------------------------------------------------------------------
# EPS: buoc perturb (nhich) cho sai phan huu han.
#   - Qua NHO  -> bi sai so lam tron nuot (dac biet o float32).
#   - Qua LON  -> xap xi dao ham kem chinh xac (cong thuc la gioi han khi eps->0).
#   - 1e-6 o float64 la "diem ngot" kinh dien.
EPS = 1e-6

# Nguong chap nhan: sai so lon nhat giua 2 cach tinh phai NHO hon muc nay.
# Voi float64, sai so thuong roi vao ~1e-9, nen 1e-4 la nguong rat rong rai.
TOL = 1e-4


# ---------------------------------------------------------------------------
# [TRAINER] Nhan xet cua Muse (2026-10-12):
#   Code Bai 1 DUNG hoan toan, khong can sua logic. Bo sung 4 diem de sau nay
#   ban tu ap dung gradient check cho model THAT:
# ---------------------------------------------------------------------------
# 1. O day dung SAI SO TUYET DOI |ana-num| < 1e-4. Voi model mini (gradient
#    ~ O(1)) thi on. Khi check model that, gradient co the cuc nho (vanishing,
#    ~1e-8) hoac cuc lon -> hay dung SAI SO TUONG DOI (chuan cua Karpathy):
#        rel_err = |ana - num| / max(1.0, |ana| + |num|),  nguong < 1e-5 (float64)
#    Toi GIU NGUYEN check tuyet doi o day vi de hieu hon cho bai hoc dau tien.
# 2. Gradient check CHI tin duoc khi forward DETERMINISTIC: tat dropout,
#    model.eval(), khoa moi nguon ngau nhien. Neu khong, 2 lan forward cho 2 so
#    khac nhau -> FAIL oan du code cua ban dung 100%.
# 3. torch.manual_seed(42) chi gieo cho CPU. Khi chay tren GPU, them
#    torch.cuda.manual_seed_all(42) de ket qua tai lap duoc hoan toan.
# 4. Thoi gian in o bonus (3.6ms vs 2.2ms) chu yeu la NOISE o quy mo nay -
#    dung suy dien "float64 nhanh hon float32". Bai hoc Green that su chi hien
#    ro khi chay tren du lieu/model lon (AMP/GPU, Bai 3).


# ---------------------------------------------------------------------------
# 1. MO HINH MINI: 2 lop tuyen tinh + tanh, loss MSE
# ---------------------------------------------------------------------------
def make_params(dtype=torch.float64):
    """Khoi tao bo tham so + du lieu vao, co dinh seed de ket qua lap lai duoc.

    Kich thuoc (rat nho, chu y de check bang tay neu muon):
        x  : (3,)        - vector dau vao 3 chieu
        W1 : (4, 3)      - lop 1: 3 -> 4
        b1 : (4,)        - bias lop 1
        W2 : (1, 4)      - lop 2: 4 -> 1
        b2 : (1,)        - bias lop 2
        t  : (1,)        - target (gia tri muc tieu muon du doan)
    """
    # Dat seed TRUOC khi randn -> moi lan chay ra cung bo so ngau nhien.
    # (Quan trong cho khoa hoc: ket qua phai tai lap duoc.)
    torch.manual_seed(42)

    # requires_grad=True: bao PyTorch "hay theo doi phep tinh tren tensor nay
    # de sau nay .backward() tinh duoc gradient". Day la cong tac bat autograd.
    W1 = torch.randn(4, 3, requires_grad=True, dtype=dtype)
    b1 = torch.randn(4, requires_grad=True, dtype=dtype)
    W2 = torch.randn(1, 4, requires_grad=True, dtype=dtype)
    b2 = torch.randn(1, requires_grad=True, dtype=dtype)

    # x, t KHONG can gradient: chung la du lieu, khong phai tham so can hoc.
    x = torch.randn(3, dtype=dtype)
    t = torch.randn(1, dtype=dtype)
    return {"W1": W1, "b1": b1, "W2": W2, "b2": b2}, x, t


def forward(params, x, t):
    """Lan truyen THUAN (forward pass): x -> du doan -> loss (scalar).

    Kien truc:  x --[Linear]--> z1 --[tanh]--> h --[Linear]--> out
                                                 |
                                                 v
                                              MSE(out, t)
    """
    W1, b1, W2, b2 = params["W1"], params["b1"], params["W2"], params["b2"]

    # Lop 1: z1 = x @ W1.T + b1   -> shape: (3,) @ (3,4) = (4,)
    # (Dung .t() vi ta luu W dang (out_features, in_features) theo chuan PyTorch.)
    z1 = x @ W1.t() + b1

    # Kich hoat phi tuyen tanh: neu khong co no, 2 lop tuyen tinh chong len nhau
    # van chi la 1 ham tuyen tinh -> model khong hoc duoc gi phuc tap.
    h = torch.tanh(z1)

    # Lop 2: out = h @ W2.T + b2  -> shape: (4,) @ (4,1) = (1,)
    out = h @ W2.t() + b2

    # Loss MSE: trung binh binh phuong sai so. .mean() tren 1 phan tu = chinh no,
    # nhung viet tong quat de sau nay mo rong batch.
    loss = ((out - t) ** 2).mean()
    return loss


# ---------------------------------------------------------------------------
# 2. CACH 1 - GRADIENT GIAI TICH (do autograd tinh)
# ---------------------------------------------------------------------------
def analytic_grads(params, x, t):
    """Chay forward 1 lan + .backward() 1 lan, thu thap gradient cua moi param.

    Tra ve: dict {ten_param: tensor gradient cung shape voi param}.
    """
    # Xoa gradient cu (neu co) de dam bao phep do sach.
    # (O day param moi tinh nen grad=None, nhung viet cho dung thoi quen train loop.)
    for p in params.values():
        if p.grad is not None:
            p.grad.zero_()

    loss = forward(params, x, t)

    # .backward(): PyTorch di NGUOC qua do thi tinh toan (computational graph),
    # ap dung chain rule tu dong, cong don dL/dp vao p.grad cua moi leaf tensor.
    loss.backward()

    # .clone(): copy gradient ra tensor moi. Neu khong clone, tensor grad goc
    # se bi anh huong khi ta perturb param o buoc numeric ben duoi.
    return {name: p.grad.clone() for name, p in params.items()}


# ---------------------------------------------------------------------------
# 3. CACH 2 - GRADIENT SO (sai phan huu han, KHONG dung autograd)
# ---------------------------------------------------------------------------
def numeric_grad(param, params, x, t, eps=EPS):
    """Uoc luong gradient cua `param` bang cong thuc trung tam:
            dL/dp_i ~ [ L(p_i + eps) - L(p_i - eps) ] / (2 * eps)

    Y tuong cuc ky don gian: nhich tung vo huong cua param len/xuong
    mot chut, do loss thay doi bao nhieu -> suy ra do doc (dao ham).

    LUU Y QUAN TRONG:
      - Chi forward (khong .backward) -> hoan toan doc lap voi autograd.
      - Dung p.data / torch.no_grad() khi sua param de KHONG lam ban graph.
      - Sau moi lan nhich phai TRA param ve gia tri goc (xem code duoi).
    """
    grad = torch.zeros_like(param)  # tensor cung shape, de chua ket qua

    # flat = "lam phang" param thanh vector 1 chieu de duyet tung vo huong.
    # .data: lay du lieu thuan tuy, tach khoi autograd (sua khong tao graph).
    flat_param = param.data.view(-1)
    flat_grad = grad.view(-1)

    # torch.no_grad(): moi phep tinh trong khoi nay deu KHONG duoc theo doi
    # boi autograd -> tranh rac graph va tiet kiem bo nho.
    with torch.no_grad():
        for i in range(flat_param.numel()):
            orig = flat_param[i].item()  # luu gia tri goc de tra lai sau

            # Buoc 1: nhich LEN +eps, forward -> L+
            flat_param[i] = orig + eps
            loss_plus = forward(params, x, t).item()

            # Buoc 2: nhich XUONG -eps (tu gia tri goc), forward -> L-
            flat_param[i] = orig - eps
            loss_minus = forward(params, x, t).item()

            # Buoc 3: TRA param ve gia tri goc (bat buoc! neu quen, cac lan
            # lap sau se do tren param da bi lech -> ket qua sai hoan toan).
            flat_param[i] = orig

            # Cong thuc sai phan trung tam (do chinh xac bac O(eps^2),
            # tot hon cong thuc lech 1 phia [L(x+eps)-L(x)]/eps chi O(eps)).
            flat_grad[i] = (loss_plus - loss_minus) / (2 * eps)

    return grad


# ---------------------------------------------------------------------------
# 4. SO SANH 2 CACH -> PASS / FAIL
# ---------------------------------------------------------------------------
def gradient_check(params, x, t, eps=EPS, tol=TOL, verbose=True):
    """Doi chieu gradient giai tich vs gradient so cho MOI param.

    Tra ve: (passed: bool, max_error: float)
    """
    ana = analytic_grads(params, x, t)  # cach 1: autograd
    all_ok, worst = True, 0.0

    for name, p in params.items():
        num = numeric_grad(p, params, x, t, eps)   # cach 2: sai phan
        # Sai so lon nhat tren tung vo huong cua param nay.
        err = (ana[name] - num).abs().max().item()
        worst = max(worst, err)
        ok = err < tol
        all_ok &= ok
        if verbose:
            status = "PASS" if ok else "FAIL"
            print(f"  [{status}] {name:>3}: max |grad_autograd - grad_numeric| = {err:.3e}")

    return all_ok, worst


# ---------------------------------------------------------------------------
# 5. BONUS (Green checkpoint): float32 vs float64 - tu do so lieu
# ---------------------------------------------------------------------------
def compare_dtypes():
    """Chay gradient check o ca float32 va float64, do thoi gian + sai so.

    Muc dich giao duc: tu thay vi sao KHONG check gradient o float32.
    (float32 chi duoc dung de DEMO o day - khong assert, chi in so lieu.)
    """
    print("\n--- Bonus: float32 vs float64 (Green checkpoint) ---")
    for dtype in (torch.float32, torch.float64):
        params, x, t = make_params(dtype)
        t0 = time.perf_counter()
        ana = analytic_grads(params, x, t)
        # Do sai so tong (khong assert): lay max tren tat ca param.
        worst = 0.0
        for name, p in params.items():
            num = numeric_grad(p, params, x, t)
            worst = max(worst, (ana[name] - num).abs().max().item())
        dt = time.perf_counter() - t0
        print(f"  {str(dtype):>15}: max_error = {worst:.3e} | thoi gian = {dt*1000:6.1f} ms")
    print("  -> float32: sai so lam tron (~1e-7) lan at buoc eps=1e-6\n"
          "     nen gradient so khong con la 'chuan vang' de doi chieu.")


# ---------------------------------------------------------------------------
# 6. MAIN
# ---------------------------------------------------------------------------
def main():
    print("=" * 64)
    print("Bai 1 - Gradient check: doi chieu autograd vs sai phan huu han")
    print("=" * 64)

    # Bai chinh BAT BUOC chay o float64 (ly do xem compare_dtypes / cau hoi 1).
    params, x, t = make_params(torch.float64)

    print("\nKiem tra tung param (nguong max_error < 1e-4):")
    passed, worst = gradient_check(params, x, t)

    print(f"\nSai so lon nhat tong the: {worst:.3e}")
    if passed:
        print("KET QUA: 4/4 PASS - autograd cho gradient DUNG.")
    else:
        # raise SystemExit(1): bao loi cho shell/CI biet script that bai.
        print("KET QUA: FAIL - co gradient khong khop, can debug!")
        raise SystemExit(1)

    # Bonus: tu do so lieu float32 vs float64 (khong anh huong PASS/FAIL).
    compare_dtypes()


if __name__ == "__main__":
    main()
