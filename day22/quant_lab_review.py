#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Lab Quantization Day 22 (tuan 4 - Green LLM): quantize that,
do that, dung tin con so tren model card.

Chay tren CPU, PyTorch, khong can GPU, khong can tai model lon.

Bai 1: affine quantization (symmetric + asymmetric), tinh
    scale, zero-point, sai so lam tron. Edge case: tensor
    hang so (max == min) -> scale 0 -> phai guard, khong chia
    cho 0.
Bai 2: outlier pha thang do per-tensor nhu the nao, per-
    channel cuu duoc bao nhieu (INT8 + mo rong INT4).
Bai 3: weight-only quantize MLP that o INT8/INT4, do accuracy,
    bo nho, do tre batch 1 va 256 (perf_counter), uoc luong
    nang luong theo ti le byte doc.
Bai 4: bang benchmark + quyet dinh + regression set 8 input.

Quy tac: chi doi SO BIT (va do hat khi so sanh per-tensor vs
per-channel), moi thu khac giu nguyen: cung seed, cung model
da train, cung tap test. Do tre cua ban "gia lap" (dequantize
ve FP32 roi tinh FP32) duoc bao THAT, khong gia vo nhanh hon.

Cach chay (venv Day 15):
    ~/workspace/venvs/day15/bin/python quant_lab.py
"""

import copy
import time

import torch
import torch.nn as nn

LATENCY_RUNS = 200
WARMUP_RUNS = 20


def quantize_dequantize(x, bits=8, symmetric=False):
    """Luong tu affine roi giai ma nguoc (fake quantize).

    Cong thuc buoi sang:
        s = (max - min) / (2^b - 1) (asymmetric)
        s = max(|min|, |max|) / (2^(b-1) - 1) (symmetric)
        x_q = clamp(round(x / s) + z, lo, hi)
        x_hat = s * (x_q - z)

    Args:
        x: tensor bat ky shape, kieu float.
        bits: so bit, phai trong [2, 16].
        symmetric: True -> muc doi xung, z = 0; False ->
            muc 0..2^b-1, z = round(-min / s).

    Returns:
        (x_hat, scale, zero_point): x_hat cung shape voi x,
        scale la float, zero_point la int.

    Raises:
        ValueError: tensor rong, bits ngoai [2, 16].
    """
    if x.numel() == 0:
        raise ValueError("tensor khong duoc rong.")
    if not 2 <= bits <= 16:
        raise ValueError("bits phai trong [2, 16].")
    xmin = float(x.min())
    xmax = float(x.max())
    if symmetric:
        max_level = 2 ** (bits - 1) - 1
        # Guard: tensor hang so 0 -> max_abs = 0, tranh chia 0.
        # Scale 1.0 thi moi gia tri 0 van map ve 0 dung.
        scale = max(abs(xmin), abs(xmax)) / max_level
        if scale == 0.0:
            scale = 1.0
        zero_point = 0
        lo, hi = -max_level, max_level
    else:
        # Guard: tensor hang so (max == min) -> range 0.
        # [TRAINER] Ban goc dat scale=1.0, z=round(-min) roi tra
        # thang x.clone(): x_hat dung, nhung cap (scale, z) tra
        # ve KHONG tu nhat quan - ap lai cong thuc voi chinh
        # scale/z do cho gia tri khong nguyen (vd 3.7: z=-4,
        # x_q=0 -> x_hat=4.0) thi sai. Ai dung scale/z nay de
        # quantize tensor khac se dinh sai so am tham. Ban sua:
        # chon scale/z sao cho gia tri hang so nam dung o mot
        # dau muc cua thang do, roi di qua nhanh tinh chung -
        # x_hat van khoi phuc dung va bo ba (x_hat, scale, z)
        # nhat quan voi nhau.
        if xmax == xmin:
            if xmax == 0.0:
                scale, zero_point = 1.0, 0
            elif xmax > 0:
                scale = xmax / (2 ** bits - 1)
                zero_point = 0
            else:
                scale = -xmin / (2 ** bits - 1)
                zero_point = 2 ** bits - 1
            lo, hi = 0, 2 ** bits - 1
        else:
            scale = (xmax - xmin) / (2 ** bits - 1)
            zero_point = int(round(-xmin / scale))
            lo, hi = 0, 2 ** bits - 1
    x_q = torch.clamp(torch.round(x / scale) + zero_point,
                      lo, hi)
    x_hat = scale * (x_q - zero_point)
    return x_hat, scale, zero_point


def quantize_per_channel(w, bits=8, symmetric=False):
    """Luong tu rieng tung hang (channel) cua ma tran 2D.

    Moi hang co scale/zero-point rieng. Tra ve ma tran cung
    shape da giai ma nguoc.
    """
    rows = []
    for i in range(w.shape[0]):
        row_hat, _, _ = quantize_dequantize(w[i], bits=bits,
                                            symmetric=symmetric)
        rows.append(row_hat)
    return torch.stack(rows)


def mse(a, b):
    """Sai so binh phuong trung binh giua 2 tensor."""
    return float(((a - b) ** 2).mean())


def bai1():
    """Bai 1: affine quantization + so sanh INT8/INT4, lech 1 phia."""
    print("=" * 72)
    print("BAI 1 - Affine quantization")
    print("=" * 72)
    x = torch.tensor([-1.2, -0.3, 0.0, 0.3, 0.6])
    x_hat, s, z = quantize_dequantize(x, bits=8, symmetric=False)
    print(f"INT8 asymmetric: scale ~{s:.5f} (ky vong 0.00706) "
          f"| zero-point: {z}")
    print(f"x_hat: {[round(v, 5) for v in x_hat.tolist()]}")
    err8 = float((x_hat - x).abs().max())
    print(f"sai so lon nhat INT8: {err8:.5f} (<= scale/2 = "
          f"{s / 2:.5f}, bang dung o diem lam tron nua muc).")
    x_hat4, s4, _ = quantize_dequantize(x, bits=4, symmetric=False)
    err4 = float((x_hat4 - x).abs().max())
    print(f"sai so lon nhat INT4: {err4:.5f} (gap INT8 "
          f"{err4 / err8:.1f}x, ky vong ~17x = ti le scale).")
    # (b) Tensor lech han 1 phia: [0.5, 2.5].
    y = torch.linspace(0.5, 2.5, 21)
    y_sym, s_sym, _ = quantize_dequantize(y, bits=8,
                                          symmetric=True)
    y_asym, s_asym, _ = quantize_dequantize(y, bits=8,
                                            symmetric=False)
    err_sym = float((y_sym - y).abs().max())
    err_asym = float((y_asym - y).abs().max())
    # Muc thang do that su dung: (max-min)/scale tren tong muc.
    used_sym = int(round((2.5 - 0.5) / s_sym)) + 1
    used_asym = int(round((2.5 - 0.5) / s_asym)) + 1
    print(f"\nTensor [0.5, 2.5], INT8:")
    print(f"  symmetric: scale={s_sym:.5f}, sai so max="
          f"{err_sym:.5f}, dung ~{used_sym}/255 muc (40%).")
    print(f"  asymmetric: scale={s_asym:.5f}, sai so max="
          f"{err_asym:.5f}, dung het {used_asym}/256 muc.")
    print(f"  Ket luan: asymmetric tot hon {err_sym / err_asym:.1f}x "
          f"vi dung tron thang do; symmetric bo phi nua muc am "
          f"ma du lieu khong bao gio dung toi.")
    # Edge case: tensor hang so va tensor 0.
    const, s_c, z_c = quantize_dequantize(
        torch.full((5,), 3.0), bits=8)
    zeros, s_z, z_z = quantize_dequantize(torch.zeros(5), bits=8,
                                          symmetric=True)
    print(f"Edge case hang so 3.0: khoi phuc dung = "
          f"{bool(torch.allclose(const, torch.full((5,), 3.0)))} "
          f"(scale={s_c}, z={z_c}, khong chia cho 0).")
    print(f"Edge case toan 0 (symmetric): khoi phuc dung = "
          f"{bool(torch.allclose(zeros, torch.zeros(5)))} "
          f"(scale={s_z}, z={z_z}).")
    try:
        quantize_dequantize(x, bits=1)
        print("CANH BAO: bits=1 khong bi chan!")
    except ValueError as exc:
        print(f"Edge case bits=1: chan dung ({exc}).")


def bai2():
    """Bai 2: outlier pha thang do per-tensor, per-channel cuu."""
    print("\n" + "=" * 72)
    print("BAI 2 - Outlier va per-channel")
    print("=" * 72)
    torch.manual_seed(0)
    w = torch.randn(4, 1000)
    w[0] *= 50.0
    for bits in (8, 4):
        print(f"--- INT{bits} ---")
        w_tensor, _, _ = quantize_dequantize(w, bits=bits)
        w_channel = quantize_per_channel(w, bits=bits)
        print(f"{'channel':>8} | {'MSE per-tensor':>14} | "
              f"{'MSE per-channel':>15}")
        for i in range(4):
            print(f"{i:>8} | {mse(w_tensor[i], w[i]):>14.4f} | "
                  f"{mse(w_channel[i], w[i]):>15.4f}")
        mse_normal_tensor = sum(mse(w_tensor[i], w[i])
                                for i in range(1, 4)) / 3
        mse_normal_channel = sum(mse(w_channel[i], w[i])
                                 for i in range(1, 4)) / 3
        print(f"MSE trung binh 3 channel thuong: per-tensor "
              f"{mse_normal_tensor:.4f} vs per-channel "
              f"{mse_normal_channel:.4f} (per-channel tot hon "
              f"{mse_normal_tensor / mse_normal_channel:.1f}x).")
    print("Ket luan: outlier (channel 0, x50) lam hai NHAT cac "
          "channel BINH THUONG o per-tensor - chung bi don vao "
          "vai muc, MSE tang vot; per-channel cho moi channel "
          "thang do rieng nen sai so tro lai muc binh thuong. "
          "O INT4, ngay ca per-channel cung bat dau duoi "
          "(xem bang).")


def train_mlp():
    """Train MLP nho tren du lieu tong hop (code de bai)."""
    torch.manual_seed(0)
    x = torch.randn(2000, 2)
    y = ((x[:, 0] ** 2 + x[:, 1] ** 2) > 1.0).long()
    xtr, xte, ytr, yte = x[:1600], x[1600:], y[:1600], y[1600:]
    mlp = nn.Sequential(nn.Linear(2, 32), nn.ReLU(),
                        nn.Linear(32, 32), nn.ReLU(),
                        nn.Linear(32, 2))
    opt = torch.optim.AdamW(mlp.parameters(), lr=0.01)
    for _ in range(200):
        opt.zero_grad()
        loss = nn.CrossEntropyLoss()(mlp(xtr), ytr)
        loss.backward()
        opt.step()
    return mlp, xte, yte


@torch.no_grad()
def accuracy(model, x, y):
    """Ti le du doan dung tren tap test."""
    model.eval()
    return (model(x).argmax(1) == y).float().mean().item()


def quantize_model_weights(model, bits, per_channel=False,
                           symmetric=False):
    """Weight-only quantize: chi trong so Linear, activation
    van FP32. Lam tren ban copy, khong pha model goc."""
    new_model = copy.deepcopy(model)
    for layer in new_model:
        if isinstance(layer, nn.Linear):
            if per_channel:
                w_hat = quantize_per_channel(layer.weight.data,
                                             bits=bits,
                                             symmetric=symmetric)
            else:
                w_hat, _, _ = quantize_dequantize(
                    layer.weight.data, bits=bits,
                    symmetric=symmetric)
            layer.weight.data = w_hat
    return new_model


def latency_ms(model, x, runs=1000):
    """Do tre TRUNG VI (ms/lan) tren nhieu lan chay rieng le.

    Trung vi thay vi trung binh: do tre o micro-giay de bi
    nhieu boi nhiet do CPU/tien trinh nen, trung binh de bi
    keo lech boi vai lan cham dot bien. Co warmup truoc.
    """
    import statistics
    with torch.no_grad():
        model.eval()
        for _ in range(WARMUP_RUNS):
            model(x)
        times = []
        for _ in range(runs):
            t0 = time.perf_counter()
            model(x)
            times.append(time.perf_counter() - t0)
    return statistics.median(times) * 1000.0


# Regression set: 8 input dai dien, ke ca bien quyet dinh
# (r^2 gan 1) va xa tam. Moi phien ban quantize sau nay phai
# cho cung du doan voi ban FP32 tren ca 8 input nay.
REGRESSION_X = torch.tensor([
    [0.0, 0.0], [1.0, 0.0], [0.0, 1.0], [0.7, 0.7],
    [0.71, 0.71], [2.0, 2.0], [-0.7, -0.7], [0.5, 0.86],
])
# [TRAINER] Nhan cua [0.5, 0.86] o ban goc ghi 1, nhung
# 0.5^2 + 0.86^2 = 0.25 + 0.7396 = 0.9896 < 1 -> theo dung
# luat sinh du lieu (r^2 > 1) nhan phai la 0. Luu y them:
# regression_agreement() hien chi so du doan voi model FP32
# nen REGRESSION_Y chua duoc dung o dau - nhan sai nam im o
# day se thanh "diem neo sai" ngay khi co ai cham accuracy
# tren regression set bang nhan nay.
REGRESSION_Y = torch.tensor([0, 0, 0, 0, 1, 1, 0, 0])


@torch.no_grad()
def regression_agreement(model, ref_model):
    """Ti le input regression ma model dong y voi ban goc."""
    model.eval()
    ref_model.eval()
    pred = model(REGRESSION_X).argmax(1)
    ref = ref_model(REGRESSION_X).argmax(1)
    return (pred == ref).float().mean().item(), pred.tolist()


def bai3_bai4():
    """Bai 3 + 4: weight-only MLP, benchmark, quyet dinh."""
    print("\n" + "=" * 72)
    print("BAI 3 - Weight-only quantize MLP that")
    print("=" * 72)
    mlp, xte, yte = train_mlp()
    acc_fp = accuracy(mlp, xte, yte)
    print(f"Accuracy FP32: {acc_fp:.3f}")
    n_params = sum(p.numel() for p in mlp.parameters())
    print(f"So tham so: {n_params:,}")
    for name, bpp in [("FP32", 4), ("INT8", 1), ("INT4", 0.5)]:
        print(f"{name}: ~{n_params * bpp / 1024:.1f} KB trong so.")
    x1 = torch.randn(1, 2)
    x256 = torch.randn(256, 2)
    variants = {
        "FP32": mlp,
        "INT8": quantize_model_weights(mlp, 8),
        "INT4": quantize_model_weights(mlp, 4),
        # [TRAINER] Ban goc them symmetric=True chi rieng cho
        # ban nay: so voi "INT4" per-tensor thi doi HAI bien
        # cung luc (do hat + doi xung), dung cai bay ablation
        # ma docstring dau file cam ("chi doi SO BIT va do
        # hat"). So do lai sach: per-channel asymmetric cho
        # acc 0.990, regression 75% (ban goc: 0.988, 38%) -
        # per-channel THAT RA giup nhieu, chi la van chua qua
        # gate 8/8. Bo symmetric de chi con 1 bien thay doi.
        "INT4 per-channel": quantize_model_weights(
            mlp, 4, per_channel=True),
    }
    print("\n" + "=" * 72)
    print("BAI 4 - Bang benchmark (so do that tren CPU nay)")
    print("=" * 72)
    print(f"{'Phien ban':<16} | {'Acc':>6} | {'Bo nho':>7} | "
          f"{'Tre b1':>8} | {'Tre b256':>9} | {'Regr':>5}")
    rows = {}
    for name, model in variants.items():
        acc = accuracy(model, xte, yte)
        lat1 = latency_ms(model, x1)
        lat256 = latency_ms(model, x256)
        agree, _ = regression_agreement(model, mlp)
        bpp = {"FP32": 4, "INT8": 1, "INT4": 0.5,
               "INT4 per-channel": 0.5}[name]
        mem = n_params * bpp / 1024
        rows[name] = (acc, mem, lat1, lat256, agree)
        print(f"{name:<16} | {acc:>6.3f} | {mem:>6.1f}KB | "
              f"{lat1:>7.3f}ms | {lat256:>8.3f}ms | "
              f"{agree:>4.0%}")
    print("\nDo tre do duoc (trung vi 1000 lan) deu o muc micro-"
          "giay va chenh nhau trong pham vi nhieu cua may nho "
          "nay - ban 'gia lap' (dequantize ve FP32 roi tinh "
          "FP32) KHONG nhanh hon that, dung nhu Cau 3 cua de "
          "bai: muon nhanh that can kernel INT8/INT4 that. "
          "Khong duoc doc bang do tre nay thanh 'INT4 cham "
          "hon' hay 'INT8 nhanh hon' - do la nhieu do, khong "
          "phai tin hieu.")
    print("Nang luong (uoc luong, khong phai so do dien): "
          "batch 1 memory-bound, energy/token ti le voi byte "
          "trong so phai doc moi lan decode -> INT8/FP32 = "
          "0.25x, INT4/FP32 = 0.125x.")
    # Quyet dinh suy tu SO DO, khong viet san truoc.
    acc_fp = rows["FP32"][0]
    acc8, regr8 = rows["INT8"][0], rows["INT8"][4]
    acc4pc, regr4pc = rows["INT4 per-channel"][0], \
        rows["INT4 per-channel"][4]
    print("\nQUYET DINH (suy tu so do tren):")
    print(f"(1) Chatbot can giu chat luong: chon INT8 "
          f"weight-only - accuracy {acc8:.3f} bang FP32 "
          f"({acc_fp:.3f}), regression {regr8:.0%}, bo nho "
          f"giam 4x, nang luong doc giam 4x.")
    if regr4pc >= 1.0 and acc_fp - acc4pc <= 0.01:
        print("(2) May yeu offline: INT4 per-channel DAT gate "
              "regression, duoc phep dung voi bo nho giam 8x.")
    else:
        print(f"(2) May yeu offline: INT4 per-channel KHONG "
              f"dat gate (accuracy {acc4pc:.3f}, regression "
              f"chi {regr4pc:.0%} < 100% - ca hai ban INT4 deu "
              f"lat lung tung gan bien quyet dinh). Ket luan: "
              f"tren bang chung nay, INT4 CHUA duoc release "
              f"ngay ca cho may yeu; dung INT8, hoac cai thien "
              f"INT4 bang per-group/GPTQ roi do lai.")
    print("(3) Regression set (8 input, ke ca bien r^2 gan 1): "
          "(0,0), (1,0), (0,1), (0.7,0.7), (0.71,0.71), (2,2), "
          "(-0.7,-0.7), (0.5,0.86) - nhan that theo r^2 > 1. "
          "Moi phien ban quantize sau nay phai dong y 8/8 voi "
          "ban FP32 truoc khi release.")
    print(f"\nRegression agreement: " +
          ", ".join(f"{n} {r[4]:.0%}" for n, r in rows.items())
          + ".")


def main():
    """Chay lab day du: Bai 1 -> 4."""
    t0 = time.perf_counter()
    print(f"torch: {torch.__version__} (CPU)")
    bai1()
    bai2()
    bai3_bai4()
    print(f"\nToan bo lab chay trong {time.perf_counter() - t0:.1f}s "
          f"tren CPU.")


if __name__ == "__main__":
    main()
