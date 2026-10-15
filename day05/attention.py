#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Tu code self-attention tu con so 0 (Day 05 - buoi toi).

Muc dich:
    Bai 1: Single-head scaled dot-product attention co causal mask.
    Bai 2: Multi-head attention (4 heads) - tach/ghep heads dung chuan.
    Bai 3: Do thoi gian theo do dai cau L -> tu thay do phuc tap O(L^2).

Cach chay:
    python attention.py

Ky vong:
    Bai 1 qua 3 assert, in ma tran attention 4x4 (nua tren duong cheo = 0).
    Bai 2 qua assert, final dung shape (4, 16), 4 heads nhin khac nhau.
    Bai 3 in bang thoi gian 4 gia tri L + cau tra loi TODO 7.

Chi can PyTorch (CPU la du). Seed co dinh de ket qua tai lap duoc.
"""

import math
import time

import torch
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# Thiet lap chung (dung cho ca 3 bai)
# ---------------------------------------------------------------------------
torch.manual_seed(5)  # de ket qua cua moi nguoi giong nhau

tokens = ["mèo", "ăn", "cá", "ngon"]
L = len(tokens)       # do dai cau = 4
d_model = 16          # chieu embedding moi token
d_k = 8               # chieu cua Q/K/V o single-head

# "Embedding" gia lap bang so ngau nhien - muc tieu toi nay la CO CHE
# attention, khong phai embedding (xem Q1 cua Day 04 de hieu vi sao
# embedding ngau nhien van dung duoc o day).
X = torch.randn(L, d_model)      # (4, 16)

# 3 ma tran chieu cho single-head (Bai 1 & 3): 16 -> 8.
Wq = torch.randn(d_model, d_k)
Wk = torch.randn(d_model, d_k)
Wv = torch.randn(d_model, d_k)


def causal_mask(n):
    """Ma tran mask tam giac tren cho causal attention.

    Tra ve mask bool shape (n, n) voi True o cac o (i, j) ma j > i,
    tuc la "token i khong duoc nhin token j dung sau no" (khong nhin
    tuong lai). Dung voi masked_fill(mask, -inf) TRUOC khi softmax.

    Args:
        n: do dai cau.

    Returns:
        Tensor bool shape (n, n).
    """
    return torch.triu(torch.ones(n, n), diagonal=1).bool()


# ---------------------------------------------------------------------------
# Bai 1 - Single-head scaled dot-product attention
# ---------------------------------------------------------------------------
def bai1():
    """Chay single-head attention tren cau 4 token, in ma tran attention."""
    # Q: truy van - moi token hoi "toi can gi?"
    # K: khoa     - moi token quang cao "toi co gi?"
    # V: gia tri  - noi dung that moi token mang theo.
    Q = X @ Wq  # (4, 8)
    K = X @ Wk  # (4, 8)
    V = X @ Wv  # (4, 8)

    # TODO 1: diem tuong dong + chia scale.
    # scores[i, j] = do hop nhau giua token i (hoi) va token j (dap).
    # Chia cho sqrt(d_k): tong cua d_k so hang co phuong sai ~d_k, khong
    # chia thi softmax bao hoa -> attention "chet cung" (xem Q1, Bay 1).
    scores = (Q @ K.T) / math.sqrt(d_k)  # (4, 4)

    # TODO 2: causal mask - gan -inf vao o (i, j) voi j > i, TRUOC softmax.
    # -inf qua softmax -> 0: token khong the lay trong so tu tuong lai.
    # (Mask SAU softmax la Bay 2: hang khong con cong bang 1.)
    scores = scores.masked_fill(causal_mask(L), float("-inf"))

    # TODO 3: softmax theo hang (dim=-1) thanh phan phoi, roi tron V.
    # weights[i, j] = token i lay bao nhieu % thong tin tu token j.
    weights = F.softmax(scores, dim=-1)  # (4, 4)
    out = weights @ V                    # (4, 8)

    # Kiem tra (giam khao tu dong - dung sua cac dong nay).
    assert weights.shape == (L, L), "sai shape weights"
    assert torch.allclose(weights.sum(dim=-1), torch.ones(L), atol=1e-5), \
        "moi hang attention phai cong bang 1"
    assert (weights.triu(diagonal=1).abs().max() < 1e-6), \
        "causal mask hong: token dang nhin trom tuong lai!"
    print("Bai 1 qua kiem tra!")

    # In ma tran attention: hang = token dang hoi, cot = token duoc nhin.
    # Nua tren duong cheo phai toan 0 (causal). "meo" chi nhin chinh no,
    # "ngon" duoc nhin ca 4 token.
    print("       ", "  ".join(f"{t:>6}" for t in tokens))
    for i, t in enumerate(tokens):
        print(f"{t:>6}", "  ".join(f"{w:6.2f}" for w in weights[i]))


# ---------------------------------------------------------------------------
# Bai 2 - Multi-head attention (4 heads)
# ---------------------------------------------------------------------------
def bai2():
    """Chay 4 heads song song roi ghep lai, in pattern nhin cua moi head."""
    n_heads = 4
    # [TRAINER] Guard 2026-10-15: d_model phai chia het cho n_heads - moi
    # head nhan duoc d_model/n_heads chieu nguyen ven. Neu khong chia het,
    # .view(L, n_heads, d_k4) se crash voi thong bao kho hieu. That bai SOM
    # voi thong bao ro rang de debug hon (giong guard d chan o Day 04).
    assert d_model % n_heads == 0, \
        f"d_model={d_model} phai chia het cho n_heads={n_heads}"
    d_k4 = d_model // n_heads  # = 4: moi head nhin 1 "lat" 4 chieu

    # 3 ma tran chieu 16->16 (giong kich thuoc nhu single-head d_k=16) + Wo
    # chieu nguoc ve d_model. Chia heads chi la CACH TO CHUC LAI phep tinh,
    # khong them tham so (xem Q3).
    Wq4 = torch.randn(d_model, d_model)
    Wk4 = torch.randn(d_model, d_model)
    Wv4 = torch.randn(d_model, d_model)
    Wo = torch.randn(d_model, d_model)

    Q4 = X @ Wq4  # (4, 16)
    K4 = X @ Wk4
    V4 = X @ Wv4

    # TODO 4: tach heads.
    # (L, d_model) -> view thanh (L, n_heads, d_k) -> transpose 2 chieu dau
    # -> (n_heads, L, d_k). Thu tu nay quan trong (Bay 3): view TRUOC,
    # transpose SAU; luon kiem tra shape sau moi buoc reshape.
    Qh = Q4.view(L, n_heads, d_k4).transpose(0, 1)  # (4, 4, 4)
    Kh = K4.view(L, n_heads, d_k4).transpose(0, 1)
    Vh = V4.view(L, n_heads, d_k4).transpose(0, 1)

    # TODO 5: attention cho ca 4 heads cung luc (them chieu head).
    # Kh.transpose(-2, -1): doi 2 chieu CUOI (L, d_k) -> (d_k, L).
    # mask (4,4) broadcast len (4,4,4) theo chieu head - dung nhu nhau.
    scores_h = (Qh @ Kh.transpose(-2, -1)) / math.sqrt(d_k4)  # (4, 4, 4)
    scores_h = scores_h.masked_fill(causal_mask(L), float("-inf"))
    weights_h = F.softmax(scores_h, dim=-1)  # (4, 4, 4)
    head_out = weights_h @ Vh                # (4, 4, 4)

    # TODO 6: ghep heads + chieu output.
    # (n_heads, L, d_k) -> transpose -> (L, n_heads, d_k) -> duoi thang
    # (L, d_model): noi 4 lat 4 chieu thanh vector 16 chieu -> @ Wo.
    # [TRAINER] Luu y 2026-10-15 (khong doi code): sau transpose tensor
    # khong con "contiguous" trong bo nho - dung .view() o day se LOI ngay,
    # phai dung .reshape() (tu chon duong an toan, copy khi can). Quy tac
    # nho: reshape sau transpose/permute; view chi khi chac chan contiguous
    # (nhu o TODO 4, Q4 vua ra tu phep @ nen contiguous).
    concat = head_out.transpose(0, 1).reshape(L, d_model)  # (4, 16)
    final = concat @ Wo                                   # (4, 16)

    # Kiem tra.
    assert final.shape == (L, d_model), "output phai ve dung d_model"
    assert torch.allclose(weights_h.sum(dim=-1),
                          torch.ones(n_heads, L), atol=1e-5)
    print("Bai 2 qua kiem tra! final shape:", tuple(final.shape))

    # So sanh: 4 heads co "nhin" khac nhau khong? (token "ngon" = index 3)
    for h in range(n_heads):
        cells = [f"{tokens[j]}:{weights_h[h, 3, j]:.2f}" for j in range(L)]
        print(f"--- head {h}: token 'ngon' nhin ->", cells)


# ---------------------------------------------------------------------------
# Bai 3 - Do do phuc tap: vi sao context dai "dot dien"
# ---------------------------------------------------------------------------
def single_head(x):
    """Single-head attention (goi code Bai 1) cho cau do dai bat ky.

    Args:
        x: tensor (Lx, d_model).

    Returns:
        Tensor (Lx, d_k): output sau attention.
    """
    Lx = x.shape[0]
    Qx, Kx, Vx = x @ Wq, x @ Wk, x @ Wv  # dung Wq/Wk/Wv cua Bai 1 (d_k=8)
    s = (Qx @ Kx.T) / math.sqrt(d_k)     # (Lx, Lx): ma tran diem Lx^2 o
    s = s.masked_fill(causal_mask(Lx), float("-inf"))
    return F.softmax(s, dim=-1) @ Vx


def bai3():
    """Do thoi gian single_head voi L tang dan, in bang + tra loi TODO 7."""
    print(f"{'L':>6}  {'thoi gian':>12}")
    prev_dt, prev_lx = None, None
    for Lx in [64, 128, 256, 512]:
        x = torch.randn(Lx, d_model)
        # [TRAINER] Do benchmark dung cach 2026-10-15 (thay cach do 1 lan):
        # (1) warmup 1 lan truoc - lan chay dau chiu moi chi phi mot lan
        #     (cap phat bo nho, cache lanh) nen cham bat thuong;
        # (2) do 5 lan lay MIN thay vi 1 lan - he dieu hanh co the xen vao
        #     bat cu luc nao (chuyen process, ngat), lay min loai duoc cac
        #     mau bi nhieu. Day la cach chuan khi do perf trong ML.
        # (3) ky vong thuc te: voi d_model=16 nho xiu tren CPU, chi phi co
        #     dinh (goi ham, chia thread) lan at phan tinh toan nen ti le
        #     do duoc se khong dep nhu 4.0x (vd 1.5x/2.1x/7.1x) - nhung duong
        #     cong van SIEU TUYEN TINH ro rang. Muon thay 4x sach, phai do o
        #     vung "compute-bound": d_model lon (vd 512) hoac tren GPU.
        _ = single_head(x)  # warmup
        dts = []
        for _ in range(5):
            t0 = time.perf_counter()
            _ = single_head(x)
            dts.append(time.perf_counter() - t0)
        dt = min(dts)
        ratio = f"(gap {dt / prev_dt:.1f}x khi L gap {Lx / prev_lx:.0f}x)" \
            if prev_dt else ""
        print(f"{Lx:>6}  {dt * 1000:10.2f} ms  {ratio}")
        prev_dt, prev_lx = dt, Lx

    # TODO 7 (tra loi viet tay, dua vao bang so vua do):
    # (1) L tang gap doi -> thoi gian tang khoang GAP 4 LAN
    #     (gap 2 binh phuong).
    # (2) Do phuc tap cua attention theo L la O(L^2): ma tran scores co LxL o,
    #     moi o la 1 phep tinh - context dai gap 8x (4k -> 32k token) thi
    #     attention ton gap 64x phep tinh (va 64x dien nang). Day la dong luc
    #     cua FlashAttention, GQA va KV-cache.


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    """Chay tuan tu Bai 1 -> 2 -> 3 (dung di tiep khi assert con do)."""
    print("=" * 64)
    print("Bai 1 - Single-head scaled dot-product attention")
    print("=" * 64)
    bai1()
    print("\n" + "=" * 64)
    print("Bai 2 - Multi-head attention (4 heads)")
    print("=" * 64)
    bai2()
    print("\n" + "=" * 64)
    print("Bai 3 - Do do phuc tap theo L")
    print("=" * 64)
    bai3()
    print("\n" + "=" * 64)
    print("XONG - Checklist: Bai 1 (3 assert + ma tran 4x4) OK;")
    print("        Bai 2 (assert + final (4,16) + 4 heads) OK;")
    print("        Bai 3 (bang thoi gian + TODO 7) OK.")
    print("=" * 64)


if __name__ == "__main__":
    main()
