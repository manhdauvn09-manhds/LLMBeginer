#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Bai 0 Day 18: so tan tay NF4 - vi sao 16 muc chia theo phan
phoi chuan lai thang int4.

Y tuong: trong so that tap trung quanh 0, nen muc luong tu phai
DAY quanh 0, THUA o duoi (NF4). int4 chia deu theo khoang gia
tri, khong quan tam phan phoi.

NF4 o day dung ban don gian hoa theo de bai: 16 khoang co xac
suat bang nhau cua phan phoi chuan, muc dai dien la trung vi
cua moi khoang (quantile tai (i+0.5)/16), chuan hoa ve [-1, 1]
de khop mien gia tri sau khi chia block cho scale.

Thi nghiem:
  (a) w ~ N(0, 1): ky vong NF4 thang (MSE thap hon int4).
  (b) w ~ U(0, 1) (torch.rand nhu de bai): NF4 mat loi the vi
      du lieu khong con dang chuan - muc day quanh 0 bi lang phi.

Cung tinh so hoc double quantization: moi block 64 trong so
can 1 scale fp32 (4 byte); nen scale xuong 8-bit (1 byte) thi
tiet kiem bao nhieu bit moi tham so.

Cach chay (CPU, khong can GPU):
    ~/workspace/venvs/day15/bin/python nf4_lab.py
"""

import torch

BLOCK_SIZE = 64
N = 100_000


def nf4_centers():
    """16 muc NF4: trung vi cua 16 khoang deu xac suat N(0, 1).

    Vi du: khoang dau tien (-inf, q(1/16)) co trung vi la
    q(1/32) ~ -1.86. Chuan hoa bang cach chia cho muc lon nhat
    de muc cuc dai bang +-1 (khop mien [-1, 1] sau chuan hoa
    block). Ket qua: muc day quanh 0, thua o duoi - dung dang
    cua bang NF4 chuan (co muc +-1.0 o hai dau).
    """
    normal = torch.distributions.Normal(0, 1)
    probs = torch.tensor([(i + 0.5) / 16 for i in range(16)])
    centers = normal.icdf(probs)
    centers = centers / centers.abs().max()
    return centers


def quantize_dequantize(w, centers, block_size=BLOCK_SIZE):
    """Luong tu hoa roi giai luong tu theo block.

    Moi block: scale = max(abs(block)), chuan hoa block ve
    [-1, 1], gan moi so vao muc centers gan nhat, roi nhan lai
    scale. Tra ve tensor cung do dai voi w.

    Args:
        w: tensor 1 chieu (trong so gia lap).
        centers: tensor 16 muc trong [-1, 1].
        block_size: so phan tu moi block (mac dinh 64).

    Raises:
        ValueError: neu w rong hoac block_size <= 0.
    """
    if w.numel() == 0:
        raise ValueError("w khong duoc rong.")
    if block_size <= 0:
        raise ValueError("block_size phai > 0.")
    # Dem lot block cuoi cho du block_size roi cat bo sau.
    n_blocks = (w.numel() + block_size - 1) // block_size
    padded = torch.zeros(n_blocks * block_size, dtype=w.dtype)
    padded[:w.numel()] = w
    blocks = padded.reshape(n_blocks, block_size)
    scale = blocks.abs().max(dim=1, keepdim=True).values
    # Block toan 0 co scale = 0: giu nguyen 0, tranh chia cho 0.
    safe_scale = torch.where(scale == 0, torch.ones_like(scale), scale)
    normalized = blocks / safe_scale
    # Khoang cach toi tung muc, chon muc gan nhat.
    dist = (normalized.unsqueeze(-1) - centers).abs()
    nearest = dist.argmin(dim=-1)
    dequant = centers[nearest] * scale
    return dequant.reshape(-1)[:w.numel()]


def quantize_dequantize_int4(w, block_size=BLOCK_SIZE):
    """Doi chung: int4 chia deu theo khoang gia tri.

    16 muc nguyen deu nhau: round(block/scale*7), clamp(-8, 7),
    giai luong tu bang q/7*scale (dung code mau cua de bai,
    chi viet vector hoa cho nhanh).
    """
    if w.numel() == 0:
        raise ValueError("w khong duoc rong.")
    n_blocks = (w.numel() + block_size - 1) // block_size
    padded = torch.zeros(n_blocks * block_size, dtype=w.dtype)
    padded[:w.numel()] = w
    blocks = padded.reshape(n_blocks, block_size)
    scale = blocks.abs().max(dim=1, keepdim=True).values
    safe_scale = torch.where(scale == 0, torch.ones_like(scale), scale)
    q = torch.round(blocks / safe_scale * 7).clamp(-8, 7)
    dequant = q / 7 * scale
    return dequant.reshape(-1)[:w.numel()]


def mse(a, b):
    """Sai so binh phuong trung binh giua goc va tai tao."""
    return ((a - b) ** 2).mean().item()


def double_quant_math(n_params, block_size=BLOCK_SIZE):
    """So hoc double quantization cho n_params tham so.

    Khong double quant: moi tham so ton 4 bit (trong so) +
    32/64 = 0.5 bit (scale fp32) = 4.5 bit/tham so.
    Co double quant: scale nen xuong 8-bit: 4 + 8/64 = 4.125
    bit/tham so (~4.13). Tiet kiem 0.375 bit/tham so.
    """
    n_scales = (n_params + block_size - 1) // block_size
    fp32_bytes = n_scales * 4
    int8_bytes = n_scales * 1
    saved_bits_per_param = (fp32_bytes - int8_bytes) * 8 / n_params
    print(f"  {n_params:,} tham so -> {n_scales:,} scale "
          f"(block {block_size}).")
    print(f"  Scale fp32: {fp32_bytes:,} byte | scale int8: "
          f"{int8_bytes:,} byte | tiet kiem: "
          f"{fp32_bytes - int8_bytes:,} byte.")
    print(f"  Bit/tham so: khong double quant = "
          f"{4 + 32 / block_size:.3f} | co double quant = "
          f"{4 + 8 / block_size:.3f} | tiet kiem = "
          f"{saved_bits_per_param:.3f} bit/tham so.")


def main():
    """Chay thi nghiem NF4 vs int4 tren 2 phan phoi."""
    torch.manual_seed(42)
    centers = nf4_centers()
    print("16 muc NF4 (chuan hoa ve [-1, 1]):")
    print(" ", [round(c, 3) for c in centers.tolist()])
    print("Nhan xet: muc day quanh 0, thua dan ve 2 duoi.")
    # (a) Phan phoi chuan - dang that cua trong so.
    w = torch.randn(N)
    w_nf4 = quantize_dequantize(w, centers)
    w_int4 = quantize_dequantize_int4(w)
    print(f"\n(a) w ~ N(0, 1), {N:,} mau, block {BLOCK_SIZE}:")
    print(f"  MSE NF4 : {mse(w, w_nf4):.6f}")
    print(f"  MSE int4: {mse(w, w_int4):.6f}")
    print(f"  NF4 thang int4: {mse(w, w_nf4) < mse(w, w_int4)} "
          f"(ti le MSE: {mse(w, w_int4) / mse(w, w_nf4):.2f}x).")
    # (b) Phan phoi deu - NF4 mat gia dinh phan phoi chuan.
    w_uniform = torch.rand(N)
    wu_nf4 = quantize_dequantize(w_uniform, centers)
    wu_int4 = quantize_dequantize_int4(w_uniform)
    print(f"\n(b) w ~ U(0, 1) (torch.rand nhu de bai), {N:,} mau:")
    print(f"  MSE NF4 : {mse(w_uniform, wu_nf4):.6f}")
    print(f"  MSE int4: {mse(w_uniform, wu_int4):.6f}")
    nf4_wins = mse(w_uniform, wu_nf4) < mse(w_uniform, wu_int4)
    print(f"  NF4 thang int4: {nf4_wins} "
          f"(ti le MSE: "
          f"{mse(w_uniform, wu_int4) / mse(w_uniform, wu_nf4):.2f}x).")
    print("  Giai thich: du lieu deu khong tap trung quanh 0 nen "
          "muc day quanh 0 cua NF4 bi lang phi; chia deu (int4) "
          "moi la toi uu.")
    # Edge case: tensor rong va block toan 0.
    try:
        quantize_dequantize(torch.tensor([]), centers)
    except ValueError as exc:
        print(f"\nEdge case tensor rong: chan dung ({exc}).")
    zero = torch.zeros(128)
    out_zero = quantize_dequantize(zero, centers)
    print(f"Edge case block toan 0: tai tao dung 0? "
          f"{bool((out_zero == 0).all())} (khong chia cho 0).")
    print("\nDouble quantization (so hoc tay):")
    double_quant_math(100_000)
    # [TRAINER] Sua so tham so GPT-2: 124.587.264 la tong cua Day 17
    # khi DA gan adapter LoRA r=4 (124.439.808 cua base + 147.456
    # adapter). Dem truc tiep tu model: GPT-2 base co 124.439.808
    # tham so. So scale doi theo: 1.944.372 scale, tiet kiem double
    # quant 5,83 MB (thay vi 5,84) - ket luan ~64,2 MB khong doi.
    print("\nAp dung cho GPT-2 (124.439.808 params, de doi chieu):")
    double_quant_math(124_439_808)
    print("  Trong so 4-bit: ~62,2 MB + scale int8 ~1,9 MB = "
          "~64,2 MB (so voi fp16 ~249 MB, fp32 ~498 MB).")


if __name__ == "__main__":
    main()
