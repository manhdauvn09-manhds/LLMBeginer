#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Day 04 - Buoi toi: DUNG EMBEDDING + MA HOA VI TRI BANG TAY
==========================================================
Muc dich:
    A1: Hieu nn.Embedding(V, d) chi la "tra cuu bang", va do tuong dong
        cosine do duoc y nghia ngu nghia (phep tinh "vua - dan ong + phu nu").
    A2: Lap trinh sinusoidal position encoding tu cong thuc, kiem chung
        tinh duy nhat va tinh "PE(pos+k) la ham tuyen tinh cua PE(pos)".
    A3: Lap trinh RoPE (xoay tung cap chieu theo vi tri), kiem chung tich
        vo huong chi phu thuoc khoang cach TUONG DOI.
    A4: Lap trinh RMSNorm va tinh "gia phai tra" cua embedding theo VRAM.

Cach chay:
    python embedding_pe.py
    # Ky vong: in ket qua + kiem chung ca 4 bai, khong bao loi assert.

Chi can PyTorch (CPU la du), khong can GPU.
"""

import torch
from torch import nn

# ---------------------------------------------------------------------------
# A1 - Embedding lookup & cosine similarity
# ---------------------------------------------------------------------------
WORDS = ["vua", "nữ_hoàng", "đàn_ông", "phụ_nữ", "mèo", "ô_tô"]


def cos_sim(u, v):
    """Do tuong dong cosine trong [-1, 1]: goc cang nho -> cang gan 1.

    Cong thuc: cos(u,v) = (u.v) / (||u|| * ||v||).
    Chia cho norm de do GOC (huong) chu khong do do dai vector - 2 vector
    cung huong nhung dai ngan khac nhau van cho cos = 1.
    Doc nhanh: 1 = cung huong (giong nhau), 0 = vuong goc (khong lien quan),
    -1 = nguoc huong (doi lap).
    """
    # [TRAINER] Luu y 2026-10-14 (khong doi code): neu u hoac v la vector 0
    # thi mau so = 0 -> ket qua NaN. Trong bai nay khong xay ra vi embedding
    # khoi tao ngau nhien (xac suat ra dung vector 0 coi nhu = 0). Neu tai
    # dung ham nay o noi khac (vd so sanh vector sau khi mask), hay cong eps
    # vao mau so hoac kiem tra norm > 0 truoc khi chia.
    return torch.dot(u, v) / (torch.norm(u) * torch.norm(v))


def bai_a1():
    print("=" * 64)
    print("A1 - Embedding lookup & cosine similarity")
    print("=" * 64)
    torch.manual_seed(4)  # co dinh seed: ket qua tai lap duoc (Bay 1)

    # nn.Embedding(V, d): don gian la 1 BANG TRA CUU kich thuoc (V, d).
    # emb.weight[i] = vector cua tu thu i. Chua train gi ca - vector ngau
    # nhien van minh hoa duoc Y TUONG (nhung chua co y nghia ngu nghia).
    emb = nn.Embedding(len(WORDS), 8)
    vecs = emb.weight  # shape (6, 8): 6 tu, moi tu 1 vector 8 chieu

    # Minh hoa "tra cuu bang": muon vector cua tu "vua" (index 0) thi go
    # emb(torch.tensor([0])) - ket qua phai DUNG BANG hang 0 cua bang.
    lookup = emb(torch.tensor([0]))[0]
    print(f"  Demo lookup: emb([0]) co phai vector cua 'vua' khong? "
          f"{torch.equal(lookup, vecs[0])} (day chinh la 'tra cuu bang')")

    print("\nDo tuong dong cosine tung cap tu:")
    best, worst = (None, -2.0), (None, 2.0)  # (cap tu, gia tri)
    for i in range(len(WORDS)):
        for j in range(i + 1, len(WORDS)):
            s = cos_sim(vecs[i], vecs[j]).item()
            print(f"  {WORDS[i]:<8} - {WORDS[j]:<8}: {s:+.4f}")
            if s > best[1]:
                best = ((WORDS[i], WORDS[j]), s)
            if s < worst[1]:
                worst = ((WORDS[i], WORDS[j]), s)
    print(f"\n  Cap GIONG nhau nhat : {best[0][0]} - {best[0][1]} ({best[1]:+.4f})")
    print(f"  Cap KHAC nhau nhat : {worst[0][0]} - {worst[0][1]} ({worst[1]:+.4f})")

    # Phep tinh so hoc tu: "vua - dan ong + phu nu = ?"
    # (kinh dien tu word2vec; o day embedding CHUA TRAIN nen dung CHO DOI
    #  no ra "nu_hoang" - chinh diem can nhan ra o cau Q1.)
    target = vecs[0] - vecs[2] + vecs[3]
    nearest, nearest_s = None, -2.0
    for i, w in enumerate(WORDS):
        s = cos_sim(target, vecs[i]).item()
        if s > nearest_s:
            nearest, nearest_s = w, s
    print(f"\n  v(vua) - v(dan_ong) + v(phu_nu) gan nhat voi: "
          f"'{nearest}' ({nearest_s:+.4f})")
    print("  (Embedding ngau nhien chua train -> ket qua la 1 tu bat ky.\n"
          "   Quan he ngu nghia chi XUAT HIEN sau khi train - xem Q1.)")


# ---------------------------------------------------------------------------
# A2 - Sinusoidal position encoding (lap trinh tu cong thuc)
# ---------------------------------------------------------------------------
def sinusoidal_pe(max_len, d):
    """Tra ve tensor (max_len, d) theo cong thuc:
        PE(pos, 2i)   = sin(pos / 10000^(2i/d))
        PE(pos, 2i+1) = cos(pos / 10000^(2i/d))

    Y tuong: moi cap chieu (2i, 2i+1) la 1 "kim dong ho" quay voi tan so
    rieng (chieu i cang cao -> quay cang cham). To hop d/2 kim voi tan so
    khac nhau cho moi vi tri 1 "dau van tay" duy nhat.
    Chi dung arange/sin/cos - khong dung thu vien co san.

    Vi du cu the (d=4): i=[0,1] -> div=[10000^0, 10000^(2/4)]=[1, 100].
    Voi pos=1: angle=[1/1, 1/100]=[1.0, 0.01] ->
      PE(1) = [sin(1.0), cos(1.0), sin(0.01), cos(0.01)]
    Thay ngay: cap dau quay nhanh (tan so cao), cap sau quay cham.
    """
    # [TRAINER] Guard 2026-10-14: d PHAI CHAN - moi cap chieu (2i, 2i+1) can
    # dung 1 tan so. Neu d le, pe[:, 0::2] se co nhieu cot hon angle dan den
    # loi shape kho hieu. That bai SOM voi thong bao ro rang de debug hon.
    assert d % 2 == 0, f"sinusoidal_pe yeu cau d chan, nhan d={d}"
    pos = torch.arange(max_len).unsqueeze(1)   # (L,1): cot vi tri 0..L-1
    i = torch.arange(d // 2)                   # (d/2,): chi so cap chieu
    div = 10000 ** (2 * i / d)                 # (d/2,): mau so 10000^(2i/d)
    angle = pos / div                          # (L, d/2): broadcast cot x hang
    pe = torch.zeros(max_len, d)
    pe[:, 0::2] = torch.sin(angle)             # cot chan = sin
    pe[:, 1::2] = torch.cos(angle)             # cot le  = cos
    return pe


def bai_a2():
    print("\n" + "=" * 64)
    print("A2 - Sinusoidal position encoding")
    print("=" * 64)
    d, max_len = 64, 100
    pe = sinusoidal_pe(max_len, d)
    print(f"  pe.shape = {tuple(pe.shape)}")

    # Kiem chung 1 - tinh duy nhat: khong 2 vi tri nao trung vector PE.
    # torch.cdist(pe, pe)[a, b] = khoang cach Euclid giua PE(a) va PE(b).
    # Duong cheo = 0 (tu so voi chinh no) -> chi xet ngoai duong cheo.
    # [TRAINER] Nhan xet mo rong 2026-10-14 (khong doi code): cdist o day tao
    # ma tran 100x100 = 10k so, re. Nhung dung cach nay voi max_len that
    # (vd 128k) thi ma tran LxL can ~64GB RAM - khong kha thi. Tinh duy nhat
    # cua PE duoc chung minh bang ly thuyet (cac tan so vo ti voi nhau), nen
    # o quy mo that khong ai kiem chung bang "suc manh co bap" nhu nay.
    dists = torch.cdist(pe, pe)
    mask = ~torch.eye(max_len, dtype=torch.bool)  # True o moi o ngoai cheo
    min_off_diag = dists[mask].min().item()
    print(f"  Khoang cach NHO NHAT giua 2 vi tri khac nhau: {min_off_diag:.6f}")
    assert min_off_diag > 1e-6, "Co 2 vi tri trung PE!"
    print("  => Khong vi tri nao trung nhau (tinh duy nhat OK).")

    # Kiem chung 2 - tinh tuyen tinh cua phep dich:
    # PE(pos+k) ~ PE(pos) @ M  (M la ma tran xoay khoi, ton tai that vi
    # moi cap (sin,cos) dich k don vi chinh la phep xoay goc k*omega).
    # Giai M bang binh phuong toi thieu: A @ M ~ B.
    k = 7
    A, B = pe[:-k], pe[k:]                      # (L-k, d) moi ben
    M = torch.linalg.lstsq(A, B).solution       # (d, d)
    err = (A @ M - B).abs().max().item()
    print(f"  Sai so lon nhat cua PE(pos+{k}) ~ PE(pos) @ M: {err:.2e}")
    assert err < 1e-4, "Tinh tuyen tinh cua phep dich KHONG dung!"
    print("  => Dich k vi tri la phep bien doi TUYEN TINH cua vector PE.\n"
          "     Nho vay model hoc duoc KHOANG CACH tuong doi giua cac token.")


# ---------------------------------------------------------------------------
# A3 - RoPE: xoay tung cap chieu theo vi tri
# ---------------------------------------------------------------------------
def apply_rope(x, pos, omegas):
    """Xoay moi cap chieu (x[2i], x[2i+1]) mot goc theta = pos * omega_i.

    x: (d,), omegas: (d//2,). Ma tran xoay 2x2:
        [x'] = [cos t  -sin t] [x]
        [y']   [sin t   cos t] [y]

    Vi du cu the: x=[1,2,3,4], omegas=[w0,w1], pos=p ->
      cap (1,2) xoay goc p*w0, cap (3,4) xoay goc p*w1.
    Moi cap co omega rieng -> goc xoay rieng (giong cac "kim dong ho"
    quay toc do khac nhau o A2).
    """
    x_even = x[0::2]          # cac chieu chan:  x0, x2, x4, ...
    x_odd = x[1::2]           # cac chieu le :  x1, x3, x5, ...
    theta = pos * omegas      # (d/2,): goc xoay rieng tung cap
    cos_t, sin_t = torch.cos(theta), torch.sin(theta)
    new_even = x_even * cos_t - x_odd * sin_t
    new_odd = x_even * sin_t + x_odd * cos_t
    # Ghep lai dung thu tu xen ke. torch.stack(..., dim=-1) tao ra
    # [[e0',o0'], [e1',o1'], ...] roi .reshape(-1) duoi thang thanh
    # [e0',o0',e1',o1',...] - dung thu tu ban dau cua x.
    # (Bay 2: dao thu tu o day la loi pho bien nhat - test pos=0 phai tra
    # ve dung vector goc vi goc xoay = 0.)
    # [TRAINER] Guard 2026-10-14: so chieu PHAI CHAN de chia cap (chan, le)
    # deu nhau (giong guard o sinusoidal_pe). Neu le, new_even dai hon
    # new_odd 1 phan tu -> stack() crash voi thong bao shape kho doc.
    assert x.numel() % 2 == 0, \
        f"apply_rope yeu cau so chieu chan, nhan {x.numel()}"
    return torch.stack([new_even, new_odd], dim=-1).reshape(-1)


def bai_a3():
    print("\n" + "=" * 64)
    print("A3 - RoPE (rotary position embedding)")
    print("=" * 64)
    torch.manual_seed(4)
    d = 8
    # Tan so goc tung cap - giong cong thuc sinusoidal o A2.
    omegas = 1.0 / (10000 ** (torch.arange(d // 2) * 2.0 / d))

    q = torch.randn(d)
    k = torch.randn(d)

    # Sanity check (Bay 2): pos=0 -> goc xoay = 0 -> phai tra ve vector goc.
    assert torch.allclose(apply_rope(q, 0, omegas), q, atol=1e-6)
    print("  Sanity: RoPE(pos=0) tra ve dung vector goc (OK).")

    # Kiem chung tinh TUONG DOI: dich ca q va k di +100 vi tri,
    # tich vo huong phai GIU NGUYEN (chi phu thuoc pos_q - pos_k = 6).
    dot1 = torch.dot(apply_rope(q, 10, omegas), apply_rope(k, 4, omegas))
    dot2 = torch.dot(apply_rope(q, 110, omegas), apply_rope(k, 104, omegas))
    diff = abs(dot1 - dot2).item()
    print(f"  dot(rope(q,10),  rope(k,4))   = {dot1.item():+.6f}")
    print(f"  dot(rope(q,110), rope(k,104)) = {dot2.item():+.6f}")
    print(f"  |chenh lech| = {diff:.2e}")
    assert diff < 1e-5, "RoPE mat tinh tuong doi - kiem tra cong thuc xoay!"
    print("  => Dich ca 2 vi tri di +100 ma tich vo huong KHONG DOI.")
    # Giai thich (1-2 cau, theo yeu cau de bai):
    # Phep xoay bao toan tich vo huong theo goc LECH (pos_q - pos_k); dich
    # dong deu ca 2 pos giu nguyen goc lech do. Nho vay attention score giua
    # q va k chi "cam nhan" duoc KHOANG CACH tuong doi giua 2 token, model
    # hieu duoc THU TU tu ma khong can biet vi tri tuyet doi cua chung.


# ---------------------------------------------------------------------------
# A4 - RMSNorm + tinh VRAM cho embedding
# ---------------------------------------------------------------------------
def rmsnorm(x, gamma, eps=1e-6):
    """RMSNorm(x) = x / sqrt(mean(x^2) + eps) * gamma.

    Khac LayerNorm o cho: KHONG tru mean, chi chia cho RMS (can bac 2 trung
    binh). Re hon, va du tot cho LLM (Llama dung RMSNorm). gamma hoc duoc,
    khoi tao toan 1 (luc dau = phep chuan hoa thuan tuy).
    """
    # [TRAINER] Cai tien 2026-10-14: mean(-1, keepdim=True) thay cho mean().
    # Voi vector 1D (demo nay) ket qua GIONG HET 100%, nhung ban cu se SAI
    # LANG LE neu ai do tai dung ham cho batch 2D (no se chuan hoa ca batch
    # chung lai voi nhau thay vi chuan hoa tung token doc lap). Chuan hoa
    # luon phai theo chieu dac trung (chieu cuoi, moi token 1 RMS rieng) -
    # day la loi rat pho bien khi copy code norm tu demo sang model that.
    return x / torch.sqrt(x.pow(2).mean(-1, keepdim=True) + eps) * gamma


def bai_a4():
    print("\n" + "=" * 64)
    print("A4 - RMSNorm + tinh VRAM cho embedding")
    print("=" * 64)
    torch.manual_seed(4)
    d = 64
    x = torch.randn(d)
    gamma = torch.ones(d)  # khoi tao gamma = 1
    y = rmsnorm(x, gamma)
    ms = y.pow(2).mean().item()
    print(f"  mean(y^2) sau RMSNorm = {ms:.6f} (ky vong ~1)")
    assert abs(ms - 1.0) < 1e-5, "RMSNorm sai cong thuc!"
    print("  => Vector ra co RMS ~ 1 (chuan hoa OK).")

    # Tinh VRAM cho bang embedding cua model gia dinh:
    # V=128k token, d=4096, bf16 (2 bytes). Adam luu them m va v - moi cai
    # bang dung kich thuoc weights -> tong x3 (Bay 3: quen x3 la sai to).
    # Tinh nham bang tay: 128k x 4096 = 524 trieu param (~524M);
    #   weights bf16: 524M x 2B ~ 1 GB; + Adam (m, v): x3 ~ 3 GB.
    V, dim, bytes_per = 128_000, 4096, 2
    n_params = V * dim
    vram_gb = n_params * bytes_per * 3 / 1024**3
    print(f"\n  So param bang embedding: {n_params:,} (~{n_params/1e6:.0f}M)")
    print(f"  VRAM embedding = {V:,} x {dim} x {bytes_per}B x 3 (Adam) "
          f"= {vram_gb:.2f} GB")
    print("  Nhan xet: chi RIENG bang embedding da ngot ~3 GB (~1/8 VRAM cua\n"
          "  GPU 24GB) - chua tinh cac layer Transformer, activation va\n"
          "  gradient. Train model co lon tren 1 GPU 24GB la khong kha thi\n"
          "  neu khong dung cac ky thuat tiet kiem bo nho (weight tying,\n"
          "  gradient checkpointing, ZeRO, ...).")


# ---------------------------------------------------------------------------
# MAIN
# ---------------------------------------------------------------------------
def main():
    bai_a1()
    bai_a2()
    bai_a3()
    bai_a4()
    print("\n" + "=" * 64)
    print("XONG - Checklist: A1 (cosine + word arithmetic) OK;")
    print("        A2 (tinh duy nhat + dich tuyen tinh) OK;")
    print("        A3 (tinh tuong doi cua RoPE) OK;")
    print("        A4 (RMSNorm + VRAM) OK.")
    print("=" * 64)


if __name__ == "__main__":
    main()
