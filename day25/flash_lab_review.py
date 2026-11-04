#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Lab FlashAttention Day 25 (tuan 4 - Green LLM).

Cung cong thuc attention, cung dap an - chi khac
cach thuc thi: attention chuan materialize ma tran
N x N, attention theo khoi (chunked) dung online
softmax + rescale de khong bao gio tao ma tran
N x N day du, SDPA de PyTorch chon backend.

Bai 1: chunked_attention dung nhu naive/SDPA o
N = 256, fp32; dap an khong doi khi doi block.
Bai 2: do N = 512 -> 4096: bo nho ly thuyet
N x N + thoi gian naive vs chunked.
Bai 3: chuoi ngan, dtype float64, backend thuc
te, va du bao ly thuyet cho chuoi rat dai.

Quy tac thi nghiem (viet truoc khi chay): thu
duy nhat thay doi la cach thuc thi attention;
con lai giu nguyen: cung seed, cung Q/K/V,
cung dtype, cung causal, cung thiet bi.

Cach chay (venv Day 15, CPU du cho phan chinh):
    ~/workspace/venvs/day15/bin/python flash_lab.py
Tren CPU, SDPA dung backend math - van dung
dap an, nhung khong do duoc toc do FlashAttention
that (can GPU + backend flash).
"""

import math
import statistics
import time

import torch
import torch.nn.functional as F


# Hang so cho ca lab: doi 1 hang so la doi luon
# thi nghiem, nen moi con so deu co ten o day.
SEED = 0
D_HEAD = 64
N_KIEM_CHUNG = 256
BLOCK_MAC_DINH = 64
WARMUP = 2
REPS = 5


def make_qkv(n, d=D_HEAD):
    """Tao Q/K/V ngau nhien, shape (1, 1, n, d).

    Ba tensor tao lien tiep bang randn, doc lap
    nhau. Ham nay KHONG dat seed - main() dat
    seed truoc moi lan goi de ket qua lap lai
    duoc, va cung 1 bo Q/K/V duoc dung cho ca
    3 cach thuc thi (naive/chunked/SDPA).

    Raises:
        ValueError: n/d khong phai so nguyen >= 1.
    """
    for ten, gia_tri in (("n", n), ("d", d)):
        if isinstance(gia_tri, bool) \
                or not isinstance(gia_tri, int) \
                or gia_tri < 1:
            raise ValueError(
                f"{ten} phai la so nguyen >= 1.")
    q = torch.randn(1, 1, n, d)
    k = torch.randn(1, 1, n, d)
    v = torch.randn(1, 1, n, d)
    return q, k, v


def _kiem_tra_qkv(q, k, v, causal):
    """Kiem tra Q/K/V hop le truoc khi tinh.

    Dung chung cho ca 3 ham attention: loi vao
    sai (dtype khac nhau, device khac nhau, NaN,
    shape khong khop) phai bao ro bang ValueError
    ngay tu dau, khong de crash kho hieu trong
    phep matmul (checklist 12, 16).

    Returns:
        (n_q, n_k, d): so query, so key, chieu
        cua moi head.
    """
    for ten, x in (("q", q), ("k", k), ("v", v)):
        if not isinstance(x, torch.Tensor):
            raise ValueError(f"{ten} phai la tensor.")
        if x.dim() < 2:
            raise ValueError(f"{ten} phai co it "
                             "nhat 2 chieu (..., N, d).")
    if q.shape[:-2] != k.shape[:-2] \
            or q.shape[:-2] != v.shape[:-2]:
        raise ValueError("q/k/v phai cung shape batch.")
    if q.size(-1) != k.size(-1):
        raise ValueError("q va k phai cung chieu d.")
    # [TRAINER] Guard moi: k/v lech so key thi naive/chunked crash
    # RuntimeError kho hieu trong matmul, con SDPA tu cat bot k
    # am tham roi tra dap an sai ma khong bao loi (da kiem chung
    # bang chay that: sdpa(k N=16, v N=8) trung khop 0.00 voi
    # sdpa(k cat con 8, v N=8)).
    if k.size(-2) != v.size(-2):
        raise ValueError("k va v phai cung so key.")
    if q.dtype != k.dtype or q.dtype != v.dtype:
        raise ValueError("q/k/v phai cung dtype.")
    if not q.is_floating_point():
        raise ValueError("q/k/v phai la so thuc.")
    if q.device != k.device or q.device != v.device:
        raise ValueError("q/k/v phai cung device.")
    if q.size(-2) == 0 or k.size(-2) == 0 \
            or q.size(-1) == 0:
        raise ValueError("N va d khong duoc bang 0.")
    for ten, x in (("q", q), ("k", k), ("v", v)):
        if not bool(torch.isfinite(x).all()):
            raise ValueError(f"{ten} phai huu han.")
    if not isinstance(causal, bool):
        raise ValueError("causal phai la bool.")
    return q.size(-2), k.size(-2), q.size(-1)


def naive_attention(q, k, v, causal=True):
    """Attention chuan: materialize ma tran N x N.

    Tinh toan bo diem so QK^T / sqrt(d), softmax
    ca hang, roi nhan V. Co y tao ma tran N x N
    day du - do la cai ma FlashAttention tranh.

    Args:
        q, k, v: tensor (..., N, d) / (..., N, d_v),
            cung dtype so thuc, cung device, huu han.
        causal: True thi key o vi tri sau query
            bi cam (mask -inf).

    Returns:
        Tensor cung shape batch voi q, chieu cuoi
        bang chieu cua v.
    """
    n_q, n_k, d = _kiem_tra_qkv(q, k, v, causal)
    scores = q @ k.transpose(-2, -1) / math.sqrt(d)
    if causal:
        # Chi so tren dung device cua q (checklist
        # 3): arange mac dinh o CPU se loi khi q
        # o GPU.
        chi_query = torch.arange(
            n_q, device=q.device).unsqueeze(1)
        chi_key = torch.arange(
            n_k, device=q.device).unsqueeze(0)
        scores = scores.masked_fill(
            chi_key > chi_query, float("-inf"))
    return torch.softmax(scores, dim=-1) @ v


def chunked_attention(q, k, v, block=BLOCK_MAC_DINH,
                      causal=True):
    """Attention theo khoi voi online softmax.

    Duyet K/V theo tung khoi cot, giu 3 thong ke
    chay theo tung query: m (max chay), l (tong
    chay), acc (tu so chay). Moi khoi moi den:
    m_new = max(m, max khoi), rescale ket qua cu
    bang alpha = exp(m - m_new), roi cong don
    khoi moi. Cuoi cung tra acc / l. Khong bao
    gio tao ma tran N x N day du: moi khoi chi
    co scores (..., N, block).

    Thu tu bat buoc trong moi khoi: ap causal
    mask TRUOC khi lay max cua khoi - neu lay
    max truoc roi moi mask, vi tri tuong lai se
    lot vao max va lam lech ca hang softmax.

    Args:
        q, k, v: nhu naive_attention.
        block: so key / khoi, so nguyen >= 1.
            Block chi doi cach nhom tinh toan,
            khong duoc doi dap an (xem Bai 1).
        causal: nhu naive_attention.

    Returns:
        Tensor cung shape batch voi q, chieu cuoi
        bang chieu cua v. Hang nao khong co key
        hop le nao (chi xay ra khi toan bo hang
        bi mask) tra NaN, giong naive_attention.
    """
    n_q, n_k, d = _kiem_tra_qkv(q, k, v, causal)
    if isinstance(block, bool) \
            or not isinstance(block, int) or block < 1:
        raise ValueError("block phai la so nguyen >= 1.")
    # State chay: shape (..., N, 1) de broadcast
    # dung theo tung query; acc co chieu cua v.
    m = torch.full(q.shape[:-1] + (1,),
                   float("-inf"), dtype=q.dtype,
                   device=q.device)
    l = torch.zeros(q.shape[:-1] + (1,),
                    dtype=q.dtype, device=q.device)
    acc = torch.zeros(q.shape[:-1] + (v.size(-1),),
                      dtype=q.dtype, device=q.device)
    chi_query = torch.arange(
        n_q, device=q.device).unsqueeze(1)
    for j0 in range(0, n_k, block):
        j1 = min(j0 + block, n_k)
        k_khoi = k[..., j0:j1, :]
        v_khoi = v[..., j0:j1, :]
        scores = q @ k_khoi.transpose(-2, -1) \
            / math.sqrt(d)
        if causal:
            chi_key = torch.arange(
                j0, j1, device=q.device).unsqueeze(0)
            scores = scores.masked_fill(
                chi_key > chi_query, float("-inf"))
        # Max cua khoi, roi max chay moi.
        m_khoi = scores.amax(dim=-1, keepdim=True)
        m_moi = torch.maximum(m, m_khoi)
        # Rescale ket qua cu: alpha = exp(m - m_moi).
        # Truong hop ca m va m_moi deu -inf (hang
        # chua co key hop le nao) thi alpha = nan,
        # phai ve 0 - neu khong nan lan sang acc/l.
        alpha = torch.exp(m - m_moi)
        alpha = torch.nan_to_num(alpha, nan=0.0)
        p = torch.exp(scores - m_moi)
        # Khoi toan bo bi mask thi m_moi = -inf,
        # p phai ve 0 chu khong de nan.
        p = torch.where(torch.isfinite(m_moi), p,
                        torch.zeros_like(p))
        l = l * alpha + p.sum(dim=-1, keepdim=True)
        acc = acc * alpha + p @ v_khoi
        m = m_moi
    return acc / l


def sdpa_attention(q, k, v, causal=True):
    """Attention qua SDPA cua PyTorch.

    Mot dong goi F.scaled_dot_product_attention:
    PyTorch tu chon backend nhanh nhat phan cung
    ho tro (flash / memory-efficient / math).
    Backend thuc te phai kiem tra bang
    check_backend(), khong duoc doan - tren CPU
    no la backend math.
    """
    _kiem_tra_qkv(q, k, v, causal)
    return F.scaled_dot_product_attention(
        q, k, v, is_causal=causal)


def scores_bytes(n, dtype=torch.float32):
    """Bo nho ly thuyet cua ma tran diem N x N.

    Ma tran diem so fp32 co N x N phan tu, moi
    phan tu chiem element_size byte. Day la con
    so ly thuyet, chua ke ban sao khi backward -
    dung de thay toc do tang binh phuong cua N.

    Returns:
        So byte, so nguyen.

    Raises:
        ValueError: n khong phai so nguyen >= 1,
            dtype khong phai so thuc.
    """
    if isinstance(n, bool) or not isinstance(n, int) \
            or n < 1:
        raise ValueError("n phai la so nguyen >= 1.")
    if dtype not in (torch.float16, torch.bfloat16,
                     torch.float32, torch.float64):
        raise ValueError("dtype phai la so thuc.")
    moi_phan_tu = torch.empty(0, dtype=dtype) \
        .element_size()
    return n * n * moi_phan_tu


def bench(fn, warmup=WARMUP, reps=REPS):
    """Do thoi gian forward: TRUNG VI ms / lan.

    Lay trung vi cua tung lan chay rieng le (sau
    warmup), duoi torch.no_grad() de khong xay
    do thi autograd. Dung time.perf_counter()
    (checklist 5).

    Args:
        fn: ham khong doi so, tra tensor dau ra.
        warmup: so lan chay khoi dong bo qua.
        reps: so lan chay lay trung vi.

    Returns:
        Trung vi thoi gian, don vi mili-giay / lan.
    """
    for ten, gia_tri, toi_thieu in (
            ("warmup", warmup, 0), ("reps", reps, 1)):
        if isinstance(gia_tri, bool) \
                or not isinstance(gia_tri, int) \
                or gia_tri < toi_thieu:
            raise ValueError(
                f"{ten} phai la so nguyen >= {toi_thieu}.")
    if not callable(fn):
        raise ValueError("fn phai goi duoc.")
    with torch.no_grad():
        for _ in range(warmup):
            fn()
        # [TRAINER] Tren GPU, fn() chi day lenh vao hang doi roi
        # tra ve ngay; khong synchronize thi perf_counter do
        # thoi gian day lenh chu khong phai thoi gian kernel chay
        # xong. Tren CPU nhanh nay khong chay, so do khong doi.
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        thoi_gian = []
        for _ in range(reps):
            t0 = time.perf_counter()
            fn()
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            thoi_gian.append(time.perf_counter() - t0)
    return float(statistics.median(thoi_gian) * 1000.0)


def check_backend():
    """In va tra ve thiet bi dang dung.

    Tren CUDA: in ten GPU + 3 co backend SDPA.
    Co bat khong co nghia backend do se chay -
    PyTorch van co the roi ve math tuy dtype va
    head_dim. Tren CPU: SDPA dung backend math.

    Returns:
        "cuda" hoac "cpu".
    """
    if torch.cuda.is_available():
        print(f"CUDA: {torch.cuda.get_device_name(0)}")
        print("flash_sdp_enabled:",
              torch.backends.cuda.flash_sdp_enabled())
        print("mem_efficient_sdp_enabled:",
              torch.backends.cuda
              .mem_efficient_sdp_enabled())
        print("math_sdp_enabled:",
              torch.backends.cuda.math_sdp_enabled())
        return "cuda"
    print("Khong co CUDA: SDPA tren CPU dung "
          "backend math - van dung dap an, nhung "
          "bai do toc do FlashAttention can GPU.")
    return "cpu"


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


def _sai_so_lon_nhat(a, b):
    """Sai so tuyet doi lon nhat giua 2 tensor."""
    return float((a - b).abs().max())


def tu_kiem_tra():
    """Tu kiem tra bang assert, chay truoc lab chinh.

    Kiem tra guard, tinh dung cua chunked o nhieu
    N/block/causal (ke ca khoi le cuoi va hinh
    chu nhat), bat bien theo block, gradient, va
    2 bay cua de bai (mask sau max, quen rescale)
    bang truong hop doi khang dung tay. Chi IN
    khi assert da qua (checklist 13).
    """
    torch.manual_seed(123)
    q_nho = torch.randn(1, 1, 16, 16)
    k_nho = torch.randn(1, 1, 16, 16)
    v_nho = torch.randn(1, 1, 16, 16)
    ket_qua = [
        _expect_value_error(
            lambda: chunked_attention(
                q_nho, k_nho[:, :, :, :8], v_nho),
            "q/k lech chieu d"),
        _expect_value_error(
            lambda: chunked_attention(
                q_nho, k_nho.double(), v_nho),
            "k sai dtype"),
        # [TRAINER] Test cho guard k/v cung so key moi them.
        _expect_value_error(
            lambda: chunked_attention(
                q_nho, k_nho, torch.randn(1, 1, 8, 16)),
            "k/v lech so key"),
        _expect_value_error(
            lambda: chunked_attention(
                q_nho, k_nho, v_nho, block=0),
            "block=0"),
        _expect_value_error(
            lambda: chunked_attention(
                q_nho, k_nho, v_nho, block="64"),
            "block dang chuoi"),
        _expect_value_error(
            lambda: chunked_attention(
                q_nho, k_nho, v_nho, causal="yes"),
            "causal dang chuoi"),
        _expect_value_error(
            lambda: chunked_attention(
                torch.empty(1, 1, 0, 16), k_nho, v_nho),
            "N=0"),
        _expect_value_error(
            lambda: chunked_attention(
                torch.full_like(q_nho, float("nan")),
                k_nho, v_nho),
            "q chua NaN"),
        _expect_value_error(
            lambda: make_qkv(0),
            "make_qkv n=0"),
        _expect_value_error(
            lambda: scores_bytes(0),
            "scores_bytes n=0"),
        _expect_value_error(
            lambda: bench(42),
            "bench fn khong goi duoc"),
    ]
    assert all(ket_qua), "Co guard khong chan dung!"

    # Dung: nhieu N/block/causal, ke ca N le
    # khong chia het cho block va hinh chu nhat.
    for n, blk in ((8, 3), (16, 64), (33, 8),
                   (64, 32)):
        torch.manual_seed(7)
        q = torch.randn(1, 1, n, 32)
        k = torch.randn(1, 1, n, 32)
        v = torch.randn(1, 1, n, 32)
        for causal in (True, False):
            sai = _sai_so_lon_nhat(
                naive_attention(q, k, v, causal),
                chunked_attention(q, k, v, blk, causal))
            assert sai <= 1e-5, \
                f"chunked sai o N={n} block={blk}."
            assert bool(torch.isfinite(
                chunked_attention(q, k, v, blk,
                                  causal)).all()), \
                "chunked phai huu han."
    torch.manual_seed(7)
    q = torch.randn(1, 2, 5, 16)
    k = torch.randn(1, 2, 7, 16)
    v = torch.randn(1, 2, 7, 16)
    for causal in (True, False):
        sai = _sai_so_lon_nhat(
            naive_attention(q, k, v, causal),
            chunked_attention(q, k, v, 3, causal))
        assert sai <= 1e-5, "chunked hinh chu nhat sai."

    # Bat bien theo block: cung Q/K/V, block
    # 16/32/128 phai cho cung dap an (muc tron).
    torch.manual_seed(7)
    q = torch.randn(1, 1, 64, 32)
    k = torch.randn(1, 1, 64, 32)
    v = torch.randn(1, 1, 64, 32)
    cac_out = [chunked_attention(q, k, v, blk)
               for blk in (16, 32, 128)]
    for out in cac_out[1:]:
        assert _sai_so_lon_nhat(cac_out[0], out) \
            <= 1e-5, "doi block khong duoc doi dap an."

    # Truong hop doi khang dung tay (N=4, block=2):
    # key tuong lai rat lon (bay mask sau max) va
    # khoi sau lon hon khoi truoc o vi tri hop le
    # (bay quen rescale). Chunked phai bang naive
    # tuyet doi o day.
    q = torch.zeros(1, 1, 4, 2)
    k = torch.zeros(1, 1, 4, 2)
    v = torch.eye(4).unsqueeze(0).unsqueeze(0)
    q[0, 0] = torch.tensor([[1., 0.], [1., 0.],
                            [1., 0.], [0., 1.]])
    k[0, 0] = torch.tensor([[1., 0.], [1., 0.],
                            [1., 0.], [10., 0.]])
    sai = _sai_so_lon_nhat(
        naive_attention(q, k, v, True),
        chunked_attention(q, k, v, 2, True))
    assert sai == 0.0, "doi khang causal phai khop 0."

    # Gradient: chunked phai cho gradient theo q
    # gan nhu naive (exact ca chieu nguoc).
    torch.manual_seed(1)
    q = torch.randn(1, 1, 16, 32, requires_grad=True)
    k = torch.randn(1, 1, 16, 32)
    v = torch.randn(1, 1, 16, 32)
    ref = naive_attention(q, k, v, True)
    ref.sum().backward()
    grad_ref = q.grad.clone()
    q.grad = None
    out = chunked_attention(q, k, v, 4, True)
    out.sum().backward()
    assert _sai_so_lon_nhat(grad_ref, q.grad) <= 1e-5, \
        "gradient chunked phai gan naive."

    # Float64: muc lam tron nho hon nhieu.
    torch.manual_seed(0)
    q = torch.randn(1, 1, 64, 64, dtype=torch.float64)
    k = torch.randn(1, 1, 64, 64, dtype=torch.float64)
    v = torch.randn(1, 1, 64, 64, dtype=torch.float64)
    sai = _sai_so_lon_nhat(
        naive_attention(q, k, v, True),
        chunked_attention(q, k, v, 32, True))
    assert sai <= 1e-12, "float64 phai khop 1e-12."

    # Bo nho ly thuyet: 512 -> 1 MiB, 4096 -> 64 MiB.
    assert scores_bytes(512) == 512 * 512 * 4
    assert scores_bytes(4096) == 4096 * 4096 * 4
    assert scores_bytes(512, torch.float64) \
        == 512 * 512 * 8
    print("Tu kiem tra: PASS (tat ca assert deu qua).")


def bai1(q, k, v):
    """Bai 1: chunked dung nhu naive/SDPA o N=256.

    Cung 1 bo Q/K/V, causal=True, fp32. Doi block
    32/64/128 va so voi naive - dap an phai dung
    yen trong muc lam tron, chi thu tu tinh thay
    doi.
    """
    print("\n" + "=" * 72)
    print("BAI 1 - Chunked vs naive vs SDPA (N=256)")
    print("=" * 72)
    print("Thu duy nhat thay doi la cach thuc thi "
          "attention; seed, Q/K/V, dtype, causal, "
          "thiet bi giu nguyen.")
    ref = naive_attention(q, k, v, causal=True)
    sdpa = sdpa_attention(q, k, v, causal=True)
    print(f"  naive vs SDPA: max abs diff = "
          f"{_sai_so_lon_nhat(ref, sdpa):.2e}")
    cac_out = {}
    sai_lon_nhat = 0.0
    for blk in (32, 64, 128):
        out = chunked_attention(q, k, v, blk,
                                causal=True)
        cac_out[blk] = out
        sai = _sai_so_lon_nhat(ref, out)
        sai_lon_nhat = max(sai_lon_nhat, sai)
        print(f"  block={blk}: naive vs chunked max "
              f"abs diff = {sai:.2e}")
    # [TRAINER] Checklist de bai yeu cau sai so so voi ca naive
    # va SDPA; truoc day moi in naive vs SDPA va naive vs
    # chunked, thieu cap SDPA vs chunked - in them tu dung 2
    # tensor da co san, khong doi cach tinh nao.
    print(f"  SDPA vs chunked (block="
          f"{BLOCK_MAC_DINH}): max abs diff = "
          f"{_sai_so_lon_nhat(sdpa, cac_out[BLOCK_MAC_DINH]):.2e}")
    # Bat bien theo block: so tung cap block.
    chenh_block = max(
        _sai_so_lon_nhat(cac_out[32], cac_out[64]),
        _sai_so_lon_nhat(cac_out[32], cac_out[128]),
        _sai_so_lon_nhat(cac_out[64], cac_out[128]))
    print(f"  Chenh lon nhat giua cac block: "
          f"{chenh_block:.2e}")
    if sai_lon_nhat <= 1e-6:
        muc = ("nam trong muc lam tron fp32 nhu de "
               "bai ky vong (<= 1e-6)")
    elif sai_lon_nhat <= 1e-5:
        muc = ("lon hon 1e-6 nhung van o muc lam "
               "tron fp32")
    else:
        muc = ("CANH BAO: vuot muc lam tron fp32 - "
               "kiem tra lai rescale/mask")
    print(f"Ket luan: chunked {muc} (sai so lon "
          f"nhat {sai_lon_nhat:.2e}). Doi block chi "
          f"doi cach nhom tinh toan, rescale lam "
          f"ket qua cuoi cung bang dung softmax ca "
          f"hang - vi vay block anh huong toc do/bo "
          f"nho, khong duoc phep anh huong dap an.")
    return {"sai_lon_nhat": sai_lon_nhat,
            "chenh_block": chenh_block}


def bai2():
    """Bai 2: N tang - bo nho N x N va thoi gian.

    Moi N tao Q/K/V moi (seed dat lai truoc moi
    N de lap lai duoc), bench naive vs chunked
    bang cung ham bench, cung thiet bi, duoi
    no_grad. Bo nho la con so ly thuyet (fp32),
    khong phai bo nho do duoc - chunked khong
    bao gio tao ma tran N x N, nen bo nho phu
    cua no theo cong thuc chi la N x block.
    """
    print("\n" + "=" * 72)
    print("BAI 2 - N = 512 -> 4096: bo nho va toc do")
    print("=" * 72)
    print("Thu duy nhat thay doi la N; cach thuc "
          "thi, dtype, causal, thiet bi giu nguyen.")
    ket_qua = {}
    for n in (512, 1024, 2048, 4096):
        torch.manual_seed(SEED)
        q, k, v = make_qkv(n, D_HEAD)
        mib = scores_bytes(n) / (2 ** 20)
        # Warmup 3 + 10 lan lay trung vi: o N=4096
        # bench 5 lan cho ti le dao dong 0.92-1.34x
        # tuy tai may, 10 lan on dinh o 1.28-1.33x.
        t_naive = bench(
            lambda: naive_attention(q, k, v, True),
            warmup=3, reps=10)
        t_chunk = bench(
            lambda: chunked_attention(
                q, k, v, BLOCK_MAC_DINH, True),
            warmup=3, reps=10)
        ket_qua[n] = (mib, t_naive, t_chunk)
        print(f"  N={n}: ma tran N x N = {mib:.0f} MiB "
              f"| naive={t_naive:.1f} ms "
              f"| chunked={t_chunk:.1f} ms "
              f"| chunked/naive={t_chunk / t_naive:.2f}x")
    mib_512, _, _ = ket_qua[512]
    mib_4096, t_n_4096, t_c_4096 = ket_qua[4096]
    _, t_n_512, _ = ket_qua[512]
    ti_le_mib = mib_4096 / mib_512
    ti_le_naive = t_n_4096 / t_n_512
    ti_le_chunk = t_c_4096 / t_n_4096
    if ti_le_chunk <= 1.05:
        toc_do = (f"chunked khong cham hon naive "
                  f"dang ke o N=4096 "
                  f"({ti_le_chunk:.2f}x)")
    else:
        toc_do = (f"chunked (vong lap Python tren "
                  f"CPU) cham hon naive "
                  f"{ti_le_chunk:.2f}x o N=4096 - "
                  f"day la ban mo phong de kiem "
                  f"chung dap an, khong phai kernel "
                  f"fused chay trong SRAM, nen dung "
                  f"suy ra FlashAttention that cham "
                  f"hon")
    # Tang thoi gian so voi tang phep tinh (64x khi
    # N tang 8x): vuot xa 64x la dau hieu
    # memory-bound - ma tran lon vuot cache nen
    # phai qua lai bo nho nhieu lan, khong chi
    # tinh nhieu hon.
    if ti_le_naive > 80:
        tang_naive = (f"thoi gian naive tang "
                      f"{ti_le_naive:.1f}x, nhanh "
                      f"hon ca ti le phep tinh 64x - "
                      f"dau hieu memory-bound")
    else:
        tang_naive = (f"thoi gian naive tang "
                      f"{ti_le_naive:.1f}x")
    # [TRAINER] Luu y (chua sua so in ra - de ban quyet): con so
    # "N x block" trong ket luan duoi moi tinh buffer scores cua
    # 1 khoi; ban chunked nay con cap p = exp(scores - m) them
    # 1 buffer N x block nua va acc N x d_v, nen dinh bo nho phu
    # thuc te ~2*N*block + N*d_v byte (N=4096: ~3 MiB thay vi
    # 1 MiB). Doi con so se doi so lieu ly thuyet ban da ghi
    # trong ket luan, nen Trainer chi ghi chu, khong tu sua.
    print(f"Ket luan: N tang 8x thi ma tran N x N "
          f"tang {ti_le_mib:.0f}x (bo nho tang binh "
          f"phuong); {tang_naive}; {toc_do}. Bo nho "
          f"phu cua chunked theo cong thuc chi "
          f"N x block = {4096 * BLOCK_MAC_DINH * 4 / 2 ** 20:.1f} "
          f"MiB o N=4096, thay cho 64 MiB cua ma "
          f"tran day du.")
    return {"ti_le_mib": ti_le_mib,
            "ti_le_naive": ti_le_naive,
            "ti_le_chunk_4096": ti_le_chunk}


def bai3(thiet_bi):
    """Bai 3: chuoi ngan, dtype, backend, du bao.

    Moi truong hop chi doi dung 1 thu so voi
    Bai 1. Truong hop can GPU (ep backend math,
    fp32 tren GPU cu) KHONG chay duoc o day vi
    khong co CUDA - ghi ro de ngo, khong suy
    ra tu so CPU.
    """
    print("\n" + "=" * 72)
    print("BAI 3 - Khi nao FlashAttention khong giup")
    print("=" * 72)
    print("Thu duy nhat thay doi la N hoac dtype; "
          "cach thuc thi, seed, causal giu nguyen.")
    # Chuoi ngan: N=32, 64.
    ket_qua_ngan = {}
    for n in (32, 64):
        torch.manual_seed(SEED)
        q, k, v = make_qkv(n, D_HEAD)
        sai = _sai_so_lon_nhat(
            naive_attention(q, k, v, True),
            chunked_attention(q, k, v, BLOCK_MAC_DINH,
                              True))
        t_naive = bench(
            lambda: naive_attention(q, k, v, True),
            warmup=WARMUP, reps=20)
        t_chunk = bench(
            lambda: chunked_attention(
                q, k, v, BLOCK_MAC_DINH, True),
            warmup=WARMUP, reps=20)
        ket_qua_ngan[n] = (sai, t_naive, t_chunk)
        print(f"  N={n}: sai so={sai:.2e} | "
              f"naive={t_naive:.3f} ms "
              f"| chunked={t_chunk:.3f} ms "
              f"| ti le={t_chunk / t_naive:.2f}x")
    ti_le_64 = ket_qua_ngan[64][2] / ket_qua_ngan[64][1]
    if 0.8 <= ti_le_64 <= 1.25:
        ngan = ("chenh lech khong dang ke - ma tran "
                "N x N von da nho, nut that khong "
                "con o bo nho")
    # [TRAINER] Nhanh else cu mac dinh ti le > 1.25: neu do duoc
    # ti le < 0.8 (chunked NHANH hon) no van in "cham hon
    # 0.50x" - ket luan nguoc han so do. Da chung minh bang
    # bench gia lap ti le 0.50: code cu in sai cau nay. Tach
    # rieng nhanh nhanh hon, giu nguyen nhanh cham hon o else.
    elif ti_le_64 < 0.8:
        ngan = (f"chunked nhanh hon naive o chuoi "
                f"ngan trong lan do nay (chi ton "
                f"{ti_le_64:.2f}x thoi gian) - o N "
                f"rat nho tren CPU chenh lech nay "
                f"thuong nam trong nhieu do va chi "
                f"phi co dinh cua tung ham, khong doi "
                f"ket luan chung: loi ich bo nho chi "
                f"ro khi N lon")
    else:
        ngan = (f"chunked cham hon {ti_le_64:.2f}x - "
                f"chi phi co dinh cua rescale va "
                f"bookkeeping theo khoi at het phan "
                f"tinh toan khi N nho; dung nhu ly "
                f"do FlashAttention chi giup khi N lon")
    print(f"  Chuoi ngan: {ngan}.")
    # Dtype float64: dap an van dung, muc tron
    # nho hon - day la truong hop chay duoc tren
    # CPU, khac voi truong hop GPU/fp32 can ep
    # backend (khong co CUDA nen de ngo).
    torch.manual_seed(SEED)
    q = torch.randn(1, 1, 64, D_HEAD,
                    dtype=torch.float64)
    k = torch.randn(1, 1, 64, D_HEAD,
                    dtype=torch.float64)
    v = torch.randn(1, 1, 64, D_HEAD,
                    dtype=torch.float64)
    sai_64 = _sai_so_lon_nhat(
        naive_attention(q, k, v, True),
        chunked_attention(q, k, v, 32, True))
    print(f"  Float64 (N=64): sai so={sai_64:.2e} - "
          f"dap an van dung, chi khac muc tron.")
    # Backend thuc te + du bao ly thuyet.
    if thiet_bi == "cuda":
        backend = ("GPU: backend thuc te xem o "
                   "check_backend() dau lab.")
    else:
        backend = ("CPU: SDPA dung backend math - "
                   "ket luan ve toc do FlashAttention "
                   "that de ngo, khong suy ra tu so "
                   "CPU; truong hop ep backend math "
                   "tren GPU khong chay duoc o day.")
    mib_32k = scores_bytes(32768) / (2 ** 20)
    mib_128k = scores_bytes(131072) / (2 ** 20)
    # [TRAINER] Luu y nhu o Bai 2: aux_128k moi tinh buffer
    # scores cua 1 khoi (N x block), chua tinh buffer p va
    # acc; giu nguyen so in ra de ban tu quyet co doi khong.
    aux_128k = 131072 * BLOCK_MAC_DINH * 4 \
        / (2 ** 20)
    print(f"  Backend: {backend}")
    print(f"  Du bao LY THUYET (1 head, fp32): "
          f"N=32768 thi ma tran diem = "
          f"{mib_32k:.0f} MiB; N=131072 thi "
          f"{mib_128k:.0f} MiB, trong khi bo nho "
          f"phu cua chunked (block=64) chi "
          f"{aux_128k:.0f} MiB - do la tiet kiem "
          f"phan cung that, can GPU moi do duoc "
          f"chuoi toi da that su.")
    return {"ti_le_64": ti_le_64, "sai_64": sai_64,
            "mib_128k": mib_128k, "aux_128k": aux_128k}


def main():
    """Chay ca lab: backend, tu kiem tra, 3 bai."""
    print("Lab FlashAttention Day 25 - "
          f"torch {torch.__version__}")
    thiet_bi = check_backend()
    tu_kiem_tra()
    # Dat lai seed SAU tu kiem tra (no dung seed
    # rieng), de Q/K/V cua Bai 1 xac dinh.
    torch.manual_seed(SEED)
    q, k, v = make_qkv(N_KIEM_CHUNG, D_HEAD)
    kq1 = bai1(q, k, v)
    kq2 = bai2()
    kq3 = bai3(thiet_bi)
    print("\nBANG KET QUA: N=256 sai so lon nhat "
          f"{kq1['sai_lon_nhat']:.2e}, chenh giua "
          f"cac block {kq1['chenh_block']:.2e} | "
          f"N=4096 ma tran 64 MiB, chunked/naive "
          f"{kq2['ti_le_chunk_4096']:.2f}x | N=64 ti "
          f"le chunked/naive {kq3['ti_le_64']:.2f}x, "
          f"float64 sai so {kq3['sai_64']:.2e}.")
    print("Ket luan chung: chunked_attention la "
          "exact attention - dap an bang naive/SDPA "
          "trong muc lam tron, bo nho phu O(N) thay "
          "cho O(N^2). Toc do FlashAttention that "
          "can GPU + backend flash moi do duoc; "
          "tren CPU chi ket luan ve dap an va bo "
          "nho, khong ket luan ve toc do.")


if __name__ == "__main__":
    main()
