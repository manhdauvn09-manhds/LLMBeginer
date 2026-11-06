#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Lab serving Day 27 (tuan 4 - Green LLM).

Serving hieu qua la bai toan he thong: prefill
compute-bound (1 lan / request), decode
memory-bound (lap lai moi token), KV cache la
nut that bo nho so 1.

Bai A: dung Transformer nho chay that (4 lop,
d_model=256), do KV cache / token, TTFT
(prefill) va TPOT (decode) o 3 do dai prompt
128 / 512 / 2048, decode 50 token.
Bai B: mo phong thuan Python (khong can GPU):
PagedAttention (lang phi bo nho) va continuous
batching (thoi gian cho) tren 20 request gia
lap do dai 50-500 token.
Bai C: speculative decoding bang so (bang
alpha -> speedup), prefix caching, va
quantized serving (KV cache INT8).

Quy tac thi nghiem (viet truoc khi chay):
thu duy nhat thay doi la do dai prompt (Bai A)
hoac cach phuc vu (Bai B); con lai giu nguyen:
cung mo hinh, cung phan cung, cung batch
size = 1, cung so token decode. Bao toc do
luon kem boi canh: batch size, do dai prompt /
dap an, da warm-up hay chua (xem bai A).

Cach chay (venv Day 15, CPU du):
    ~/workspace/venvs/day15/bin/python serving_lab.py
Tren CPU, so tuyet doi (ms) chi dung cho may
nay; ket luan dung duoc la ti le / xu huong
khi prompt dai ra, khong phai con so tuyet
doi cua GPU serving that.
"""

import math
import random
import statistics
import time
from collections import deque

import torch
import torch.nn as nn
import torch.nn.functional as F


# Hang so cho ca lab: doi 1 hang so la doi
# luon thi nghiem, nen moi con so deu co ten.
SEED = 0
D_MODEL = 256
N_LAYER = 4
N_HEAD = 4
D_FF = 1024
VOCAB = 1000
MAX_LEN = 4096
DO_DAI_PROMPT = (128, 512, 2048)
N_DECODE = 50
WARMUP = 2
REPS = 3
BLOCK_TOKEN = 16
MAX_LEN_NAIVE = 512
N_REQUEST = 20
MAX_BATCH = 4
HE_SO_PREFIX = 200


def _kiem_tra_so_nguyen(ten, gia_tri, toi_thieu=1):
    """Guard dung chung cho tham so so nguyen.

    Raises:
        ValueError: gia tri la bool, khong phai
            int, hoac nho hon toi_thieu.
    """
    if isinstance(gia_tri, bool) \
            or not isinstance(gia_tri, int) \
            or gia_tri < toi_thieu:
        raise ValueError(
            f"{ten} phai la so nguyen >= {toi_thieu}.")


def _kiem_tra_so_thuc(ten, gia_tri, lon_hon=None,
                      lon_hon_bang=None):
    """Guard dung chung cho tham so so thuc.

    Chi chap nhan int/float (tru bool), huu han.
    Kieu sai (vd chuoi "0.5") phai nem
    ValueError, khong de TypeError tran ra
    tu phep so sanh (checklist 12).

    Raises:
        ValueError: kieu sai, NaN/Inf, hoac
            vuot nguong cho truoc.
    """
    if isinstance(gia_tri, bool) \
            or not isinstance(gia_tri, (int, float)):
        raise ValueError(f"{ten} phai la so.")
    if not math.isfinite(float(gia_tri)):
        raise ValueError(f"{ten} phai huu han.")
    if lon_hon is not None \
            and not float(gia_tri) > lon_hon:
        raise ValueError(f"{ten} phai > {lon_hon}.")
    if lon_hon_bang is not None \
            and not float(gia_tri) >= lon_hon_bang:
        raise ValueError(
            f"{ten} phai >= {lon_hon_bang}.")


def kv_bytes_per_token(n_layer, d_model,
                       bytes_per_param):
    """KV cache / token (byte) theo cong thuc.

    Moi token, moi lop, luu ca K va V, moi
    vector dai d_model: 2 * n_layer * d_model
    phan tu, moi phan tu chiem bytes_per_param
    byte (FP32 = 4, FP16 = 2, INT8/FP8 = 1).

    Args:
        n_layer: so lop, int >= 1.
        d_model: chieu an, int >= 1.
        bytes_per_param: int trong (1, 2, 4, 8).

    Returns:
        So byte / token, so nguyen.

    Raises:
        ValueError: tham so sai kieu / pham vi.
    """
    _kiem_tra_so_nguyen("n_layer", n_layer)
    _kiem_tra_so_nguyen("d_model", d_model)
    if isinstance(bytes_per_param, bool) \
            or bytes_per_param not in (1, 2, 4, 8):
        raise ValueError(
            "bytes_per_param phai la 1, 2, 4 hoac 8.")
    return 2 * n_layer * d_model * bytes_per_param


def kv_cache_bytes(n_layer, d_model, bytes_per_param,
                   n_tokens):
    """KV cache (byte) cho n_tokens token.

    Bang kv_bytes_per_token * n_tokens. Day la
    con so ly thuyet theo cong thuc; lab se
    doi chieu voi cache that dem tu tensor
    (xem tu_kiem_tra).

    Raises:
        ValueError: n_tokens khong phai int >= 0.
    """
    if isinstance(n_tokens, bool) \
            or not isinstance(n_tokens, int) \
            or n_tokens < 0:
        raise ValueError(
            "n_tokens phai la so nguyen >= 0.")
    return kv_bytes_per_token(n_layer, d_model,
                              bytes_per_param) * n_tokens


class Block(nn.Module):
    """1 block Transformer co ho tro KV cache.

    Attention tu cai (Q/K/V rieng) thay vi
    nn.MultiheadAttention: can tach K, V ra
    de cache, va noi cache cu vao K, V moi.
    Quy uoc su dung (ghi ro de khong hieu
    nham): prefill chi chay khi kv=None (toan
    bo prompt, mask causal); decode chi chay
    1 token / lan khi kv co san (moi key da
    luu deu la qua khu, khong can mask).

    Args:
        d_model: chieu an, int >= 1, chia het
            cho n_head.
        n_head: so head, int >= 1.
        d_ff: chieu an MLP, int >= 1.
    """

    def __init__(self, d_model, n_head, d_ff):
        super().__init__()
        _kiem_tra_so_nguyen("d_model", d_model)
        _kiem_tra_so_nguyen("n_head", n_head)
        _kiem_tra_so_nguyen("d_ff", d_ff)
        if d_model % n_head != 0:
            raise ValueError(
                "d_model phai chia het cho n_head.")
        self.d_model = d_model
        self.n_head = n_head
        self.d_head = d_model // n_head
        self.d_ff = d_ff
        self.q_proj = nn.Linear(d_model, d_model)
        self.k_proj = nn.Linear(d_model, d_model)
        self.v_proj = nn.Linear(d_model, d_model)
        self.out_proj = nn.Linear(d_model, d_model)
        self.ln1 = nn.LayerNorm(d_model)
        self.ln2 = nn.LayerNorm(d_model)
        self.mlp = nn.Sequential(
            nn.Linear(d_model, d_ff), nn.GELU(),
            nn.Linear(d_ff, d_model))

    def forward(self, x, kv=None):
        """Chay 1 block, tra (out, (K, V) moi).

        Args:
            x: tensor float (batch, seq, d_model),
                huu han, khong rong.
            kv: None (prefill) hoac (K, V) da
                cache, moi tensor (batch, n_head,
                seq_cu, d_head), cung dtype/device
                voi x. Decode chi ho tro seq = 1.

        Returns:
            (out, (K, V)): out cung shape voi x;
            K, V gom ca cache cu + token moi.

        Raises:
            ValueError: x sai dtype/shape/chua
                NaN-Inf/rong, kv sai shape/dtype/
                device, decode voi seq != 1.
        """
        if not isinstance(x, torch.Tensor):
            raise ValueError("x phai la tensor.")
        if x.dim() != 3:
            raise ValueError("x phai co 3 chieu.")
        if not x.is_floating_point():
            raise ValueError("x phai la so thuc.")
        if x.size(0) == 0 or x.size(1) == 0 \
                or x.size(2) != self.d_model:
            raise ValueError(
                "x rong hoac chieu cuoi khac d_model.")
        if not bool(torch.isfinite(x).all()):
            raise ValueError("x phai huu han.")
        if kv is not None:
            if not isinstance(kv, tuple) \
                    or len(kv) != 2:
                raise ValueError(
                    "kv phai la tuple (K, V).")
            if x.size(1) != 1:
                raise ValueError(
                    "decode co cache chi ho tro "
                    "1 token / lan.")
            for ten, t in (("K", kv[0]), ("V", kv[1])):
                if not isinstance(t, torch.Tensor) \
                        or t.dim() != 4 \
                        or t.size(0) != x.size(0) \
                        or t.size(1) != self.n_head \
                        or t.size(3) != self.d_head \
                        or t.dtype != x.dtype \
                        or t.device != x.device:
                    raise ValueError(
                        f"cache {ten} sai shape/dtype"
                        "/device.")
            if kv[0].size(2) != kv[1].size(2):
                raise ValueError(
                    "K va V trong cache phai cung "
                    "do dai.")
        batch, seq, _ = x.shape
        h = self.ln1(x)
        q = self.q_proj(h).view(
            batch, seq, self.n_head,
            self.d_head).transpose(1, 2)
        k = self.k_proj(h).view(
            batch, seq, self.n_head,
            self.d_head).transpose(1, 2)
        v = self.v_proj(h).view(
            batch, seq, self.n_head,
            self.d_head).transpose(1, 2)
        if kv is not None:
            k = torch.cat([kv[0], k], dim=2)
            v = torch.cat([kv[1], v], dim=2)
        scores = q @ k.transpose(-2, -1) \
            / math.sqrt(self.d_head)
        if kv is None and seq > 1:
            # Mask causal o prefill: chi so tren
            # dung device cua x (checklist 3).
            chi_query = torch.arange(
                seq, device=x.device).unsqueeze(1)
            chi_key = torch.arange(
                k.size(2), device=x.device).unsqueeze(0)
            scores = scores.masked_fill(
                chi_key > chi_query, float("-inf"))
        attn = torch.softmax(scores, dim=-1) @ v
        attn = attn.transpose(1, 2).contiguous().view(
            batch, seq, self.d_model)
        x = x + self.out_proj(attn)
        x = x + self.mlp(self.ln2(x))
        return x, (k, v)


class MiniTransformer(nn.Module):
    """Transformer nho de do serving that.

    Embedding token + vi tri hoc duoc (toi da
    MAX_LEN), n_layer block co KV cache, norm
    cuoi, LM head. Moi con so lay tu cau hinh
    cua chinh instance (self.*), khong dung
    hang toan cuc trong guard (bai hoc Day 26:
    guard phai hoi instance, khong hoi hang).

    Args:
        vocab: kich thuoc tu dien, int >= 1.
        d_model, n_layer, n_head, d_ff: nhu Block.
        max_len: do dai toi da, int >= 1.
    """

    def __init__(self, vocab=VOCAB, d_model=D_MODEL,
                 n_layer=N_LAYER, n_head=N_HEAD,
                 d_ff=D_FF, max_len=MAX_LEN):
        super().__init__()
        _kiem_tra_so_nguyen("vocab", vocab)
        _kiem_tra_so_nguyen("d_model", d_model)
        _kiem_tra_so_nguyen("n_layer", n_layer)
        _kiem_tra_so_nguyen("n_head", n_head)
        _kiem_tra_so_nguyen("d_ff", d_ff)
        _kiem_tra_so_nguyen("max_len", max_len)
        if d_model % n_head != 0:
            raise ValueError(
                "d_model phai chia het cho n_head.")
        self.vocab = vocab
        self.d_model = d_model
        self.n_layer = n_layer
        self.n_head = n_head
        self.d_ff = d_ff
        self.max_len = max_len
        self.tok_emb = nn.Embedding(vocab, d_model)
        self.pos_emb = nn.Embedding(max_len, d_model)
        self.blocks = nn.ModuleList([
            Block(d_model, n_head, d_ff)
            for _ in range(n_layer)
        ])
        self.ln_f = nn.LayerNorm(d_model)
        self.lm_head = nn.Linear(d_model, vocab,
                                 bias=False)

    def forward(self, tokens, kv_caches=None):
        """Chay mo hinh, tra (logits, caches moi).

        Args:
            tokens: tensor long (batch, seq),
                gia tri trong [0, vocab).
            kv_caches: None (prefill) hoac list
                dai n_layer, moi phan tu la (K, V)
                nhu Block.forward. Decode chi
                ho tro seq = 1.

        Returns:
            (logits, caches): logits (batch, seq,
            vocab); caches la list (K, V) moi
            cho moi lop.

        Raises:
            ValueError: tokens sai dtype/shape/
                pham vi, kv_caches sai do dai,
                tong do dai vuot max_len.
        """
        if not isinstance(tokens, torch.Tensor):
            raise ValueError("tokens phai la tensor.")
        if tokens.dim() != 2:
            raise ValueError("tokens phai co 2 chieu.")
        if tokens.dtype != torch.long:
            raise ValueError("tokens phai la long.")
        if tokens.size(0) == 0 or tokens.size(1) == 0:
            raise ValueError("tokens khong duoc rong.")
        if int(tokens.min()) < 0 \
                or int(tokens.max()) >= self.vocab:
            raise ValueError(
                "tokens vuot ngoai [0, vocab).")
        if kv_caches is not None:
            if not isinstance(kv_caches, list) \
                    or len(kv_caches) \
                    != self.n_layer:
                raise ValueError(
                    "kv_caches phai la list dai "
                    "n_layer.")
            # [TRAINER] Phai kiem tra cau truc
            # tung (K, V) NGAY tai day: ban goc lay
            # kv_caches[0][0] truoc khi Block kip
            # kiem tra, nen phan tu sai cau truc
            # (vd so nguyen, tensor don) lam ro
            # TypeError / IndexError ra ngoai thay
            # vi ValueError - lech quy uoc "moi loi
            # dau vao la ValueError" cua ca lab.
            for cap in kv_caches:
                if not isinstance(cap, tuple) \
                        or len(cap) != 2 \
                        or not isinstance(
                            cap[0], torch.Tensor) \
                        or not isinstance(
                            cap[1], torch.Tensor):
                    raise ValueError(
                        "moi phan tu kv_caches phai "
                        "la tuple (K, V) tensor.")
            past = kv_caches[0][0].size(2)
            if tokens.size(1) != 1:
                raise ValueError(
                    "decode co cache chi ho tro "
                    "1 token / lan.")
        else:
            past = 0
        if past + tokens.size(1) > self.max_len:
            raise ValueError(
                "tong do dai vuot qua max_len.")
        vi_tri = torch.arange(
            past, past + tokens.size(1),
            device=tokens.device)
        x = self.tok_emb(tokens) \
            + self.pos_emb(vi_tri).unsqueeze(0)
        caches_moi = []
        for i, block in enumerate(self.blocks):
            kv = None if kv_caches is None \
                else kv_caches[i]
            x, kv_moi = block(x, kv)
            caches_moi.append(kv_moi)
        logits = self.lm_head(self.ln_f(x))
        return logits, caches_moi

    def kv_bytes_per_token(self, bytes_per_param=4):
        """KV cache / token cua chinh model nay.

        Dung n_layer/d_model cua instance (bai
        hoc Day 26: hoi instance, khong hoi
        hang toan cuc). bytes_per_param la
        dtype muon tinh cho (FP32 = 4, FP16 = 2,
        INT8 = 1), khong nhat thiet bang dtype
        thuc te dang chay.
        """
        return kv_bytes_per_token(
            self.n_layer, self.d_model, bytes_per_param)

    def cache_bytes_thuc_te(self, caches):
        """Dem byte cache that tu tensor (K, V).

        Cong numel * element_size cua ca K va V,
        ca lop. Dung de doi chieu voi cong thuc
        ly thuyet trong tu_kiem_tra.

        Raises:
            ValueError: caches khong phai list
                dai n_layer.
        """
        if not isinstance(caches, list) \
                or len(caches) != self.n_layer:
            raise ValueError(
                "caches phai la list dai n_layer.")
        tong = 0
        for k, v in caches:
            tong += k.numel() * k.element_size()
            tong += v.numel() * v.element_size()
        return tong

def bench(fn, warmup=WARMUP, reps=REPS):
    """Do thoi gian: TRUNG VI ms / lan chay fn.

    Dung time.perf_counter() (checklist 5),
    duoi no_grad. Tren GPU dong bo CUDA sau
    warmup va sau moi lan chay (checklist 21);
    tren CPU nhanh dong bo khong chay. Warm-up
    2-3 lan de loai thoi gian load/compile/
    cap phat lan dau (bay 1 trong de bai).

    Returns:
        Trung vi thoi gian, mili-giay / lan.
    """
    _kiem_tra_so_nguyen("warmup", warmup, toi_thieu=0)
    _kiem_tra_so_nguyen("reps", reps)
    if not callable(fn):
        raise ValueError("fn phai goi duoc.")
    with torch.no_grad():
        for _ in range(warmup):
            fn()
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


def _phan_vi(mau, p):
    """Percentile nearest-rank cua 1 list so.

    Sap xep tang dan, lay phan tu o hang
    ceil(p / 100 * n) (toi thieu hang 1).
    p phai trong (0, 100]. nearest-rank khac
    trung vi: p50 cua [1..10] la 5, khong
    phai 5.5 - doc so thi nho dung dinh nghia.

    Raises:
        ValueError: mau rong / khong phai list,
            p ngoai (0, 100].
    """
    if not isinstance(mau, (list, tuple)) \
            or len(mau) == 0:
        raise ValueError("mau khong duoc rong.")
    _kiem_tra_so_thuc("p", p, lon_hon=0)
    if float(p) > 100:
        raise ValueError("p phai <= 100.")
    ds = sorted(float(x) for x in mau)
    hang = max(1, math.ceil(
        float(p) / 100.0 * len(ds)))
    return ds[hang - 1]


def bench_day_du(fn, warmup=WARMUP, reps=10):
    """Do thoi gian: tra (TRUNG VI, p95) ms.

    [TRAINER] Ban goc chi co bench() tra trung
    vi, ma brief Day 27 yeu cau bao ca p95 cho
    TTFT / TPOT: trung vi giu nhip trung tam
    nhung giau duoi phan phoi - nguoi dung xau
    nhat (request cham) khong thay o dau ca.
    reps mac dinh 10 (thay vi 3 cua bench) de
    p95 co y nghia hon; trung vi van tinh
    giong het bench() nen so chinh khong doi
    cach do. Sync CUDA giong bench.
    """
    _kiem_tra_so_nguyen("warmup", warmup, toi_thieu=0)
    _kiem_tra_so_nguyen("reps", reps)
    if not callable(fn):
        raise ValueError("fn phai goi duoc.")
    with torch.no_grad():
        for _ in range(warmup):
            fn()
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        thoi_gian = []
        for _ in range(reps):
            t0 = time.perf_counter()
            fn()
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            thoi_gian.append(
                (time.perf_counter() - t0) * 1000.0)
    return (float(statistics.median(thoi_gian)),
            float(_phan_vi(thoi_gian, 95)))


def _tao_prompt(do_dai, seed=SEED):
    """Tao prompt gia ngau nhien (batch = 1).

    Token id ngau nhien trong [0, VOCAB) -
    chi de do toc do, khong phai van ban that.
    Seed rieng theo do dai de lap lai duoc.

    Returns:
        Tensor long (1, do_dai).
    """
    _kiem_tra_so_nguyen("do_dai", do_dai)
    gen = torch.Generator()
    gen.manual_seed(seed + do_dai)
    return torch.randint(0, VOCAB, (1, do_dai),
                         generator=gen)


def prefill(model, prompt):
    """Prefill 1 prompt, tra (logits, caches).

    Chay ca prompt 1 lan, duoi no_grad (chi
    do toc do, khong train).

    Raises:
        ValueError: model khong phai
            MiniTransformer, prompt sai.
    """
    if not isinstance(model, MiniTransformer):
        raise ValueError("model phai la MiniTransformer.")
    model.eval()
    with torch.no_grad():
        return model(prompt)


def decode_greedy(model, caches, token_dau, n_steps):
    """Decode tham lam n_steps token, dung cache.

    Moi buoc: dua 1 token vao, lay argmax cua
    logits cuoi lam token tiep theo, cache
    duoc noi dai them 1 token / lop. Tra list
    token sinh ra (khong gom token_dau).

    Raises:
        ValueError: n_steps khong phai int >= 1,
            caches sai, token_dau ngoai vocab.
    """
    _kiem_tra_so_nguyen("n_steps", n_steps)
    if not isinstance(model, MiniTransformer):
        raise ValueError("model phai la MiniTransformer.")
    if not isinstance(caches, list) \
            or len(caches) != model.n_layer:
        raise ValueError(
            "caches phai la list dai n_layer.")
    if isinstance(token_dau, bool) \
            or not isinstance(token_dau, int) \
            or token_dau < 0 \
            or token_dau >= model.vocab:
        raise ValueError(
            "token_dau vuot ngoai [0, vocab).")
    model.eval()
    token_hien_tai = token_dau
    ket_qua = []
    with torch.no_grad():
        for _ in range(n_steps):
            x = torch.tensor([[token_hien_tai]])
            logits, caches = model(x, caches)
            # detach truoc khi doi sang int: logits
            # mang requires_grad qua tham so model
            # (bai hoc Day 26: float()/int() tren
            # tensor co grad sinh UserWarning).
            token_hien_tai = int(logits[0, -1]
                                 .argmax().detach())
            ket_qua.append(token_hien_tai)
    return ket_qua, caches


def _kiem_tra_danh_sach_do_dai(lengths, ten="lengths"):
    """Guard dung chung cho list do dai request.

    Tra list(int) da sao chep. Moi phan tu
    phai la int (tru bool) >= 1.

    Raises:
        ValueError: list rong, phan tu sai kieu
            / pham vi, hoac khong phai list/tuple.
    """
    if not isinstance(lengths, (list, tuple)):
        raise ValueError(f"{ten} phai la list/tuple.")
    if len(lengths) == 0:
        raise ValueError(f"{ten} khong duoc rong.")
    for i, gia_tri in enumerate(lengths):
        if isinstance(gia_tri, bool) \
                or not isinstance(gia_tri, int) \
                or gia_tri < 1:
            raise ValueError(
                f"{ten}[{i}] phai la so nguyen >= 1.")
    return list(lengths)


def _kiem_tra_arrivals(arrivals, n):
    """Chuan hoa thoi diem den cua n request.

    [TRAINER] Ban goc khong co arrivals: ca 2
    mo phong deu gia dinh moi request den o
    t = 0, trong khi brief Day 27 yeu cau xet
    request "den lech pha nhau" - request den
    muon phai cho duoc cong vao thoi gian cho
    cua chinh no, khong duoc chay truoc khi no
    den. None -> tat ca den o t = 0 (giu nguyen
    hanh vi va moi con so cua ban goc).

    Returns:
        List thoi diem den (int/float >= 0),
        dai n.

    Raises:
        ValueError: arrivals khong phai list /
            tuple dai n, phan tu am / NaN / sai
            kieu.
    """
    if arrivals is None:
        return [0] * n
    if not isinstance(arrivals, (list, tuple)) \
            or len(arrivals) != n:
        raise ValueError(
            "arrivals phai la list/tuple cung do "
            "dai voi lengths.")
    for i, a in enumerate(arrivals):
        _kiem_tra_so_thuc(f"arrivals[{i}]", a,
                          lon_hon_bang=0)
    return list(arrivals)


def paged_waste(lengths, max_len=MAX_LEN_NAIVE,
                block=BLOCK_TOKEN):
    """Lang phi bo nho naive vs PagedAttention.

    Naive: cap phat toi da max_len cho moi
    request, thua max_len - L token. Paged:
    cap theo block (lam tron len boi cua
    block), thua phan le cua block cuoi, toi
    da block - 1 token / request.

    Args:
        lengths: do dai that cua moi request.
        max_len: cap phat toi da kieu naive,
            int >= do dai lon nhat.
        block: token / block, int >= 1.

    Returns:
        {"naive_waste": int, "paged_waste": int,
         "naive_alloc": int, "paged_alloc": int,
         "naive_waste_pct": float,
         "paged_waste_pct": float} - ti le
        lang phi = thua / tong cap phat * 100.

    Raises:
        ValueError: lengths rong/sai, max_len
            nho hon do dai lon nhat, block < 1.
    """
    ds = _kiem_tra_danh_sach_do_dai(lengths)
    _kiem_tra_so_nguyen("max_len", max_len)
    _kiem_tra_so_nguyen("block", block)
    if max_len < max(ds):
        raise ValueError(
            "max_len phai >= do dai lon nhat.")
    thua_naive = sum(max_len - l for l in ds)
    thua_paged = sum(
        -(-l // block) * block - l for l in ds)
    cap_naive = max_len * len(ds)
    cap_paged = sum(-(-l // block) * block
                    for l in ds)
    return {"naive_waste": thua_naive,
            "paged_waste": thua_paged,
            "naive_alloc": cap_naive,
            "paged_alloc": cap_paged,
            "naive_waste_pct":
                thua_naive / cap_naive * 100.0,
            "paged_waste_pct":
                thua_paged / cap_paged * 100.0}


def simulate_static(lengths, max_batch=MAX_BATCH,
                    arrivals=None):
    """Mo phong static batching.

    Gom request theo thu tu thanh batch toi da
    max_batch, chay tuan tu: batch chi xong
    khi request DAI NHAT xong, request ngan
    trong batch phai cho.

    [TRAINER] arrivals (mac dinh None = moi
    request den luc t = 0, giong ban goc): batch
    khong the bat dau truoc khi request den
    muon nhat trong batch da den; thoi gian
    cho cua 1 request = hoan thanh - thoi diem
    den - do dai (truoc day thieu so hang thoi
    diem den vi gia dinh den = 0).

    Returns:
        {"sum_completion": float, "avg_wait": float,
         "makespan": float} - sum_completion la
        tong thoi gian hoan thanh cua moi
        request; avg_wait la thoi gian cho
        trung binh = (hoan thanh - thoi diem
        den - do dai); makespan la thoi gian
        request cuoi cung xong.

    Raises:
        ValueError: lengths / max_batch /
            arrivals sai.
    """
    ds = _kiem_tra_danh_sach_do_dai(lengths)
    _kiem_tra_so_nguyen("max_batch", max_batch)
    arr = _kiem_tra_arrivals(arrivals, len(ds))
    thoi_gian = 0
    tong_hoan_thanh = 0
    tong_cho = 0
    for i in range(0, len(ds), max_batch):
        batch = ds[i:i + max_batch]
        den = arr[i:i + max_batch]
        thoi_gian = max(thoi_gian, max(den)) \
            + max(batch)
        for l, a in zip(batch, den):
            tong_hoan_thanh += thoi_gian
            tong_cho += thoi_gian - a - l
    return {"sum_completion": float(tong_hoan_thanh),
            "avg_wait": tong_cho / len(ds),
            "makespan": float(thoi_gian)}


def simulate_continuous(lengths, max_batch=MAX_BATCH,
                        arrivals=None):
    """Mo phong continuous batching.

    Request nao xong thi roi ngay, request moi
    chen vao o buoc tiep theo (slot trong thi
    lap day tu hang doi FIFO). Moi buoc decode
    ton 1 don vi thoi gian cho moi request
    dang chay. Dung deque cho hang doi va list
    cho tap dang chay (20 request, buoc toi
    da 500 - khong can heap phuc tap).

    [TRAINER] arrivals (mac dinh None = den o
    t = 0 het, giong ban goc): request chi vao
    hang doi khi da den (sap xep theo thoi
    diem den, cung thoi diem thi theo thu tu
    goc); chua co gi chay ma request ke tiep
    chua den thi dong ho nhay thang toi luc
    no den (khong quay vong rong, khong cho
    request chay truoc khi den).

    Returns:
        Giong simulate_static (cung dinh nghia
        sum_completion / avg_wait / makespan,
        de so sanh truc tiep).

    Raises:
        ValueError: lengths / max_batch /
            arrivals sai.
    """
    ds = _kiem_tra_danh_sach_do_dai(lengths)
    _kiem_tra_so_nguyen("max_batch", max_batch)
    arr = _kiem_tra_arrivals(arrivals, len(ds))
    # Thu tu nap: den som truoc, hoa thi theo
    # thu tu trong list goc.
    thu_tu = sorted(range(len(ds)),
                    key=lambda i: (arr[i], i))
    ke_tiep = 0
    hang_doi = deque()
    # Moi phan tu dang chay: [con_lai, do_dai_goc,
    # thoi_diem_den].
    dang_chay = []
    thoi_gian = 0
    tong_hoan_thanh = 0
    tong_cho = 0
    while ke_tiep < len(thu_tu) or hang_doi \
            or dang_chay:
        while ke_tiep < len(thu_tu) \
                and arr[thu_tu[ke_tiep]] <= thoi_gian:
            i = thu_tu[ke_tiep]
            ke_tiep += 1
            hang_doi.append((ds[i], arr[i]))
        while hang_doi and len(dang_chay) < max_batch:
            l, a = hang_doi.popleft()
            dang_chay.append([l, l, a])
        if not dang_chay:
            # Chua request nao den: nhay dong ho
            # toi thoi diem den ke tiep.
            thoi_gian = arr[thu_tu[ke_tiep]]
            continue
        thoi_gian += 1
        con_lai = []
        for con, goc, a in dang_chay:
            con -= 1
            if con == 0:
                tong_hoan_thanh += thoi_gian
                tong_cho += thoi_gian - a - goc
            else:
                con_lai.append([con, goc, a])
        dang_chay = con_lai
    return {"sum_completion": float(tong_hoan_thanh),
            "avg_wait": tong_cho / len(ds),
            "makespan": float(thoi_gian)}


def speculative_speedup(alpha, draft_speedup=None):
    """Toc do tang cua speculative decoding.

    Moi lan kiem chung, model lon doc trong so
    1 lan va cho ra alpha token dung (trung
    binh) + 1 token sua o cuoi: alpha + 1 token
    / lan doc. Neu draft_speedup=None (chi phi
    draft khong dang ke): tra alpha + 1.
    Neu cho draft_speedup = r (draft nhanh hon
    model lon r lan / token): moi vong draft
    ton (alpha + 1) / r lan doc, toc do tang
    rong = (alpha + 1) / (1 + (alpha + 1) / r).

    Args:
        alpha: so token dung trung binh / vong,
            so >= 0.
        draft_speedup: so > 0, hoac None.

    Returns:
        Toc do tang (float, lan).

    Luu y [TRAINER]: alpha o day la SO TOKEN
    chap nhan ky vong moi vong (so dem, >= 0,
    co the > 1), KHONG phai ti le chap nhan
    theo tung token nhu trong tai lieu
    speculative decoding goc. Neu co ti le chap
    nhan beta / token va gamma token nhap moi
    vong thi so token ky vong / vong la
    (1 - beta ** (gamma + 1)) / (1 - beta) -
    phai doi beta, gamma ve alpha theo dung
    nghia do truoc khi dung cong thuc alpha + 1
    o day, neu khong se tinh thieu (vd beta =
    0.7, gamma = 4 cho ky vong ~2.77 token, ma
    alpha + 1 chi cho 1.7x).
    """
    _kiem_tra_so_thuc("alpha", alpha, lon_hon_bang=0)
    if draft_speedup is not None:
        _kiem_tra_so_thuc("draft_speedup",
                          draft_speedup, lon_hon=0)
        return (alpha + 1.0) / (
            1.0 + (alpha + 1.0) / draft_speedup)
    return alpha + 1.0


def breakeven_draft_speedup(alpha):
    """Draft phai nhanh hon bao nhieu lan moi co lai.

    Giai (alpha + 1) / (1 + (alpha + 1) / r) > 1
    theo r: r > (alpha + 1) / alpha (voi
    alpha > 0). alpha = 0: draft khong doan
    dung token nao, toc do rong luon < 1 - tra
    None (khong co nguong huu han nao co lai).

    Returns:
        Float (so lan), hoac None khi alpha = 0.
    """
    _kiem_tra_so_thuc("alpha", alpha, lon_hon_bang=0)
    if alpha == 0:
        return None
    return (alpha + 1.0) / alpha


def _expect_value_error(fn, mo_ta):
    """Chay fn, doi no nem ValueError, in ket qua."""
    try:
        fn()
    except ValueError as exc:
        print(f"  guard {mo_ta}: chan dung ({exc})")
        return True
    print(f"  CANH BAO: guard {mo_ta} KHONG chan!")
    return False


def _sai_so_lon_nhat(a, b):
    """Sai so tuyet doi lon nhat giua 2 tensor."""
    return float(
        (a.detach() - b.detach()).abs().max())


def tu_kiem_tra():
    """Tu kiem tra bang assert, chay truoc lab.

    Kiem tra guard, cong thuc KV (doi chieu
    voi cache that), tinh dung cua KV cache
    (prefill + decode co cache bang prefill
    day du), lang phi paged (tinh tay), 2 mo
    phong batching (truong hop tinh tay va
    truong hop do dai bang nhau), va cong thuc
    speculative decoding. Chi IN khi assert
    da qua (checklist 13).
    """
    torch.manual_seed(123)
    ket_qua = [
        _expect_value_error(
            lambda: kv_bytes_per_token(0, 256, 4),
            "kv n_layer=0"),
        _expect_value_error(
            lambda: kv_bytes_per_token(4, 256, 3),
            "kv bytes_per_param=3"),
        _expect_value_error(
            lambda: kv_cache_bytes(4, 256, 4, -1),
            "kv n_tokens am"),
        _expect_value_error(
            lambda: Block(255, 4, 64), "Block d%head"),
        _expect_value_error(
            lambda: MiniTransformer(vocab=0),
            "MiniTransformer vocab=0"),
        _expect_value_error(
            lambda: paged_waste([], 512, 16),
            "paged lengths rong"),
        _expect_value_error(
            lambda: paged_waste([600], 512, 16),
            "paged do dai vuot max_len"),
        _expect_value_error(
            lambda: paged_waste([100], 512, 0),
            "paged block=0"),
        _expect_value_error(
            lambda: simulate_static([], 4),
            "static lengths rong"),
        _expect_value_error(
            lambda: simulate_continuous([100], 0),
            "continuous max_batch=0"),
        _expect_value_error(
            lambda: speculative_speedup(-0.5),
            "speculative alpha am"),
        _expect_value_error(
            lambda: speculative_speedup(1, 0),
            "speculative draft_speedup=0"),
        _expect_value_error(
            lambda: decode_greedy(
                MiniTransformer(vocab=50, d_model=32,
                                n_layer=2, n_head=2,
                                d_ff=64),
                [], 0, 1),
            "decode caches sai"),
        _expect_value_error(
            lambda: bench(42), "bench fn khong goi"),
        # [TRAINER] Guard moi kem cac sua o vong
        # review Day 27: _phan_vi, arrivals cua 2
        # mo phong, va cau truc (K, V) cua
        # kv_caches (truoc day ro TypeError).
        _expect_value_error(
            lambda: _phan_vi([], 95),
            "phan vi mau rong"),
        _expect_value_error(
            lambda: _phan_vi([1.0], 101),
            "phan vi p > 100"),
        _expect_value_error(
            lambda: simulate_static(
                [10, 10], 2, arrivals=[0]),
            "static arrivals thieu phan tu"),
        _expect_value_error(
            lambda: simulate_continuous(
                [10, 10], 2, arrivals=[0, -1]),
            "continuous arrival am"),
        _expect_value_error(
            lambda: MiniTransformer(
                vocab=50, d_model=32, n_layer=2,
                n_head=2, d_ff=64).forward(
                    torch.zeros(1, 1,
                                dtype=torch.long),
                    [1, 2]),
            "kv_caches phan tu khong phai (K, V)"),
    ]
    assert all(ket_qua), "Co guard khong chan dung!"

    # [TRAINER] _phan_vi nearest-rank: p95 cua
    # [1..20] la phan tu thu ceil(0.95*20)=19;
    # p50 cua [1..10] la 5 (khong phai 5.5).
    assert _phan_vi(
        [float(i) for i in range(1, 21)], 95) == 19.0
    assert _phan_vi(
        [float(i) for i in range(1, 11)], 50) == 5.0
    assert _phan_vi([7.5], 95) == 7.5

    # [TRAINER] Arrivals lech pha, tinh tay:
    # continuous [10, 10], batch 1, den [0, 9]:
    # req1 chay 0->10; req2 den t=9 nhung phai
    # cho slot, chay 10->20: tong hoan thanh
    # 30, cho TB (0 + 1) / 2.
    kq_den = simulate_continuous(
        [10, 10], 1, arrivals=[0, 9])
    assert kq_den["sum_completion"] == 30
    assert kq_den["avg_wait"] == 0.5
    # Static [10, 10], batch 2, den [0, 9]: batch
    # cho req muon nhat, bat dau t=9, xong t=19:
    # tong 38, cho TB ((19-0-10) + (19-9-10))/2.
    kq_den_tinh = simulate_static(
        [10, 10], 2, arrivals=[0, 9])
    assert kq_den_tinh["sum_completion"] == 38
    assert kq_den_tinh["avg_wait"] == 4.5
    # Khong truyen arrivals thi phai y het ban
    # goc (tat ca den t = 0).
    assert simulate_continuous(
        [10, 10], 1)["sum_completion"] == 30
    assert simulate_static(
        [10, 10], 2)["sum_completion"] == 20

    # Test duong: token id 0 la hop le, forward
    # voi tokens toan 0 phai chay duoc (khong
    # nham 0 voi "rong" hay "sai").
    assert MiniTransformer(
        vocab=50, d_model=32, n_layer=2, n_head=2,
        d_ff=64).forward(
            torch.zeros(1, 4,
                        dtype=torch.long))[0].shape \
        == (1, 4, 50)

    # Cong thuc KV: 4 lop, d=256, FP32 -> 8192
    # byte/token; FP16 -> 4096; INT8 -> 2048.
    assert kv_bytes_per_token(4, 256, 4) == 8192
    assert kv_bytes_per_token(4, 256, 2) == 4096
    assert kv_bytes_per_token(4, 256, 1) == 2048
    assert kv_cache_bytes(4, 256, 4, 2000) \
        == 8192 * 2000

    # KV cache dung: model nho, prefill 5 token
    # roi decode 3 token co cache phai bang
    # prefill day du 8 token (logits cuoi).
    torch.manual_seed(7)
    nho = MiniTransformer(vocab=50, d_model=32,
                          n_layer=2, n_head=2, d_ff=64)
    nho.eval()
    tokens = torch.randint(0, 50, (1, 8))
    with torch.no_grad():
        logits_day, _ = nho(tokens)
        logits_ngan, caches = nho(tokens[:, :5])
        _, caches = nho(tokens[:, 5:6], caches)
        _, caches = nho(tokens[:, 6:7], caches)
        logits_cache, caches = nho(tokens[:, 7:8],
                                   caches)
    assert _sai_so_lon_nhat(
        logits_day[:, -1], logits_cache[:, -1]) \
        <= 1e-5, "decode co cache phai bang prefill."
    # Cache that dem tu tensor phai bang cong thuc.
    assert nho.cache_bytes_thuc_te(caches) \
        == kv_cache_bytes(2, 32, 4, 8), \
        "byte cache that phai bang cong thuc."
    # Decode tham lam tra dung so token, dung vocab.
    sinh, _ = decode_greedy(
        nho, caches, int(tokens[0, -1]), 3)
    assert len(sinh) == 3
    assert all(0 <= t < 50 for t in sinh)

    # Paged: tinh tay [50, 500, 16, 17], block 16,
    # max_len 512: naive thua 1465, paged thua 41.
    kq = paged_waste([50, 500, 16, 17], 512, 16)
    assert kq["naive_waste"] == 1465
    assert kq["paged_waste"] == 41
    assert kq["naive_alloc"] == 2048
    assert kq["paged_alloc"] == 64 + 512 + 16 + 32
    assert abs(kq["naive_waste_pct"]
               - 1465 / 2048 * 100) <= 1e-9
    assert abs(kq["paged_waste_pct"]
               - 41 / 624 * 100) <= 1e-9

    # Batching, truong hop tinh tay: [1, 10, 10, 10],
    # batch 2: static sum = 60, continuous sum = 42.
    assert simulate_static(
        [1, 10, 10, 10], 2)["sum_completion"] == 60
    assert simulate_continuous(
        [1, 10, 10, 10], 2)["sum_completion"] == 42
    assert simulate_continuous(
        [1, 10, 10, 10], 2)["avg_wait"] \
        == (0 + 0 + 1 + 10) / 4
    # Do dai bang nhau: continuous khong thang
    # static (cung sum_completion).
    assert simulate_static(
        [200] * 20, 4)["sum_completion"] \
        == simulate_continuous(
            [200] * 20, 4)["sum_completion"]
    # Continuous khong bao gio te hon static.
    random.seed(0)
    ds = [random.randint(50, 500) for _ in range(20)]
    assert simulate_continuous(ds, 4)["sum_completion"] \
        <= simulate_static(ds, 4)["sum_completion"]

    # Speculative: alpha 0 -> 1x, 4 -> 5x; alpha
    # 0.5, draft nhanh gap 10 -> 1.5/1.15; nguong
    # hoa von cho alpha 0.5 la 3.0; alpha 0 thi
    # khong co nguong huu han.
    assert speculative_speedup(0) == 1.0
    assert speculative_speedup(4) == 5.0
    assert abs(speculative_speedup(0.5, 10)
               - 1.5 / 1.15) <= 1e-9
    assert breakeven_draft_speedup(0.5) == 3.0
    assert breakeven_draft_speedup(0) is None
    print("Tu kiem tra: PASS (tat ca assert deu qua).")

def _ket_luan_ti_le(ti_le, ten_nhanh, ten_cham,
                    nguong_duoi=0.8, nguong_tren=1.25):
    """Ket luan ti le 3 nhanh (checklist 22).

    < nguong_duoi: nhanh hon; trong khoang:
    gan nhu hoa; > nguong_tren: cham hon.
    Khong dung else mac dinh "cham hon X"
    vi se in nguoc khi ti le < 1.
    """
    if ti_le < nguong_duoi:
        return (f"{ten_nhanh} nhanh hon "
                f"({ti_le:.2f}x thoi gian)")
    if ti_le <= nguong_tren:
        return (f"{ten_nhanh} gan nhu hoa voi "
                f"{ten_cham} ({ti_le:.2f}x)")
    return (f"{ten_nhanh} cham hon {ten_cham} "
            f"{ti_le:.2f}x")


def bai_a():
    """Bai A: KV cache + TTFT / TPOT (3 do dai).

    Thu duy nhat thay doi la do dai prompt;
    cung mo hinh, cung phan cung, batch = 1,
    cung 50 token decode. TTFT = thoi gian
    prefill (trung vi, da warm-up); TPOT =
    thoi gian decode trung binh / token.
    KV cache tinh theo cong thuc, doi chieu
    voi cache that o tu_kiem_tra.
    """
    print("\n" + "=" * 72)
    print("BAI A - KV cache + prefill vs decode")
    print("=" * 72)
    print("Thu duy nhat thay doi la do dai prompt; "
          "cung mo hinh, cung phan cung, batch = 1, "
          "cung 50 token decode.")
    torch.manual_seed(SEED)
    model = MiniTransformer()
    n_tham_so = sum(p.numel()
                    for p in model.parameters())
    kv_fp32 = model.kv_bytes_per_token(4)
    kv_fp16 = model.kv_bytes_per_token(2)
    print(f"  Model: {N_LAYER} lop, d_model={D_MODEL}, "
          f"{n_tham_so} tham so "
          f"({n_tham_so * 4 / 2 ** 20:.1f} MiB FP32).")
    print(f"  KV cache / token: FP32 {kv_fp32} byte, "
          f"FP16 {kv_fp16} byte.")
    print(f"  Vi du: 1 request 2.000 token FP16 = "
          f"{kv_fp16 * 2000 / 2 ** 20:.0f} MiB; "
          f"64 request = "
          f"{kv_fp16 * 2000 * 64 / 2 ** 20:.0f} MiB.")
    ket_qua = {}
    for do_dai in DO_DAI_PROMPT:
        prompt = _tao_prompt(do_dai)
        # [TRAINER] Do bang bench_day_du de co ca
        # trung vi (so chinh, giong ban goc) lan
        # p95 (brief Day 27 yeu cau tach TTFT /
        # TPOT va bao ca p95, khong gop 1 so duy
        # nhat). p95 cua TPOT = p95 tong thoi gian
        # decode / so token (percentile la ham
        # don dieu nen chia deu duoc).
        ttft, ttft_p95 = bench_day_du(
            lambda: prefill(model, prompt))
        _, caches = prefill(model, prompt)
        token_dau = int(prompt[0, -1])
        # Bench decode: moi lan chay decode lai
        # 50 token tu cung cache ban dau - dung
        # cache da prefill, khong tinh prefill
        # vao thoi gian decode.
        t_decode, t_decode_p95 = bench_day_du(
            lambda: decode_greedy(
                model, caches, token_dau, N_DECODE))
        tpot = t_decode / N_DECODE
        tpot_p95 = t_decode_p95 / N_DECODE
        kv_mb = kv_fp32 * do_dai / 2 ** 20
        ket_qua[do_dai] = (ttft, tpot, kv_mb)
        print(f"  Prompt {do_dai}: TTFT={ttft:.1f} ms "
              f"(p95 {ttft_p95:.1f}) | "
              f"TPOT={tpot:.2f} ms/token "
              f"(p95 {tpot_p95:.2f}) "
              f"({1000.0 / tpot:.0f} token/s) | "
              f"KV cache={kv_mb:.1f} MiB")
    ti_le_ttft = ket_qua[2048][0] / ket_qua[128][0]
    ti_le_tpot = ket_qua[2048][1] / ket_qua[128][1]
    if ti_le_ttft <= 20:
        ttft = (f"TTFT tang {ti_le_ttft:.1f}x khi "
                f"prompt dai gap 16x - gan tuyen "
                f"tinh, phan MLP (O(n)) dang chi "
                f"phoi o do dai nay")
    elif ti_le_ttft <= 40:
        ttft = (f"TTFT tang {ti_le_ttft:.1f}x khi "
                f"prompt dai gap 16x - nhanh hon "
                f"tuyen tinh, attention O(n^2) "
                f"bat dau lo ro")
    else:
        ttft = (f"TTFT tang {ti_le_ttft:.1f}x khi "
                f"prompt dai gap 16x - sieu tuyen "
                f"tinh ro, attention O(n^2) chi phoi")
    if 0.8 <= ti_le_tpot <= 1.25:
        tpot = (f"TPOT gan nhu khong doi "
                f"({ti_le_tpot:.2f}x) khi prompt dai "
                f"ra - decode moi buoc chi doc trong "
                f"so + KV, khong tinh lai ca prompt")
    elif ti_le_tpot < 0.8:
        tpot = (f"TPOT giam con {ti_le_tpot:.2f}x - "
                f"nam trong nhieu do CPU, khong doi "
                f"ket luan chung")
    else:
        tpot = (f"TPOT tang {ti_le_tpot:.2f}x khi "
                f"prompt dai ra - KV cache dai hon "
                f"lam moi buoc decode doc nhieu hon")
    print(f"Ket luan: {ttft}; {tpot}. Prefill "
          f"compute-bound, decode memory-bound - "
          f"dung nhu ly thuyet buoi sang.")
    return {"ttft": {d: ket_qua[d][0]
                     for d in DO_DAI_PROMPT},
            "tpot": {d: ket_qua[d][1]
                     for d in DO_DAI_PROMPT},
            "kv_fp32": kv_fp32, "kv_fp16": kv_fp16,
            "model": model}


def bai_b():
    """Bai B: PagedAttention + continuous batching.

    20 request do dai ngau nhien 50-500 (seed
    0, dung nhu de bai). So naive vs paged
    (bo nho) va static vs continuous (thoi
    gian cho). Them truong hop doi chieu do
    dai bang nhau de tra loi cau hoi cua de.
    """
    print("\n" + "=" * 72)
    print("BAI B - PagedAttention + continuous "
          "batching")
    print("=" * 72)
    random.seed(SEED)
    lengths = [random.randint(50, 500)
               for _ in range(N_REQUEST)]
    print(f"  20 request, do dai {min(lengths)}-"
          f"{max(lengths)} token, tong "
          f"{sum(lengths)} token.")
    lang_phi = paged_waste(lengths, MAX_LEN_NAIVE,
                           BLOCK_TOKEN)
    print(f"  PagedAttention: naive thua "
          f"{lang_phi['naive_waste']} token "
          f"({lang_phi['naive_waste_pct']:.1f}% bo "
          f"nho cap phat) vs paged thua "
          f"{lang_phi['paged_waste']} token "
          f"({lang_phi['paged_waste_pct']:.1f}%).")
    tinh = simulate_static(lengths, MAX_BATCH)
    lien = simulate_continuous(lengths, MAX_BATCH)
    giam = (tinh["sum_completion"]
            - lien["sum_completion"]) \
        / tinh["sum_completion"] * 100.0
    print(f"  Static: tong hoan thanh "
          f"{tinh['sum_completion']:.0f} | cho TB "
          f"{tinh['avg_wait']:.1f} | makespan "
          f"{tinh['makespan']:.0f}.")
    print(f"  Continuous: tong hoan thanh "
          f"{lien['sum_completion']:.0f} | cho TB "
          f"{lien['avg_wait']:.1f} | makespan "
          f"{lien['makespan']:.0f}.")
    if giam < 5:
        thang = (f"continuous chi giam {giam:.1f}% "
                 f"tong hoan thanh - do dai cac "
                 f"request khong chenh nhau nhieu")
    elif giam <= 20:
        thang = (f"continuous giam {giam:.1f}% tong "
                 f"hoan thanh - request ngan thoat "
                 f"sot, khong phai cho request dai")
    else:
        thang = (f"continuous giam {giam:.1f}% tong "
                 f"hoan thanh - thang lon nho do dai "
                 f"chenh lech manh")
    print(f"Ket luan: {thang}. Paged giam lang phi "
          f"tu {lang_phi['naive_waste_pct']:.1f}% "
          f"xuong {lang_phi['paged_waste_pct']:.1f}%.")
    # Doi chieu: do dai bang nhau thi continuous
    # khong thang static (cung sum_completion).
    bang = [200] * N_REQUEST
    tinh_bang = simulate_static(bang, MAX_BATCH)
    lien_bang = simulate_continuous(bang, MAX_BATCH)
    print(f"  Doi chieu (20 request deu 200 token): "
          f"static tong hoan thanh "
          f"{tinh_bang['sum_completion']:.0f} = "
          f"continuous "
          f"{lien_bang['sum_completion']:.0f} - "
          f"continuous chi thang khi do dai chenh "
          f"lech, khong thang khi moi request dai "
          f"nhu nhau.")
    return {"lang_phi": lang_phi,
            "tinh": tinh, "lien": lien,
            "giam_pct": giam}


def bai_c(model, kv_fp16):
    """Bai C: speculative, prefix cache, INT8.

    C1: bang alpha -> speedup (draft khong
    dang ke) + alpha = 0.5 voi draft nhanh
    gap 10 lan va nguong hoa von.
    C2: 20 request chung system prompt 200
    token - KV tiet kiem + thoi gian prefill
    tiet kiem (do that prefill 200 token).
    C3: KV cache FP16 -> INT8, toc do decode
    ly thuyet + cai gia phai tra.
    """
    print("\n" + "=" * 72)
    print("BAI C - Speculative, prefix cache, INT8")
    print("=" * 72)
    print("  C1 speculative decoding (draft khong "
          "dang ke: speedup = alpha + 1):")
    for alpha in (0, 1, 2, 3, 4):
        print(f"     alpha={alpha}: "
              f"{speculative_speedup(alpha):.1f}x")
    net = speculative_speedup(0.5, 10)
    nguong = breakeven_draft_speedup(0.5)
    print(f"     alpha=0.5, draft nhanh gap 10x: "
          f"toc do rong {net:.2f}x - van dang "
          f"dung vi draft re; draft phai nhanh hon "
          f"{nguong:.1f}x moi co lai o alpha=0.5, "
          f"con alpha=0 thi khong bao gio co lai.")
    print("  C2 prefix caching (20 request, chung "
          f"system prompt {HE_SO_PREFIX} token):")
    # [TRAINER] So request dung chung prefix la
    # N_REQUEST - 1 (tru request dau tien phai tu
    # cache): ban goc go tay "19" o 3 cho - doi
    # N_REQUEST la cac so nay sai am tham trong
    # khi code van chay so khac (bai hoc Day 26:
    # so trong ket luan phai tinh tu so do / hang
    # so, khong go tay). Gia tri in ra khong doi.
    so_dung_chung = N_REQUEST - 1
    kv_tiet_kiem = so_dung_chung * HE_SO_PREFIX \
        * kv_fp16
    prompt_200 = _tao_prompt(HE_SO_PREFIX)
    t_prefill_200 = bench(
        lambda: prefill(model, prompt_200))
    print(f"     KV tiet kiem: {so_dung_chung} "
          f"request x "
          f"{HE_SO_PREFIX} token x {kv_fp16} byte "
          f"= {kv_tiet_kiem / 2 ** 20:.1f} MiB "
          f"(request dau tien van phai cache).")
    print(f"     Prefill tiet kiem: "
          f"{so_dung_chung} x "
          f"{t_prefill_200:.1f} ms = "
          f"{so_dung_chung * t_prefill_200:.0f} ms (do that "
          f"prefill 200 token). Khong giup gi khi "
          f"chi 1 request, khong chung prefix, "
          f"hoac block prefix da bi evict khoi "
          f"cache.")
    print("  C3 quantized serving (KV FP16 -> INT8):")
    kv_int8 = model.kv_bytes_per_token(1)
    print(f"     KV / token: FP16 {kv_fp16} byte -> "
          f"INT8 {kv_int8} byte (giam "
          f"{kv_fp16 / kv_int8:.1f}x). Decode "
          f"memory-bound nen nhanh len toi da "
          f"{kv_fp16 / kv_int8:.1f}x theo ly "
          f"thuyet (gioi han tren - thuc te thap "
          f"hon do chi phi dequant). Cai gia: "
          f"chat luong co the giam - do bang "
          f"accuracy/perplexity truoc-sau nhu "
          f"gate Day 22, khong doan bang cam giac.")
    return {"net_alpha_0_5": net,
            "nguong_alpha_0_5": nguong,
            "kv_tiet_kiem_mib": kv_tiet_kiem / 2 ** 20,
            "prefill_200_ms": t_prefill_200}


def tra_loi_tu_kiem_tra():
    """Tra loi ngan 3 cau tu kiem tra trong de."""
    print("\nTRA LOI TU KIEM TRA (3 cau trong de):")
    print("  1. Prefill xu ly nhieu token cung luc "
          "nen moi lan doc trong so duoc chia cho "
          "nhieu token - gioi han o tinh toan. "
          "Decode moi buoc chi 1 token nhung van "
          "doc toan bo trong so + KV cache - gioi "
          "han o bang thong bo nho.")
    print("  2. PagedAttention go nut that bo nho: "
          "cap phat theo block dung nhu cau nen "
          "cung GPU chua duoc nhieu request hon, "
          "GPU luon co viec - throughput tang ma "
          "mo hinh khong doi gi.")
    print("  3. Speculative decoding: token cuoi "
          "deu duoc model lon kiem chung nen dau "
          "ra giong het model lon. Quantization "
          "doi chinh trong so/phep tinh nen dau ra "
          "co the khac. Chon quantization khi bo "
          "nho la nut that; chon speculative khi "
          "chat luong khong duoc doi va van ban "
          "du de doan.")


def main():
    """Chay ca lab: tu kiem tra, 3 bai, tom tat."""
    print("Lab serving Day 27 - "
          f"torch {torch.__version__}")
    print("Thiet bi: "
          f"{'cuda' if torch.cuda.is_available() else 'cpu'} "
          f"- so ms chi dung cho may nay, ket luan "
          f"dung duoc la ti le / xu huong.")
    tu_kiem_tra()
    kq_a = bai_a()
    kq_b = bai_b()
    kq_c = bai_c(kq_a["model"], kq_a["kv_fp16"])
    # [TRAINER] Bang ket qua chua co cot nang
    # luong (kWh/token hoac token/kWh) theo brief
    # Day 27: lab khong do cong suat (W) nen
    # khong co so lieu tinh nang luong - khong
    # bia so. Chi so hieu suat hien co la token/s
    # suy tu TPOT o Bai A; muon co cot nang luong
    # phai do cong suat that khi chay serving.
    print("\nBANG KET QUA: TTFT 128/512/2048 = "
          f"{kq_a['ttft'][128]:.1f}/"
          f"{kq_a['ttft'][512]:.1f}/"
          f"{kq_a['ttft'][2048]:.1f} ms | TPOT = "
          f"{kq_a['tpot'][128]:.2f}/"
          f"{kq_a['tpot'][512]:.2f}/"
          f"{kq_a['tpot'][2048]:.2f} ms/token | KV "
          f"FP32/token {kq_a['kv_fp32']} byte | lang "
          f"phi naive "
          f"{kq_b['lang_phi']['naive_waste_pct']:.1f}% "
          f"-> paged "
          f"{kq_b['lang_phi']['paged_waste_pct']:.1f}% "
          f"| continuous giam tong hoan thanh "
          f"{kq_b['giam_pct']:.1f}% | speculative "
          f"alpha=0.5 rong {kq_c['net_alpha_0_5']:.2f}x "
          f"| prefix tiet kiem "
          f"{kq_c['kv_tiet_kiem_mib']:.1f} MiB.")
    print("Ket luan chung: prefill compute-bound, "
          "decode memory-bound; KV cache la nut "
          "that bo nho so 1; PagedAttention + "
          "continuous batching tang throughput bang "
          "cach dung bo nho / GPU khon hon, khong "
          "doi mo hinh.")
    tra_loi_tu_kiem_tra()


if __name__ == "__main__":
    main()
