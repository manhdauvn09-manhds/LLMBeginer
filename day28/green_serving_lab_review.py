"""Lab Day 28 - Mini-project 4: Green serving va Pareto.

Benchmark 6 cau hinh phuc vu (distilgpt2, CPU)
tren cung prompt set, do du 3 truc: chat luong
(PPL) - do tre (TTFT/TPOT, p50 va p95) - nang
luong (Joules/request), roi tim Pareto frontier
va chon cau hinh theo SLO.

"Thu duy nhat thay doi la ___" (quy tac Day 24
+ Day 28): moi lan chay chi doi 1 bien (dtype,
hoac batch size, hoac use_cache). Moi thu khac
- cung prompt set, cung concurrency, cung
warmup, cung phan cung - giu nguyen. So sanh
INT8 batch 32 voi FP16 batch 1 roi ket luan
"INT8 nhanh gap 5 lan" la ket luan vo gia tri.

Nang luong: MAY NAY KHONG DO CONG SUAT THAT.
Dung cong suat GIA DINH 65 W (laptop CPU) cho
MOI cau hinh, ghi ro o day va trong bao cao -
cung mot cach tinh cho moi cau hinh, theo dung
de bai. Muon so nang luong that phai do cong
suat bang codecarbon / dong ho do dien.

May chay: CPU (khong GPU), torch CPU. So ms /
J chi dung cho may nay; ket luan dung duoc la
thu hang / frontier / xu huong. Tren CPU, FP16
thuong khong nhanh hon FP32 - neu do ra nhu
vay thi do la phat hien, khong phai loi.

Chay: python green_serving_lab.py
Ket qua: in bang 6 cau hinh + frontier + 3 lua
chon theo SLO, luu results.json va pareto.png
cung thu muc voi file nay.
"""

# ============================================================
# [TRAINER] REVIEW Day 28 (ban green_serving_lab_review.py)
# Kiem chung doc lap: tu_kiem_tra PASS khi chay lai;
# frontier tinh lai tu results.json dung {B} theo 3 truc
# code da chon (ppl, ttft_p95, joules). NHUNG co 2 diem
# chinh can sua truoc khi ket luan "frontier chi co B":
#  (1) TTFT/TPOT trong code la CHI PHI KHAU HAO theo
#      request (chia cho kich thuoc batch), khong phai do
#      tre nguoi dung trai nghiem - xem comment o
#      _sinh_mot_batch va chon_chatbot.
#  (2) So lieu cau hinh C khoi phuc tu log lan chay khac,
#      thieu 3 truong TPOT - xem comment o
#      _doc_ket_qua_da_co.
# 3 diem nhe: p95 qua it mau o batch lon (do_mot_cau_hinh),
# frontier chua dung truc TPOT (_kiem_tra_diem), hinh ve
# 2 truc trong khi frontier tinh 3 truc (ve_pareto).
# Moi cho sua deu co comment [TRAINER] ngay trong code.
# ============================================================

import json
import math
import os
import statistics
import time

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import transformers
from transformers import AutoModelForCausalLM
from transformers import AutoTokenizer

MODEL = "distilgpt2"
# Cung prompt set cho MOI cau hinh - khong doi
# giua cac lan chay.
PROMPTS = [
    "The future of artificial intelligence is",
    "To save energy, a data center should",
    "The capital of France is",
    "In machine learning, overfitting means",
    "A transformer model uses attention to",
    "The quick brown fox jumps over",
    "Renewable energy is important because",
    "To train a neural network, we need",
    "The largest planet in the solar system is",
    "A good benchmark should measure",
]
# Cau co dinh de tinh perplexity (chat luong) -
# khong trung prompt set.
PPL_TEXTS = [
    "The cat sat on the mat and looked out "
    "of the window.",
    "Machine learning models learn patterns "
    "from data.",
    "The sun rises in the east and sets in "
    "the west.",
]
# Cong suat GIA DINH (xem docstring dau file):
# 65 W CPU laptop; doi thanh 250 neu chay GPU.
ASSUMED_WATTS = 65.0
N_REPS = 3  # moi cau hinh chay 3 lan, bao TB +- LC
WARMUP_PASSES = 2  # so luot chay khoi dong / rep
MEASURE_PASSES = 3  # so luot do that / rep
N_DECODE = 16  # so token sinh them / request
SLO_CHATBOT_TTFT_P95 = 0.300  # giay, chatbot
# Batch offline chap nhan PPL tang toi da 5%
# so voi cau hinh goc FP32 batch 1 (cau hinh A).
PPL_TANG_TOI_DA = 0.05
CONFIGS = [
    {"name": "A", "dtype": torch.float32, "batch": 1,
     "use_cache": True},
    {"name": "B", "dtype": torch.float32, "batch": 8,
     "use_cache": True},
    {"name": "C", "dtype": torch.float16, "batch": 1,
     "use_cache": True},
    {"name": "D", "dtype": torch.float16, "batch": 8,
     "use_cache": True},
    {"name": "E", "dtype": torch.float16, "batch": 32,
     "use_cache": True},
    {"name": "F", "dtype": torch.float32, "batch": 8,
     "use_cache": False},
]
DUONG_KET_QUA = os.path.join(
    os.path.dirname(os.path.abspath(__file__)),
    "results.json")
DUONG_PARETO = os.path.join(
    os.path.dirname(os.path.abspath(__file__)),
    "pareto.png")


def _kiem_tra_so_nguyen(ten, gia_tri, toi_thieu=1):
    """Guard: so nguyen (tru bool) >= toi_thieu."""
    if isinstance(gia_tri, bool) \
            or not isinstance(gia_tri, int) \
            or gia_tri < toi_thieu:
        raise ValueError(
            f"{ten} phai la so nguyen >= {toi_thieu}.")


def _kiem_tra_so_thuc(ten, gia_tri, lon_hon_bang=None,
                      lon_hon=None):
    """Guard: so thuc huu han, trong khoang neu co.

    Chi nhan int/float (tru bool). NaN/Inf bi
    chan. Moi loi dau vao deu la ValueError
    (quy uoc chung cua ca lab, xem Day 27).
    """
    if isinstance(gia_tri, bool) \
            or not isinstance(gia_tri, (int, float)):
        raise ValueError(f"{ten} phai la so.")
    gia_tri = float(gia_tri)
    if not math.isfinite(gia_tri):
        raise ValueError(f"{ten} phai huu han.")
    if lon_hon_bang is not None \
            and gia_tri < lon_hon_bang:
        raise ValueError(f"{ten} phai >= {lon_hon_bang}.")
    if lon_hon is not None and gia_tri <= lon_hon:
        raise ValueError(f"{ten} phai > {lon_hon}.")
    return gia_tri


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


def _nang_luong(watts, giay, n_req):
    """Joules/request = cong suat x thoi gian / n.

    Cong suat o day la GIA DINH (xem docstring
    dau file), khong phai cong suat do duoc.
    """
    _kiem_tra_so_thuc("watts", watts, lon_hon=0)
    _kiem_tra_so_thuc("giay", giay, lon_hon_bang=0)
    _kiem_tra_so_nguyen("n_req", n_req)
    return watts * giay / n_req


def _kiem_tra_cau_hinh(cfg):
    """Guard cau hinh: du 4 khoa, kieu dung.

    dtype phai la torch.float32 / float16,
    batch int >= 1, use_cache bool, name la
    chuoi khong rong.

    Returns:
        Ban sao dict cau hinh (khong sua goc).
    """
    if not isinstance(cfg, dict):
        raise ValueError("cau hinh phai la dict.")
    for khoa in ("name", "dtype", "batch",
                 "use_cache"):
        if khoa not in cfg:
            raise ValueError(
                f"cau hinh thieu khoa '{khoa}'.")
    if not isinstance(cfg["name"], str) \
            or not cfg["name"]:
        raise ValueError("name phai la chuoi khong "
                         "rong.")
    if cfg["dtype"] not in (torch.float32,
                            torch.float16):
        raise ValueError(
            "dtype phai la torch.float32/float16.")
    _kiem_tra_so_nguyen("batch", cfg["batch"])
    if not isinstance(cfg["use_cache"], bool):
        raise ValueError("use_cache phai la bool.")
    return dict(cfg)


def _ten_dtype(dtype):
    """Ten ngan cua dtype: FP32 / FP16."""
    if dtype == torch.float32:
        return "FP32"
    return "FP16"


def _bien_thay_doi(cfg, goc):
    """Bien nao khac giua cfg va cau hinh goc.

    So 3 khoa dtype / batch / use_cache, tra
    list ten bien khac nhau (rong neu giong
    het). Dung de in "thu duy nhat thay doi"
    - tinh tu du lieu, khong go tay.
    """
    _kiem_tra_cau_hinh(cfg)
    _kiem_tra_cau_hinh(goc)
    khac = []
    if cfg["dtype"] != goc["dtype"]:
        khac.append("dtype")
    if cfg["batch"] != goc["batch"]:
        khac.append("batch size")
    if cfg["use_cache"] != goc["use_cache"]:
        khac.append("use_cache")
    return khac


def _cap_doi_cong_bang(configs):
    """Cap cau hinh chi khac nhau dung 1 bien.

    Day moi la cap so sanh cong bang theo quy
    tac 1 bien (Day 24/28): doi 2 bien cung
    luc thi ket luan khong con gia tri. Tra
    list tuple (ten_a, ten_b, bien_doi).

    Raises:
        ValueError: configs rong / trung ten /
            cau hinh sai.
    """
    if not isinstance(configs, (list, tuple)) \
            or len(configs) == 0:
        raise ValueError("configs khong duoc rong.")
    ten = [c["name"] for c in configs]
    if len(set(ten)) != len(ten):
        raise ValueError("ten cau hinh bi trung.")
    cap = []
    for i in range(len(configs)):
        for j in range(i + 1, len(configs)):
            khac = _bien_thay_doi(configs[i],
                                  configs[j])
            if len(khac) == 1:
                cap.append((configs[i]["name"],
                            configs[j]["name"],
                            khac[0]))
    return cap

def _tai_tokenizer():
    """Tai tokenizer distilgpt2.

    Padding trai (left) de token cuoi cung cua
    moi hang trong batch luon la token that
    cuoi cua prompt do - can cho vong lap sinh
    tung buoc o _sinh_mot_batch. Pad token =
    EOS (distilgpt2 khong co pad rieng).
    """
    tok = AutoTokenizer.from_pretrained(MODEL)
    tok.pad_token = tok.eos_token
    tok.padding_side = "left"
    return tok


def _tai_chi_model(dtype):
    """Tai distilgpt2 theo dtype, che do eval."""
    if dtype not in (torch.float32, torch.float16):
        raise ValueError(
            "dtype phai la torch.float32/float16.")
    model = AutoModelForCausalLM.from_pretrained(
        MODEL, dtype=dtype)
    model.eval()
    return model


def tai_model(dtype):
    """Tai tokenizer + distilgpt2 theo dtype.

    Returns:
        (tokenizer, model) o che do eval.
    """
    return _tai_tokenizer(), _tai_chi_model(dtype)


@torch.no_grad()
def perplexity(tok, model, texts):
    """PPL trung binh tren cac cau co dinh.

    Cang thap cang tot. Loss cua HF da dich
    chuyen noi bo (du doan token t+1 tu t),
    nen so token duoc du doan la len - 1 / cau
    - dung so do lam trong so khi gop.

    Raises:
        ValueError: texts rong / phan tu rong.
    """
    if not isinstance(texts, (list, tuple)) \
            or len(texts) == 0:
        raise ValueError("texts khong duoc rong.")
    tong_nll = 0.0
    tong_token = 0
    for cau in texts:
        if not isinstance(cau, str) or not cau:
            raise ValueError(
                "moi cau phai la chuoi khong rong.")
        ids = tok(cau, return_tensors="pt").input_ids
        out = model(ids, labels=ids)
        n = ids.shape[1] - 1
        tong_nll += float(out.loss) * n
        tong_token += n
    if tong_token == 0:
        raise ValueError(
            "cau qua ngan, khong co token de doan.")
    return float(np.exp(tong_nll / tong_token))


@torch.no_grad()
def _sinh_mot_batch(tok, model, batch, use_cache,
                    n_decode=N_DECODE):
    """Sinh tham lam 1 batch, do TTFT/TPOT THAT.

    Thay vi dung generate() roi uoc luong tho
    (tong thoi gian / so token), do truc tiep
    tung buoc: buoc prefill dau tien = TTFT,
    cac buoc decode sau = TPOT (Day 27: tach
    that, khong gop 1 so). Tra ca uoc luong
    tho cua khung mau de so sanh o Buoc 3.

    use_cache=True: moi buoc decode chi dua 1
    token moi + KV cache cu. use_cache=False:
    moi buoc phai dua lai TOAN BO chuoi (khong
    cache thi khong co gi de dung lai) - dat
    hon ro, dung nhu ly thuyet.

    Args:
        batch: list prompt (chuoi khong rong).
        use_cache: bool.
        n_decode: so token sinh them, int >= 1.

    Returns:
        (ttft_s, tpot_s, ttft_tho_s, tpot_tho_s)
        - ca 4 deu la giay / request (/ token
        cho TPOT); tho = tong thoi gian / so
        token (cach uoc luong cua khung mau).

    Raises:
        ValueError: batch rong / prompt rong,
            use_cache khong phai bool, n_decode
            sai, model khong phai GPT2.
    """
    if not isinstance(batch, (list, tuple)) \
            or len(batch) == 0:
        raise ValueError("batch khong duoc rong.")
    for cau in batch:
        if not isinstance(cau, str) or not cau:
            raise ValueError(
                "moi prompt phai la chuoi khong "
                "rong.")
    if not isinstance(use_cache, bool):
        raise ValueError("use_cache phai la bool.")
    _kiem_tra_so_nguyen("n_decode", n_decode)
    so_req = len(batch)
    enc = tok(batch, return_tensors="pt",
              padding=True)
    input_ids = enc.input_ids
    attention_mask = enc.attention_mask
    # Prefill: dua ca prompt 1 lan.
    t0 = time.perf_counter()
    out = model(input_ids=input_ids,
                attention_mask=attention_mask,
                use_cache=use_cache)
    t_prefill = time.perf_counter() - t0
    past = out.past_key_values if use_cache else None
    logits = out.logits
    # Decode: moi buoc 1 token, argmax tham lam.
    t_decode = 0.0
    for _ in range(n_decode):
        token = logits[:, -1].argmax(
            dim=-1, keepdim=True)
        if use_cache:
            input_ids = token
            attention_mask = torch.cat(
                [attention_mask,
                 torch.ones_like(token)], dim=1)
            t1 = time.perf_counter()
            out = model(input_ids=input_ids,
                        attention_mask=attention_mask,
                        past_key_values=past,
                        use_cache=True)
            t_decode += time.perf_counter() - t1
            past = out.past_key_values
        else:
            input_ids = torch.cat(
                [input_ids, token], dim=1)
            attention_mask = torch.cat(
                [attention_mask,
                 torch.ones_like(token)], dim=1)
            t1 = time.perf_counter()
            out = model(input_ids=input_ids,
                        attention_mask=attention_mask,
                        use_cache=False)
            t_decode += time.perf_counter() - t1
        logits = out.logits
    # [TRAINER] DIEM CHINH 1 - ttft/tpot o day la CHI PHI
    # KHAU HAO theo request (thoi gian batch / so request),
    # KHONG phai do tre nguoi dung trai nghiem. Trong 1
    # batch, MOI request deu phai cho het t_prefill moi
    # nhan token dau, va moi buoc decode moi request deu
    # cho het thoi gian cua buoc do. Do tre trai nghiem:
    #   TTFT_exp = t_prefill (khong chia so_req)
    #   TPOT_exp = t_decode / n_decode (khong chia so_req)
    # So lieu Day 28 (p95, tinh lai tu results.json):
    # B khau hao 77 ms nhung trai nghiem ~613 ms (x8);
    # A trai nghiem 143 ms. Neu frontier dung truc TTFT
    # trai nghiem thi frontier la {A, B} chu khong chi
    # {B}: A thang do tre, B thang nang luong, PPL hoa.
    # Cach sua: tra them ttft_exp/tpot_exp, dung ban trai
    # nghiem cho frontier truc do tre + SLO, giu ban khau
    # hao cho truc nang luong (J tinh tu tong thoi gian
    # nhu hien tai la dung).
    ttft = t_prefill / so_req
    tpot = t_decode / so_req / n_decode
    tong = t_prefill + t_decode
    return (ttft, tpot, tong / so_req,
            tong / so_req / n_decode)


def _mot_luot(tok, model, cfg):
    """1 luot qua ca prompt set, tra mau do.

    Gom prompt thanh batch theo cfg, moi batch
    do bang _sinh_mot_batch. Tra (list TTFT,
    list TPOT, tong giay MODEL do, so request)
    - list co 1 phan tu / batch (moi phan tu la
    gia tri / request cua batch do); tong giay
    chi tinh thoi gian model, khong tinh
    tokenize.
    """
    cfg = _kiem_tra_cau_hinh(cfg)
    mau_ttft = []
    mau_tpot = []
    tong_giay = 0.0
    for i in range(0, len(PROMPTS), cfg["batch"]):
        batch = PROMPTS[i:i + cfg["batch"]]
        ttft, tpot, tho_ttft, _ = _sinh_mot_batch(
            tok, model, batch, cfg["use_cache"])
        # Tong thoi gian model / batch = TTFT tho
        # x so request (khong tinh tokenize - viec
        # chuan bi giong nhau o moi cau hinh nen
        # khong duoc lam lech J/request).
        tong_giay += tho_ttft * len(batch)
        mau_ttft.append(ttft)
        mau_tpot.append(tpot)
    return mau_ttft, mau_tpot, tong_giay, len(PROMPTS)


def do_mot_cau_hinh(tok, model, cfg):
    """Do 1 cau hinh: 3 rep, bao TB +- do lech.

    Moi rep: warmup WARMUP_PASSES luot (bo
    qua, de may on dinh) roi do MEASURE_PASSES
    luot. Moi rep cho 1 bo (TTFT p50/p95,
    TPOT p50/p95, J/request); ket qua cuoi la
    trung binh +- do lech chuan mau qua 3 rep
    (statistics.stdev, n - 1). PPL do 1 lan
    (tien dinh, khong phu thuoc lan chay).

    Returns:
        Dict ket qua 1 cau hinh (xem bai_a).
    """
    cfg = _kiem_tra_cau_hinh(cfg)
    ppl = perplexity(tok, model, PPL_TEXTS)
    ttft_p50 = []
    ttft_p95 = []
    tpot_p50 = []
    tpot_p95 = []
    nang_luong = []
    # [TRAINER] Diem nhe - p95 tinh RIENG tung rep roi
    # moi lay trung binh qua 3 rep, nhung so mau moi rep
    # phu thuoc batch: batch 1 co 30 mau/rep, batch 8 chi
    # 6, batch 32 (hieu luc 10 prompt) chi 3 mau/rep -
    # p95 cua 3 mau thuc chat la gia tri lon nhat, rat
    # nhieu. Nen gom mau cua ca 3 rep roi tinh percentile
    # 1 lan (batch 32 se co 9 mau), hoac tang
    # MEASURE_PASSES rieng cho batch lon; kem theo la
    # ttft_p95_std cua E rat nho (0,006) de gay cam giac
    # chac chan gia tao.
    for rep in range(N_REPS):
        # In tien do moi rep: lan chay dai (chuc
        # phut tren CPU) phai cho thay no con
        # song, khong duoc im lang hang chuc
        # phut nhu 1 tien trinh treo.
        print(f"  rep {rep + 1}/{N_REPS} (warmup "
              f"{WARMUP_PASSES} + do "
              f"{MEASURE_PASSES})...", flush=True)
        for _ in range(WARMUP_PASSES):
            _mot_luot(tok, model, cfg)
        mau_ttft = []
        mau_tpot = []
        giay = 0.0
        n_req = 0
        for _ in range(MEASURE_PASSES):
            a, b, g, n = _mot_luot(tok, model, cfg)
            mau_ttft.extend(a)
            mau_tpot.extend(b)
            giay += g
            n_req += n
        ttft_p50.append(float(np.median(mau_ttft)))
        ttft_p95.append(_phan_vi(mau_ttft, 95))
        tpot_p50.append(float(np.median(mau_tpot)))
        tpot_p95.append(_phan_vi(mau_tpot, 95))
        nang_luong.append(_nang_luong(
            ASSUMED_WATTS, giay, n_req))

    def _tb_lc(ds):
        """Trung binh +- do lech chuan mau."""
        if len(ds) == 1:
            return float(ds[0]), 0.0
        return (float(statistics.mean(ds)),
                float(statistics.stdev(ds)))

    kq = {"name": cfg["name"],
          "dtype": _ten_dtype(cfg["dtype"]),
          "batch": cfg["batch"],
          "use_cache": cfg["use_cache"],
          "ppl": ppl}
    for khoa, ds in (("ttft_p50", ttft_p50),
                     ("ttft_p95", ttft_p95),
                     ("tpot_p50", tpot_p50),
                     ("tpot_p95", tpot_p95),
                     ("joules_per_req", nang_luong)):
        tb, lc = _tb_lc(ds)
        kq[khoa] = tb
        kq[khoa + "_std"] = lc
    return kq

def _kiem_tra_diem(diem):
    """Guard diem Pareto: du 3 truc so huu han.

    Truc: ppl, ttft_p95, joules_per_req - ca 3
    cang nho cang tot. Tra dict da chuan hoa
    (name giu nguyen, 3 truc float).
    """
    if not isinstance(diem, dict):
        raise ValueError("diem phai la dict.")
    for khoa in ("name", "ppl", "ttft_p95",
                 "joules_per_req"):
        if khoa not in diem:
            raise ValueError(
                f"diem thieu khoa '{khoa}'.")
    if not isinstance(diem["name"], str) \
            or not diem["name"]:
        raise ValueError("name phai la chuoi khong "
                         "rong.")
    truc = {}
    for khoa in ("ppl", "ttft_p95",
                 "joules_per_req"):
        truc[khoa] = _kiem_tra_so_thuc(
            khoa, diem[khoa], lon_hon_bang=0)
    return {"name": diem["name"], **truc}


# [TRAINER] Diem nhe - frontier chi dung 3 truc (ppl,
# ttft_p95, joules_per_req) trong khi lab do ca TPOT
# p50/p95. Lan nay ket luan khong doi vi B cung thang
# TPOT p95 (46,6 ms, tot nhat trong cac cau hinh co so),
# nhung ve nguyen tac: hoac them tpot_p95 lam truc thu
# 4, hoac ghi 1 dong ly do loai TPOT khoi frontier (vd
# "TPOT da phan anh qua joules vi cung ti le thoi
# gian") - tranh de nguoi doc tu hoi sao do ma khong
# dung. Luu y: C dang thieu tpot_p95 nen neu them truc
# thi phai do lai C truoc (lien quan diem chinh 2).

def dominates(a, b):
    """True neu a chi phoi b (Pareto).

    a tot hon hoac bang b o CA 3 truc (ppl,
    ttft_p95, joules_per_req) VA tot hon han
    o IT NHAT 1 truc. "Tot hon" = nho hon vi
    ca 3 truc deu cang nho cang tot. Hai diem
    bang nhau o ca 3 truc thi khong ai chi
    phoi ai (can tot hon han o 1 truc).
    """
    a = _kiem_tra_diem(a)
    b = _kiem_tra_diem(b)
    tot_hon_hoac_bang = all(
        a[k] <= b[k] for k in
        ("ppl", "ttft_p95", "joules_per_req"))
    tot_hon_han = any(
        a[k] < b[k] for k in
        ("ppl", "ttft_p95", "joules_per_req"))
    return tot_hon_hoac_bang and tot_hon_han


def tim_frontier(results):
    """Tim frontier + cau hinh bi chi phoi.

    frontier = cau hinh KHONG bi cau hinh nao
    chi phoi. Moi cau hinh bi chi phoi kem ten
    ke chi phoi no (ke dau tien tim thay) de
    viet cau "___ bi ___ chi phoi vi ___".

    Returns:
        (frontier, dominated) - 2 list dict,
        dominated co them khoa "bi_chi_phoi_boi"
        (ten ke chi phoi) va "ly_do" (chuoi
        mo ta 3 truc cua ke chi phoi tot hon).
    """
    if not isinstance(results, (list, tuple)) \
            or len(results) == 0:
        raise ValueError("results khong duoc rong.")
    ds = [_kiem_tra_diem(r) for r in results]
    frontier = []
    dominated = []
    for r in ds:
        ke_chi_phoi = None
        for khac in ds:
            if khac["name"] == r["name"]:
                continue
            if dominates(khac, r):
                ke_chi_phoi = khac
                break
        if ke_chi_phoi is None:
            frontier.append(r)
        else:
            ly_do = (
                f"PPL {ke_chi_phoi['ppl']:.2f} <= "
                f"{r['ppl']:.2f}, TTFT p95 "
                f"{ke_chi_phoi['ttft_p95'] * 1000:.0f} "
                f"<= {r['ttft_p95'] * 1000:.0f} ms, "
                f"{ke_chi_phoi['joules_per_req']:.1f} "
                f"<= {r['joules_per_req']:.1f} J/req")
            dominated.append(
                {**r, "bi_chi_phoi_boi":
                 ke_chi_phoi["name"],
                 "ly_do": ly_do})
    return frontier, dominated


def ve_pareto(results, ten_frontier, duong_ra):
    """Ve scatter Pareto: x = J/request, y = PPL.

    Frontier mau vang, noi bang duong vang
    (sap xep theo x tang dan); bi chi phoi
    mau xam; ghi ten A-F canh moi diem. Luu
    thanh file PNG (khong hien cua so).

    Raises:
        ValueError: results rong, ten_frontier
            chua ten la, duong ra khong ghi duoc.
    """
    if not isinstance(results, (list, tuple)) \
            or len(results) == 0:
        raise ValueError("results khong duoc rong.")
    ds = [_kiem_tra_diem(r) for r in results]
    ten_ds = {r["name"] for r in ds}
    if not set(ten_frontier) <= ten_ds:
        raise ValueError(
            "ten_frontier chua ten la.")
    # [TRAINER] Diem nhe - hinh chi ve 2 truc (Joules,
    # PPL) trong khi frontier tinh tren 3 truc (them
    # ttft_p95): 1 diem bi chi phoi vi TTFT van co the
    # nam ngay tren duong vang trong hinh va trong nhu
    # thuoc frontier. Them ghi chu trong title/legend,
    # vd "(frontier tinh tren 3 truc: PPL, TTFT p95,
    # Joules - hinh chi chieu 2 truc)".
    fig, ax = plt.subplots(figsize=(8, 5))
    for r in ds:
        tren_frontier = r["name"] in ten_frontier
        ax.scatter(
            r["joules_per_req"], r["ppl"],
            c="gold" if tren_frontier else "gray",
            s=120,
            edgecolors="black",
            zorder=3)
        ax.annotate(r["name"],
                    (r["joules_per_req"], r["ppl"]),
                    textcoords="offset points",
                    xytext=(8, 6), fontsize=12,
                    fontweight="bold")
    diem_frontier = sorted(
        [r for r in ds if r["name"] in ten_frontier],
        key=lambda r: r["joules_per_req"])
    ax.plot([r["joules_per_req"]
             for r in diem_frontier],
            [r["ppl"] for r in diem_frontier],
            color="gold", linewidth=2, zorder=2)
    ax.set_xlabel("Nang luong (Joules / request) "
                  "- cang nho cang tot")
    ax.set_ylabel("PPL - cang nho cang tot")
    ax.set_title("Pareto frontier Day 28: "
                 "PPL vs Joules/request "
                 "(vang = frontier, xam = bi chi phoi)")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(duong_ra, dpi=150)
    plt.close(fig)
    return duong_ra


def chon_chatbot(frontier):
    """Chon cau hinh cho chatbot tuong tac.

    SLO: TTFT p95 < 300 ms (nguoi dung cho
    tung token). Trong cac diem frontier dat
    SLO, chon diem re nhat (J/request thap
    nhat) - dat SLO roi thi khong can tra
    them dien cho do tre thua. Tra None neu
    khong diem frontier nao dat SLO (ke ca
    khi diem bi chi phoi dat - diem bi chi
    phoi khong bao gio duoc chon).
    """
    # [TRAINER] SLO 300 ms la nguong DO TRE NGUOI DUNG
    # ("nguoi dung cho tung token" - xem docstring), nhung
    # ttft_p95 truyen vao day la so KHAU HAO (diem chinh 1)
    # nen B "dat SLO" o 77 ms trong khi nguoi dung that
    # cho ~613 ms. Sau khi co ttft_p95 trai nghiem, ham
    # nay phai so SLO tren so trai nghiem; khi do cau tra
    # loi Day 28 doi thanh: chatbot chon A (143 ms, dat
    # SLO), B chi con thang o bai offline/nang luong.
    ds = [_kiem_tra_diem(r) for r in frontier]
    dat = [r for r in ds
           if r["ttft_p95"] < SLO_CHATBOT_TTFT_P95]
    if not dat:
        return None
    return min(dat, key=lambda r: r["joules_per_req"])


def chon_offline(results):
    """Chon cau hinh cho tac vu batch offline.

    SLO: re nhat co the, PPL tang toi da 5%
    so voi cau hinh goc FP32 batch 1 (A).
    Tra (diem_duoc_chon, tiet_kiem_pct) - tiet
    kiem % nang luong so voi A; None neu
    khong co diem nao (ke ca A) dat dieu kien.
    """
    ds = [_kiem_tra_diem(r) for r in results]
    goc = next((r for r in ds if r["name"] == "A"),
               None)
    if goc is None:
        raise ValueError(
            "thieu cau hinh goc A de lam chuan.")
    nguong = goc["ppl"] * (1.0 + PPL_TANG_TOI_DA)
    dat = [r for r in ds if r["ppl"] <= nguong]
    if not dat:
        return None, 0.0
    chon = min(dat, key=lambda r: r["joules_per_req"])
    tiet_kiem = (goc["joules_per_req"]
                 - chon["joules_per_req"]) \
        / goc["joules_per_req"] * 100.0
    return chon, tiet_kiem

def _expect_value_error(fn, mo_ta):
    """Chay fn, doi no nem ValueError, in ket qua."""
    try:
        fn()
    except ValueError as exc:
        print(f"  guard {mo_ta}: chan dung ({exc})")
        return True
    print(f"  CANH BAO: guard {mo_ta} KHONG chan!")
    return False


def tu_kiem_tra():
    """Tu kiem tra bang assert, chay truoc lab.

    Kiem tra guard, _phan_vi, cong thuc nang
    luong, dominates + frontier (tinh tay, gom
    cau 1 trong de), cap doi cong bang, chon
    theo SLO tren so gia, ve_pareto ghi file,
    va tinh dung cua vong lap sinh (cache vs
    khong cache cho cung token tiep theo, tren
    model nho ngau nhien).
    """
    ket_qua = [
        _expect_value_error(
            lambda: _phan_vi([], 95),
            "phan vi mau rong"),
        _expect_value_error(
            lambda: _phan_vi([1.0], 101),
            "phan vi p > 100"),
        _expect_value_error(
            lambda: _kiem_tra_cau_hinh(
                {"name": "X", "dtype": torch.float32,
                 "batch": 0, "use_cache": True}),
            "cau hinh batch=0"),
        _expect_value_error(
            lambda: _kiem_tra_cau_hinh(
                {"name": "X", "dtype": torch.float32,
                 "batch": 8}),
            "cau hinh thieu use_cache"),
        _expect_value_error(
            lambda: dominates(
                {"name": "X", "ppl": 10.0,
                 "ttft_p95": 0.1},
                {"name": "Y", "ppl": 10.0,
                 "ttft_p95": 0.1,
                 "joules_per_req": 1.0}),
            "dominates thieu truc"),
        _expect_value_error(
            lambda: _sinh_mot_batch(
                None, None, [], True),
            "sinh batch rong"),
        _expect_value_error(
            lambda: _nang_luong(65.0, 10.0, 0),
            "nang luong n_req=0"),
        _expect_value_error(
            lambda: ve_pareto(
                [{"name": "A", "ppl": 10.0,
                  "ttft_p95": 0.1,
                  "joules_per_req": 1.0}],
                ["Z"], "/tmp/pareto_test.png"),
            "ve pareto ten la"),
        _expect_value_error(
            lambda: chon_offline(
                [{"name": "B", "ppl": 10.0,
                  "ttft_p95": 0.1,
                  "joules_per_req": 1.0}]),
            "chon offline thieu A"),
    ]
    assert all(ket_qua), "Co guard khong chan dung!"

    # _phan_vi nearest-rank (giong Day 27).
    assert _phan_vi(
        [float(i) for i in range(1, 21)], 95) == 19.0
    assert _phan_vi(
        [float(i) for i in range(1, 11)], 50) == 5.0
    # Nang luong: 65 W x 10 s / 20 req = 32.5 J.
    assert _nang_luong(65.0, 10.0, 20) == 32.5

    # Cau 1 trong de: Y chi phoi X (tot hon ca 3
    # truc). Bang nhau ca 3 truc thi khong ai
    # chi phoi ai; tot hon 1 truc + bang 2 truc
    # con lai thi co chi phoi.
    x = {"name": "X", "ppl": 24.0, "ttft_p95": 0.4,
         "joules_per_req": 30.0}
    y = {"name": "Y", "ppl": 22.0, "ttft_p95": 0.35,
         "joules_per_req": 25.0}
    assert dominates(y, x)
    assert not dominates(x, y)
    assert not dominates(x, dict(x, name="X2"))
    assert dominates(
        {"name": "P", "ppl": 20.0, "ttft_p95": 0.35,
         "joules_per_req": 25.0}, y)
    assert not dominates(
        {"name": "Q", "ppl": 21.0, "ttft_p95": 0.36,
         "joules_per_req": 25.0}, y)

    # Frontier tinh tay: D bi A chi phoi, A/B/C
    # khong ai chi phoi ai (moi diem thang 1
    # truc, thua truc khac).
    tong_hop = [
        {"name": "A", "ppl": 10.0, "ttft_p95": 0.10,
         "joules_per_req": 10.0},
        {"name": "B", "ppl": 12.0, "ttft_p95": 0.08,
         "joules_per_req": 20.0},
        {"name": "C", "ppl": 11.0, "ttft_p95": 0.09,
         "joules_per_req": 15.0},
        {"name": "D", "ppl": 20.0, "ttft_p95": 0.20,
         "joules_per_req": 30.0},
    ]
    frontier, dominated = tim_frontier(tong_hop)
    assert {r["name"] for r in frontier} == {"A", "B",
                                              "C"}
    assert len(dominated) == 1
    assert dominated[0]["name"] == "D"
    assert dominated[0]["bi_chi_phoi_boi"] == "A"

    # Cap doi cong bang: 7 cap chi khac 1 bien;
    # A-D khac 2 bien (dtype + batch) nen khong
    # co trong danh sach.
    cap = _cap_doi_cong_bang(CONFIGS)
    assert len(cap) == 7
    assert ("A", "B", "batch size") in cap
    assert ("B", "F", "use_cache") in cap
    assert ("A", "D", "dtype") not in cap
    assert _bien_thay_doi(CONFIGS[3], CONFIGS[0]) == [
        "dtype", "batch size"]
    assert _bien_thay_doi(CONFIGS[0], CONFIGS[0]) == []

    # Chon theo SLO tren so gia: chatbot lay
    # diem frontier dat TTFT p95 < 300 ms re
    # nhat (B); offline lay re nhat trong
    # nguong PPL <= 1.05 x PPL cua A (C).
    gia = [
        {"name": "A", "ppl": 20.0, "ttft_p95": 0.10,
         "joules_per_req": 10.0},
        {"name": "B", "ppl": 20.5, "ttft_p95": 0.20,
         "joules_per_req": 5.0},
        {"name": "C", "ppl": 20.8, "ttft_p95": 0.40,
         "joules_per_req": 3.0},
    ]
    assert chon_chatbot(gia)["name"] == "B"
    chon, tiet_kiem = chon_offline(gia)
    assert chon["name"] == "C"
    assert abs(tiet_kiem - 70.0) <= 1e-9

    # Ve pareto ghi file that (xoa sau khi kiem).
    duong_thu = "/tmp/pareto_test.png"
    ve_pareto(gia, ["A", "B", "C"], duong_thu)
    assert os.path.exists(duong_thu)
    assert os.path.getsize(duong_thu) > 0
    os.remove(duong_thu)

    # Vong lap sinh: cache vs khong cache cho
    # cung token tiep theo (model nho ngau
    # nhien, seed co dinh). Padding trai de
    # token cuoi moi hang la token that.
    from transformers import GPT2Config
    from transformers import GPT2LMHeadModel
    torch.manual_seed(0)
    nho = GPT2LMHeadModel(GPT2Config(
        n_layer=2, n_head=2, n_embd=64,
        n_positions=128, vocab_size=50257))
    nho.eval()
    tok, _ = tai_model(torch.float32)
    cau = ["The cat sat on the mat", "Hi"]
    enc = tok(cau, return_tensors="pt", padding=True)
    with torch.no_grad():
        out = nho(input_ids=enc.input_ids,
                  attention_mask=enc.attention_mask,
                  use_cache=True)
        token = out.logits[:, -1].argmax(
            dim=-1, keepdim=True)
        past = out.past_key_values
        mask_moi = torch.cat(
            [enc.attention_mask,
             torch.ones_like(token)], dim=1)
        out_cache = nho(
            input_ids=token,
            attention_mask=mask_moi,
            past_key_values=past, use_cache=True)
        ids_day = torch.cat(
            [enc.input_ids, token], dim=1)
        out_khong = nho(
            input_ids=ids_day,
            attention_mask=mask_moi,
            use_cache=False)
    sai_so = float(
        (out_cache.logits[:, -1].detach()
         - out_khong.logits[:, -1].detach())
        .abs().max())
    assert sai_so <= 1e-4, "cache phai cho cung logits."
    assert int(out_cache.logits[:, -1].argmax()) \
        == int(out_khong.logits[:, -1].argmax())
    # _sinh_mot_batch chay duoc ca 2 che do,
    # tra thoi gian duong huu han.
    for cache in (True, False):
        ttft, tpot, tho_ttft, tho_tpot = \
            _sinh_mot_batch(tok, nho, cau, cache, 4)
        for gia_tri in (ttft, tpot, tho_ttft,
                        tho_tpot):
            assert math.isfinite(gia_tri)
            assert gia_tri > 0
    print("Tu kiem tra: PASS (tat ca assert deu qua).")


def _doc_ket_qua_da_co():
    """Doc results.json cu (neu co) de chay tiep.

    Tra list ket qua da do (rong neu chua co
    file / file hong). Dung de chay tiep sau
    khi lan chay truoc bi ngat (vd treo may)
    ma khong do lai cau hinh da xong.
    """
    if not os.path.exists(DUONG_KET_QUA):
        return []
    try:
        with open(DUONG_KET_QUA) as f:
            ds = json.load(f)
    except (ValueError, OSError) as exc:
        print(f"  Canh bao: khong doc duoc "
              f"{DUONG_KET_QUA} ({exc}) - do lai "
              f"tu dau.")
        return []
    if not isinstance(ds, list):
        print("  Canh bao: results.json khong phai "
              "list - do lai tu dau.")
        return []
    # [TRAINER] DIEM CHINH 2 - ham nay chi kiem tra entry
    # co khoa "name" la coi nhu "da do xong", khong kiem
    # tra du truong. Bang chung ngay trong results.json
    # Day 28: cau hinh C thieu tpot_p50_std, tpot_p95,
    # tpot_p95_std (theo ghi chu trong file: khoi phuc tu
    # log lan chay 07/11 bi ket o cau hinh D, so da lam
    # tron) - tuc so C do o PHIEN KHAC, dieu kien khac,
    # roi tron vao cung bang/frontier, pha dung quy tac
    # "cung cach do" cua chinh lab nay. Cach sua: (a) khi
    # resume, chi chap nhan entry du bo khoa chuan cua
    # do_mot_cau_hinh (co ca cac khoa _std), thieu thi do
    # lai cau hinh do; (b) so lieu khoi phuc thu cong thi
    # hoac do lai trong cung phien, hoac loai khoi
    # frontier va ghi ro ly do - khong de chung "da xong"
    # mot cach am tham.
    return [r for r in ds
            if isinstance(r, dict) and r.get("name")]


def _luu_ket_qua(ket_qua):
    """Luu results.json, sap theo thu tu A-F.

    Luu NGAY sau moi cau hinh xong (checkpoint)
    - lan chay dai ma bi ngat giua chung thi
    cac cau hinh da xong khong mat.
    """
    thu_tu = {c["name"]: i
              for i, c in enumerate(CONFIGS)}
    ket_qua.sort(key=lambda r: thu_tu.get(
        r["name"], 99))
    with open(DUONG_KET_QUA, "w") as f:
        json.dump(ket_qua, f, indent=2)


def bai_a():
    """Bai A: benchmark cong bang 6 cau hinh.

    Moi cau hinh: PPL (3 cau co dinh) + TTFT /
    TPOT p50/p95 + J/request (TB +- LC qua 3
    rep). In "thu duy nhat thay doi" so voi A
    (tinh tu du lieu) va cac cap doi cong bang
    (chi khac 1 bien). Model chi tai 1 lan /
    dtype (D va E chung FP16) de bot tai lai
    va bot xao tron bo nho. Ket qua luu ngay
    sau moi cau hinh; neu results.json da co
    cau hinh nao thi bo qua cau hinh do (chay
    tiep sau khi bi ngat).
    """
    print("\n" + "=" * 72)
    print("BAI A - Benchmark 6 cau hinh")
    print("=" * 72)
    print(f"Model {MODEL} tren CPU, {N_REPS} rep, "
          f"warmup {WARMUP_PASSES} luot/rep, do "
          f"{MEASURE_PASSES} luot/rep, sinh "
          f"{N_DECODE} token/request.")
    print(f"Nang luong = cong suat GIA DINH "
          f"{ASSUMED_WATTS:.0f} W x thoi gian / so "
          f"request (khong do cong suat that).")
    print("Prompt set co 10 cau; batch 32 chi "
          "gom duoc 10 prompt / batch (hieu luc "
          "10, khong phai 32).")
    print("Cap doi cong bang (chi khac 1 bien): "
          + ", ".join(
              f"{a}-{b} ({v})" for a, b, v
              in _cap_doi_cong_bang(CONFIGS)))
    ket_qua = _doc_ket_qua_da_co()
    da_co = {r["name"] for r in ket_qua}
    if da_co:
        print(f"  Chay tiep: da co "
              f"{', '.join(sorted(da_co))} trong "
              f"results.json, chi do cac cau hinh "
              f"con lai.")
    tok = None
    model = None
    dtype_hien_tai = None
    try:
        for cfg in CONFIGS:
            doi = _bien_thay_doi(cfg, CONFIGS[0])
            print(f"\n--- Cau hinh {cfg['name']} "
                  f"({_ten_dtype(cfg['dtype'])}, batch "
                  f"{cfg['batch']}, use_cache="
                  f"{cfg['use_cache']}) ---")
            if cfg["name"] in da_co:
                print("  Bo qua, da co trong "
                      "results.json.")
                continue
            if not doi:
                print("  Thu duy nhat thay doi la: "
                      "khong doi (cau hinh goc).")
            else:
                print("  Thu duy nhat thay doi la (so "
                      f"voi A): {', '.join(doi)}.")
            if tok is None:
                tok = _tai_tokenizer()
            if dtype_hien_tai != cfg["dtype"]:
                if model is not None:
                    del model
                model = _tai_chi_model(cfg["dtype"])
                dtype_hien_tai = cfg["dtype"]
            kq = do_mot_cau_hinh(tok, model, cfg)
            print(f"  PPL={kq['ppl']:.2f} | TTFT p50="
                  f"{kq['ttft_p50'] * 1000:.0f}+-"
                  f"{kq['ttft_p50_std'] * 1000:.0f} ms, "
                  f"p95={kq['ttft_p95'] * 1000:.0f}+-"
                  f"{kq['ttft_p95_std'] * 1000:.0f} ms | "
                  f"TPOT p50={kq['tpot_p50'] * 1000:.1f}"
                  f"+-{kq['tpot_p50_std'] * 1000:.1f} "
                  f"ms, p95={kq['tpot_p95'] * 1000:.1f}"
                  f"+-{kq['tpot_p95_std'] * 1000:.1f} "
                  f"ms | J/req={kq['joules_per_req']:.2f}"
                  f"+-{kq['joules_per_req_std']:.2f}")
            if cfg["name"] == "A":
                # Buoc 3 (de bai): so phuong phap tho
                # cua khung mau voi do that tung buoc.
                _so_sanh_uoc_luong_tho(tok, model, cfg)
            ket_qua.append(kq)
            _luu_ket_qua(ket_qua)
    finally:
        if model is not None:
            del model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    print(f"\nDa luu {DUONG_KET_QUA}")
    return ket_qua


def _so_sanh_uoc_luong_tho(tok, model, cfg):
    """Buoc 3: uoc luong tho vs do that (cau A).

    Khung mau uoc TTFT ~= tong thoi gian /
    request (gom ca decode) va TPOT ~= tong /
    16 - do 1 luot, in ti le chenh lech.
    """
    tho_ttft = []
    tho_tpot = []
    that_ttft = []
    that_tpot = []
    for i in range(0, len(PROMPTS), cfg["batch"]):
        batch = PROMPTS[i:i + cfg["batch"]]
        a, b, c, d = _sinh_mot_batch(
            tok, model, batch, cfg["use_cache"])
        that_ttft.append(a)
        that_tpot.append(b)
        tho_ttft.append(c)
        tho_tpot.append(d)
    ti_le_ttft = float(np.median(tho_ttft)) \
        / float(np.median(that_ttft))
    ti_le_tpot = float(np.median(tho_tpot)) \
        / float(np.median(that_tpot))
    print(f"  [Buoc 3] Uoc luong tho vs do that: "
          f"TTFT tho cao hon {ti_le_ttft:.1f}x "
          f"(vi gom ca {N_DECODE} token decode "
          f"vao TTFT); TPOT tho "
          f"{ti_le_tpot:.2f}x so voi that "
          f"(tho chia deu tong thoi gian, trong "
          f"khi prefill dat hon decode / token).")


def bai_b(ket_qua=None):
    """Bai B: tim + ve Pareto frontier.

    Doc results.json (hoac dung ket qua vua do),
    in cau "___ bi ___ chi phoi vi ___", sap
    frontier tu re nhat toi chinh xac nhat, ve
    pareto.png.
    """
    print("\n" + "=" * 72)
    print("BAI B - Pareto frontier")
    print("=" * 72)
    if ket_qua is None:
        with open(DUONG_KET_QUA) as f:
            ket_qua = json.load(f)
    frontier, dominated = tim_frontier(ket_qua)
    if dominated:
        for r in dominated:
            print(f"  {r['name']} bi "
                  f"{r['bi_chi_phoi_boi']} chi phoi "
                  f"vi {r['ly_do']}.")
    else:
        print("  Khong co cau hinh nao bi chi phoi "
              "- theo bai hoc, co the chua do du "
              "truc (thuong thieu nang luong hoac "
              "p95); kiem tra lai Pham A.")
    theo_gia = sorted(
        frontier, key=lambda r: r["joules_per_req"])
    print("  Frontier, tu re nhat toi chinh xac "
          "nhat: " + " -> ".join(
              f"{r['name']} "
              f"({r['joules_per_req']:.1f} J/req, "
              f"PPL {r['ppl']:.2f})"
              for r in theo_gia))
    ve_pareto(ket_qua,
              [r["name"] for r in frontier],
              DUONG_PARETO)
    print(f"  Da luu {DUONG_PARETO}")
    return frontier, dominated


def bai_c(ket_qua, frontier, dominated):
    """Bai C: chon theo SLO + Green checkpoint.

    Chatbot: TTFT p95 < 300 ms, trong frontier
    dat SLO thi chon re nhat. Batch offline:
    re nhat trong nguong PPL <= 1.05 x PPL A.
    Green checkpoint: 1 cau hinh loai vi bi chi
    phoi + 1 cau hinh chon kem ly do SLO. Moi
    lua chon tinh tu so do, khong go tay ten.
    """
    print("\n" + "=" * 72)
    print("BAI C - Chon theo SLO + Green checkpoint")
    print("=" * 72)
    re_nhat = min(
        frontier, key=lambda r: r["joules_per_req"])
    chinh_xac_nhat = min(
        frontier, key=lambda r: r["ppl"])
    chon_bot = chon_chatbot(frontier)
    if chon_bot is None:
        print("  Chatbot (TTFT p95 < 300 ms): KHONG "
              "diem frontier nao dat SLO - can doi "
              "SLO hoac them cau hinh (vd batch "
              "lon hon / model nho hon).")
    else:
        print(f"  Chatbot (TTFT p95 < 300 ms): chon "
              f"{chon_bot['name']} (TTFT p95 "
              f"{chon_bot['ttft_p95'] * 1000:.0f} "
              f"ms, {chon_bot['joules_per_req']:.1f} "
              f"J/req, PPL {chon_bot['ppl']:.2f}).")
        if chon_bot["name"] != re_nhat["name"]:
            print(f"    Khong chon re nhat "
                  f"{re_nhat['name']} vi TTFT p95 "
                  f"cua no "
                  f"{re_nhat['ttft_p95'] * 1000:.0f} "
                  f"ms khong dat SLO 300 ms.")
        if chon_bot["name"] != chinh_xac_nhat["name"]:
            print(f"    Khong chon chinh xac nhat "
                  f"{chinh_xac_nhat['name']} vi dat "
                  f"SLO roi thi khong can tra them "
                  f"{chon_bot['joules_per_req']:.1f} "
                  f"-> "
                  f"{chinh_xac_nhat['joules_per_req']:.1f} "
                  f"J/req cho PPL thap hon.")
    chon_off, tiet_kiem = chon_offline(ket_qua)
    if chon_off is None:
        print("  Batch offline (PPL tang <= 5% so "
              "voi A): khong diem nao dat - ke ca "
              "A cung vuot nguong (khong the).")
    else:
        print(f"  Batch offline (PPL tang <= 5% so "
              f"voi A): chon {chon_off['name']} "
              f"(PPL {chon_off['ppl']:.2f}, "
              f"{chon_off['joules_per_req']:.1f} "
              f"J/req) - tiet kiem "
              f"{tiet_kiem:.1f}% nang luong so voi "
              f"A.")
    print("  Green checkpoint:")
    if dominated:
        r = dominated[0]
        print(f"    Loai {r['name']} vi bi "
              f"{r['bi_chi_phoi_boi']} chi phoi "
              f"({r['ly_do']}).")
    else:
        print("    Khong loai duoc cau hinh nao - "
              "can do them truc truoc khi ket luan.")
    if chon_bot is not None:
        print(f"    Chon {chon_bot['name']} cho "
              f"chatbot vi dat SLO TTFT p95 < 300 "
              f"ms voi nang luong thap nhat trong "
              f"so diem frontier dat SLO.")
    elif chon_off is not None:
        print(f"    Chon {chon_off['name']} cho "
              f"batch offline vi re nhat trong "
              f"nguong PPL +5%.")
    return {"chatbot": chon_bot, "offline": chon_off,
            "tiet_kiem_pct": tiet_kiem}


def tra_loi_tu_kiem_tra():
    """Tra loi ngan 3 cau tu kiem tra trong de."""
    print("\nTRA LOI TU KIEM TRA (3 cau trong de):")
    print("  1. X bi Y chi phoi vi Y tot hon o ca "
          "3 truc (PPL 22 < 24, TTFT 350 < 400 "
          "ms, 25 < 30 J/req) - loai X ngay, du X "
          "trong van on.")
    print("  2. Trung binh giau trai nghiem te "
          "nhat: 19/20 request xong trong 100 ms "
          "nhung 1 request ket 3 giay vi toi dung "
          "luc batch day - trung binh van dep "
          "(~245 ms) nhung p95 bat dung 5% nguoi "
          "dung den dui do.")
    print("  3. Sai vi doi 2 bien cung luc "
          "(quantization + batch size) - do la so "
          "sanh batch size, khong phai so sanh "
          "quantization. Dung: giu batch bang "
          "nhau, chi doi dtype; muon ket luan ve "
          "batch thi giu dtype bang nhau, chi doi "
          "batch.")


def main():
    """Chay ca lab: tu kiem tra, 3 bai, tom tat."""
    print("Lab Green serving Day 28 - Mini-project 4")
    print(f"torch {torch.__version__}, "
          f"transformers {transformers.__version__}")
    print("Thiet bi: "
          f"{'cuda' if torch.cuda.is_available() else 'cpu'} "
          f"- so ms / J chi dung cho may nay, ket "
          f"luan dung duoc la thu hang / frontier "
          f"/ xu huong.")
    tu_kiem_tra()
    ket_qua = bai_a()
    frontier, dominated = bai_b(ket_qua)
    lua_chon = bai_c(ket_qua, frontier, dominated)
    ten_bot = lua_chon["chatbot"]["name"] \
        if lua_chon["chatbot"] else "khong co"
    ten_off = lua_chon["offline"]["name"] \
        if lua_chon["offline"] else "khong co"
    print("\nBANG KET QUA: " + "; ".join(
        f"{r['name']} PPL {r['ppl']:.2f}, TTFT p95 "
        f"{r['ttft_p95'] * 1000:.0f} ms, "
        f"{r['joules_per_req']:.1f} J/req"
        for r in ket_qua))
    print(f"Frontier: "
          f"{', '.join(r['name'] for r in frontier)}"
          f" | Bi chi phoi: "
          f"{', '.join(r['name'] for r in dominated) or 'khong co'}"
          f" | Chatbot: {ten_bot} | Offline: "
          f"{ten_off} "
          f"(tiet kiem {lua_chon['tiet_kiem_pct']:.1f}% "
          f"so voi A).")
    print("Ket luan chung: khong co cau hinh tot "
          "nhat tuyet doi - chi co diem tren "
          "frontier + SLO cua san pham; cau hinh "
          "bi chi phoi loai ngay, khong can tranh "
          "cai.")
    tra_loi_tu_kiem_tra()


if __name__ == "__main__":
    main()
