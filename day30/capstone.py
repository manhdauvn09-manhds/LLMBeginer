"""Capstone GreenLLM - Day 30 (buoi toi, Phase 1).

Chuoi capstone thu nho tren CPU: mo hinh ngon ngu ky tu (char-level)
huan luyen tren doan van ban tieng Viet, do baseline FP32 roi so voi
bien the luong tu hoa dong INT8, tinh chia Chinchilla, ap Green
checkpoint de ra quyet dinh.

Chi thu thay doi la bien the mo hinh (FP32 -> INT8): cung mot may,
cung mot bo danh gia (tap test), cung mot ham measure(). Nang luong
do XEN KE A/B nhieu vong + lay trung vi ti le tung vong (sua theo
review Trainer 1ab9c39: do rieng 2 phien lat verdict -1,25% vs
-28,2% vi thoi gian sinh 100 token chi ~0,2 giay, nam duoi muc
nhieu cua moi truong do).

GHI CHU (nguoi hoc dien 1 dong truoc Phase 2): khai niem mo nhat
trong 30 ngay la gi - de on lai truoc khi buoc sang Phase 2.
"""

# ============================================================
# [TRAINER] REVIEW Day 30 - Capstone (capstone_review.py)
# Kiem chung doc lap: chay lai toan bo capstone tren may
# Trainer - PPL tai lap dung 7,78 ca baseline lan INT8;
# cong thuc Chinchilla dung (assert 6*N*D = C tu kiem tra
# trong code la cach kiem chung rat tot). DAC BIET: nguoi
# hoc da bat duoc loi trong DAP AN MAU cua practice -
# practice ghi N = sqrt(1e21/120) ~ 9,1e9, nhung gia tri
# dung la 2,89e9 (9,1e9 la ket qua cua sqrt(1e21/12)).
# Trainer da sua dap an practice theo dung so nay.
# 1 diem chinh + 1 diem nhe:
#  (1) CHINH: verdict nhanh (b) nang luong LAT giua hai
#      lan chay doc lap - xem comment o green_checkpoint.
#  (2) Nhe: eval_loss lay trung binh khong trong so qua
#      cac batch (comment o eval_loss).
# ============================================================

import copy
import json
import math
import statistics
import time

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

torch.manual_seed(42)

# ---- Buoc 0: du lieu (khau 1: Data) ----
# Doan van tieng Viet ~2.000-3.000 ky tu, viet ve AI xanh.
TEXT = (
    "Tri tue nhan tao dang thay doi cach con nguoi lam viec, hoc tap "
    "va sang tao. Nhung mo hinh ngon ngu lon co the viet van ban, "
    "dich thuat, tra loi cau hoi va ho tro lap trinh voi do chinh xac "
    "ngay cang cao. Dang sau moi cau tra loi troi chay la hang nghin "
    "ty phep tinh duoc thuc hien trong cac trung tam du lieu, noi "
    "hang chuc nghin bo xu ly do hoa chay lien tuc suot nhieu tuan "
    "de huan luyen mot mo hinh duy nhat. Chi phi nang luong cua qua "
    "trinh nay khong he nho, va no dat ra mot cau hoi quan trong cho "
    "ca nganh: lam sao de tri tue nhan tao vua manh me vua ben vung. "
    "Hieu qua nang luong khong chi la van de tiet kiem tien dien. "
    "Moi mo hinh lon khi dua vao su dung se phuc vu hang trieu luot "
    "truy van moi ngay, va nang luong tieu thu trong giai doan suy "
    "luan thuong vuot xa nang luong huan luyen sau mot thoi gian dai "
    "van hanh. Vi vay cac ky su bat dau quan tam den nhung mo hinh "
    "nho hon nhung duoc huan luyen tot hon, thay vi chay dua tang so "
    "tham so mot cach vo toi va. Luat Chinchilla da chi ra rang voi "
    "cung mot ngan sach tinh toan, mot mo hinh vua phai huan luyen "
    "tren nhieu du lieu thuong cho ket qua tot hon mot mo hinh kho "
    "lo nhung thieu du lieu. Ben canh viec chon kich thuoc mo hinh "
    "hop ly, nhieu ky thuat toi uu cung giup giam chi phi suy luan. "
    "Luong tu hoa la mot trong nhung ky thuat pho bien nhat: thay vi "
    "luu tru trong so bang so thuc 32 bit, mo hinh co the dung so "
    "nguyen 8 bit, giup giam dung luong bo nho va tang toc do tinh "
    "toan. Ky thuat cat tia loai bo nhung ket noi it quan trong, con "
    "chung cat tri thuc chuyen kien thuc tu mo hinh lon sang mo hinh "
    "nho gon hon. Moi ky thuat deu phai danh doi mot phan chat luong "
    "de lay hieu qua, va nhiem vu cua nguoi lam mo hinh la do luong "
    "su danh doi do mot cach trung thuc. Mot bao cao tot khong chi "
    "khoe con so dep ma con noi ro dieu kien do: cung mot may, cung "
    "mot bo du lieu, cung mot cach do. Neu hai mo hinh duoc do tren "
    "hai phan cung khac nhau thi moi so sanh deu vo nghia. Green "
    "checkpoint la mot khung kiem tra don gian: bien the toi uu chi "
    "duoc coi la tot hon khi chat luong khong giam qua mot nguong "
    "cho phep, dong thoi nang luong tieu thu giam du nhieu, va ket "
    "qua co the lap lai khi do lan thu hai. Khung kiem tra nay giup "
    "nguoi lam mo hinh tranh duoc cam do to ve so lieu. Cuoi cung, "
    "tri tue nhan tao xanh khong phai la mot tinh nang them vao sau "
    "cung ma la mot cach tiep can xuyen suot vong doi mo hinh: tu "
    "khau thu thap du lieu, chon kien truc, huan luyen, toi uu, cho "
    "den trien khai va bao cao. Moi quyet dinh o mot khau deu anh "
    "huong den nang luong tieu thu o cac khau sau. Khi nguoi hoc "
    "nam duoc tu duy nay, ho khong chi xay dung duoc mo hinh tot ma "
    "con xay dung duoc mo hinh co trach nhiem voi moi truong va voi "
    "chi phi van hanh lau dai."
)

# Chia 80/10/10 theo thu tu thoi gian, KHONG xao tron: neu test lan
# vao train thi moi so do sau do deu vo nghia (loi so 1 capstone).
N_TOTAL = len(TEXT)
N_TRAIN = int(N_TOTAL * 0.8)
N_VAL = int(N_TOTAL * 0.1)
TEXT_TRAIN = TEXT[:N_TRAIN]
TEXT_VAL = TEXT[N_TRAIN:N_TRAIN + N_VAL]
TEXT_TEST = TEXT[N_TRAIN + N_VAL:]
print(f"train={len(TEXT_TRAIN)} val={len(TEXT_VAL)} "
      f"test={len(TEXT_TEST)} ky tu")
assert len(TEXT_TRAIN) + len(TEXT_VAL) + len(TEXT_TEST) == N_TOTAL
assert 2000 <= N_TOTAL <= 3000, f"doan van {N_TOTAL} ky tu, can 2-3k"

CHARS = sorted(set(TEXT))
STOI = {c: i for i, c in enumerate(CHARS)}
VOCAB = len(CHARS)
print(f"vocab={VOCAB} ky tu rieng biet")

SEQ_LEN = 64
BATCH = 32
EPOCHS = 3
LR = 3e-3
D_MODEL = 64
N_HEAD = 2
GEN_TOKENS = 500
# 1 lan do sinh 500 token (~1 giay) thay vi 100 token (~0,2 giay):
# 100 token nam ngay duoi muc nhieu cua moi truong do nen verdict
# nang luong lat giua 2 lan chay doc lap (review Trainer 1ab9c39).
N_VONG_XEN_KE = 5
POWER_W = 65.0  # cong suat gia dinh cua CPU laptop, de uoc luong


def encode(text):
    """Ma hoa chuoi thanh tensor id (moi ky tu -> 1 so nguyen)."""
    return torch.tensor([STOI[c] for c in text], dtype=torch.long)


def make_windows(ids, stride=16):
    """Cat chuoi id thanh cac cua so (input, target) dich 1 ky tu.

    Vi du: ids = [5, 9, 2, ...], cua so dau tien co input la 64 id
    dau, target la 64 id ke tiep (dich sang phai 1 buoc) - mo hinh
    hoc doan ky tu tiep theo. Buoc truot 16 de co du mau ma van
    nhanh tren CPU.
    """
    xs, ys = [], []
    for i in range(0, len(ids) - SEQ_LEN, stride):
        xs.append(ids[i:i + SEQ_LEN])
        ys.append(ids[i + 1:i + SEQ_LEN + 1])
    assert xs, "chuoi qua ngan, khong cat duoc cua so nao"
    return torch.stack(xs), torch.stack(ys)


IDS_TRAIN = encode(TEXT_TRAIN)
IDS_VAL = encode(TEXT_VAL)
IDS_TEST = encode(TEXT_TEST)
X_TRAIN, Y_TRAIN = make_windows(IDS_TRAIN)
X_VAL, Y_VAL = make_windows(IDS_VAL)
X_TEST, Y_TEST = make_windows(IDS_TEST)
print(f"cua so: train={len(X_TRAIN)} val={len(X_VAL)} "
      f"test={len(X_TEST)} (moi cua so {SEQ_LEN} ky tu)")


# ---- Buoc 1: baseline (khau 2-3: Model + Train) ----
class TinyLM(nn.Module):
    """Mo hinh ngon ngu ky tu nho: embedding -> Transformer -> Linear.

    Dau vao (batch, seq) id ky tu; dau ra (batch, seq, vocab) logits
    cho ky tu tiep theo o moi vi tri.
    """

    def __init__(self, vocab_size, d_model=64, n_head=2):
        super().__init__()
        self.emb = nn.Embedding(vocab_size, d_model)
        layer = nn.TransformerEncoderLayer(d_model=d_model,
                                           nhead=n_head,
                                           batch_first=True)
        self.enc = nn.TransformerEncoder(layer, num_layers=1)
        self.head = nn.Linear(d_model, vocab_size)

    def forward(self, x):
        seq_len = x.size(1)
        # Mat na nhan qua: vi tri i chi duoc "nhin" cac vi tri <= i,
        # khong duoc nhin truoc ky tu can doan (True = cam).
        mask = torch.triu(torch.ones(seq_len, seq_len,
                                     dtype=torch.bool,
                                     device=x.device), diagonal=1)
        h = self.emb(x)
        h = self.enc(h, mask=mask)
        return self.head(h)


def count_params(model):
    """Dem so tham so co the huan luyen cua mo hinh."""
    return sum(p.numel() for p in model.parameters()
               if p.requires_grad)


def state_bytes(model):
    """Tong so byte cua state_dict (trong so luong tu hoa nhe hon).

    Vi du: tensor float32 co 1.000 phan tu chiem 4.000 byte, tensor
    qint8 cung so phan tu chi chiem 1.000 byte. Mo hinh luong tu
    hoa co cac muc dac biet trong state_dict: _packed_params.dtype
    (1 doi tuong dtype, khong phai tensor) va _packed_params
    (tuple chua tensor trong so that) - phai kiem tra KIEU truoc
    khi goi .numel() (bai hoc Day 22/AGENTS.md diem 11).
    """
    def _bytes(value):
        if isinstance(value, torch.Tensor):
            return value.numel() * value.element_size()
        if isinstance(value, (tuple, list)):
            return sum(_bytes(v) for v in value)
        return 0  # dtype/so vo huong: khong chiem bo nho dang ke

    total = 0
    for value in model.state_dict().values():
        total += _bytes(value)
    return total


def eval_loss(model, x, y, batch=32):
    """Loss trung binh THEO TRONG SO TOKEN tren mot tap.

    Cong don NLL (loss * so token tung batch) roi chia cho tong
    token - batch cuoi it mau khong duoc tinh ngang batch day
    (sua theo review Trainer 1ab9c39; dung chuan Day 28: tong
    NLL / tong token). Khong tinh grad.
    """
    model.eval()
    tong_nll = 0.0
    tong_token = 0
    with torch.no_grad():
        for i in range(0, len(x), batch):
            xb, yb = x[i:i + batch], y[i:i + batch]
            logits = model(xb)
            loss = nn.functional.cross_entropy(
                logits.reshape(-1, VOCAB), yb.reshape(-1))
            n_token = int(yb.numel())
            tong_nll += float(loss.item()) * n_token
            tong_token += n_token
    # [TRAINER] Diem nhe - trung binh cong cac loss theo
    # batch ma khong trong so theo so token: batch cuoi
    # it cua so hon van duoc tinh ngang hang batch day.
    # O capstone nay tap test chi co 12 cua so (1 batch)
    # nen khong anh huong, nhung dung lai quen xau - o
    # Day 28 chinh nguoi hoc da gop PPL theo trong so
    # token (tong_nll / tong_token). Nen giu cung mot
    # chuan: cong loss * so token tung batch roi chia
    # cho tong token.
    # -> SUA: da doi sang cong theo trong so token o tren.
    assert tong_token > 0, "tap du lieu rong"
    loss_tb = tong_nll / tong_token
    assert math.isfinite(loss_tb), "loss khong huu han"
    return loss_tb


def train_baseline():
    """Huan luyen baseline 3 epoch, tra ve mo hinh da huan luyen."""
    model = TinyLM(VOCAB, D_MODEL, N_HEAD)
    print(f"baseline: {count_params(model):,} tham so, "
          f"{state_bytes(model) / 1024:.1f} KiB")
    opt = torch.optim.AdamW(model.parameters(), lr=LR)
    loader = DataLoader(TensorDataset(X_TRAIN, Y_TRAIN),
                        batch_size=BATCH, shuffle=True,
                        generator=torch.Generator().manual_seed(42))
    for epoch in range(EPOCHS):
        model.train()
        for xb, yb in loader:
            opt.zero_grad()
            logits = model(xb)
            loss = nn.functional.cross_entropy(
                logits.reshape(-1, VOCAB), yb.reshape(-1))
            loss.backward()
            opt.step()
        train_loss = eval_loss(model, X_TRAIN, Y_TRAIN)
        val_loss = eval_loss(model, X_VAL, Y_VAL)
        print(f"epoch {epoch + 1}/{EPOCHS}: train loss={train_loss:.4f} "
              f"val loss={val_loss:.4f}")
    return model


# ---- Buoc 2: ham do (khau 4 + 6: Eval + Serve) ----
def generate(model, seed_ids, n_tokens=GEN_TOKENS):
    """Sinh tham lam n_tokens ky tu tiep theo tu seed (argmax).

    Moi buoc chi dua 64 ky tu gan nhat vao mo hinh roi lay ky tu co
    xac suat cao nhat; lap lai cho den du so token.
    """
    model.eval()
    ctx = list(seed_ids)
    with torch.no_grad():
        for _ in range(n_tokens):
            window = torch.tensor([ctx[-SEQ_LEN:]], dtype=torch.long)
            logits = model(window)
            nxt = int(torch.argmax(logits[0, -1]).item())
            ctx.append(nxt)
    return ctx[len(seed_ids):]


def measure(model, seed_ids):
    """Do (perplexity test, giay/100 token) bang cung mot cach do.

    Tra ve dict co ppl, sec_per_100, mwh_per_token (uoc luong bang
    thoi gian x 65 W - ghi ro day la UOC LUONG, khong phai so do
    CodeCarbon that). Thoi gian lay TRUNG VI 5 lan chay, sau 1 lan
    warmup (bai hoc Day 29: trung vi triet duoc 1-2 lan do tre do
    VM dung chung giat; practice ghi "trung binh 3 lan" nhung trung
    binh bi keo lech khi 1 lan tre gap 10 lan, nen doi sang trung vi
    van giu dung tinh than "nhieu lan roi gop lai").
    """
    model.eval()
    # Perplexity = exp(loss trung binh) tren tap test.
    test_loss = eval_loss(model, X_TEST, Y_TEST)
    ppl = math.exp(test_loss)
    assert math.isfinite(ppl) and ppl > 1.0, f"ppl vo ly: {ppl}"
    # Sinh GEN_TOKENS token: 1 lan warmup roi 5 lan bam gio lay
    # trung vi, quy ve giay/100 token (GEN_TOKENS = 500 de 1 lan
    # do dai hon han nhieu: 100 token chi ~0,2 giay, nam ngay
    # duoi muc nhieu cua moi truong do - review Trainer 1ab9c39).
    generate(model, seed_ids)
    times = []
    for _ in range(5):
        t0 = time.perf_counter()
        generate(model, seed_ids)
        times.append(time.perf_counter() - t0)
    sec_per_100 = statistics.median(times) * 100 / GEN_TOKENS
    # Nang luong uoc luong: giay/token x 65 W = joule/token, doi
    # sang mWh (1 Wh = 3600 J, 1 mWh = 3,6 J). Chu y chia cho
    # 100 (so token cua don vi sec_per_100), khong chia cho
    # GEN_TOKENS - 2 so nay chi bang nhau khi GEN_TOKENS = 100.
    joule_per_token = (sec_per_100 / 100) * POWER_W
    mwh_per_token = joule_per_token / 3.6
    return {"ppl": ppl, "sec_per_100": sec_per_100,
            "mwh_per_token": mwh_per_token}


def rel_diff(a, b):
    """Chen lech tuong doi |a-b|/a, dung de kiem tra lap lai duoc."""
    return abs(a - b) / abs(a)


def _do_1_lan(model, seed_ids, n_tokens):
    """Sinh n_tokens token 1 lan, tra so giay (de do xen ke)."""
    t0 = time.perf_counter()
    generate(model, seed_ids, n_tokens)
    return time.perf_counter() - t0


def do_xen_ke(co_so, bien_the, seed_ids, n_vong=5,
              n_tokens=GEN_TOKENS):
    """Do xen ke A/B nhieu vong, tra ti le thoi gian tung vong.

    Moi vong: sinh n_tokens token bang baseline roi NGAY LAP TUC
    bang bien the (A/B/A/B...), tinh ti le t_bien / t_co_so cua
    vong do - va DOI THU TU o vong le (B truoc, A sau) de diet
    ca lech do "ben do sau luc nao cung cham hon". Tai may thay
    doi cham theo thoi gian nen do rieng 2 phien (do het
    baseline roi moi INT8) co the LAT verdict: lan chay cua
    nguoi hoc -1,25%, Trainer chay lai -28,2%. Xen ke diet
    lech nay. Tra (ti_le_trung_vi, ti_le_thap_nhat,
    ti_le_cao_nhat, danh sach ti le). Moi mo hinh warmup 1 lan
    truoc vong dau (khong warmup lai tung vong - mo hinh van
    nong). Thoi gian bam bang time.perf_counter().
    """
    if not isinstance(co_so, nn.Module) \
            or not isinstance(bien_the, nn.Module):
        raise ValueError("mo hinh phai la nn.Module.")
    if not isinstance(n_vong, int) or n_vong < 1:
        raise ValueError(f"n_vong phai la so nguyen >= 1, "
                         f"nhan duoc {n_vong}")
    if not isinstance(n_tokens, int) or n_tokens < 1:
        raise ValueError(f"n_tokens phai la so nguyen >= 1, "
                         f"nhan duoc {n_tokens}")
    co_so.eval()
    bien_the.eval()
    generate(co_so, seed_ids, n_tokens)
    generate(bien_the, seed_ids, n_tokens)
    ti_les = []
    for vong in range(n_vong):
        if vong % 2 == 0:
            t_a = _do_1_lan(co_so, seed_ids, n_tokens)
            t_b = _do_1_lan(bien_the, seed_ids, n_tokens)
        else:
            t_b = _do_1_lan(bien_the, seed_ids, n_tokens)
            t_a = _do_1_lan(co_so, seed_ids, n_tokens)
        assert t_a > 0 and t_b > 0, "thoi gian do phai duong"
        ti_les.append(t_b / t_a)
    return (float(statistics.median(ti_les)), min(ti_les),
            max(ti_les), ti_les)


# ---- Buoc 4: luat Chinchilla (khau 3: quyet dinh ngan sach) ----
def chinchilla_split(c_flops):
    """Chia ngan sach C (FLOPs) thanh so tham so N va so token D.

    Theo quy tac thuc nghiem: N ~ D/20 va C ~ 6*N*D, suy ra
    N = sqrt(C/120), D = 20*N. Tra ve (N, D).
    """
    if not isinstance(c_flops, (int, float)):
        raise ValueError(f"C phai la so, nhan duoc {type(c_flops)}")
    if not math.isfinite(c_flops) or c_flops <= 0:
        raise ValueError(f"C phai duong va huu han, nhan duoc {c_flops}")
    n_params = math.sqrt(c_flops / 120.0)
    return n_params, 20.0 * n_params


# [TRAINER] DIEM CHINH - dieu kien (c) hien tai chi do
# lai TRONG CUNG MOT PHIEN chay (cung tai may, cung
# trang thai nhiet/nhieu), nen no chi chung minh nhieu
# ngan han thap, khong chung minh duoc ket luan (b)
# on dinh giua cac lan chay doc lap. Bang chung cu the:
# lan chay cua nguoi hoc cho thoi gian INT8 chi giam
# 1,25% (verdict KHONG DAT nhanh b), con lan chay lai
# doc lap cua Trainer tren cung code, cung seed, cho
# thoi gian giam 28,2% (verdict DAT ca 3 nhanh) - chi
# PPL la tai lap dung (7,78). Voi mo hinh 1 lop rat
# nho, thoi gian sinh 100 token chi ~0,2 giay, chenh
# lech FP32/INT8 nam ngay duoi muc nhieu cua moi
# truong do. Cach cung co truoc khi tin verdict (b):
#  - do XEN KE A/B/A/B nhieu vong va tinh ti le cua
#    tung vong roi lay trung vi cac ti le (diet lech
#    do tai may thay doi theo thoi gian);
#  - tang GEN_TOKENS / so lan do de thoi gian 1 lan do
#    dai hon nhieu nhieu lan;
#  - bao verdict (b) kem khoang dao dong giua cac lan
#    chay doc lap, khong chi 1 con so.
# Ket luan KHONG DAT cua nguoi hoc van la cach bao cao
# dung (khong sua so), nhung ve mat bang chung ca hai
# verdict deu chua du manh - day la gioi han cua phep
# do thoi gian tren mo hinh ti hon, khong phai loi code.

def green_checkpoint(ppl_pct, energy_pct, repeat_diff_ppl,
                     repeat_diff_time, energy_lo_pct,
                     energy_hi_pct):
    """Ap Green checkpoint: var chi "tot hon" neu ca 3 dieu kien dung.

    Nhan vao cac PHAN TRAM da tinh san (de test truc tiep tung
    nhanh): (a) perplexity tang khong qua 5% so voi baseline;
    (b) nang luong/token giam it nhat 20% - lay theo TRUNG VI
    ti le xen ke A/B, kem khoang dao dong (thap..cao) giua cac
    vong; (c) do lan 2 chenh lech < 3% CA ppl VA thoi gian -
    "ket qua" cua measure() gom ca 2 so, nen chi xet ppl thi
    chua du: thoi gian khong lap lai duoc thi ket luan ve nang
    luong cung khong dung vung.
    Tra ve (dat_hay_khong, danh sach ly do).
    """
    ok_a = ppl_pct <= 5.0
    ok_b = energy_pct <= -20.0
    ok_c = repeat_diff_ppl < 0.03 and repeat_diff_time < 0.03
    reasons = [
        f"(a) ppl {ppl_pct:+.1f}% (nguong <= +5%): "
        f"{'DAT' if ok_a else 'KHONG DAT'}",
        f"(b) nang luong/token {energy_pct:+.1f}% (trung vi "
        f"xen ke, khoang {energy_lo_pct:+.1f}%.."
        f"{energy_hi_pct:+.1f}%) (nguong <= -20%): "
        f"{'DAT' if ok_b else 'KHONG DAT'}",
        f"(c) do lan 2 chenh ppl {repeat_diff_ppl * 100:.2f}%"
        f" + thoi gian {repeat_diff_time * 100:.1f}%"
        f" (nguong < 3%): {'DAT' if ok_c else 'KHONG DAT'}",
    ]
    return ok_a and ok_b and ok_c, reasons


def tu_kiem_tra():
    """Tu kiem tra bang assert (fail la dung).

    Kiem: eval_loss cong theo trong so token (batch 32 vs
    batch 5 phai cho cung so); green_checkpoint dung tung
    nhanh (a)/(b)/(c); do_xen_ke tra ti le hop le.
    """
    # eval_loss: tap test 14 cua so, batch=5 cat thanh 5/5/4 -
    # lay trung binh khong trong so se lech (kiem chung: cach
    # cu lech 7,1e-05 giua batch 32 va batch 5), cong theo
    # token thi 2 cach chia phai cho cung so.
    mo_hinh_thu = TinyLM(VOCAB, D_MODEL, N_HEAD)
    loss_32 = eval_loss(mo_hinh_thu, X_TEST, Y_TEST,
                        batch=32)
    loss_5 = eval_loss(mo_hinh_thu, X_TEST, Y_TEST, batch=5)
    assert abs(loss_32 - loss_5) < 1e-6, (
        f"eval_loss phai khong phu thuoc cach chia batch: "
        f"{loss_32:.6f} vs {loss_5:.6f}")
    # green_checkpoint: (a) ppl +8% -> rot; ca 3 dat -> DAT;
    # (c) thoi gian do lan 2 chenh 5% -> rot.
    dat, _ = green_checkpoint(8.0, -40.0, 0.01, 0.01,
                              -45.0, -35.0)
    assert not dat, "ppl +8% vuot nguong 5% phai rot"
    dat, _ = green_checkpoint(2.0, -25.0, 0.01, 0.01,
                              -30.0, -21.0)
    assert dat, "ca 3 nhanh dat thi phai DAT"
    dat, _ = green_checkpoint(2.0, -25.0, 0.01, 0.05,
                              -30.0, -21.0)
    assert not dat, "thoi gian do lan 2 chenh 5% phai rot"
    # do_xen_ke: 5 vong, 50 token tren 2 mo hinh giong het
    # nhau - ti le phai duong, trung vi nam trong [thap, cao]
    # va gan 1 (cung mo hinh thi khong the chenh nhieu). Dung
    # 5 vong vi trung vi cua 2 gia tri chi la trung binh cua
    # chung - 1 lan tre cung du lam lech (tu kiem tra bat duoc
    # ti le 3,55 o ban dau tien dung n_vong=2).
    ti_le_tv, ti_le_thap, ti_le_cao, ti_les = do_xen_ke(
        mo_hinh_thu, copy.deepcopy(mo_hinh_thu),
        IDS_TEST[:16].tolist(), n_vong=5, n_tokens=50)
    assert len(ti_les) == 5 and all(t > 0 for t in ti_les)
    assert ti_le_thap <= ti_le_tv <= ti_le_cao
    assert 0.5 <= ti_le_tv <= 2.0, (
        f"2 mo hinh giong het nhau thi ti le phai ~1, got "
        f"{ti_le_tv:.2f}")
    print("Tu kiem tra: PASS.")


def main():
    """Chay tron chuoi capstone: train, do, luong tu hoa, ket luan."""
    tu_kiem_tra()
    # Dat lai seed 42 truoc khi train: tu_kiem_tra() da tieu thu
    # mot phan luong RNG toan cuc (khoi tao mo hinh thu), khong
    # dat lai thi baseline khoi tao khac lan chay goc va PPL
    # khong tai lap dung 7,78 nua.
    torch.manual_seed(42)
    baseline = train_baseline()
    # Seed sinh lay 16 ky tu dau cua tap test (mo hinh chua thay
    # trong huan luyen, nhung thuoc cung doan van).
    seed = IDS_TEST[:16].tolist()

    results = {}
    results["baseline"] = measure(baseline, seed)
    print(f"baseline: ppl={results['baseline']['ppl']:.2f}, "
          f"{results['baseline']['sec_per_100']:.3f} giay/100 token, "
          f"{results['baseline']['mwh_per_token']:.4f} mWh/token "
          f"(uoc luong 65 W)")

    # ---- Buoc 3: bien the toi uu (khau 5: Optimize) ----
    # Luong tu hoa dong INT8 tu chinh baseline, KHONG huan luyen
    # lai; do lai bang CUNG ham measure(), CUNG tap test.
    quantized = torch.quantization.quantize_dynamic(
        baseline, {nn.Linear}, dtype=torch.qint8)
    quantized.eval()
    print(f"INT8: {state_bytes(quantized) / 1024:.1f} KiB "
          f"(baseline {state_bytes(baseline) / 1024:.1f} KiB)")
    assert state_bytes(quantized) < state_bytes(baseline), (
        "luong tu hoa phai nhe hon baseline")
    # Tat fastpath cua Transformer khi do mo hinh luong tu hoa:
    # Linear luong tu hoa de .weight duoi dang method (khong phai
    # tensor) nen ham kiem tra fastpath cua TransformerEncoderLayer
    # crash; duong slow path goi self.linear1(x) thi chay dung.
    # out_proj cua MultiheadAttention khong bi luong tu hoa (PyTorch
    # danh dau NonDynamicallyQuantizableLinear) nen attention van
    # chay binh thuong. Bat lai ngay sau khi do xong.
    torch.backends.mha.set_fastpath_enabled(False)
    try:
        results["int8"] = measure(quantized, seed)
        # Kiem tra lap lai duoc (dieu kien (c)): do lan 2, so ca
        # ppl va thoi gian. PPL la phep do xac dinh (eval + no_grad)
        # nen 2 lan chay phai gan nhu giong het nhau; neu chenh
        # > 3% thi nghi ngay quen model.eval() (dropout 0.1 con
        # bat se lam so dao dong). Thoi gian thi co the chenh lon
        # do VM dung chung giat - do la ket qua do, khong phai loi
        # code, nen chi bao cao chu khong assert.
        repeat = measure(quantized, seed)
        # Do xen ke A/B (nhanh (b) lay theo ti le nay, khong lay
        # theo 2 lan measure() rieng biet - review 1ab9c39).
        ti_le_tv, ti_le_thap, ti_le_cao, ti_les = do_xen_ke(
            baseline, quantized, seed, n_vong=N_VONG_XEN_KE)
        ti_le_chuoi = ", ".join(f"{t:.3f}" for t in ti_les)
        print(f"xen ke A/B {N_VONG_XEN_KE} vong x {GEN_TOKENS} "
              f"token: ti le INT8/baseline = {ti_le_chuoi}")
    finally:
        torch.backends.mha.set_fastpath_enabled(True)
    print(f"INT8: ppl={results['int8']['ppl']:.2f}, "
          f"{results['int8']['sec_per_100']:.3f} giay/100 token, "
          f"{results['int8']['mwh_per_token']:.4f} mWh/token "
          f"(uoc luong 65 W)")

    diff_ppl = rel_diff(results["int8"]["ppl"], repeat["ppl"])
    diff_time = rel_diff(results["int8"]["sec_per_100"],
                         repeat["sec_per_100"])
    print(f"do lan 2 INT8: ppl={repeat['ppl']:.2f} "
          f"(chenh {diff_ppl * 100:.2f}%), thoi gian chenh "
          f"{diff_time * 100:.1f}% (may dung chung co the giat, "
          f"chi tham khao)")
    assert diff_ppl < 0.03, f"ppl do lan 2 chenh {diff_ppl:.3f} > 3%"

    # ---- Buoc 4: Chinchilla ----
    # LUU Y: dap an Cau 2 trong practice tung ghi C=1e21 -> N ~ 9,1B.
    # Kiem tra lai bang cong thuc: N = sqrt(1e21/120) = 2,89e9
    # (9,1B la ket qua khi chia cho 12 thay vi 120, tuc C=1e22).
    # Trainer da sua dap an practice o commit 1ab9c39 (N = 2,9B,
    # D = 57,7B). So in ra duoi day da duoc assert tai tao dung
    # C (6*N*D = C).
    print("\nLuat Chinchilla (N ~ D/20, C ~ 6*N*D):")
    for c_flops in [1e18, 1e21]:
        n_params, n_tokens = chinchilla_split(c_flops)
        # Tu kiem tra: ap lai cong thuc phai tai tao dung C va D/N=20.
        assert abs(6 * n_params * n_tokens - c_flops) / c_flops < 1e-9
        assert abs(n_tokens / n_params - 20.0) < 1e-9
        print(f"  C={c_flops:.0e}: N={n_params / 1e9:.2f}B tham so, "
              f"D={n_tokens / 1e9:.1f}B token")
    print("  GPT-3: 175B tham so / 300B token = 1,7 token/tham so, "
          "thap hon nhieu so voi muc toi uu ~20 -> GPT-3 THUA mo "
          "hinh, THIEU du lieu (Chinchilla 70B/1400B token thang "
          "GPT-3 voi chi phi suy luan re hon).")

    # ---- Buoc 5: bang tong ket + Green checkpoint (khau 7) ----
    base, var = results["baseline"], results["int8"]
    ppl_pct = (var["ppl"] - base["ppl"]) / base["ppl"] * 100
    time_pct = ((var["sec_per_100"] - base["sec_per_100"])
                / base["sec_per_100"] * 100)
    energy_pct_rieng = ((var["mwh_per_token"]
                         - base["mwh_per_token"])
                        / base["mwh_per_token"] * 100)
    # Nhanh (b) lay theo TRUNG VI ti le xen ke, khong lay theo
    # 2 lan measure() rieng biet (do rieng lat verdict giua 2
    # lan chay doc lap - review Trainer 1ab9c39).
    energy_pct = (ti_le_tv - 1) * 100
    energy_lo_pct = (ti_le_thap - 1) * 100
    energy_hi_pct = (ti_le_cao - 1) * 100
    print("\nBang tong ket capstone (cung may, cung tap test, "
          "cung ham measure):")
    print(f"  {'Bien the':<14} {'PPL':>8} {'giay/100tok':>12} "
          f"{'mWh/token':>11}")
    print(f"  {'Baseline FP32':<14} {base['ppl']:>8.2f} "
          f"{base['sec_per_100']:>12.3f} {base['mwh_per_token']:>11.4f}")
    print(f"  {'INT8 dynamic':<14} {var['ppl']:>8.2f} "
          f"{var['sec_per_100']:>12.3f} {var['mwh_per_token']:>11.4f}")
    print(f"  Thay doi (do rieng): ppl {ppl_pct:+.1f}%, thoi gian "
          f"{time_pct:+.1f}%, nang luong {energy_pct_rieng:+.1f}%")
    print(f"  Thay doi (xen ke trung vi): nang luong "
          f"{energy_pct:+.1f}% (khoang {energy_lo_pct:+.1f}%.."
          f"{energy_hi_pct:+.1f}%) - day la so dung cho (b)")
    passed, reasons = green_checkpoint(
        ppl_pct, energy_pct, diff_ppl, diff_time, energy_lo_pct,
        energy_hi_pct)
    print("\nGreen checkpoint:")
    for reason in reasons:
        print(f"  {reason}")
    verdict = "DAT" if passed else "KHONG DAT"
    print(f"\nKet luan: INT8 {verdict} Green checkpoint.")
    if passed:
        print("  Bien the vua giu chat luong trong nguong (+5%), vua "
              "tiet kiem nang luong (>= 20%), lai lap lai duoc.")
    else:
        print("  Khong sua so: bao cao dung nhu vay, tim bien the "
              "khac (vd INT8 + fine-tune nhe) neu muon cai thien.")
    if energy_lo_pct <= -20.0 <= energy_hi_pct:
        print("  Canh bao: khoang dao dong cua (b) cat ngang "
              "nguong -20% - verdict (b) chua vung, can them "
              "vong do hoac may on dinh hon.")
    if diff_time >= 0.03:
        print(f"  Luu y: thoi gian 2 lan measure() rieng biet chenh "
              f"{diff_time * 100:.1f}% - (c) khong dat vi nhieu "
              "phien do; nhanh (b) thi lay theo ti le xen ke nen "
              "vung hon (xem khoang dao dong o (b)).")
    results["green_checkpoint"] = verdict
    results["repeat_ppl_diff"] = diff_ppl
    results["repeat_time_diff"] = diff_time
    results["xen_ke_ti_le"] = ti_les
    results["energy_pct_median"] = energy_pct
    results["energy_pct_lo"] = energy_lo_pct
    results["energy_pct_hi"] = energy_hi_pct
    with open("capstone_results.json", "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)
    print("\nDa luu capstone_results.json")
    print("Hoan tat capstone Day 30 - Phase 1.")


if __name__ == "__main__":
    main()
