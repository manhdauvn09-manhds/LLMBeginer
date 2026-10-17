#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Mini-project 1: TinyGPT tu so 0 (Day 07 - buoi toi).

Tong hop tuan 1 (Ngay 1 -> 6) thanh mot mo hinh chay duoc that
(~0.8M tham so): du lieu + dataloader (M1), tokenizer char-level (M2),
embedding + vi tri (M3), causal self-attention (M4), Transformer block
(M5), train loop + checkpoint (M6), sampling + model card (M7).

Cach chay:
    python tinygpt.py
File "input.txt": neu chua co, script tu tai tiny-shakespeare; neu
khong co mang thi dung doan van ban Viet du phong kem san trong code.

Ky vong (accept):
    M0: chay khong loi import, in duoc config.
    M1: x.shape == (32, 64), y[0,:5] == x[0,1:6].
    M2: round-trip dung 100% tren 100 doan trich ngau nhien.
    M3: h.shape == (B, T, n_embd); in tong so tham so.
    M4: khop torch SDPA voi sai so < 1e-5.
    M5: forward + backward 1 batch: loss huu han, khong NaN.
    M6: sau 2000 iters, train loss giam ro ret; ckpt.pt load lai duoc.
    M7: sinh van ban giong phong cach du lieu train; co MODEL_CARD.md.

Chi can PyTorch (CPU la du). Seed co dinh de tai lap duoc.
"""

import json
import math
import os
import time
import urllib.request

import torch
import torch.nn as nn
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# M0 - Khung file va config (moi sieu tham so dat o day, khong hardcode)
# ---------------------------------------------------------------------------
config = {
    "data": "input.txt",  # van ban train (tu tai neu chua co)
    "block_size": 64,     # T: do dai ngu canh (mo hinh nhin toi da 64 ky tu)
    "batch_size": 32,     # B: so chuoi moi batch
    "n_layer": 4,         # so Transformer block
    "n_head": 4,          # so attention head moi block
    "n_embd": 128,        # chieu embedding
    "lr": 3e-4,           # learning rate cho AdamW
    "max_iters": 2000,    # so buoc train
    "eval_every": 200,    # cu bao nhieu buoc thi do loss 1 lan
    "eval_batches": 20,   # do loss trung binh tren bao nhieu batch
    "weight_decay": 0.01, # chong overfit (Day 2)
    "grad_clip": 1.0,     # cat gradient de khong no (Day 2)
    "seed": 1337,         # tai lap duoc (Day 9 se dao sau)
}

torch.manual_seed(config["seed"])

# Van ban Viet du phong: chi dung khi khong tai duoc du lieu va chua co
# input.txt. Du nho de chay het pipeline (se overfit - chap nhan duoc
# cho muc dich demo).
FALLBACK_TEXT = (
    "xin chao cac ban. hom nay chung ta hoc lap trinh python. "
    "python la ngon ngu don gian va manh me. "
    "tri tue nhan tao dang thay doi the gioi. "
    "mo hinh ngon ngu hoc tu du lieu van ban rat lon. "
    "chung ta bat dau tu mo hinh nho de hieu cach no hoat dong. "
) * 40


# ---------------------------------------------------------------------------
# M1 - Du lieu + dataloader (Ngay 1)
# ---------------------------------------------------------------------------
def ensure_data(path):
    """Dam bao co file du lieu; tra ve noi dung van ban.

    Thu tu: file co san -> tai tiny-shakespeare -> van ban du phong.
    """
    if os.path.exists(path):
        print(f"M1: doc du lieu tu {path} (co san)")
        with open(path, encoding="utf-8") as f:
            return f.read()
    url = ("https://raw.githubusercontent.com/karpathy/char-rnn/master/"
           "data/tinyshakespeare/input.txt")
    try:
        print(f"M1: dang tai tiny-shakespeare tu github...")
        with urllib.request.urlopen(url, timeout=60) as r:
            text = r.read().decode("utf-8")
        with open(path, "w", encoding="utf-8") as f:
            f.write(text)
        print(f"M1: da tai xong, luu vao {path}")
        return text
    except Exception as e:  # khong mang / bi chan: dung du phong
        print(f"M1: khong tai duoc ({e}); dung van ban Viet du phong")
        with open(path, "w", encoding="utf-8") as f:
            f.write(FALLBACK_TEXT)
        return FALLBACK_TEXT


def build_splits(text):
    """Ma hoa toan bo van ban thanh tensor so nguyen, tach 90/10.

    Returns:
        Tuple (train_data, val_data): tensor 1 chieu cac token id.
    """
    data = torch.tensor(encode(text), dtype=torch.long)
    n = int(0.9 * len(data))  # 90% train, 10% val
    return data[:n], data[n:]


def get_batch(split, train_data, val_data):
    """Lay 1 batch ngau nhien: x (B, T), y (B, T).

    y la x dich phai 1 token: y[i, t] = x[i, t+1] - nhiem vu "du doan
    token ke tiep". Offset ngau nhien moi lan goi de model khong hoc
    thuoc thu tu du lieu (Bay 1: quen dich -> model hoc copy, loss ao).
    """
    data = train_data if split == "train" else val_data
    B, T = config["batch_size"], config["block_size"]
    # Vi tri bat dau ngau nhien; tru T+1 de y con du 1 token phia sau.
    ix = torch.randint(0, len(data) - T - 1, (B,))
    x = torch.stack([data[i:i + T] for i in ix])
    y = torch.stack([data[i + 1:i + T + 1] for i in ix])
    return x, y


# ---------------------------------------------------------------------------
# M2 - Tokenizer char-level + round-trip (Ngay 3)
# ---------------------------------------------------------------------------
# Vocab = toan bo ky tu xuat hien trong du lieu (don gian nhung du cho
# mini-project). Xay sau khi co text - bien toan cuc dat trong main().
stoi = {}  # char -> id
itos = {}  # id -> char


def build_vocab(text):
    """Xay vocab tu tap ky tu cua du lieu."""
    global stoi, itos
    chars = sorted(set(text))
    stoi = {ch: i for i, ch in enumerate(chars)}
    itos = {i: ch for i, ch in enumerate(chars)}
    return len(chars)


def encode(s):
    """Ma hoa chuoi -> list id (ky tu la -> id 0)."""
    return [stoi.get(ch, 0) for ch in s]


def decode(ids):
    """Giai ma list id -> chuoi."""
    return "".join(itos[i] for i in ids)


def test_roundtrip(text, n=100):
    """Round-trip phai dung 100% tren n doan trich ngau nhien."""
    for _ in range(n):
        i = torch.randint(0, len(text) - 50, (1,)).item()
        s = text[i:i + 50]
        assert decode(encode(s)) == s, f"round-trip hong tai: {s!r}"
    # Cau tieng Viet co dau: chi test khi vocab co du ky tu (du lieu Viet).
    # Voi du lieu tieng Anh, ky tu la duoc anh xa ve id 0 theo dung thiet
    # ke (Hint M2) nen khong the round-trip - bo qua cho nay la dung.
    vn = "xin chào Việt Nam"
    if all(ch in stoi for ch in vn):
        assert decode(encode(vn)) == vn
        print(f"M2: round-trip dung 100% tren {n} doan trich + cau tieng Viet")
    else:
        print(f"M2: round-trip dung 100% tren {n} doan trich "
              "(bo qua cau tieng Viet: vocab tieng Anh khong co dau)")


# ---------------------------------------------------------------------------
# M3 - Embedding + vi tri (Ngay 4): nam trong TinyGPT (xem M5)
# ---------------------------------------------------------------------------
def count_emb_params(vocab_size):
    """Dem tham so embedding, bao cao MB (float32 = 4 byte/param)."""
    n = vocab_size * config["n_embd"] + config["block_size"] * config["n_embd"]
    print(f"M3: tham so embedding = {n:,} (~{n * 4 / 1e6:.3f} MB)")
    return n


# ---------------------------------------------------------------------------
# M4 - Causal self-attention (Ngay 5): milestone de sai nhat
# ---------------------------------------------------------------------------
class Head(nn.Module):
    """Mot attention head: q, k, v = Linear(n_embd, head_size)(x).

    Args:
        n_embd: chieu embedding dau vao.
        head_size: chieu cua moi head.
        block_size: do dai toi da de tao san mask.
    """

    def __init__(self, n_embd, head_size, block_size):
        super().__init__()
        self.q = nn.Linear(n_embd, head_size, bias=False)
        self.k = nn.Linear(n_embd, head_size, bias=False)
        self.v = nn.Linear(n_embd, head_size, bias=False)
        # Mask tam giac duoi tao 1 LAN: o (i, j) = 1 neu j <= i
        # (token i duoc nhin token j dung truoc no).
        self.register_buffer("tril",
                             torch.tril(torch.ones(block_size, block_size)))

    def forward(self, x):
        B_, T_, _ = x.shape
        # [TRAINER] Guard 2026-10-17: giong loi da sua o Day 6 - mask tril
        # chi tao san den block_size. Neu T_ vuot qua, self.tril[:T_, :T_]
        # khong bao loi ma am tham tra ve ma tran thieu kich thuoc ->
        # masked_fill crash voi thong bao broadcast kho hieu.
        # That bai SOM voi thong bao ro rang de debug hon.
        assert T_ <= self.tril.shape[0], \
            f"do dai chuoi {T_} vuot qua block_size={self.tril.shape[0]}"
        q, k, v = self.q(x), self.k(x), self.v(x)
        # scores[i, j] = do hop q_i voi k_j; chia sqrt(head_size) de
        # softmax khong bao hoa (quen chia -> gradient ~ 0, Day 5).
        scores = (q @ k.transpose(-2, -1)) / math.sqrt(q.shape[-1])
        # BAT BUOC: cam nhin tuong lai truoc softmax. Quen mask la loi
        # "gian lan" kinh dien: loss dep ao nhung sample ra rac (Q1).
        scores = scores.masked_fill(self.tril[:T_, :T_] == 0, float("-inf"))
        attn = F.softmax(scores, dim=-1)
        return attn @ v


class MultiHeadAttention(nn.Module):
    """Nhieu head chay song song, ghep lai roi chieu ve n_embd."""

    def __init__(self, n_embd, n_head, block_size):
        super().__init__()
        # [TRAINER] Guard 2026-10-17: giong Day 6 (transformer_block.py) -
        # neu n_embd khong chia het cho n_head, head_size bi cat ngan do
        # phep // -> cat cac head ra sai chieu (B, T, h*head_size != d) ->
        # self.proj crash o forward voi thong bao dim kho hieu.
        # That bai SOM ngay khi khoi tao de thay loi config.
        assert n_embd % n_head == 0, "n_embd phai chia het cho n_head"
        head_size = n_embd // n_head
        self.heads = nn.ModuleList(
            [Head(n_embd, head_size, block_size) for _ in range(n_head)])
        self.proj = nn.Linear(n_embd, n_embd)  # gop head -> n_embd

    def forward(self, x):
        # Noi output cac head theo chieu cuoi: (B, T, h*head_size)=(B,T,d).
        out = torch.cat([h(x) for h in self.heads], dim=-1)
        return self.proj(out)


def test_sdpa():
    """So head tu viet voi torch SDPA: sai so phai < 1e-5 (M4 accept)."""
    torch.manual_seed(0)
    n_embd, head_size, T_ = 32, 8, 16
    head = Head(n_embd, head_size, T_).eval()
    x = torch.randn(2, T_, n_embd)
    with torch.no_grad():
        q, k, v = head.q(x), head.k(x), head.v(x)
        manual = head(x)
        # SDPA chuan cua torch, is_causal=True = tu ap mask nhan qua.
        ref = F.scaled_dot_product_attention(q, k, v, is_causal=True)
    diff = (manual - ref).abs().max().item()
    assert diff < 1e-5, f"khop SDPA kem: diff = {diff}"
    print(f"M4: attention tu viet khop torch SDPA (sai so {diff:.2e} < 1e-5)")


# ---------------------------------------------------------------------------
# M5 - Transformer block + TinyGPT (Ngay 6)
# ---------------------------------------------------------------------------
class Block(nn.Module):
    """Mot block: x + Attn(LN(x)) roi x + MLP(LN(x)) (pre-norm).

    Norm nam TRONG nhanh phu, khong chan duong residual -> gradient chay
    thang qua phep cong (Day 6). Quen "+ x" thi sau 4 layer gradient chet.
    """

    def __init__(self, n_embd, n_head, block_size):
        super().__init__()
        self.ln1 = nn.LayerNorm(n_embd)
        self.ln2 = nn.LayerNorm(n_embd)
        self.attn = MultiHeadAttention(n_embd, n_head, block_size)
        self.mlp = nn.Sequential(
            nn.Linear(n_embd, 4 * n_embd),  # mo rong 4x: "nha kho kien thuc"
            nn.GELU(),
            nn.Linear(4 * n_embd, n_embd),  # thu ve n_embd
        )

    def forward(self, x):
        x = x + self.attn(self.ln1(x))
        x = x + self.mlp(self.ln2(x))
        return x


class TinyGPT(nn.Module):
    """GPT thu nho: embedding -> n_layer block -> LN -> LM head rieng."""

    def __init__(self, vocab_size):
        super().__init__()
        cfg = config
        self.tok_emb = nn.Embedding(vocab_size, cfg["n_embd"])
        self.pos_emb = nn.Embedding(cfg["block_size"], cfg["n_embd"])
        self.blocks = nn.Sequential(*[
            Block(cfg["n_embd"], cfg["n_head"], cfg["block_size"])
            for _ in range(cfg["n_layer"])])
        self.ln_f = nn.LayerNorm(cfg["n_embd"])
        self.lm_head = nn.Linear(cfg["n_embd"], vocab_size, bias=False)

    def forward(self, x):
        """Nhan x (B, T) token id, tra logits (B, T, V)."""
        B_, T_ = x.shape
        # Cong vector token + vector vi tri (hoc duoc, du dung o co nay).
        # [TRAINER] Sua 2026-10-17: them device=x.device. Ban Day 6
        # (transformer_block.py) ban da viet dung torch.arange(t,
        # device=idx.device); sang file nay bi "regression" mat device ->
        # tren CPU van chay, nhung ai .to('cuda') model se crash vi
        # pos_emb nhan tensor CPU trong khi weight da len GPU.
        h = self.tok_emb(x) + self.pos_emb(torch.arange(T_, device=x.device))
        assert h.shape == (B_, T_, config["n_embd"]), "sai shape embedding"
        h = self.blocks(h)
        h = self.ln_f(h)
        return self.lm_head(h)

    def count_params(self):
        """Tong so tham so cua model."""
        return sum(p.numel() for p in self.parameters())


def test_forward_backward(model, train_data, val_data):
    """M5 accept: forward + backward 1 batch: loss huu han, khong NaN."""
    model.train()
    x, y = get_batch("train", train_data, val_data)
    logits = model(x)
    loss = F.cross_entropy(logits.view(-1, logits.shape[-1]), y.view(-1))
    assert torch.isfinite(loss), "loss khong huu han!"
    loss.backward()
    for n_, p in model.named_parameters():
        assert p.grad is not None and not torch.isnan(p.grad).any(), \
            f"grad loi tai {n_}"
    model.zero_grad()  # xoa grad thu nghiem truoc khi train that
    print(f"M5: forward+backward OK, loss batch dau = {loss.item():.4f}")


# ---------------------------------------------------------------------------
# M6 - Train loop + checkpoint (Ngay 2 + Ngay 7)
# ---------------------------------------------------------------------------
@torch.no_grad()
def estimate_loss(model, train_data, val_data):
    """Loss trung binh tren eval_batches batch (khong tinh gradient)."""
    model.eval()
    out = {}
    for split in ("train", "val"):
        losses = []
        for _ in range(config["eval_batches"]):
            x, y = get_batch(split, train_data, val_data)
            logits = model(x)
            losses.append(
                F.cross_entropy(logits.view(-1, logits.shape[-1]),
                                y.view(-1)).item())
        out[split] = sum(losses) / len(losses)
    model.train()
    return out


def train(model, train_data, val_data):
    """Vong train chinh. Tra ve lich su loss + thoi gian (giay)."""
    opt = torch.optim.AdamW(model.parameters(), lr=config["lr"],
                            weight_decay=config["weight_decay"])
    history = []
    t0 = time.perf_counter()
    for it in range(1, config["max_iters"] + 1):
        x, y = get_batch("train", train_data, val_data)
        logits = model(x)
        loss = F.cross_entropy(logits.view(-1, logits.shape[-1]), y.view(-1))
        opt.zero_grad()  # Bay 2: quen dong nay -> grad cong don -> NaN
        loss.backward()
        # Cat gradient: chong 1 batch "di" lam no tung loss (Day 2).
        torch.nn.utils.clip_grad_norm_(model.parameters(), config["grad_clip"])
        opt.step()
        if it % config["eval_every"] == 0 or it == 1:
            el = estimate_loss(model, train_data, val_data)
            history.append((it, el["train"], el["val"]))
            print(f"  iter {it:>4}: train loss {el['train']:.4f} | "
                  f"val loss {el['val']:.4f}")
    dt = time.perf_counter() - t0
    # Checkpoint: luu de tai su dung / nop bai (M6 accept).
    torch.save({"model": model.state_dict(), "config": config,
                "iter": config["max_iters"]}, "ckpt.pt")
    print(f"M6: train xong {config['max_iters']} iters trong {dt / 60:.1f} "
          f"phut; da luu ckpt.pt")
    return history, dt


def test_checkpoint(vocab_size, train_data, val_data):
    """M6 accept: ckpt.pt load lai duoc va cho output nhu cu."""
    sd = torch.load("ckpt.pt", weights_only=True)["model"]
    m2 = TinyGPT(vocab_size)
    m2.load_state_dict(sd)  # no neu khop -> checkpoint hop le
    m2.eval()
    x, _ = get_batch("val", train_data, val_data)
    torch.manual_seed(1)
    m1 = TinyGPT(vocab_size)
    m1.load_state_dict(sd)
    m1.eval()
    with torch.no_grad():
        assert torch.equal(m1(x), m2(x)), "load checkpoint cho output khac!"
    print("M6: ckpt.pt load lai duoc, output khop")


def save_loss_log(history):
    """Luu log loss ra CSV; ve PNG neu co matplotlib."""
    with open("loss_log.csv", "w") as f:
        f.write("iter,train_loss,val_loss\n")
        for it, tr, va in history:
            f.write(f"{it},{tr:.4f},{va:.4f}\n")
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        iters = [h[0] for h in history]
        plt.plot(iters, [h[1] for h in history], label="train")
        plt.plot(iters, [h[2] for h in history], label="val")
        plt.xlabel("iter")
        plt.ylabel("loss")
        plt.legend()
        plt.savefig("loss_curve.png")
        print("M6: da ve loss_curve.png")
    except ImportError:
        print("M6: chua co matplotlib -> bo qua ve do thi "
              "(xem loss_log.csv)")


# ---------------------------------------------------------------------------
# M7 - Sampling + model card + bao cao xanh (Ngay 7)
# ---------------------------------------------------------------------------
@torch.no_grad()
def generate(model, prompt, max_new=200, temperature=0.8):
    """Sinh tiep max_new token tu prompt (temperature sampling).

    Moi buoc: lay logits cua token CUOI -> chia temperature -> softmax
    -> boc tham 1 token (multinomial) -> noi vao chuoi. Temperature cao
    -> phan phoi "det" hon -> van ban "dien" hon.
    """
    assert temperature > 0, "temperature phai > 0"
    model.eval()
    # [TRAINER] Sua 2026-10-17: dat idx len dung device cua model (giong
    # loi pos_emb o TinyGPT.forward). Tren CPU khong anh huong, nhung neu
    # model da .to('cuda') ma idx con o CPU thi forward crash ngay buoc 1.
    dev = next(model.parameters()).device
    idx = torch.tensor([encode(prompt)], dtype=torch.long, device=dev)
    for _ in range(max_new):
        # Cat context ve block_size: model chi nho duoc T token gan nhat.
        logits = model(idx[:, -config["block_size"]:])
        probs = F.softmax(logits[:, -1, :] / temperature, dim=-1)
        nxt = torch.multinomial(probs, num_samples=1)
        idx = torch.cat([idx, nxt], dim=1)
    return decode(idx[0].tolist())


def write_model_card(history, dt, samples, vocab_size, total_params):
    """Viet MODEL_CARD.md (<= 20 dong): bao cao cua mini-project."""
    tr0, va0 = history[0][1], history[0][2]
    tr1, va1 = history[-1][1], history[-1][2]
    toks = config["max_iters"] * config["batch_size"] * config["block_size"]
    card = [
        "# MODEL_CARD.md - TinyGPT (Day 07 mini-project)",
        f"1. Kien truc: char-level Transformer {config['n_layer']} layer x "
        f"{config['n_head']} head, n_embd={config['n_embd']}, MLP 4x + GELU, "
        "pre-norm, causal mask.",
        f"2. Du lieu: {config['data']}, vocab {vocab_size} ky tu, "
        "tach 90/10 train/val.",
        f"3. Sieu tham so: block_size={config['block_size']}, "
        f"batch={config['batch_size']}, AdamW lr={config['lr']} "
        f"wd={config['weight_decay']}, {config['max_iters']} iters, "
        f"seed={config['seed']}.",
        f"4. Tham so: {total_params:,} (~{total_params / 1e6:.2f}M).",
        f"5. Loss: train {tr0:.2f} -> {tr1:.2f}, val {va0:.2f} -> {va1:.2f} "
        f"(sau {config['max_iters']} iters, {dt / 60:.1f} phut CPU).",
        f"6. Sample tot nhat (T=0.7): {samples[0.7][:110]!r}",
        f"7. Sample 'dien' nhat (T=1.3): {samples[1.3][:110]!r}",
        "8. Gioi han da biet: char-level + context 64 nen van ban con sai "
        "chinh ta; model nho (~1M) khong hieu nghia sau; chi nho pattern "
        "ngan.",
        f"9. Xanh: {toks / dt:,.0f} tokens/s; train gap doi iters ~ gap doi "
        "dien nang (chi phi tuyen tinh theo so buoc).",
    ]
    assert len(card) <= 20, "model card qua 20 dong!"
    with open("MODEL_CARD.md", "w", encoding="utf-8") as f:
        f.write("\n".join(card) + "\n")
    print("M7: da viet MODEL_CARD.md (9 dong)")


# ---------------------------------------------------------------------------
# Main: chay tuan tu M0 -> M7, dat accept moi sang milestone tiep
# ---------------------------------------------------------------------------
def main():
    """Pipeline mini-project: du lieu -> model -> train -> sample."""
    print("=" * 64)
    print("M0 - Config")
    print("=" * 64)
    print(json.dumps(config, indent=2))
    print("M0: import OK, in duoc config")

    print("\n" + "=" * 64)
    print("M1 - Du lieu + dataloader")
    print("=" * 64)
    text = ensure_data(config["data"])
    print(f"M1: do dai van ban = {len(text):,} ky tu")
    vocab_size = build_vocab(text)
    print(f"M1: vocab size = {vocab_size}")
    train_data, val_data = build_splits(text)
    x, y = get_batch("train", train_data, val_data)
    assert x.shape == (config["batch_size"], config["block_size"])
    assert torch.equal(y[0, :5], x[0, 1:6]), "y phai la x dich phai 1!"
    print(f"M1: x.shape = {tuple(x.shape)}, y[0,:5] == x[0,1:6] dung")
    print("M1: 10 token dau cua batch:", x[0, :10].tolist(), "->",
          repr(decode(x[0, :10].tolist())))

    print("\n" + "=" * 64)
    print("M2 - Tokenizer char-level")
    print("=" * 64)
    test_roundtrip(text)

    print("\n" + "=" * 64)
    print("M3 - Embedding + vi tri")
    print("=" * 64)
    count_emb_params(vocab_size)

    print("\n" + "=" * 64)
    print("M4 - Causal self-attention")
    print("=" * 64)
    test_sdpa()

    print("\n" + "=" * 64)
    print("M5 - Transformer block + TinyGPT")
    print("=" * 64)
    model = TinyGPT(vocab_size)
    total = model.count_params()
    print(f"M5: tong so tham so = {total:,} (~{total / 1e6:.2f}M)")
    test_forward_backward(model, train_data, val_data)

    print("\n" + "=" * 64)
    print("M6 - Train loop + checkpoint (~vai phut tren CPU)")
    print("=" * 64)
    history, dt = train(model, train_data, val_data)
    test_checkpoint(vocab_size, train_data, val_data)
    save_loss_log(history)

    print("\n" + "=" * 64)
    print("M7 - Sampling + model card + bao cao xanh")
    print("=" * 64)
    prompt = text[:30]  # lay 30 ky tu dau du lieu lam prompt mau
    print(f"M7: prompt = {prompt!r}")
    samples = {}
    for temp in (0.7, 1.0, 1.3):
        torch.manual_seed(7)  # cung seed de so sanh cong bang 3 nhiet do
        s = generate(model, prompt, max_new=200, temperature=temp)
        samples[temp] = s[len(prompt):]
        print(f"--- temperature = {temp}:")
        print("   ", samples[temp][:200].replace("\n", " "))
    write_model_card(history, dt, samples, vocab_size, total)

    print("\n" + "=" * 64)
    print("XONG - Checklist: M0-M7 dat accept; co ckpt.pt, loss_log.csv,")
    print("        MODEL_CARD.md (+ loss_curve.png neu co matplotlib).")
    print("=" * 64)


if __name__ == "__main__":
    main()
