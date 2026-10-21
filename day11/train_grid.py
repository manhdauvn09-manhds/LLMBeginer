#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Train that 9 run cho Scaling Lab - Phan B (Day 11).

Grid: 3 kich thuoc mo hinh (S/M/L) x 3 token budget (50k/150k/400k).
Quy tac NGHEM NGAT (tranh Bay 1, 2):
    - Khoa learning rate cho ca 9 run (chi khac N va D).
    - Budget = so token that nhin thay: steps = D // (batch*block_size).
      Het budget la dung, ke ca loss van giam ("budget la hop dong").
    - Moi run: seed rieng, dem N that bang sum(p.numel()).

Dau ra: runs.jsonl - moi dong 1 run:
    {"name", "N", "D", "loss", "walltime_s", "kwh"}

Cach chay:
    python train_grid.py --smoke     # smoke test bat buoc: S1, 50 buoc
    python train_grid.py             # chay full 9 run
    python train_grid.py --run10     # train run 10 (M x 800k token)

kWh uoc luong voi gia dinh CPU 30W (giong bang mau cua tro giang) vi VM
khong do duoc dien nang that - GHI RO trong bao cao.

Chi can PyTorch (CPU).
"""

import argparse
import json
import math
import os
import time
import urllib.request

import torch
import torch.nn as nn
import torch.nn.functional as F

# ---------------------------------------------------------------------------
# Cau hinh grid (copy tu de bai) + sieu tham so train (KHOA cho moi run)
# ---------------------------------------------------------------------------
CONFIGS = {
    "S": dict(n_layer=2, n_embd=64, n_head=2),    # ~0.4M params
    "M": dict(n_layer=4, n_embd=128, n_head=4),   # ~1.2M params
    "L": dict(n_layer=6, n_embd=192, n_head=6),   # ~3.0M params
}
TOKEN_BUDGETS = [50_000, 150_000, 400_000]  # D cho moi run
BATCH_SIZE = 32
BLOCK_SIZE = 64
LR = 3e-4            # khoa cung cho ca 9 run (Bay 1)
WATTS = 30           # gia dinh CPU 30W de uoc luong kWh
DATA_URL = ("https://raw.githubusercontent.com/karpathy/char-rnn/master/"
            "data/tinyshakespeare/input.txt")


# ---------------------------------------------------------------------------
# Du lieu: tai tiny-shakespeare (giong Day 7), tokenizer char-level
# ---------------------------------------------------------------------------
def ensure_data(path="input.txt"):
    """Tai du lieu neu chua co. Tra ve van ban."""
    if os.path.exists(path):
        with open(path, encoding="utf-8") as f:
            return f.read()
    print("  dang tai du lieu...")
    with urllib.request.urlopen(DATA_URL, timeout=120) as r:
        text = r.read().decode("utf-8")
    with open(path, "w", encoding="utf-8") as f:
        f.write(text)
    return text


def build_tokenizer(text):
    """Char-level tokenizer. Tra ve (encode, decode, vocab_size)."""
    chars = sorted(set(text))
    stoi = {ch: i for i, ch in enumerate(chars)}
    itos = {i: ch for i, ch in enumerate(chars)}
    return (lambda s: [stoi.get(ch, 0) for ch in s],
            lambda ids: "".join(itos[i] for i in ids), len(chars))


# ---------------------------------------------------------------------------
# Model: TinyGPT tham so hoa (khoa Day 7, gon lai)
# ---------------------------------------------------------------------------
class Block(nn.Module):
    """Transformer block chuan pre-norm: x + Attn(LN(x)), x + MLP(LN(x))."""

    def __init__(self, n_embd, n_head, block_size):
        super().__init__()
        self.ln1 = nn.LayerNorm(n_embd)
        self.ln2 = nn.LayerNorm(n_embd)
        self.n_head = n_head
        hs = n_embd // n_head
        self.qkv = nn.Linear(n_embd, 3 * n_embd, bias=False)
        self.proj = nn.Linear(n_embd, n_embd, bias=False)
        self.register_buffer("tril", torch.tril(torch.ones(block_size,
                                                           block_size)))
        self.mlp = nn.Sequential(nn.Linear(n_embd, 4 * n_embd), nn.GELU(),
                                 nn.Linear(4 * n_embd, n_embd))

    def forward(self, x):
        B_, T_, _ = x.shape
        # Nhanh attention (pre-norm): 1 phep chieu QKV roi tach head.
        h = self.ln1(x)
        q, k, v = self.qkv(h).chunk(3, dim=-1)
        q = q.view(B_, T_, self.n_head, -1).transpose(1, 2)
        k = k.view(B_, T_, self.n_head, -1).transpose(1, 2)
        v = v.view(B_, T_, self.n_head, -1).transpose(1, 2)
        scores = (q @ k.transpose(-2, -1)) / math.sqrt(q.shape[-1])
        scores = scores.masked_fill(self.tril[:T_, :T_] == 0, float("-inf"))
        o = (F.softmax(scores, dim=-1) @ v).transpose(1, 2)
        o = o.reshape(B_, T_, -1)
        x = x + self.proj(o)
        # Nhanh MLP (pre-norm) + residual.
        return x + self.mlp(self.ln2(x))


class TinyGPT(nn.Module):
    """GPT thu nho voi kich thuoc cau hinh duoc."""

    def __init__(self, vocab_size, n_layer, n_embd, n_head,
                 block_size=BLOCK_SIZE):
        super().__init__()
        self.tok_emb = nn.Embedding(vocab_size, n_embd)
        self.pos_emb = nn.Embedding(block_size, n_embd)
        self.blocks = nn.Sequential(*[
            Block(n_embd, n_head, block_size) for _ in range(n_layer)])
        self.ln_f = nn.LayerNorm(n_embd)
        self.lm_head = nn.Linear(n_embd, vocab_size, bias=False)
        self.block_size = block_size

    def forward(self, x):
        B_, T_ = x.shape
        # [TRAINER] torch.arange(T_) mac dinh tao tensor tren CPU - neu x
        # nam tren GPU se crash device mismatch. Day 7 trainer da sua
        # dung loi nay (fix #3), Day 11 tai pham do copy-paste model.
        # Sua: tao arange tren CUNG device voi x.
        h = self.tok_emb(x) + self.pos_emb(torch.arange(T_, device=x.device))
        h = self.blocks(h)
        return self.lm_head(self.ln_f(h))

    def count_params(self):
        """Dem that so tham so (dung cho cot N trong bang)."""
        return sum(p.numel() for p in self.parameters())


# ---------------------------------------------------------------------------
# Train 1 run voi token budget
# ---------------------------------------------------------------------------
def get_batch(data, batch_size=BATCH_SIZE, block_size=BLOCK_SIZE):
    """Batch ngau nhien: x (B, T), y = x dich phai 1."""
    ix = torch.randint(0, len(data) - block_size - 1, (batch_size,))
    x = torch.stack([data[i:i + block_size] for i in ix])
    y = torch.stack([data[i + 1:i + block_size + 1] for i in ix])
    return x, y


def train_run(name, cfg, budget, data, encode, vocab_size, seed,
              log_path="runs.jsonl", max_steps_override=None):
    """Train 1 run den het token budget. Tra ve dict ket qua.

    steps = D // (batch_size * block_size): moi buoc "tieu" dung
    batch*block token. Het budget la dung - ke ca loss van giam.
    """
    torch.manual_seed(seed)
    model = TinyGPT(vocab_size, cfg["n_layer"], cfg["n_embd"],
                    cfg["n_head"])
    n_params = model.count_params()
    opt = torch.optim.AdamW(model.parameters(), lr=LR)
    steps = max_steps_override or (budget // (BATCH_SIZE * BLOCK_SIZE))
    print(f"[{name}] N={n_params:,} params, D={budget:,} tokens, "
          f"steps={steps}, seed={seed}")
    model.train()
    t0 = time.perf_counter()
    for step in range(1, steps + 1):
        x, y = get_batch(data)
        opt.zero_grad()
        loss = F.cross_entropy(model(x).view(-1, vocab_size), y.view(-1))
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        if step == 1 or step % max(1, steps // 5) == 0 or step == steps:
            print(f"  step {step}/{steps}: loss={loss.item():.4f}")
    walltime = time.perf_counter() - t0
    # Loss cuoi: trung binh 20 batch (bot nhieu).
    model.eval()
    with torch.no_grad():
        ls = []
        for _ in range(20):
            x, y = get_batch(data)
            ls.append(F.cross_entropy(model(x).view(-1, vocab_size),
                                      y.view(-1)).item())
    final_loss = sum(ls) / len(ls)
    kwh = walltime / 3600 * WATTS / 1000
    rec = {"name": name, "N": n_params, "D": budget, "loss": final_loss,
           "walltime_s": walltime, "kwh": kwh}
    with open(log_path, "a", encoding="utf-8") as f:
        f.write(json.dumps(rec) + "\n")
    print(f"[{name}] XONG: loss={final_loss:.4f}, "
          f"{walltime / 60:.1f} phut, kWh~{kwh:.4f}")
    return rec


def main():
    """Chay smoke test / full grid / run 10 tuy tham so dong lenh."""
    ap = argparse.ArgumentParser()
    ap.add_argument("--smoke", action="store_true",
                    help="smoke test bat buoc: S1, 50 buoc")
    ap.add_argument("--run10", action="store_true",
                    help="train run 10 (M x 800k token)")
    args = ap.parse_args()

    text = ensure_data()
    encode, _, vocab_size = build_tokenizer(text)
    data = torch.tensor(encode(text), dtype=torch.long)
    print(f"Du lieu: {len(text):,} ky tu, vocab={vocab_size}")

    if args.smoke:
        # Buoc 0 BAT BUOC: S1 voi 50 buoc - loss phai giam muot.
        print("SMOKE TEST: S1, 50 buoc (vuot budget 24 buoc de kiem tra)")
        rec = train_run("S1-smoke", CONFIGS["S"], TOKEN_BUDGETS[0], data,
                        encode, vocab_size, seed=1234,
                        log_path="runs_smoke.jsonl", max_steps_override=50)
        print(f"Smoke test: loss cuoi={rec['loss']:.4f} "
              f"(so voi ~4.2 ban dau - phai giam ro)")
        return

    if args.run10:
        # Run 10: M x 800k token (gap doi compute cua M3).
        rec = train_run("R10", CONFIGS["M"], 800_000, data, encode,
                        vocab_size, seed=999, log_path="runs_run10.jsonl")
        print(f"Run 10: loss that = {rec['loss']:.4f} (so voi du bao!)")
        return

    # Full grid 9 run.
    if os.path.exists("runs.jsonl"):
        os.remove("runs.jsonl")
    order = [("S1", "S", 0), ("S2", "S", 1), ("S3", "S", 2),
             ("M1", "M", 0), ("M2", "M", 1), ("M3", "M", 2),
             ("L1", "L", 0), ("L2", "L", 1), ("L3", "L", 2)]
    t0 = time.perf_counter()
    for i, (name, size, bi) in enumerate(order):
        train_run(name, CONFIGS[size], TOKEN_BUDGETS[bi], data, encode,
                  vocab_size, seed=1000 + i)
    print(f"\nTONG: 9 run xong trong {(time.perf_counter() - t0) / 60:.1f} "
          "phut. Ket qua trong runs.jsonl")


if __name__ == "__main__":
    main()
