#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Ghep mot Transformer block chay duoc (Day 06 - buoi toi).

Muc dich:
    - RMSNorm tu viet (khong dung nn.LayerNorm co san).
    - CausalSelfAttention: multi-head, mask nhan qua, scale 1/sqrt(d_head).
    - MLP: d -> 4d -> d voi GELU ("nha kho kien thuc" cua block).
    - TransformerBlock theo cong thuc pre-norm:
        x' = x + Attn(Norm(x));  y = x' + MLP(Norm(x')).
    - MiniGPT: embedding + N block + norm cuoi + LM head chia trong so
      (weight tying) voi embedding.
    - 5 unit test bat buoc PASS + train thu mini-task thay loss giam
      manh (< 1.0 sau 200 step).

Cach chay:
    python transformer_block.py

Ky vong:
    5/5 test in PASS; train 200 step: loss tu ~4.2 giam xuong < 1.0.

Chi can PyTorch (CPU la du, chay vai giay).
"""

import math
import time

import torch
import torch.nn as nn
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# Cau hinh mini de chay nhanh tren CPU
# ---------------------------------------------------------------------------
torch.manual_seed(6)

B, T, V = 4, 16, 64        # batch, do dai chuoi, vocab size
D, N_HEAD, N_LAYER = 64, 4, 2  # chieu embedding, so head, so block
T_MAX = 128               # do dai chuoi toi da cho pos embedding + mask
EPS = 1e-6                # epsilon chong chia cho 0 trong norm


# ---------------------------------------------------------------------------
# Buoc 1 - RMSNorm
# ---------------------------------------------------------------------------
class RMSNorm(nn.Module):
    """Chuan hoa theo RMS (root mean square).

    Cong thuc: y = x / sqrt(mean(x^2) + eps) * g, voi g la tham so hoc
    duoc khoi tao = 1. Khac LayerNorm o cho KHONG tru mean - re hon ma
    van du tot cho LLM (Llama dung RMSNorm).

    Args:
        d: so chieu cua vector can chuan hoa.
    """

    def __init__(self, d):
        super().__init__()
        # g: he so co gian tung chieu, hoc duoc. Khoi tao = 1 tuc la luc
        # dau chi chuan hoa thuan tuy, chua phong to/thu nho gi.
        self.weight = nn.Parameter(torch.ones(d))

    def forward(self, x):
        # mean tren chieu CUOI (keepdim de broadcast duoc): ms shape (..., 1).
        # rsqrt = 1/sqrt: x * rsqrt(ms) == x / sqrt(ms), on dinh so hon.
        ms = x.pow(2).mean(-1, keepdim=True)
        return x * torch.rsqrt(ms + EPS) * self.weight


# ---------------------------------------------------------------------------
# Buoc 2 - Causal self-attention (multi-head)
# ---------------------------------------------------------------------------
class CausalSelfAttention(nn.Module):
    """Self-attention nhan qua, nhieu head.

    Moi token chi duoc nhin cac token dung TRUOC no (ke ca chinh no) -
    bat buoc khi train mo hinh sinh (GPT), neu khong la "nhin trom dap an".

    Args:
        d: chieu embedding.
        n_head: so head.
        t_max: do dai chuoi toi da (de tao san mask).
    """

    def __init__(self, d, n_head, t_max=T_MAX):
        super().__init__()
        assert d % n_head == 0, "d phai chia het cho n_head"
        self.n_head = n_head
        self.d_head = d // n_head  # chieu moi head "nhin"

        # 1 ma tran duy nhat sinh ra Q, K, V (thay vi 3 ma tran rieng) -
        # gon va chay nhanh hon. bias=False: chuan cua LLM hien dai.
        self.qkv = nn.Linear(d, 3 * d, bias=False)
        self.proj = nn.Linear(d, d, bias=False)  # gop head, chieu ve d

        # Mask tam giac duoi tao 1 LAN (khong tao lai moi forward):
        # o (i, j) = 1 nghia la token i DUOC nhin token j (j <= i).
        self.register_buffer("mask", torch.tril(torch.ones(t_max, t_max)))

    def forward(self, x):
        B_, T_, _ = x.shape
        # Tach Q, K, V tu 1 phep chieu: (B, T, d) -> 3 x (B, T, d).
        q, k, v = self.qkv(x).chunk(3, dim=-1)
        # Tach head: (B, T, d) -> (B, T, h, dh) -> (B, h, T, dh).
        q = q.view(B_, T_, self.n_head, self.d_head).transpose(1, 2)
        k = k.view(B_, T_, self.n_head, self.d_head).transpose(1, 2)
        v = v.view(B_, T_, self.n_head, self.d_head).transpose(1, 2)

        # Diem tuong dong, chia sqrt(d_head): khong chia thi dot-product
        # o khoi tao da lon -> softmax bao hoa -> gradient ~ 0 (Bay 2).
        scores = (q @ k.transpose(-2, -1)) / math.sqrt(self.d_head)
        # Cam nhin tuong lai: o bi cam (mask == 0) gan -inf TRUOC softmax
        # -> qua softmax thanh 0. Cat mask theo do dai that T_.
        mask = self.mask[:T_, :T_]
        scores = scores.masked_fill(mask == 0, float("-inf"))
        weights = F.softmax(scores, dim=-1)
        out = weights @ v  # (B, h, T, dh): tron thong tin theo trong so

        # Gop head: (B, h, T, dh) -> (B, T, h, dh) -> duoi thang (B, T, d).
        out = out.transpose(1, 2).reshape(B_, T_, -1)
        return self.proj(out)


# ---------------------------------------------------------------------------
# Buoc 3 - MLP ("nha kho kien thuc" cua block)
# ---------------------------------------------------------------------------
class MLP(nn.Module):
    """MLP: Linear(d -> 4d) -> GELU -> Linear(4d -> d).

    Vi sao mo rong 4d roi thu ve? Hai ma tran day tao nhieu chieu an de
    luu pattern + phi tuyen GELU -> capacity lon hon han giu nguyen d.
    Day la trade-off chuan cua moi LLM (GPT, LLaMA).
    """

    def __init__(self, d):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d, 4 * d),  # mo rong: d -> 4d
            nn.GELU(),            # phi tuyen mem (thay ReLU)
            nn.Linear(4 * d, d),  # thu ve: 4d -> d
        )

    def forward(self, x):
        return self.net(x)


# ---------------------------------------------------------------------------
# Buoc 4 - TransformerBlock (pre-norm + residual)
# ---------------------------------------------------------------------------
class TransformerBlock(nn.Module):
    """Mot block Transformer dung cong thuc pre-norm.

    x' = x + Attn(Norm1(x));  y = x' + MLP(Norm2(x')).

    DIEM THEN CHOT: norm chi nam TRONG nhanh phu (truoc attn/mlp), khong
    bao gio chan duong residual x -> x + .... Nho vay gradient co "duong
    cao toc" chay thang qua phep cong, train sau nhieu lop van on.
    Viet nham thanh norm(x + attn(x)) la post-norm tra hinh (Bay 3).
    """

    def __init__(self, d, n_head):
        super().__init__()
        # Hai norm RIENG BIET cho 2 nhanh (khong dung chung).
        self.norm1 = RMSNorm(d)
        self.norm2 = RMSNorm(d)
        self.attn = CausalSelfAttention(d, n_head)
        self.mlp = MLP(d)

    def forward(self, x):
        x = x + self.attn(self.norm1(x))  # nhanh attention + residual
        x = x + self.mlp(self.norm2(x))   # nhanh MLP + residual
        return x


# ---------------------------------------------------------------------------
# Buoc 5 - MiniGPT + weight tying
# ---------------------------------------------------------------------------
class MiniGPT(nn.Module):
    """GPT thu nho: embedding + N block + norm cuoi + LM head.

    Weight tying: LM head DUNG CHUNG ma tran voi token embedding
    (logits = h @ tok_emb.weight.T), khong tao ma tran moi -> tiet kiem
    V x d tham so va ep 2 dau "nhat quan" voi nhau.
    """

    def __init__(self, v=V, d=D, n_head=N_HEAD, n_layer=N_LAYER,
                 t_max=T_MAX):
        super().__init__()
        self.tok_emb = nn.Embedding(v, d)   # bang vector tung token
        self.pos_emb = nn.Embedding(t_max, d)  # vector vi tri (hoc duoc)
        self.blocks = nn.ModuleList(
            [TransformerBlock(d, n_head) for _ in range(n_layer)])
        self.ln_f = RMSNorm(d)  # norm cuoi truoc khi ra logits

    def forward(self, idx):
        """Nhan idx (B, T) token id, tra logits (B, T, V)."""
        _, t = idx.shape
        assert t <= T_MAX, "chuoi dai hon T_MAX"
        pos = torch.arange(t, device=idx.device)
        # Cong embedding token + embedding vi tri
        # (mo hinh biet "ai" + "o dau").
        h = self.tok_emb(idx) + self.pos_emb(pos)
        for block in self.blocks:
            h = block(h)
        h = self.ln_f(h)
        # Weight tying: tai su dung tok_emb.weight lam LM head.
        logits = h @ self.tok_emb.weight.T
        return logits

    def count_params(self):
        """Tong so tham so (embedding dem 1 lan nho weight tying)."""
        return sum(p.numel() for p in self.parameters())


# ---------------------------------------------------------------------------
# Buoc 6 - 5 unit test (bat buoc PASS het)
# ---------------------------------------------------------------------------
def _fresh_model():
    """Tao model moi voi seed co dinh de test tai lap duoc."""
    torch.manual_seed(6)
    return MiniGPT()


def test_shapes():
    """Logits phai co shape (B, T, V)."""
    model = _fresh_model()
    idx = torch.randint(0, V, (B, T))
    logits = model(idx)
    assert logits.shape == (B, T, V), f"sai shape: {logits.shape}"
    print("PASS test_shapes: logits shape =", tuple(logits.shape))


def test_causal():
    """Kiem tra tinh nhan qua theo ca hai chieu.

    Doi token tuong lai -> qua khu KHONG doi;
    doi token qua khu -> tuong lai PHAI doi (Bay: mask nguoc chieu)."""
    model = _fresh_model().eval()
    base = torch.randint(0, V, (B, T))

    # Chieu 1: chi doi token CUOI -> cac vi tri truoc do phai giu nguyen.
    alt_future = base.clone()
    alt_future[:, -1] = (alt_future[:, -1] + 1) % V
    o1 = model(base)
    o2 = model(alt_future)
    assert torch.allclose(o1[:, :-1, :], o2[:, :-1, :], atol=1e-5), \
        "token qua khu bi anh huong boi tuong lai: mask hong!"
    print("PASS test_causal (chieu 1): doi token cuoi khong anh huong qua khu")

    # Chieu 2: doi token DAU -> cac vi tri sau PHAI thay doi (neu khong,
    # nghia la attention khong truyen thong tin gi ca).
    alt_past = base.clone()
    alt_past[:, 0] = (alt_past[:, 0] + 1) % V
    o3 = model(alt_past)
    assert not torch.allclose(o1[:, 1:, :], o3[:, 1:, :], atol=1e-5), \
        "doi qua khu ma tuong lai khong doi: attention khong chay?"
    print("PASS test_causal (chieu 2): doi token dau lam doi tuong lai")


def test_residual_grad():
    """Sau backward: moi tham so deu co grad, khong NaN (residual chay)."""
    model = _fresh_model()
    idx = torch.randint(0, V, (B, T))
    tgt = torch.randint(0, V, (B * T,))
    loss = F.cross_entropy(model(idx).view(-1, V), tgt)
    loss.backward()
    for name, p in model.named_parameters():
        assert p.grad is not None, f"{name} khong co grad"
        assert not torch.isnan(p.grad).any(), f"{name} grad bi NaN"
    print("PASS test_residual_grad: moi param deu co grad, khong NaN")


def test_tying():
    """Weight tying: bang embedding chi dem 1 lan trong tong params.

    Cong thuc dem (voi attention khong bias, MLP co bias):
      moi block = attn(3d^2 + d^2) + mlp(4d^2+4d + 4d^2+d) + 2 norm (2d)
                = 12d^2 + 7d
      tong = V*d (tok_emb) + T_MAX*d (pos_emb) + n_layer*block + d (ln_f).
    Neu ai them ma tran lm_head rieng thi tong se lech -> test no ngay.
    """
    model = _fresh_model()
    total = model.count_params()
    per_block = 12 * D * D + 7 * D
    expected = V * D + T_MAX * D + N_LAYER * per_block + D
    assert total == expected, f"tong params {total} != {expected}: " \
        "nghi co ma tran thua (mat weight tying?)"
    # Kiem tra truc tiep: khong ton tai tham so nao ngoai tok_emb.weight
    # co the dong vai tro LM head rieng (ten khong chua 'lm_head'/'head').
    names = [n for n, _ in model.named_parameters()]
    assert not any("lm_head" in n or "head" in n for n in names), \
        f"phat hien head rieng: {names}"
    print(f"PASS test_tying: tong {total:,} params, embedding dem dung 1 lan")


def test_deterministic():
    """Cung seed + eval mode -> 2 lan forward cho output giong het nhau."""
    torch.manual_seed(6)
    model = MiniGPT().eval()
    idx = torch.randint(0, V, (B, T))
    out1 = model(idx)
    out2 = model(idx)
    assert torch.equal(out1, out2), "forward khong deterministic!"
    print("PASS test_deterministic: 2 lan forward giong het nhau")


# ---------------------------------------------------------------------------
# Buoc 7 - Train thu mini-task (bonus sanity check)
# ---------------------------------------------------------------------------
def train_sanity(steps=200, lr=3e-4):
    """Train next-token prediction tren chuoi lap [3,7,3,7,...].

    Task nay qua de nen model DUNG phai hoc duoc: loss giam manh ve
    < 1.0. Neu loss dung yen -> block co bug (xem Bay 2, 3).

    Ghi chu ve loss buoc 0: embedding khoi tao chuan N(0,1) nen logits
    ban dau co do lech chuan ~sqrt(d)=8 -> cross-entropy khoi dau ~30
    chu khong phai ln(V)~4.2. Model that dung init nho hon (vd std=0.02).
    Dieu quan trong o day khong phai so ban dau ma la loss GIAM manh.
    """
    torch.manual_seed(6)
    model = MiniGPT()
    opt = torch.optim.AdamW(model.parameters(), lr=lr)

    # 64 chuoi [3,7,3,7,...] dai 16. Input = bo token cuoi, target = dich
    # trai 1: du doan token ke tiep (sau 3 la 7, sau 7 la 3).
    x = torch.tensor([[3, 7] * 8] * 64)
    inp, tgt = x[:, :-1], x[:, 1:].reshape(-1)

    t0 = time.perf_counter()
    init_loss = None
    for step in range(steps):
        opt.zero_grad()
        logits = model(inp)  # (64, 15, 64)
        loss = F.cross_entropy(logits.reshape(-1, V), tgt)
        if step == 0:
            init_loss = loss.item()
        loss.backward()
        opt.step()
        if step % 50 == 0 or step == steps - 1:
            print(f"  step {step:>3}: loss = {loss.item():.4f}")
    dt = time.perf_counter() - t0
    final = loss.item()
    assert final < 1.0, f"loss cuoi {final:.4f} >= 1.0: model khong hoc duoc!"
    print(f"  => loss giam tu {init_loss:.4f} xuong {final:.4f} trong "
          f"{dt:.1f}s (CPU): block chay DUNG.")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    """Chay 5 unit test roi train thu (dung di tiep khi test con do)."""
    print("=" * 64)
    print("5 unit test cho Transformer block")
    print("=" * 64)
    test_shapes()
    test_causal()
    test_residual_grad()
    test_tying()
    test_deterministic()
    print("\n" + "=" * 64)
    print("Train thu mini-task: [3,7,3,7,...] -> next token")
    print("=" * 64)
    train_sanity()
    print("\n" + "=" * 64)
    print("XONG - 5/5 test PASS, train loss giam dat.")
    print("=" * 64)


if __name__ == "__main__":
    main()
