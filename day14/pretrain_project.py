#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Mini-project: Pretraining co ngan sach (Day 14 - buoi toi).

Tong hop tuan 2: du lieu (Ngay 8) -> tai lap (9) -> on dinh (10) ->
ngan sach (11) -> danh gia (12) -> do luong (13).

Dau ra: experiment report 3 dong (baseline + 2 ablation), moi dong co
val loss, task score, thoi gian, kWh, kWh/1.000 token, verdict green.

Quy tac sat (va cach code nay tuan thu):
  - Ngan sach viet TRUOC khi chay (Bẫy 1).
  - Ablation doi dung 1 bien - kiem tra TU DONG bang assert_single_change
    (bai hoc Day 13: claim "1 bien" phai verify bang code, khong bang
    mat; truoc day SDPA lech 4.096 params ma docstring van ghi 1 bien).
  - So dien ghi kem nguon (do RAPL/NVML hay uoc luong TDP).

Drill muc 0 (dap an):
  1) t0 = time.perf_counter(); run(); dt = time.perf_counter() - t0
     (khong dung time.time() - dong ho tuong co the bi chinh).
  2) try: train() finally: tracker.stop() (crash cung khong mat so).
  3) Tach thanh 3 lan chay: baseline, chi tang batch, chi tang lr.

Cach chay:
    python pretrain_project.py
"""

import hashlib
import json
import math
import random
import time

import torch
import torch.nn as nn
import torch.nn.functional as F

# ---------------------------------------------------------------------------
# Phan A - Ngan sach + manifest (VIET TRUOC KHI CHAY)
# ---------------------------------------------------------------------------
# TODO (Phan A): dien ngan sach TRUOC KHI chay - giu nguyen trong
# report du co vuot (Bay 1: ngan sach viet sau ket qua la vo nghia).
BUDGET = {
    "max_minutes": 30,      # chay tren CPU, du kien chi vai phut
    "max_kwh": 0.10,        # uoc luong CodeCarbon TDP, du kien ~0.001
    "note": "chay tren CPU cua VM (khong GPU)",
}
SEED = 42


def seed_all(seed):
    """Seed MOI nguon ngau nhien (random + torch + cuda neu co).

    Chi seed 1 nguon ma quen nguon kia la du de ket qua troi (cau hoi 2).
    """
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def count_params(model):
    """Tong so tham so (dung cho cot N va kiem tra Chinchilla)."""
    return sum(p.numel() for p in model.parameters())


# ---------------------------------------------------------------------------
# Phan B - Du lieu mini + dedup (Ngay 8)
# ---------------------------------------------------------------------------
CORPUS = [
    "meo ngoi tren tham", "cho chay trong vuon", "chim hot tren cay",
    "meo uong sua buoi sang", "cho sua khi co khach", "chim bay qua song",
    "troi mua nen duong uot", "nang len sau con mua",
    # 10 cau cua rieng toi, cung phong cach ngan gon.
    "meo nam phoi nang", "cho dao ho trong san", "ca boi duoi ao",
    "trang sang tren troi", "gio thoi qua dong", "la roi day san",
    "be doc sach buoi toi", "com chin thom lung",
    "song chay ra bien", "hoa no trong vuon",
    # 2 cau TRUNG co tinh de kiem tra dedup co bat duoc khong.
    "meo ngoi tren tham", "troi mua nen duong uot",
]


def dedup(lines):
    """Loai cau trung, GIU NGUYEN thu tu (de hash on dinh)."""
    seen, out = set(), []
    for ln in lines:
        if ln not in seen:
            seen.add(ln)
            out.append(ln)
    return out


def sha256_text(text):
    """Hash SHA-256 cua text (ghi vao manifest de doi chieu lan sau)."""
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def prepare_data():
    """Dedup -> tach train/val -> tokenizer muc tu -> manifest.

    Returns:
        Tuple (train_seqs, val_seqs, vocab, pad_id, manifest).
    """
    raw = len(CORPUS)
    lines = dedup(CORPUS)
    print(f"Dedup: {raw} cau -> {len(lines)} cau duy nhat "
          f"(bat duoc {raw - len(lines)} cau trung)")
    # 2 cau cuoi lam val, con lai lam train.
    train_lines, val_lines = lines[:-2], lines[-2:]
    # Contamination check (Ngay 12): khong cau val nao nam trong train.
    assert not set(val_lines) & set(train_lines), \
        "RO RI: cau val trung cau train!"
    print(f"Train: {len(train_lines)} cau | Val: {len(val_lines)} cau "
          f"(held-out, khong trung train)")
    # Tokenizer muc tu: vocab = cac tu xuat hien + <pad> + <unk>.
    words = sorted({w for ln in lines for w in ln.split()})
    vocab = ["<pad>", "<unk>"] + words
    stoi = {w: i for i, w in enumerate(vocab)}
    pad_id = stoi["<pad>"]
    unk_id = stoi["<unk>"]

    def encode(sentence):
        return [stoi.get(w, unk_id) for w in sentence.split()]

    train_seqs = [encode(s) for s in train_lines]
    val_seqs = [encode(s) for s in val_lines]
    manifest = {
        "budget": BUDGET,
        "seed": SEED,
        "data_hash": sha256_text("\n".join(lines)),
        "n_raw": raw, "n_unique": len(lines),
        "n_train": len(train_lines), "n_val": len(val_lines),
        "vocab_size": len(vocab),
    }
    with open("manifest.json", "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2, ensure_ascii=False)
    print(f"Vocab: {len(vocab)} tu (pad_id={pad_id}) | "
          f"hash du lieu: {manifest['data_hash'][:16]}...")
    return train_seqs, val_seqs, vocab, pad_id, manifest


# ---------------------------------------------------------------------------
# Phan C - Model + vong train co health check (Ngay 7 + Ngay 10)
# ---------------------------------------------------------------------------
class TinyBlock(nn.Module):
    """Block decoder-only (ban mau co guard copy tu Ngay 7)."""

    def __init__(self, d, heads):
        super().__init__()
        # Guard Ngay 7: d phai chia het cho heads, neu khong attention
        # vo shape.
        assert d % heads == 0, "d phai chia het cho heads"
        self.attn = nn.MultiheadAttention(d, heads, batch_first=True)
        self.ln1, self.ln2 = nn.LayerNorm(d), nn.LayerNorm(d)
        self.mlp = nn.Sequential(nn.Linear(d, 4 * d), nn.GELU(),
                                 nn.Linear(4 * d, d))

    def forward(self, x):
        a, _ = self.attn(x, x, x, need_weights=False)
        x = self.ln1(x + a)
        return self.ln2(x + self.mlp(x))


class TinyLM(nn.Module):
    """LM nho kieu decoder-only."""

    def __init__(self, vocab, d=128, heads=4, layers=2, block_size=16):
        super().__init__()
        self.block_size = block_size
        self.emb = nn.Embedding(vocab, d)
        self.pos = nn.Embedding(block_size, d)
        self.blocks = nn.ModuleList(
            [TinyBlock(d, heads) for _ in range(layers)])
        self.head = nn.Linear(d, vocab)

    def forward(self, idx):
        T = idx.size(1)
        # Guard Ngay 7: chuoi dai hon block_size la loi vi tri.
        assert T <= self.block_size, "chuoi dai hon block_size"
        # Guard device (Ngay 7, tai pham o Day 11): arange phai cung
        # device voi idx.
        pos = torch.arange(T, device=idx.device)
        h = self.emb(idx) + self.pos(pos)
        for b in self.blocks:
            h = b(h)
        return self.head(h)


def make_batches(seqs, batch_size, pad_id, shuffle, seed):
    """Chia cau thanh batch, pad ve do dai lon nhat trong batch.

    Moi cau thanh 1 cap (x, y): x = ids[:-1], y = ids[1:] (du doan
    token tiep theo). Vi tri pad se bi ignore khi tinh loss.
    """
    g = torch.Generator().manual_seed(seed) if shuffle else None
    order = torch.randperm(len(seqs), generator=g).tolist() if shuffle \
        else list(range(len(seqs)))
    batches = []
    for i in range(0, len(order), batch_size):
        idxs = order[i:i + batch_size]
        pairs = [(s[:-1], s[1:]) for s in [seqs[j] for j in idxs]]
        maxlen = max(len(x) for x, _ in pairs)
        x = torch.full((len(pairs), maxlen), pad_id, dtype=torch.long)
        y = torch.full((len(pairs), maxlen), pad_id, dtype=torch.long)
        for r, (xi, yi) in enumerate(pairs):
            x[r, :len(xi)] = torch.tensor(xi)
            y[r, :len(yi)] = torch.tensor(yi)
        batches.append((x, y))
    return batches


def grad_norm(model):
    """Chuan gradient toan cuc (de kiem tra finite sau backward)."""
    total = 0.0
    for p in model.parameters():
        if p.grad is not None:
            total += p.grad.detach().pow(2).sum().item()
    return math.sqrt(total)


def train_run(cfg, train_seqs, val_seqs, vocab_size, pad_id):
    """Mot lan chay hoan chinh: seed lai tu dau, train, tra metrics.

    Health check theo dung thu tu Ngay 10:
      1) Truoc train: khong token ngoai vocab, batch khong rong.
      2) Sau backward, TRUOC opt.step: loss finite + grad norm finite.
         Neu NaN: DUNG va ghi severity, khong co train tiep.
      3) Moi epoch: in train loss + val loss.
      4) Do thoi gian bang perf_counter (khong time.time()).
    """
    seed_all(SEED)
    model = TinyLM(vocab_size, d=cfg.get("d", 128))
    opt = torch.optim.AdamW(model.parameters(), lr=cfg["lr"])
    # 1) Kiem tra du lieu truoc train (edge case: batch rong, token la).
    assert train_seqs and val_seqs, "train/val khong duoc rong"
    assert all(t < vocab_size for s in train_seqs + val_seqs for t in s), \
        "token ngoai vocab!"
    failed = None
    # 4) perf_counter (drill muc 0) - khong time.time().
    t0 = time.perf_counter()
    for epoch in range(1, cfg["epochs"] + 1):
        model.train()
        batches = make_batches(train_seqs, cfg["batch"], pad_id,
                               shuffle=True, seed=SEED + epoch)
        total_loss, total_tok = 0.0, 0
        for x, y in batches:
            opt.zero_grad()
            logits = model(x)
            loss = F.cross_entropy(logits.view(-1, vocab_size),
                                   y.view(-1), ignore_index=pad_id)
            loss.backward()
            # 2) Sau backward, TRUOC opt.step: kiem tra finite.
            gn = grad_norm(model)
            if not math.isfinite(loss.item()) or not math.isfinite(gn):
                failed = (f"NaN tai epoch {epoch} "
                          f"(loss={loss.item()}, grad_norm={gn})")
                break
            opt.step()
            n_tok = (y != pad_id).sum().item()
            total_loss += loss.item() * n_tok
            total_tok += n_tok
        if failed:
            print(f"  [SEVERITY] DUNG RUN: {failed}")
            break
        train_loss = total_loss / total_tok if total_tok else float("nan")
        # 3) Moi epoch: in train loss + val loss.
        val_loss, _ = evaluate(model, val_seqs, vocab_size, pad_id)
        print(f"  epoch {epoch}/{cfg['epochs']}: "
              f"train_loss={train_loss:.4f} val_loss={val_loss:.4f}")
    seconds = time.perf_counter() - t0
    val_loss, task_score = evaluate(model, val_seqs, vocab_size, pad_id)
    # Tong token da train = epochs * token train (moi epoch thay het).
    tokens_trained = cfg["epochs"] * sum(len(s) - 1 for s in train_seqs)
    return {"val_loss": val_loss, "task_score": task_score,
            "seconds": seconds, "params": count_params(model),
            "tokens_trained": tokens_trained, "failed": failed,
            "model": model}


# ---------------------------------------------------------------------------
# Phan D - Danh gia: val loss chua du (Ngay 12)
# ---------------------------------------------------------------------------
def evaluate(model, val_seqs, vocab_size, pad_id):
    """Tra ve (val_loss, task_score).

    val_loss: cross-entropy trung binh (bo qua pad).
    task_score: ty le vi tri du doan dung token tiep theo (held-out).
    """
    model.eval()
    total_loss, total_tok, correct = 0.0, 0, 0
    with torch.no_grad():
        for x, y in make_batches(val_seqs, 8, pad_id, shuffle=False,
                                 seed=0):
            logits = model(x)
            loss = F.cross_entropy(logits.view(-1, vocab_size),
                                   y.view(-1), ignore_index=pad_id)
            mask = y != pad_id
            n_tok = mask.sum().item()
            total_loss += loss.item() * n_tok
            total_tok += n_tok
            correct += (logits.argmax(-1)[mask] == y[mask]).sum().item()
    val_loss = total_loss / total_tok if total_tok else float("nan")
    task_score = correct / total_tok if total_tok else 0.0
    return val_loss, task_score


def generate(model, stoi, itos, prompt, n_tokens=10):
    """Sinh mau tham lam (greedy) tu prompt.

    Guard device (Ngay 7): tensor phai cung device voi model.
    """
    device = next(model.parameters()).device
    idx = torch.tensor([[stoi.get(w, stoi["<unk>"]) for w in prompt.split()]],
                       dtype=torch.long, device=device)
    model.eval()
    with torch.no_grad():
        for _ in range(n_tokens):
            idx_c = idx[:, -model.block_size:]
            nxt = model(idx_c)[:, -1, :].argmax(-1, keepdim=True)
            idx = torch.cat([idx, nxt], dim=1)
    return " ".join(itos[i] for i in idx[0].tolist())


# ---------------------------------------------------------------------------
# Phan E - Baseline + 2 ablation, moi cai dung 1 bien
# ---------------------------------------------------------------------------
def assert_single_change(baseline, ablation):
    """Kiem tra TU DONG: ablation chi khac baseline dung 1 key.

    Bai hoc Day 13: claim "1 bien" phai verify bang code. Neu khac 0
    hoac >1 key -> raise ngay, khong cho chay (khong ket luan duoc).
    """
    # [TRAINER] Ban cu chi dem key khac GIA TRI nen lot 2 lo hong:
    # (1) ablation THIEU key cua baseline (vd quen "lr") van qua
    #     guard, roi crash KeyError mo ho o giua train_run thay vi
    #     bi chan ngay tai day; (2) ablation THEM key la (vd go nham
    #     "lrr" thay "lr", hoac key train_run khong he doc nhu "wd")
    #     duoc chung nhan la "chi khac 1 key" du run chay ra GIONG
    #     HET baseline (doi 0 bien hieu dung) -> ket luan sai am tham,
    #     dung dang loi Day 13 ma guard nay sinh ra de chan. Nay bat
    #     buoc tap key cua ablation phai BANG tap key cua baseline
    #     (muon them knob moi: them vao BASELINE truoc, kem gia tri
    #     mac dinh), va dung 1 key khac gia tri.
    missing = [k for k in baseline if k not in ablation]
    extra = [k for k in ablation if k not in baseline]
    assert not missing and not extra, \
        f"ablation phai co dung tap key cua baseline " \
        f"(thieu: {missing}, thua: {extra})"
    diff = [k for k in baseline if baseline[k] != ablation[k]]
    assert len(diff) == 1, \
        f"ablation phai khac baseline dung 1 key, thay {len(diff)}: {diff}"
    return diff[0]


# ---------------------------------------------------------------------------
# Phan F - Green checkpoint: do dien co guard (Ngay 13)
# ---------------------------------------------------------------------------
def run_with_energy(cfg, train_seqs, val_seqs, vocab_size, pad_id):
    """Chay 1 cau hinh kem do dien CodeCarbon (guard finally).

    Returns:
        Dict metrics + kwh + kwh_per_1000_tokens + nguon so lieu.
    """
    from codecarbon import EmissionsTracker
    tracker = EmissionsTracker()
    tracker.start()
    try:
        metrics = train_run(cfg, train_seqs, val_seqs, vocab_size, pad_id)
    finally:
        # Guard Ngay 13: crash cung khong mat so lieu.
        tracker.stop()
    energy = tracker.final_emissions_data.energy_consumed  # kWh
    metrics["kwh"] = energy
    # Edge case guard (checklist 1): tranh chia cho 0 neu khong co token.
    if metrics["tokens_trained"] > 0:
        metrics["kwh_per_1000"] = energy / (metrics["tokens_trained"] / 1000)
    else:
        metrics["kwh_per_1000"] = float("nan")
    # Nguon so lieu: VM khong co RAPL/NVML -> UOC LUONG tu TDP.
    import os
    has_rapl = os.path.exists("/sys/class/powercap/intel-rapl")
    metrics["energy_source"] = "do (RAPL/NVML)" if has_rapl \
        else "uoc luong (TDP)"
    return metrics


def main():
    """Chay full mini-project: baseline + 2 ablation, in report."""
    train_seqs, val_seqs, vocab, pad_id, manifest = prepare_data()
    vocab_size = len(vocab)
    stoi = {w: i for i, w in enumerate(vocab)}
    itos = {i: w for i, w in enumerate(vocab)}

    # Chinchilla check (Phan A): D ~ 20*N - model co "doi data" khong?
    probe = TinyLM(vocab_size)
    n_params = count_params(probe)
    chin_tokens = 20 * n_params
    train_tokens = sum(len(s) - 1 for s in train_seqs)
    print(f"\nModel: {n_params:,} params | Chinchilla D ~ {chin_tokens:,} "
          f"tokens | train that: {train_tokens} tokens/ep | "
          f"{train_tokens * 30} tokens/30 epochs")
    print("  -> Du an mini tren corpus nho: muc dich la phuong phap "
          "(ngan sach/ablation/green), khong phai dat Chinchilla.")

    BASELINE = {"vocab": vocab_size, "lr": 3e-3, "batch": 8, "epochs": 30,
                "d": 128}
    # Chon 2 ablation tu thuc don (moi cai chi khac BASELINE 1 key).
    ablation1 = {**BASELINE, "lr": 1e-2}      # A: hoc nhanh hon?
    ablation2 = {**BASELINE, "batch": 16}     # B: on dinh hon?
    for ab in (ablation1, ablation2):
        key = assert_single_change(BASELINE, ab)
        print(f"Ablation kiem tra: chi khac key '{key}' - OK")

    runs = [("Baseline", BASELINE),
            ("Ablation A: lr 3e-3 -> 1e-2", ablation1),
            ("Ablation B: batch 8 -> 16", ablation2)]
    results = []
    for name, cfg in runs:
        print(f"\n===== {name} =====")
        m = run_with_energy(cfg, train_seqs, val_seqs, vocab_size, pad_id)
        m["name"] = name
        results.append(m)
        sample = generate(m["model"], stoi, itos, "meo", n_tokens=8)
        print(f"  Mau sinh tu 'meo': {sample}")

    # Kiem tra ngan sach (viet truoc khi chay - Bay 1).
    print("\n" + "=" * 78)
    print("EXPERIMENT REPORT (so that)")
    print("=" * 78)
    print(f"{'Cau hinh':28} | {'val loss':>8} | {'task':>5} | "
          f"{'giay':>5} | {'kWh':>8} | {'kWh/1k':>8} | Green?")
    print("-" * 78)
    base = results[0]
    for r in results:
        if r["name"] == "Baseline":
            verdict = "moc doi chieu"
        else:
            better = (r["val_loss"] < base["val_loss"]
                      or r["task_score"] > base["task_score"])
            # Green: chat luong tang ma dien KHONG tang.
            green = better and r["kwh"] <= base["kwh"] * 1.05
            verdict = ("GIU (green)" if green else
                       "BO (chat luong dien deu khong thang)")
        print(f"{r['name']:28} | {r['val_loss']:>8.4f} | "
              f"{r['task_score']:>5.2f} | {r['seconds']:>5.1f} | "
              f"{r['kwh']:>8.6f} | {r['kwh_per_1000']:>8.6f} | {verdict}")
        if r["failed"]:
            print(f"  !! Run that bai: {r['failed']}")
    total_s = sum(r["seconds"] for r in results)
    total_kwh = sum(r["kwh"] for r in results)
    print("-" * 78)
    print(f"Tong: {total_s / 60:.1f} phut / ngan sach {BUDGET['max_minutes']} "
          f"phut | {total_kwh:.6f} kWh / ngan sach {BUDGET['max_kwh']} kWh "
          f"| nguon so dien: {results[0]['energy_source']}")
    print("=" * 78)


if __name__ == "__main__":
    main()
