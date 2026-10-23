#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Profile Lab - do truoc, toi uu sau (Day 13 - buoi toi).

Muc tieu: mot bang before/after (Cau hinh | tokens/s | peak VRAM |
nhan xet) voi SO THAT do tren may nay. Quy tac sat: cai tien nao khong
lam tokens/s tang (vuot nhieu do ~5%) thi BO.

A. Model TinyGPT + du lieu tong hop (khong can download).
B. bench(): thuoc do chuan (warmup rieng, synchronize truoc khi dung
   gio, reset dem VRAM).
C. Ablation 3 cai tien (doi 1 bien moi lan): pin_memory/num_workers,
   torch.compile, SDPA thay MultiheadAttention.
D. torch.profiler: xem op nao an thoi gian nhat.
E. CodeCarbon: do nang luong + 3 cau hoi Green checkpoint.

Cach chay:
    python profile_lab.py        # chay full lab (~5-10 phut CPU)

Chi can PyTorch (+ codecarbon cho phan E, neu cai duoc).
"""

import time

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print("device:", DEVICE, "| torch:", torch.__version__)

SEED = 42
SEQ_LEN = 128
BATCH_SIZE = 32


# ---------------------------------------------------------------------------
# Phan A - "Chuot bach" do dac: model nho + du lieu tong hop
# ---------------------------------------------------------------------------
class TinyBlock(nn.Module):
    """Block decoder-only: attention + MLP (ban goc dung MHA)."""

    def __init__(self, d=256, heads=4):
        super().__init__()
        self.attn = nn.MultiheadAttention(d, heads, batch_first=True)
        self.ln1, self.ln2 = nn.LayerNorm(d), nn.LayerNorm(d)
        self.mlp = nn.Sequential(nn.Linear(d, 4 * d), nn.GELU(),
                                 nn.Linear(4 * d, d))

    def forward(self, x):
        a, _ = self.attn(x, x, x, need_weights=False)
        x = self.ln1(x + a)
        return self.ln2(x + self.mlp(x))


class TinyBlockSDPA(nn.Module):
    """Cung cau truc TinyBlock nhung dung scaled_dot_product_attention.

    De ablation cong bang: chi doi co che attention, giu nguyen norm,
    MLP va kich thuoc (d, heads).
    """

    def __init__(self, d=256, heads=4):
        super().__init__()
        self.heads = heads
        self.qkv = nn.Linear(d, 3 * d, bias=False)
        self.proj = nn.Linear(d, d, bias=False)
        self.ln1, self.ln2 = nn.LayerNorm(d), nn.LayerNorm(d)
        self.mlp = nn.Sequential(nn.Linear(d, 4 * d), nn.GELU(),
                                 nn.Linear(4 * d, d))

    def forward(self, x):
        B_, T_, D_ = x.shape
        qkv = self.qkv(x).view(B_, T_, 3, self.heads, D_ // self.heads)
        q, k, v = qkv.permute(2, 0, 3, 1, 4).unbind(0)
        a = F.scaled_dot_product_attention(q, k, v, is_causal=False)
        a = a.permute(0, 2, 1, 3).reshape(B_, T_, D_)
        x = self.ln1(x + self.proj(a))
        return self.ln2(x + self.mlp(x))


class TinyGPT(nn.Module):
    """GPT thu nho de do bottleneck (khong phai de train that)."""

    def __init__(self, vocab=1000, d=256, layers=4, block_cls=TinyBlock):
        super().__init__()
        self.emb = nn.Embedding(vocab, d)
        self.blocks = nn.ModuleList([block_cls(d) for _ in range(layers)])
        self.head = nn.Linear(d, vocab)

    def forward(self, x):
        h = self.emb(x)
        for b in self.blocks:
            h = b(h)
        return self.head(h)


def count_params(model):
    """Tong so tham so (de sau nay uoc luong FLOPs/token)."""
    return sum(p.numel() for p in model.parameters())


class SynthDS(Dataset):
    """Du lieu tong hop: khong can download, van do duoc bottleneck."""

    def __init__(self, n=4000, seq=SEQ_LEN, vocab=1000):
        # n=4000 (khong phai 2000 nhu skeleton): bench() goi next(it)
        # 110 lan (warmup 10 + steps 100), moi batch 32 mau -> can it
        # nhat 3520 mau, khong thi StopIteration giua chung.
        self.x = torch.randint(0, vocab, (n, seq))

    def __len__(self):
        return len(self.x)

    def __getitem__(self, i):
        return self.x[i]


# TODO (Phan A): in tong so tham so - 1 dong duy nhat.
print("Tong so tham so TinyGPT:",
      f"{count_params(TinyGPT()):,}")


# ---------------------------------------------------------------------------
# Phan B - bench(): thuoc do chuan (do sai o day -> moi ket luan sai)
# ---------------------------------------------------------------------------
def bench(model, loader, steps=100, warmup=10):
    """Do tokens/s + peak VRAM cua 1 cau hinh.

    3 nguyen tac do dung: (1) warmup rieng de loai chi phi mot lan
    (khoi tao CUDA, compile lan dau); (2) synchronize truoc khi dung
    gio vi CUDA chay bat dong bo; (3) reset dem VRAM truoc khi do.
    """
    model.train()
    opt = torch.optim.AdamW(model.parameters(), lr=3e-4)
    it = iter(loader)
    # 1) Warmup: bo qua - kernel/CUDA/compile can "nong may".
    for _ in range(warmup):
        x = next(it).to(DEVICE)
        opt.zero_grad()
        loss = model(x).pow(2).mean()
        loss.backward()
        opt.step()
    # 2) Reset bo dem VRAM roi moi do that.
    if DEVICE == "cuda":
        torch.cuda.reset_peak_memory_stats()
        torch.cuda.synchronize()
    t0 = time.time()
    for _ in range(steps):
        x = next(it).to(DEVICE)
        opt.zero_grad()
        loss = model(x).pow(2).mean()
        loss.backward()
        opt.step()
    # TODO (Phan B): synchronize NGAY TRUOC khi bam gio dung.
    # Vi sao? CUDA bat dong bo: loss.backward() chi XEP HANG kernel roi
    # tra ve ngay - khong synchronize ma bam gio la do thoi gian "xep
    # hang" chu khong phai "lam xong" -> tokens/s ao cao, so sanh vo
    # nghia. (Tren CPU dong nay khong can nhung giu de code dung ca 2.)
    if DEVICE == "cuda":
        torch.cuda.synchronize()
    wall = time.time() - t0

    tokens = steps * loader.batch_size * SEQ_LEN
    tps = tokens / wall
    peak = torch.cuda.max_memory_allocated() / 1e9 if DEVICE == "cuda" else 0
    return {"tokens/s": round(tps, 1), "peak_GB": round(peak, 2),
            "wall_s": round(wall, 1)}


def fresh_model(block_cls=TinyBlock, compiled=False):
    """Tao model moi tinh (cung seed) - ablation phai doi 1 bien/lan."""
    torch.manual_seed(SEED)
    model = TinyGPT(block_cls=block_cls).to(DEVICE)
    if compiled:
        model = torch.compile(model)
    return model


# ---------------------------------------------------------------------------
# Phan C - Ablation: thu 3 cai tien, giu cai do duoc
# ---------------------------------------------------------------------------
def main():
    """Chay baseline + 3 ablation, in bang before/after."""
    loader = DataLoader(SynthDS(), batch_size=BATCH_SIZE, shuffle=True)
    results = []

    print("\n--- BASELINE ---")
    base = bench(fresh_model(), loader)
    print("BASELINE:", base)
    results.append(("Baseline", base, "moc doi chieu"))

    # Thi nghiem 1 - Data pipeline: pin_memory + num_workers.
    # Du doan: dataset tong hop qua nhe (__getitem__ chi index tensor
    # co san) nen data stall gan nhu khong co -> ky vong KHONG cai thien
    # (thuong thi co tac dung voi dataset that nang).
    print("\n--- TN1: pin_memory + num_workers=4 ---")
    loader2 = DataLoader(SynthDS(), batch_size=BATCH_SIZE, shuffle=True,
                         pin_memory=True, num_workers=4,
                         persistent_workers=True)
    r1 = bench(fresh_model(), loader2)
    print("TN1:", r1)
    results.append(("+ pin_memory/num_workers", r1, ""))

    # Thi nghiem 2 - torch.compile.
    # Luu y: lan chay DAU TIEN cua compile rat cham (bien dich graph) -
    # warmup=10 trong bench() da lo viec nay (Bay 3, cau hoi 3).
    print("\n--- TN2: torch.compile ---")
    try:
        r2 = bench(fresh_model(compiled=True), loader)
        print("TN2:", r2)
        results.append(("+ torch.compile", r2, ""))
    except Exception as e:  # may yeu/compile loi -> ghi khong ap dung
        print("TN2: khong ap dung:", e)
        results.append(("+ torch.compile", None, "khong ap dung"))

    # Thi nghiem 3 - SDPA thay MultiheadAttention.
    # Ky vong: peak VRAM giam ro (attention O(L^2) -> O(L)); tren CPU
    # khong co FlashAttention nen toc do co the khong doi nhieu.
    print("\n--- TN3: SDPA ---")
    print("SDPA kha dung:",
          hasattr(F, "scaled_dot_product_attention"))
    r3 = bench(fresh_model(block_cls=TinyBlockSDPA), loader)
    print("TN3:", r3)
    results.append(("+ SDPA", r3, ""))

    # Bang before/after - chi giu cai tang tokens/s vuot nhieu ~5%.
    print("\n" + "=" * 72)
    print("BANG BEFORE/AFTER (so that do tren may nay)")
    print("=" * 72)
    print(f"{'Cau hinh':28} | {'tokens/s':>10} | {'peak GB':>8} | Nhan xet")
    print("-" * 72)
    base_tps = base["tokens/s"]
    for name, r, note in results:
        if r is None:
            print(f"{name:28} | {'n/a':>10} | {'n/a':>8} | {note}")
            continue
        gain = (r["tokens/s"] - base_tps) / base_tps * 100
        if name == "Baseline":
            verdict = "moc doi chieu"
        elif gain > 5:
            verdict = f"GIU (+{gain:.1f}%)"
        else:
            verdict = f"BO ({gain:+.1f}%, trong nhieu do)"
        print(f"{name:28} | {r['tokens/s']:>10} | {r['peak_GB']:>8} | "
              f"{verdict}")
    print("=" * 72)
    return results


# ---------------------------------------------------------------------------
# Phan D - Trace profiler: kernel/op nao an thoi gian nhat?
# ---------------------------------------------------------------------------
def run_profiler():
    """Chay torch.profiler 20 buoc, in 12 op ton kem nhat."""
    from torch.profiler import ProfilerActivity, profile
    loader = DataLoader(SynthDS(), batch_size=BATCH_SIZE, shuffle=True)
    model = fresh_model()
    opt = torch.optim.AdamW(model.parameters())
    xb = next(iter(loader)).to(DEVICE)
    acts = [ProfilerActivity.CPU]
    if DEVICE == "cuda":
        acts.append(ProfilerActivity.CUDA)
    sort_key = ("self_cuda_time_total" if DEVICE == "cuda"
                else "self_cpu_time_total")
    with profile(activities=acts, record_shapes=True) as prof:
        for _ in range(20):
            opt.zero_grad()
            loss = model(xb).pow(2).mean()
            loss.backward()
            opt.step()
    print(prof.key_averages().table(sort_by=sort_key, row_limit=12))
    # prof.export_chrome_trace("trace.json")  # mo bang chrome://tracing
    # Nhan xet tu trace (so that tren CPU):
    # 1) aten::mm (52.6%) + aten::addmm (18.2%) ~ 70% thoi gian la nhan
    #    ma tran (GEMM) cua cac lop Linear (QKV/proj/MLP) - bottleneck la
    #    COMPUTE, khong phai du lieu.
    # 2) Khop du doan TN1: dataset tong hop qua nhe (__getitem__ chi
    #    index tensor) nen data stall ~ 0 -> pin_memory/num_workers vo
    #    dung, chi them overhead IPC.
    # 3) SDPA tren CPU dung kernel flash_attention_for_cpu nhung chi
    #    chiem ~9% -> doi attention khong cham duoc bottleneck chinh.


# ---------------------------------------------------------------------------
# Phan E - Green checkpoint: CodeCarbon
# ---------------------------------------------------------------------------
def run_energy():
    """Do nang luong baseline bang CodeCarbon + tra loi 3 cau hoi."""
    from codecarbon import EmissionsTracker
    import os
    loader = DataLoader(SynthDS(), batch_size=BATCH_SIZE, shuffle=True)
    tracker = EmissionsTracker()
    tracker.start()
    r = bench(fresh_model(), loader, steps=100)
    em = tracker.stop()
    energy = tracker.final_emissions_data.energy_consumed
    print(f"CO2: {em:.6f} kg | dien: {energy:.6f} kWh "
          f"(cho {r['tokens/s']} tokens/s)")
    # Cau 1: do hay uoc luong? May ao khong co cam bien RAPL/NVML.
    rapl = os.path.exists("/sys/class/powercap/intel-rapl")
    print(f"1) CodeCarbon dang UOC LUONG (khong phai do). Bang chung: "
          f"may ao nay {'co' if rapl else 'khong co'} cam bien RAPL "
          f"(/sys/class/powercap/intel-rapl), khong co GPU/NVML. "
          f"So uoc luong tu TDP chi dung de ang chung do lon.")
    # Cau 2: phan cung that.
    import platform
    cpu = open("/proc/cpuinfo").read().split("model name")[1].split(
        ":")[1].split("\n")[0].strip()
    ram = os.popen("free -g | awk '/Mem:/ {print $2}'").read().strip()
    print(f"2) Phan cung: {cpu} | RAM {ram} GB | {platform.platform()} "
          f"| chay tren VM (CPU-only, khong GPU).")
    # Cau 3: kWh cho 1 trieu token.
    tps = r["tokens/s"]
    sec_per_1m = 1_000_000 / tps
    # Nang luong tren token tu lan do vua roi:
    tokens_bench = 100 * BATCH_SIZE * SEQ_LEN
    kwh_per_1m = energy / tokens_bench * 1_000_000
    print(f"3) Voi {tps} tokens/s, 1 trieu token mat {sec_per_1m:.0f} giay "
          f"~ {kwh_per_1m:.6f} kWh (suy tu lan do CodeCarbon vua roi).")


if __name__ == "__main__":
    main()
    print("\n" + "#" * 72 + "\n")
    run_profiler()
    print("\n" + "#" * 72 + "\n")
    run_energy()
