#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Training loop tai lap + checkpoint & resume (Day 09 - buoi toi).

Muc dich:
    - seed_everything: gieo hat toan dien (random/numpy/torch/cuda).
    - Checkpoint DAY DU 6 thu: epoch, model, optimizer, scheduler,
      RNG states, seed.
    - Thi nghiem 1 (tai lap): train 2 lan doc lap (2 process rieng) ->
      SHA-256 trong so cuoi + loss cuoi Y HET nhau.
    - Thi nghiem 2 (resume): train 6 epoch lien tuc == train 3 epoch ->
      "sap nguon" -> resume 3 epoch (H_A == H_B).

Cach chay:
    python train_repro.py            # chay ca 2 thi nghiem
    python train_repro.py --worker ...  # che do worker (duoc goi noi bo)

Ghi chu: de bai nhac setup_day9.py "da viet san" nhung file do khong co
trong repo, nen phan setup du lieu duoc nhung thang vao make_data().

Chi can PyTorch (CPU la du).
"""

import argparse
import hashlib
import json
import os
import random
import subprocess
import sys

import torch
import torch.nn as nn

try:
    import numpy as np
    _HAS_NUMPY = True
except ImportError:  # torch CPU van chay duoc khi thieu numpy
    np = None
    _HAS_NUMPY = False

# ---------------------------------------------------------------------------
# Cau hinh (moi sieu tham so dat o day)
# ---------------------------------------------------------------------------
SEED = 42
EPOCHS = 6
BATCH_SIZE = 64
LR = 1e-3
N, D = 2000, 20          # kich thuoc du lieu tong hop
N_TRAIN = 1800           # train 1800 / val 200


# ---------------------------------------------------------------------------
# Buoc 0 - Du lieu & mo hinh (setup_day9.py duoc nhung vao day)
# ---------------------------------------------------------------------------
def make_data():
    """Tao du lieu hoi quy tong hop + chia train/val DETERMINISTIC.

    Du lieu luon giong nhau moi lan goi (dung generator rieng seed 0),
    khong phu thuoc seed cua thi nghiem - seed thi nghiem chi anh huong
    khoi tao model va thu tu shuffle.
    """
    g = torch.Generator().manual_seed(0)  # RNG rieng cho du lieu
    X = torch.randn(N, D, generator=g)
    true_w = torch.randn(D, 1, generator=g)
    y = X @ true_w + 0.1 * torch.randn(N, 1, generator=g)
    # Split deterministic: generator rieng seed 7 (nhu de bai).
    perm = torch.randperm(N, generator=torch.Generator().manual_seed(7))
    train_ds = torch.utils.data.TensorDataset(X[perm[:N_TRAIN]],
                                              y[perm[:N_TRAIN]])
    val_ds = torch.utils.data.TensorDataset(X[perm[N_TRAIN:]],
                                            y[perm[N_TRAIN:]])
    return train_ds, val_ds


class TinyMLP(nn.Module):
    """MLP nho: 20 -> 64 (ReLU) -> 1 (bai toan hoi quy)."""

    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(D, 64), nn.ReLU(), nn.Linear(64, 1))

    def forward(self, x):
        return self.net(x)


# ---------------------------------------------------------------------------
# Buoc 1 - seed_everything(seed): gieo hat toan dien
# ---------------------------------------------------------------------------
def seed_everything(seed):
    """Gieo hat cho moi nguon ngau nhien, bat che do deterministic.

    Bao gom: random (python), numpy, torch CPU, torch cuda (neu co).
    cudnn.deterministic=True + benchmark=False phai di doi: benchmark=True
    bao cuDNN "tu chon thuat toan nhanh nhat moi lan chay" - thuat toan
    khac nhau cho ket qua float khac nhau o chu so cuoi, pha vo ngam moi
    co gang deterministic (Bay 2).
    """
    random.seed(seed)
    if _HAS_NUMPY:
        np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    # Nang cao: bat che do deterministic that - op nao khong ho tro se
    # bao loi ngay thay vi am tham cho ket qua khac nhau. Voi MLP nay
    # (Linear/ReLU/MSELoss/Adam) thi an toan.
    torch.use_deterministic_algorithms(True)
    return seed


# ---------------------------------------------------------------------------
# Tien ich: hash trong so + checkpoint
# ---------------------------------------------------------------------------
def state_dict_hash(model):
    """SHA-256 tren bytes cua toan bo tensor trong state_dict.

    Dung de so sanh 2 run: hash giong het nhau = trong so giong nhau
    den tung bit (manh hon nhieu so voi "loss gan bang").
    """
    h = hashlib.sha256()
    for p in model.state_dict().values():
        # contiguous + bytes(storage): khong can numpy van lay duoc bytes.
        h.update(bytes(p.detach().cpu().contiguous().untyped_storage()))
    return h.hexdigest()


def save_checkpoint(path, epoch, model, opt, sched, loader_gen, seed):
    """Luu checkpoint DAY DU 6 thu (quy tac: stateful thi luu het).

    Thieu optimizer state (momentum Adam) -> huong di sau resume chech.
    Thieu RNG state -> thu tu batch sau resume khac -> gradient khac.
    Thieu scheduler state -> lr khong khop. Chi luu model.state_dict()
    la Bay 1: resume duoc nhung khong giong run lien tuc.
    """
    ckpt = {
        "epoch": epoch,
        "seed": seed,
        "model": model.state_dict(),
        "optimizer": opt.state_dict(),   # momentum buffers cua Adam!
        "scheduler": sched.state_dict(),  # step count cua StepLR
        "rng": {
            "python": random.getstate(),
            "numpy": np.random.get_state() if _HAS_NUMPY else None,
            "torch": torch.get_rng_state(),
            "cuda": torch.cuda.get_rng_state_all()
                    if torch.cuda.is_available() else None,
            # TRANG THAI RIENG cua DataLoader shuffle: no dung Generator
            # doc lap, khong nam trong torch.get_rng_state()!
            "dataloader": loader_gen.get_state(),
        },
    }
    torch.save(ckpt, path)
    # Giu checkpoint moi nhat + moi 3 epoch mot ban (do ton dia).
    d = os.path.dirname(path) or "."
    for f in os.listdir(d):
        if f.startswith("ckpt_epoch") and f.endswith(".pt"):
            e = int(f[len("ckpt_epoch"):-len(".pt")])
            if e != epoch and e % 3 != 0:
                os.remove(os.path.join(d, f))


def load_checkpoint(path, model, opt, sched, loader_gen):
    """Nap checkpoint, phuc hoi TOAN BO trang thai. Tra ve (epoch, seed).

    Thu tu cuc ky quan trong: nap RNG state TRUOC khi lam bat cu viec
    ngau nhien nao tiep theo.
    """
    ckpt = torch.load(path, weights_only=False)
    model.load_state_dict(ckpt["model"])
    opt.load_state_dict(ckpt["optimizer"])
    sched.load_state_dict(ckpt["scheduler"])
    rng = ckpt["rng"]
    random.setstate(rng["python"])
    if _HAS_NUMPY and rng["numpy"] is not None:
        np.random.set_state(rng["numpy"])
    torch.set_rng_state(rng["torch"])
    if torch.cuda.is_available() and rng["cuda"] is not None:
        torch.cuda.set_rng_state_all(rng["cuda"])
    loader_gen.set_state(rng["dataloader"])
    return ckpt["epoch"], ckpt["seed"]


# ---------------------------------------------------------------------------
# Buoc 2 - train_one_run: vong lap train (+ resume)
# ---------------------------------------------------------------------------
def train_one_run(seed, epochs, log_path, ckpt_dir, resume_from=None):
    """Train 1 run; neu resume_from thi nap checkpoint va train tiep.

    Ghi moi epoch 1 dong JSONL vao log_path:
        {"epoch": e, "train_loss": ..., "val_loss": ..., "lr": ...}

    Returns:
        Tuple (loss_train_cuoi, loss_val_cuoi, hash_trong_so, losses).
    """
    # Seed TRUOC KHI tao model/DataLoader: nn.Linear khoi tao trong so
    # bang RNG cua torch - seed sau la qua muon (Q2, thu pham so 3).
    seed_everything(seed)
    train_ds, val_ds = make_data()

    # DIEM 90% tutorial bo sot: shuffle cua DataLoader dung RNG RIENG,
    # khong an theo torch.manual_seed. Phai truyen generator co seed.
    # (num_workers=0 de tranh phuc tap worker_init_fn - co comment.)
    loader_gen = torch.Generator().manual_seed(seed)
    loader = torch.utils.data.DataLoader(
        train_ds, batch_size=BATCH_SIZE, shuffle=True, generator=loader_gen)
    val_x, val_y = val_ds.tensors

    model = TinyMLP()
    opt = torch.optim.Adam(model.parameters(), lr=LR)
    sched = torch.optim.lr_scheduler.StepLR(opt, step_size=2, gamma=0.5)
    criterion = nn.MSELoss()
    os.makedirs(ckpt_dir, exist_ok=True)

    start_epoch = 1
    if resume_from is not None:
        # Nap checkpoint TRUOC vong lap, tiep tuc tu epoch+1.
        last_epoch, _ = load_checkpoint(resume_from, model, opt, sched,
                                        loader_gen)
        start_epoch = last_epoch + 1

    # Resume thi GHI TIEP log (append), run moi thi ghi de (write).
    log_mode = "a" if resume_from is not None else "w"
    losses = []
    with open(log_path, log_mode) as logf:
        for epoch in range(start_epoch, epochs + 1):
            model.train()
            total, n = 0.0, 0
            for xb, yb in loader:
                opt.zero_grad()  # quen dong nay -> grad cong don -> NaN
                loss = criterion(model(xb), yb)
                loss.backward()
                opt.step()
                total += loss.item() * len(xb)
                n += len(xb)
            train_loss = total / n
            model.eval()
            with torch.no_grad():
                val_loss = criterion(model(val_x), val_y).item()
            sched.step()
            lr = opt.param_groups[0]["lr"]
            losses.append((epoch, train_loss, val_loss))
            logf.write(json.dumps({"epoch": epoch, "train_loss": train_loss,
                                   "val_loss": val_loss, "lr": lr}) + "\n")
            logf.flush()
            ckpt_path = os.path.join(ckpt_dir, f"ckpt_epoch{epoch}.pt")
            save_checkpoint(ckpt_path, epoch, model, opt, sched,
                            loader_gen, seed)

    return train_loss, val_loss, state_dict_hash(model), losses


# ---------------------------------------------------------------------------
# Che do worker: 1 process train 1 run (duoc goi boi thi nghiem)
# ---------------------------------------------------------------------------
def run_worker(args):
    """Worker: train xong in RESULT de tien trinh cha doc."""
    tl, vl, h, _ = train_one_run(args.seed, args.epochs, args.log,
                                 args.ckpt_dir, args.resume_from)
    # In theo format co dinh de cha parse (khong in gi khac ra stdout).
    print(f"RESULT hash={h} train_loss={tl:.10f} val_loss={vl:.10f}",
          flush=True)


def launch_worker(seed, epochs, log, ckpt_dir, tag, resume_from=None):
    """Chay 1 worker tren process RIENG, tra ve dict ket qua."""
    cmd = [sys.executable, os.path.abspath(__file__), "--worker",
           "--seed", str(seed), "--epochs", str(epochs),
           "--log", log, "--ckpt-dir", ckpt_dir, "--tag", tag]
    if resume_from:
        cmd += ["--resume-from", resume_from]
    # stderr giu lai de thay loi worker (neu co); stdout chi co RESULT.
    proc = subprocess.run(cmd, capture_output=True, text=True)
    if proc.returncode != 0:
        raise RuntimeError(f"worker {tag} loi:\n{proc.stderr[-2000:]}")
    line = [l for l in proc.stdout.splitlines() if l.startswith("RESULT")]
    if not line:
        raise RuntimeError(
            f"worker {tag} khong in RESULT:\n{proc.stderr[-2000:]}")
    parts = dict(p.split("=") for p in line[0].split()[1:])
    return parts


# ---------------------------------------------------------------------------
# Buoc 4 - Thi nghiem 1: chung minh TAI LAP (2 process doc lap)
# ---------------------------------------------------------------------------
def run_experiment_1():
    """Train 2 lan cung seed tren 2 process rieng -> hash + loss y het."""
    print("=" * 64)
    print("Thi nghiem 1 - TAI LAP: 2 process doc lap, cung seed=42")
    print("=" * 64)
    r1 = launch_worker(SEED, EPOCHS, "run_exp1a.jsonl", "ckpts_exp1a", "exp1a")
    r2 = launch_worker(SEED, EPOCHS, "run_exp1b.jsonl", "ckpts_exp1b", "exp1b")
    print(f"  Run 1: hash={r1['hash'][:16]}... train_loss={r1['train_loss']}")
    print(f"  Run 2: hash={r2['hash'][:16]}... train_loss={r2['train_loss']}")
    ok = (r1["hash"] == r2["hash"]
          and r1["train_loss"] == r2["train_loss"]
          and r1["val_loss"] == r2["val_loss"])
    print("  => PASS: tai lap tuyet doi (hash + loss y het tung bit)"
          if ok else "  => FAIL: con cho ro ri randomness!")
    print("=" * 64)
    return ok


# ---------------------------------------------------------------------------
# Buoc 5 - Thi nghiem 2: mo phong CRASH va resume
# ---------------------------------------------------------------------------
def run_experiment_2():
    """Run A (6 epoch lien tuc) == Run B (3 epoch -> crash -> resume 3).

    "Crash" that: process worker ket thuc han (model mat khoi RAM), roi
    process moi nap checkpoint va train tiep.
    """
    print("=" * 64)
    print("Thi nghiem 2 - RESUME: crash o epoch 3 roi train tiep")
    print("=" * 64)
    # Run A: 6 epoch lien tuc.
    ra = launch_worker(SEED, EPOCHS, "run_A.jsonl", "ckpts_A", "exp2a")
    print(f"  Run A (lien tuc 6 epoch): hash={ra['hash'][:16]}...")
    # Run B pha 1: train 3 epoch roi "sap nguon" (process ket thuc).
    launch_worker(SEED, 3, "run_B.jsonl", "ckpts_B", "exp2b_p1")
    print("  Run B pha 1: train 3 epoch -> SAP NGUON (process ket thuc)")
    # Run B pha 2: process MOI, model MOI, nap checkpoint, train tiep.
    rb = launch_worker(SEED, EPOCHS, "run_B.jsonl", "ckpts_B", "exp2b_p2",
                       resume_from=os.path.join("ckpts_B", "ckpt_epoch3.pt"))
    print(f"  Run B pha 2: resume tu ckpt_epoch3 -> hash={rb['hash'][:16]}...")

    h_ok = ra["hash"] == rb["hash"]
    # So sanh loss tung epoch 4,5,6 (doc tu 2 file log JSONL).
    def read_log(path):
        return {r["epoch"]: r for r in
                map(json.loads, open(path, encoding="utf-8"))}
    la, lb = read_log("run_A.jsonl"), read_log("run_B.jsonl")
    loss_ok = all(la[e]["train_loss"] == lb[e]["train_loss"]
                  and la[e]["val_loss"] == lb[e]["val_loss"]
                  for e in (4, 5, 6))
    print("  Loss epoch 4,5,6 giong het nhau:", loss_ok)
    ok = h_ok and loss_ok
    print("  => PASS: H_A == H_B - resume khong phan biet duoc voi "
          "chay lien tuc" if ok else "  => FAIL: checkpoint thieu trang thai!")
    print("=" * 64)
    return ok


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    """Chay tuan tu thi nghiem 1 -> thi nghiem 2, bao cao PASS/FAIL."""
    ok1 = run_experiment_1()
    print()
    ok2 = run_experiment_2()
    print()
    print("=" * 64)
    if ok1 and ok2:
        print("XONG - Ca 2 thi nghiem PASS: tai lap + resume dat.")
    else:
        print("CHUA DAT - xem FAIL o tren de tim cho ro ri.")
    print("=" * 64)
    # Ghi chu ve checkpoint muc epoch vs step (checklist muc 5):
    # O day "crash" dung bien epoch nen checkpoint epoch la du. Voi LLM
    # that, crash co the roi vao GIUA epoch (da train 70% batch): muon
    # resume chinh xac phai checkpoint muc STEP - luu ca vi tri dang
    # dung trong DataLoader (so batch da thay) chu khong chi epoch.
    return ok1 and ok2


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--worker", action="store_true")
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument("--epochs", type=int, default=EPOCHS)
    parser.add_argument("--log", default="run.log")
    parser.add_argument("--ckpt-dir", default="ckpts")
    parser.add_argument("--resume-from", default=None)
    parser.add_argument("--tag", default="w")
    args = parser.parse_args()
    if args.worker:
        run_worker(args)
    else:
        sys.exit(0 if main() else 1)
