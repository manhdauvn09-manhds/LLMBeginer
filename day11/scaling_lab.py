#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Scaling Lab - tu 9 run nho den quyet dinh ngan sach (Day 11).

Phan A (lam ngay): dung bang 9 run mau de fit duong log-log
loss = a * C^b, du bao loss cho run 10 TRUOC KHI train that, roi chon
run toi uu theo tieu chi cai thien loss/kWh (Green checkpoint).

Phan B (train that, co the de qua dem): chay lai 9 run bang train_grid.py
roi thay so that vao bang `runs` duoi day.

Cach chay:
    python scaling_lab.py

Chi can numpy.
"""

import math
import os

import numpy as np

# ---------------------------------------------------------------------------
# Bang 9 run MAU (tro giang chay truoc tren Kaggle, tiny char-GPT).
# NGUON SO LIEU: mau - thay bang so that cua ban sau khi chay Phan B.
# (name, N params, D tokens, loss cuoi, kWh)
# ---------------------------------------------------------------------------
runs = [
    ("S1", 4e5, 5e4, 3.02, 0.0054), ("S2", 4e5, 1.5e5, 2.87, 0.0162),
    ("S3", 4e5, 4e5, 2.73, 0.0432),
    ("M1", 1.2e6, 5e4, 2.90, 0.0162), ("M2", 1.2e6, 1.5e5, 2.74, 0.0486),
    ("M3", 1.2e6, 4e5, 2.60, 0.1296),
    ("L1", 3e6, 5e4, 2.76, 0.0405), ("L2", 3e6, 1.5e5, 2.62, 0.1215),
    ("L3", 3e6, 4e5, 2.50, 0.3240),
]

# Run 10 (du bao): gap doi compute cua M3 (run re nhat-dat-loss-thap).
# M3: C = 6*1.2e6*4e5 = 2.88e12 -> run 10: C = 5.76e12 (M x 800k token).
RUN10_N, RUN10_D = 1.2e6, 8e5


def compute_of(n_params, n_tokens):
    """Compute C = 6*N*D (uoc luong FLOPs train cua Transformer)."""
    return 6 * n_params * n_tokens


# ---------------------------------------------------------------------------
# TODO 1 - fit_scaling_law: fit loss = a * C^b tren thang log-log
# ---------------------------------------------------------------------------
def fit_scaling_law(runs):
    """Fit power law loss = a * C^b bang hoi quy tuyen tinh tren log10.

    Vi sao fit tren log: loss = a*C^b <=> log(loss) = log(a) + b*log(C)
    la duong thang; C trai nhieu bac 10^11-10^12 nen fit truc tiep se
    mat on dinh so hoc. He so b la "do doc cai thien": C gap 10 lan thi
    loss nhan voi 10^b.

    Returns:
        Tuple (a, b).
    """
    xs = np.log10([compute_of(n, d) for _, n, d, _, _ in runs])
    ys = np.log10([loss for _, _, _, loss, _ in runs])
    b, loga = np.polyfit(xs, ys, 1)  # ys = b*xs + loga
    return 10 ** loga, b


# ---------------------------------------------------------------------------
# TODO 2 - predict_loss: du bao loss cho run 10 (TRUOC KHI train that)
# ---------------------------------------------------------------------------
def predict_loss(a, b, n_params, n_tokens):
    """Du bao loss = a * C^b voi C = 6*N*D cua cau hinh moi."""
    return a * compute_of(n_params, n_tokens) ** b


# ---------------------------------------------------------------------------
# TODO 3 - green checkpoint: chon run theo cai thien loss/kWh
# ---------------------------------------------------------------------------
def best_run_per_kwh(runs):
    """Chon run co diem hieu qua cao nhat.

    Diem = (loss_run_truoc - loss_run) / kWh_run, voi cac run xep theo C
    tang dan. Y nghia: moi kWh bo them ra mua duoc bao nhieu diem loss.
    Run re nhat lam moc (khong co "run truoc" de so).

    Returns:
        Ten run tot nhat.
    """
    ordered = sorted(runs, key=lambda r: compute_of(r[1], r[2]))
    best, best_score = None, float("-inf")
    prev_loss = None
    for name, n, d, loss, kwh in ordered:
        if prev_loss is not None:
            score = (prev_loss - loss) / kwh
            print(f"  {name}: C={compute_of(n, d):.2e}, loss={loss:.2f}, "
                  f"kWh={kwh:.4f} -> diem={score:.2f} loss/kWh")
            if score > best_score:
                best, best_score = name, score
        else:
            print(f"  {name}: C={compute_of(n, d):.2e}, loss={loss:.2f}, "
                  f"kWh={kwh:.4f} -> moc so sanh")
        prev_loss = loss
    return best


# ---------------------------------------------------------------------------
# Phan tich bo sung kieu engineer (ngoai 3 TODO)
# ---------------------------------------------------------------------------
def report_same_compute(runs):
    """So 2 run cung compute C.

    Tien do vao data hay vao params: voi cung C, cach chia N/D nao cho
    loss thap hon?
    """
    by_c = {}
    for name, n, d, loss, kwh in runs:
        by_c.setdefault(compute_of(n, d), []).append((name, n, d, loss))
    print("\nCung compute, khac cach chia N/D:")
    for c in sorted(by_c):
        if len(by_c[c]) > 1:
            for name, n, d, loss in sorted(by_c[c]):
                print(f"  C={c:.2e}: {name} (N={n:.0e}, D={d:.0e}) "
                      f"-> loss {loss:.2f}")


def load_real_runs(path="runs.jsonl"):
    """Doc bang 9 run THAT tu train_grid.py (Phan B).

    [TRAINER] Docstring cu ghi "tra ve (runs, 'THAT') hoac
    (None, 'MAU')" nhung code thuc te chi tra ve list runs hoac None -
    docstring sai khien nguoi doc tuong ham tra tuple. Da sua
    docstring khop hanh vi that cua code.
    Tra ve list runs, hoac None neu chua co file runs.jsonl.
    """
    import json
    if not os.path.exists(path):
        return None
    recs = [json.loads(l) for l in open(path, encoding="utf-8")]
    runs = [(r["name"], r["N"], r["D"], r["loss"], r["kwh"]) for r in recs]
    return runs


def load_run10_actual(path="runs_run10.jsonl"):
    """Doc loss THAT cua run 10 (de so prediction vs actual).

    [TRAINER] Ban cu dung json.loads() tren TOAN BO noi dung file ->
    crash JSONDecodeError "Extra data" khi file co >1 dong. train_grid
    --run10 ghi che do APPEND nen chay lai run 10 lan 2 la file da co
    2 dong (da tai hien loi). Sua: chi doc dong khong rong CUOI CUNG
    (run moi nhat la run can so sanh).
    """
    import json
    if not os.path.exists(path):
        return None
    lines = [l for l in open(path, encoding="utf-8") if l.strip()]
    return json.loads(lines[-1]) if lines else None


def main():
    """Chay Phan A (so mau) + Phan B (so that, neu co)."""
    print("=" * 64)
    print("Phan A - Fit scaling law tu 9 run mau")
    print("=" * 64)
    a, b = fit_scaling_law(runs)
    print(f"loss = {a:.3f} * C^({b:.3f})")
    print(f"(C tang gap 10 -> loss nhan voi 10^{b:.3f} = {10 ** b:.3f})")

    print("\nDu bao run 10 (TRUOC KHI train that):")
    pred = predict_loss(a, b, RUN10_N, RUN10_D)  # run 10: M x 800k token
    c10 = compute_of(RUN10_N, RUN10_D)
    print(f"  Run 10: N={RUN10_N:.1e}, D={RUN10_D:.1e}, C={c10:.2e}")
    print(f"  Du bao loss run 10: {pred:.3f}  <- train that roi so sanh!")

    report_same_compute(runs)

    print("\nGreen checkpoint - diem hieu qua loss/kWh (xep theo C tang):")
    best = best_run_per_kwh(runs)
    print(f"\nRun tot nhat theo loss/kWh: {best}")

    # ---- Phan B: fit lai bang so that ----
    real = load_real_runs()
    if real:
        print("\n" + "=" * 64)
        print("Phan B - Fit lai bang SO THAT (train_grid.py, 9 run)")
        print("=" * 64)
        a2, b2 = fit_scaling_law(real)
        print(f"loss = {a2:.3f} * C^({b2:.3f})  (so that)")
        # Run 10 that: lay N, D that tu file.
        r10 = load_run10_actual()
        if r10:
            pred2 = predict_loss(a2, b2, r10["N"], r10["D"])
            actual = r10["loss"]
            err = abs(pred2 - actual) / actual * 100
            print(f"\nRun 10 that: N={r10['N']:,}, D={r10['D']:,}, "
                  f"C={compute_of(r10['N'], r10['D']):.2e}")
            print(f"  Du bao (fit so that): {pred2:.4f}")
            print(f"  Thuc te train:        {actual:.4f}")
            print(f"  Do lech tuong doi: {err:.1f}%")
        print("\nGreen checkpoint (so that):")
        best2 = best_run_per_kwh(real)
        print(f"\nRun tot nhat theo loss/kWh (so that): {best2}")
    else:
        print("\n(Phan B: chua co runs.jsonl - chay 'python train_grid.py')")
    print("=" * 64)


if __name__ == "__main__":
    main()
