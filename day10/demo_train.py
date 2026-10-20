#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Demo training loop de cam HealthMonitor + 3 thi nghiem pha run.

De bai ghi file nay "da viet san" nhung thuc te chua co trong repo nen
viet lai day du theo dac ta: make_model, train_step, run_experiment,
3 ham fault_* va smoke_test o __main__.

3 "benh" duoc cai san (moi thi nghiem dung 1 benh):
    TN1 fault_duplicate: dataloader "ket", steps 5-9 tra dung 1 batch.
    TN2 fault_lr_spike: step 5 learning rate nhay 3e-4 -> 2.0.
    TN3 fault_bad_token: step 5 batch chua token id = VOCAB (ngoai range).

Cach chay:
    python demo_train.py
"""

import torch
import torch.nn as nn

from health_checks import HealthAlert, HealthMonitor, grad_norm, smoke_test

VOCAB, D, H, SEQLEN = 64, 16, 32, 8
loss_fn = nn.CrossEntropyLoss()


def make_model(seed=42):
    """Tao model + optimizer MOI TINH (moi thi nghiem bat dau khoe manh)."""
    torch.manual_seed(seed)
    model = nn.Sequential(                 # LM ti hon: embed -> MLP
        nn.Embedding(VOCAB, D),
        nn.Flatten(1),
        nn.Linear(D * SEQLEN, H), nn.ReLU(),
        nn.Linear(H, VOCAB))
    opt = torch.optim.AdamW(model.parameters(), lr=3e-4)
    return model, opt


def train_step(model, opt, batch, targets):
    """1 buoc train: bat mach (grad norm) TRUOC khi update."""
    opt.zero_grad()
    loss = loss_fn(model(batch), targets)
    loss.backward()
    gn = grad_norm(model)                  # bat mach truoc khi update
    opt.step()
    return loss.item(), gn


def run_experiment(name, fault_fn, steps=30):
    """Chay 1 thi nghiem, ghi lai canh bao DAU TIEN."""
    model, opt = make_model()              # moi thi nghiem: model khoe
    mon = HealthMonitor(vocab_size=VOCAB)
    print(f"\n===== {name} =====")
    first_alert = None
    for step in range(steps):
        batch, targets = fault_fn(step, opt)  # 3 ham fault cai "benh"
        try:
            alerts = mon.check_batch(batch, step)      # (1) du lieu TRUOC
            loss, gn = train_step(model, opt, batch, targets)
            alerts += mon.check_step(loss, model, step)  # (2) SAU backward
        except HealthAlert as e:           # loi nghiem trong -> dung ngay
            print(f"  [step {step}] DUNG RUN: {e}")
            first_alert = first_alert or f"HealthAlert: {e}"
            break
        for a in alerts:
            print(f"  [step {step}] Canh bao: {a}")
            first_alert = first_alert or a
        if step % 10 == 0:
            print(f"  [step {step}] loss={loss:.4f} grad_norm={gn:.3f}")
    print(f"  => Canh bao dau tien: {first_alert}")


# ---------------------------------------------------------------------------
# 3 "benh" da cai san
# ---------------------------------------------------------------------------
def _rand_batch():
    """1 batch ngau nhien khoe manh: (32, 8) token id + (32,) target."""
    return (torch.randint(0, VOCAB, (32, SEQLEN)),
            torch.randint(0, VOCAB, (32,)))


_DUP = None


def fault_duplicate(step, opt):
    """Dataloader 'ket': steps 5-9 tra DUNG mot batch lap lai."""
    global _DUP
    if _DUP is None:
        _DUP = _rand_batch()
    return _DUP if 5 <= step < 10 else _rand_batch()


def fault_lr_spike(step, opt):
    """Bug lich trinh LR: step 5 learning rate nhay 3e-4 -> 2.0."""
    for g in opt.param_groups:
        g['lr'] = 2.0 if step == 5 else 3e-4
    return _rand_batch()


def fault_bad_token(step, opt):
    """Mot batch chua token id = VOCAB (ngoai pham vi tu dien)."""
    b, t = _rand_batch()
    if step == 5:
        b[0, 0] = VOCAB
    return b, t


def healthy_train_fn(step):
    """Ham sinh batch khoe manh cho smoke_test."""
    return _rand_batch()


if __name__ == "__main__":
    run_experiment("TN1: batch trung lap", fault_duplicate)
    run_experiment("TN2: learning rate bung no", fault_lr_spike)
    run_experiment("TN3: token ngoai pham vi", fault_bad_token)

    print("\n===== smoke_test(50) =====")
    res = smoke_test(healthy_train_fn, VOCAB, steps=50)
    print(f"  ok={res['ok']}, loss {res['loss_start']:.4f} -> "
          f"{res['loss_end']:.4f}, max_grad_norm={res['max_grad_norm']:.3f}")
    if res["alerts"]:
        print(f"  alerts: {res['alerts']}")
    print("  => smoke_test " + ("PASS" if res["ok"] else "FAIL"))
