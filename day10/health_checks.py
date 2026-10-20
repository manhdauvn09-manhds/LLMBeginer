#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Health checks - "bac si truc" cho training loop (Day 10 - buoi toi).

Cac thanh phan:
    HealthAlert: exception nem ra khi loi nghiem trong -> dung run ngay.
    check_finite: bat NaN/Inf o loss, trong so, gradient.
    SpikeDetector: bat loss spike; qua nhieu spike lien tiep -> kill run.
    BatchDedup: bat dataloader tra batch trung lap.
    check_token_range: token id phai nam trong [0, vocab_size).
    grad_norm: chuan gradient toan cuc (da viet san theo de bai).
    smoke_test: chay thu 50 buoc train that truoc khi chay full run.
    HealthMonitor: lap rap theo checklist - du lieu truoc, loss/grad sau.

Cach dung (xem demo_train.py):
    mon = HealthMonitor(vocab_size=64)
    alerts = mon.check_batch(batch, step)       # TRUOC forward
    loss, gn = train_step(...)
    alerts += mon.check_step(loss, model, step)  # SAU backward

Chi can PyTorch (CPU la du).
"""

import hashlib
import math

import torch
import torch.nn.functional as F


class HealthAlert(Exception):
    """Loi nghiem trong: dung run ngay, dung "hy vong no tu het"."""


# ---------------------------------------------------------------------------
# TODO 1 - check_finite: bat NaN/Inf o 3 noi
# ---------------------------------------------------------------------------
def check_finite(loss, model, step):
    """Kiem tra NaN/Inf o loss, parameter va gradient.

    (a) loss: loss la float (loss.item()) -> goi vao tensor de kiem tra.
    (b) moi parameter p.data: trong so NaN la dau hieu run da hong.
    (c) moi gradient p.grad (neu khac None): grad NaN se lan sang trong
        so o buoc opt.step() ke tiep.
    Dung .detach() de khong dinh computation graph. Ghi ten param vao
    canh bao de biet "benh" o lop nao.

    Returns:
        List[str] cac canh bao (rong neu khong co gi).
    """
    warnings = []
    if not torch.isfinite(torch.tensor(loss)).all():
        warnings.append(f"[step {step}] loss khong huu han: {loss}")
    for name, p in model.named_parameters():
        if not torch.isfinite(p.data.detach()).all():
            warnings.append(f"[step {step}] trong so NaN/Inf tai '{name}'")
        if p.grad is not None and not torch.isfinite(p.grad.detach()).all():
            warnings.append(f"[step {step}] gradient NaN/Inf tai '{name}'")
    return warnings


# ---------------------------------------------------------------------------
# TODO 2 - SpikeDetector: bat loss spike va dem lien tiep
# ---------------------------------------------------------------------------
class SpikeDetector:
    """Phat hien loss spike: loss_t / loss_{t-1} > ratio => 1 spike.

    Spike don le thuong tu hoi phuc (dung giet run voi). Nhung qua
    max_consecutive spike LIEN TIEP => run dang lao doc -> raise
    HealthAlert de kill run (Green checkpoint: dung dot GPU vo ich).
    """

    def __init__(self, ratio=5.0, max_consecutive=3):
        self.ratio = ratio
        self.max_consecutive = max_consecutive
        self.prev = None
        self.streak = 0

    def step(self, loss, step):
        """Cap nhat voi loss moi. Tra ve list[str] canh bao."""
        warnings = []
        # Chi tinh ti le khi prev > 0: tranh chia cho 0 va loss am
        # (CrossEntropy khong am nhung code nen chac chan).
        if self.prev is not None and self.prev > 0:
            r = loss / self.prev
            if r > self.ratio:
                self.streak += 1
                warnings.append(
                    f"[step {step}] loss spike x{r:.1f} "
                    f"(streak {self.streak}/{self.max_consecutive})")
            else:
                self.streak = 0
        if self.streak >= self.max_consecutive:
            raise HealthAlert(
                f"{self.max_consecutive} loss spike lien tiep "
                f"(lan cuoi tai step {step}) - kill run")
        self.prev = loss
        return warnings


# ---------------------------------------------------------------------------
# TODO 3 - BatchDedup: bat batch trung lap
# ---------------------------------------------------------------------------
class BatchDedup:
    """Bat dataloader tra batch trung lap.

    Hash noi dung batch; gap lai hash trong `window` buoc gan nhat =>
    dataloader dang tra batch trung (bug kinh dien: dataloader "ket").
    Mot hash duy nhat cho ca batch vua re vua du de bat benh.
    """

    def __init__(self, window=50):
        self.window = window
        self.seen = {}  # hash -> step thay lan cuoi

    def step(self, batch, step):
        """Kiem tra batch hien tai. Tra ve list[str] canh bao."""
        warnings = []
        # bytes truc tiep tu storage (khong can numpy).
        h = hashlib.sha256(
            bytes(batch.detach().cpu().contiguous().untyped_storage())
        ).hexdigest()
        if h in self.seen and step - self.seen[h] <= self.window:
            warnings.append(
                f"[step {step}] batch trung lap (da thay o step "
                f"{self.seen[h]}) - kiem tra dataloader/shuffle")
        self.seen[h] = step
        # Don entry qua cu de dict khong phinh vo han.
        self.seen = {hh: s for hh, s in self.seen.items()
                     if step - s <= self.window}
        return warnings


# ---------------------------------------------------------------------------
# TODO 4 - check_token_range + grad_norm + smoke_test
# ---------------------------------------------------------------------------
def check_token_range(batch, vocab_size, step):
    """Token id phai nam trong [0, vocab_size). Tra ve list[str].

    Day la LOI CHI MANG: nn.Embedding se crash voi IndexError neu token
    vuot range - phai fail fast TRUOC khi forward.
    """
    bad = (batch < 0) | (batch >= vocab_size)
    if bad.any():
        return [f"[step {step}] token ngoai range [0, {vocab_size}): "
                f"min={batch.min().item()}, max={batch.max().item()}"]
    return []


def grad_norm(model):
    """Chuan gradient toan cuc (da viet san theo de bai).

    Cong thuc: sqrt(sum(||g||^2)) tren moi parameter co grad.
    """
    total = 0.0
    for p in model.parameters():
        if p.grad is not None:
            total += p.grad.detach().pow(2).sum().item()
    return total ** 0.5


class _SmokeModel(torch.nn.Module):
    """Model nho dung RIENG cho smoke_test (khong phu thuoc demo_train).

    Nhan batch (B, T) token id -> logits (B, vocab_size). Kich thuoc
    chuoi T duoc suy ra tu batch dau tien.
    """

    def __init__(self, vocab_size, seqlen):
        super().__init__()
        self.emb = torch.nn.Embedding(vocab_size, 16)
        self.net = torch.nn.Sequential(
            torch.nn.Flatten(1),
            torch.nn.Linear(16 * seqlen, 32),
            torch.nn.ReLU(),
            torch.nn.Linear(32, vocab_size))

    def forward(self, x):
        return self.net(self.emb(x))


def smoke_test(train_fn, vocab_size, steps=50):
    """Green checkpoint: chay `steps` buoc train THAT truoc khi chay dai.

    Moi buoc goi HealthMonitor day du. Tao model + optimizer MOI TINH
    trong ham (dung tai dung model da "benh").

    PASS khi: loss_end < loss_start, khong NaN, grad norm khong vot qua
    10x so voi trung vi 10 buoc dau, khong HealthAlert/canh bao.

    Args:
        train_fn: ham(step) -> (batch, targets).
        vocab_size: kich thuoc tu dien.
        steps: so buoc chay thu.

    Returns:
        Dict {'ok', 'loss_start', 'loss_end', 'max_grad_norm', 'alerts'}.
    """
    torch.manual_seed(1234)  # smoke test cung phai tai lap duoc
    model, opt, mon = None, None, HealthMonitor(vocab_size=vocab_size)
    gns, losses, alerts = [], [], []
    try:
        for step in range(steps):
            batch, targets = train_fn(step)
            if model is None:  # suy kich thuoc chuoi tu batch dau tien
                model = _SmokeModel(vocab_size, batch.shape[1])
                opt = torch.optim.AdamW(model.parameters(), lr=3e-4)
            alerts += mon.check_batch(batch, step)  # du lieu TRUOC forward
            opt.zero_grad()
            loss = F.cross_entropy(model(batch), targets)
            loss.backward()
            gns.append(grad_norm(model))
            alerts += mon.check_step(loss.item(), model, step)  # SAU backward
            opt.step()
            losses.append(loss.item())
    except HealthAlert as e:
        alerts.append(f"HealthAlert: {e}")
        return {"ok": False, "loss_start": None, "loss_end": None,
                "max_grad_norm": None, "alerts": alerts}

    loss_start, loss_end = losses[0], losses[-1]
    max_gn = max(gns)
    median = sorted(gns[:10])[len(gns[:10]) // 2]
    problems = []
    if not loss_end < loss_start:
        problems.append(
            f"loss khong giam ({loss_start:.4f} -> {loss_end:.4f})")
    if not all(math.isfinite(x) for x in losses):
        problems.append("loss co NaN/Inf")
    if not max_gn < 10 * median:
        problems.append(f"grad norm vot ({max_gn:.2f} >= 10x median "
                        f"{median:.2f})")
    if alerts:
        problems.append(f"co {len(alerts)} canh bao monitor")
    return {"ok": not problems, "loss_start": loss_start,
            "loss_end": loss_end, "max_grad_norm": max_gn,
            "alerts": alerts + problems}


# ---------------------------------------------------------------------------
# Lap rap - HealthMonitor (da viet san theo de bai, doc hieu)
# ---------------------------------------------------------------------------
class HealthMonitor:
    """Cam vao training loop. Thu tu = checklist chan doan.

    check_batch (du lieu) chay TRUOC forward - 90% van de nam o du lieu.
    check_step (loss/gradient) chay SAU backward, TRUOC opt.step() - de
    biet NaN xuat hien o khau nao (Bay 3).
    """

    def __init__(self, vocab_size, ratio=5.0, max_consecutive=3):
        self.spike = SpikeDetector(ratio, max_consecutive)
        self.dedup = BatchDedup()
        self.vocab_size = vocab_size

    def check_batch(self, batch, step):
        """Kiem tra du lieu TRUOC forward.

        Token range la LOI CHI MANG (forward chac chan crash) -> fail fast
        bang HealthAlert. Batch trung chi canh bao (train van chay duoc).
        """
        fatal = check_token_range(batch, self.vocab_size, step)
        if fatal:
            raise HealthAlert("; ".join(fatal) + " - dung truoc khi "
                                        "forward crash")
        return self.dedup.step(batch, step)

    def check_step(self, loss, model, step):
        """Kiem tra loss/gradient SAU backward.

        SpikeDetector co the raise HealthAlert -> caller dung run.
        """
        return check_finite(loss, model, step) + self.spike.step(loss, step)
