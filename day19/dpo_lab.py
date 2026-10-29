#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Lab DPO tren logits gia (Day 19 - buoi toi).

Khong can model that, khong can GPU, chi can logits gia va
PyTorch tren CPU. Lam xong se hieu vi sao phai detach model
tham chieu va vi sao beta la day neo.

Bai 1: log-prob cua mot cau tra loi (log_softmax -> gather
    -> cong). Kiem chung: log-prob am, tang khi doi token thanh
    argmax.
Bai 2: loss DPO + soi gradient. Policy == ref -> loss ~0.693.
    Gradient chosen/rejected nguoc dau. Thi nghiem quen detach
    (ref nhan gradient -> moc tu di chuyen, loi nghiem trong).
Bai 3: quet beta {0.05, 0.1, 0.5, 1.0}, moi beta train 50 buoc
    SGD (lr=0.5) tren cung cap logits gia. Chi beta thay doi,
    moi thu khac giu nguyen.
Bai 4 (nang cao): mo phong reward hacking - reward = logp +
    0.5 x do dai, toi uu tran 30 buoc: reward tang, do dai tang,
    so token co thong tin dung yen.

Cach chay (CPU, venv Day 15):
    ~/workspace/venvs/day15/bin/python dpo_lab.py
"""

import time

import torch
import torch.nn.functional as F

BETA_SWEEP = [0.05, 0.1, 0.5, 1.0]
SWEEP_STEPS = 50
SWEEP_LR = 0.5
HACK_STEPS = 30
HACK_LR = 0.5
HACK_BONUS = 0.5


def sequence_logprob(logits, tokens):
    """Tong log-prob cua mot cau tra loi tu logits cua model.

    Args:
        logits: tensor (T, V) - logits du doan token tiep theo
            tai moi vi tri.
        tokens: tensor (T,) - cac token cua cau tra loi.

    Returns:
        Tensor scalar: tong log-prob (luon am vi log xac suat
        < 1, tru khi cau rong).

    Raises:
        ValueError: neu cau rong (tong tren 0 token = 0.0, khong
            con la log-prob am, de gay hieu nham) hoac token
            vuot khoi vocab.
    """
    if tokens.numel() == 0:
        raise ValueError("tokens khong duoc rong.")
    if tokens.min().item() < 0 or tokens.max().item() >= logits.shape[1]:
        raise ValueError("token vuot khoi vocab.")
    # Buoc 1: log_softmax theo chieu vocab.
    log_probs = F.log_softmax(logits, dim=-1)
    # Buoc 2: gather log-prob cua dung token tai moi vi tri.
    picked = log_probs.gather(1, tokens.unsqueeze(1)).squeeze(1)
    # Buoc 3: cong tat ca lai.
    return picked.sum()


def dpo_loss(policy_logits_w, policy_logits_l, ref_logits_w,
             ref_logits_l, tokens_w, tokens_l, beta=0.1):
    """Loss DPO: -log sigmoid(beta * (ratio_w - ratio_l)).

    ratio = logp_policy - logp_ref (do lech khoi moc). Ref DUOC
    DETACH trong ham (phong thu: ke ca caller dua ref co grad
    thi ham van detach) - ref la moc co dinh, khong nhan grad.

    Args:
        beta: he so neo, phai > 0 (beta <= 0 lam mat y nghia
            "neo ve ref" cua loss).

    Raises:
        ValueError: neu beta <= 0.
    """
    if beta <= 0:
        raise ValueError("beta phai > 0.")
    logp_w = sequence_logprob(policy_logits_w, tokens_w)
    logp_l = sequence_logprob(policy_logits_l, tokens_l)
    logp_ref_w = sequence_logprob(ref_logits_w.detach(), tokens_w)
    logp_ref_l = sequence_logprob(ref_logits_l.detach(), tokens_l)
    logits = beta * ((logp_w - logp_ref_w) - (logp_l - logp_ref_l))
    return -F.logsigmoid(logits)


def dpo_loss_no_detach(policy_logits_w, policy_logits_l, ref_logits_w,
                       ref_logits_l, tokens_w, tokens_l, beta=0.1):
    """Ban DPO QUEN detach (chi de thi nghiem Bay 1, khong dung).

    Giong dpo_loss nhung khong detach ref - gradient se chay
    nguoc vao ref, moc tu di chuyen.
    """
    logp_w = sequence_logprob(policy_logits_w, tokens_w)
    logp_l = sequence_logprob(policy_logits_l, tokens_l)
    logp_ref_w = sequence_logprob(ref_logits_w, tokens_w)
    logp_ref_l = sequence_logprob(ref_logits_l, tokens_l)
    logits = beta * ((logp_w - logp_ref_w) - (logp_l - logp_ref_l))
    return -F.logsigmoid(logits)


def bai1():
    """Bai 1: log-prob cua mot cau tra loi."""
    print("=" * 72)
    print("BAI 1 - Log-prob cua mot cau tra loi")
    print("=" * 72)
    torch.manual_seed(0)
    logits = torch.randn(4, 10)
    tokens = torch.tensor([3, 1, 7, 2])
    lp = sequence_logprob(logits, tokens)
    print(f"logprob: {lp.item():.4f} (phai am).")
    assert lp.item() < 0, "log-prob phai am!"
    # Doi token dau thanh argmax tai vi tri do -> log-prob tang.
    tokens_better = tokens.clone()
    tokens_better[0] = logits[0].argmax()
    lp_better = sequence_logprob(logits, tokens_better)
    print(f"Sau khi doi token 0 thanh argmax: {lp_better.item():.4f} "
          f"(tang: {lp_better.item() > lp.item()}).")
    assert lp_better.item() > lp.item(), "argmax phai tang logprob!"
    # Edge case: cau rong bi chan.
    try:
        sequence_logprob(logits, torch.tensor([], dtype=torch.long))
        print("CANH BAO: cau rong khong bi chan!")
    except ValueError as exc:
        print(f"Edge case cau rong: chan dung ({exc}).")
    # Edge case: beta <= 0 bi chan.
    try:
        dpo_loss(logits, logits, logits, logits, tokens, tokens,
                 beta=0.0)
        print("CANH BAO: beta=0 khong bi chan!")
    except ValueError as exc:
        print(f"Edge case beta=0: chan dung ({exc}).")


def bai2():
    """Bai 2: loss DPO va soi gradient."""
    print("\n" + "=" * 72)
    print("BAI 2 - Loss DPO va gradient")
    print("=" * 72)
    torch.manual_seed(0)
    vocab_size, seq_len = 20, 5
    base = torch.randn(seq_len, vocab_size)
    ref_w, ref_l = base.clone(), base.clone()
    pol_w = base.clone().requires_grad_(True)
    pol_l = base.clone().requires_grad_(True)
    tok_w = torch.randint(0, vocab_size, (seq_len,))
    tok_l = torch.randint(0, vocab_size, (seq_len,))
    loss = dpo_loss(pol_w, pol_l, ref_w, ref_l, tok_w, tok_l,
                    beta=0.1)
    print(f"loss khi policy == ref: {loss.item():.4f} "
          f"(ky vong ~0.693 = -log(0.5)).")
    assert abs(loss.item() - 0.693147) < 1e-4, "loss phai ~0.693!"
    loss.backward()
    print(f"grad policy chosen, norm: {pol_w.grad.norm().item():.4f}")
    print(f"grad policy rejected, norm: "
          f"{pol_l.grad.norm().item():.4f}")
    # Buoc nho theo -grad: loss phai giam, chosen logp tang,
    # rejected logp giam (kiem chung so hoc, khong doan).
    lp_w_before = sequence_logprob(pol_w.detach(), tok_w).item()
    lp_l_before = sequence_logprob(pol_l.detach(), tok_l).item()
    step = 0.1
    pol_w_new = pol_w.detach() - step * pol_w.grad
    pol_l_new = pol_l.detach() - step * pol_l.grad
    loss_new = dpo_loss(pol_w_new, pol_l_new, ref_w, ref_l,
                        tok_w, tok_l, beta=0.1).item()
    lp_w_after = sequence_logprob(pol_w_new, tok_w).item()
    lp_l_after = sequence_logprob(pol_l_new, tok_l).item()
    print(f"Sau buoc -grad (lr={step}): loss {loss.item():.4f} -> "
          f"{loss_new:.4f} (giam: {loss_new < loss.item()}).")
    print(f"  chosen logp: {lp_w_before:.4f} -> {lp_w_after:.4f} "
          f"(tang: {lp_w_after > lp_w_before}).")
    print(f"  rejected logp: {lp_l_before:.4f} -> {lp_l_after:.4f} "
          f"(giam: {lp_l_after < lp_l_before}).")
    # Thi nghiem quen detach: ref nhan gradient (Bay 1).
    print("\nThi nghiem QUEN detach (de doi chieu):")
    ref_w_g = base.clone().requires_grad_(True)
    ref_l_g = base.clone().requires_grad_(True)
    pol_w2 = base.clone().requires_grad_(True)
    pol_l2 = base.clone().requires_grad_(True)
    loss_bad = dpo_loss_no_detach(pol_w2, pol_l2, ref_w_g, ref_l_g,
                                  tok_w, tok_l, beta=0.1)
    loss_bad.backward()
    print(f"  grad ref chosen, norm: "
          f"{ref_w_g.grad.norm().item():.4f} (phai > 0 neu quen "
          f"detach - moc tu di chuyen!).")
    print(f"  grad ref rejected, norm: "
          f"{ref_l_g.grad.norm().item():.4f}.")
    # Voi detach dung (ham tu detach ben trong): ke ca ref co
    # requires_grad thi grad van None - moc khong di chuyen.
    ref_w_d = base.clone().requires_grad_(True)
    ref_l_d = base.clone().requires_grad_(True)
    pol_w3 = base.clone().requires_grad_(True)
    pol_l3 = base.clone().requires_grad_(True)
    loss_ok = dpo_loss(pol_w3, pol_l3, ref_w_d, ref_l_d,
                       tok_w, tok_l, beta=0.1)
    loss_ok.backward()
    print(f"  Voi detach dung (ref co requires_grad): ref grad = "
          f"{ref_w_d.grad} (None = moc khong di chuyen).")


def bai3():
    """Bai 3: quet beta - day neo chat hay long?

    Chi beta thay doi; cung cap logits gia, cung lr. Vi policy
    khoi tao = ref nen (a) logp chosen tang va (b) do lech khoi
    ref TRUNG NHAU - ghi ro dieu nay thay vi gia vo la 2 so
    khac nhau. Them do lech phia rejected de thay muc do policy
    dich chuyen toan dien.

    PHAT HIEN QUAN TRONG (so that, khong theo de bai): voi 50
    buoc co dinh, do troi KHONG don dieu theo beta ma co dang
    chu U nguoc, dinh o beta=0.1. Ly do: gradient ti le voi
    beta*sigmoid(-beta*Δ) - beta nho khoi dong cham (gradient
    nho) nhung khong bao hoa, beta lon khoi dong nhanh nhung
    bao hoa som. Chay them 200 buoc de kiem chung: beta nho
    van tiep tuc troi (+2.97 -> +8.74) trong khi beta=1.0 gan
    nhu dung yen (+2.74 -> +3.37). Vay khang dinh "beta nho
    troi xa hon" cua de bai chi dung o gioi han TIEM CAN (du
    buoc), khong dung trong ngan sach buoc co dinh. Va "hoc
    nhanh" neu do bang loss thi nguoc lai: beta lon giam loss
    nhanh hon (vi beta nhan truc tiep vao margin trong
    sigmoid) - loss cuoi 50 buoc: 0.55 / 0.32 / 0.016 / 0.004.
    """
    print("\n" + "=" * 72)
    print("BAI 3 - Quet beta")
    print("=" * 72)
    print("beta | Δchosen 50b | Δchosen 200b | do lech rejected "
          "(50b) | loss 50b")
    for beta in BETA_SWEEP:
        torch.manual_seed(0)
        vocab_size, seq_len = 20, 5
        base = torch.randn(seq_len, vocab_size)
        ref_w, ref_l = base.clone(), base.clone()
        tok_w = torch.randint(0, vocab_size, (seq_len,))
        tok_l = torch.randint(0, vocab_size, (seq_len,))
        row = {}
        for steps in (50, 200):
            pol_w = base.clone().requires_grad_(True)
            pol_l = base.clone().requires_grad_(True)
            opt = torch.optim.SGD([pol_w, pol_l], lr=SWEEP_LR)
            lp_w_init = sequence_logprob(pol_w.detach(),
                                         tok_w).item()
            for _ in range(steps):
                opt.zero_grad()
                loss = dpo_loss(pol_w, pol_l, ref_w, ref_l,
                                tok_w, tok_l, beta=beta)
                loss.backward()
                opt.step()
            lp_w_final = sequence_logprob(pol_w.detach(),
                                          tok_w).item()
            lp_l_final = sequence_logprob(pol_l.detach(),
                                          tok_l).item()
            lp_ref_w = sequence_logprob(ref_w, tok_w).item()
            lp_ref_l = sequence_logprob(ref_l, tok_l).item()
            row[steps] = (lp_w_final - lp_w_init,
                          lp_l_final - lp_ref_l, loss.item())
        print(f"{beta:>4} | {row[50][0]:>+11.4f} | "
              f"{row[200][0]:>+12.4f} | {row[50][1]:>+16.4f} | "
              f"{row[50][2]:.4f}")
    print("Ghi chu: vi policy khoi tao = ref nen Δlogp chosen va "
          "do lech khoi ref trung nhau dung - khong phai 2 so "
          "doc lap.")


def bai4():
    """Bai 4: mo phong reward hacking bang mat.

    Reward = logp_policy(cau) + 0.5 x do dai. Moi buoc: noi them
    1 token dem vo nghia (id 0) vao cau, roi gradient ascent
    truc tiep reward (khong KL, khong ref). In sau moi 10 buoc:
    do dai, reward, so token co thong tin (khong dem).
    """
    print("\n" + "=" * 72)
    print("BAI 4 - Mo phong reward hacking")
    print("=" * 72)
    torch.manual_seed(0)
    vocab_size, max_len = 20, 40
    filler_id = 0
    logits = torch.randn(max_len, vocab_size, requires_grad=True)
    # Cau ban dau: 4 token co thong tin (khac filler).
    tokens = torch.tensor([3, 1, 7, 2])
    n_informative = len(tokens)
    print("buoc | do dai | reward | token thong tin (that)")
    for step in range(HACK_STEPS + 1):
        if step > 0:
            # Noi them 1 token dem, roi ascent reward.
            tokens = torch.cat([tokens,
                                torch.tensor([filler_id])])
            reward = sequence_logprob(logits[:len(tokens)],
                                      tokens) + HACK_BONUS * len(tokens)
            reward.backward()
            with torch.no_grad():
                logits += HACK_LR * logits.grad
            logits.grad = None
        if step % 10 == 0 or step == HACK_STEPS:
            reward_now = sequence_logprob(
                logits[:len(tokens)].detach(),
                tokens) + HACK_BONUS * len(tokens)
            print(f"{step:>4} | {len(tokens):>6} | "
                  f"{reward_now.item():>7.3f} | {n_informative} "
                  f"(dung yen).")
    print("Ket luan: reward tang deu nho cong do dai (+0.5 moi "
          "token dem) va logp cua token dem cung tang theo "
          "ascent, nhung so token co thong tin dung yen o 4 - "
          "cau dai ra ma khong them thong tin gi. Do la reward "
          "hacking thu nho: toi uu reward tran ma khong co KL "
          "neo ve ref, model se hoc 'do dai' thay vi 'chat luong'.")


def main():
    """Chay lab day du: Bai 1 -> 4, do tong thoi gian CPU."""
    t0 = time.perf_counter()
    bai1()
    bai2()
    bai3()
    bai4()
    dt = time.perf_counter() - t0
    print("\n" + "=" * 72)
    print(f"TOAN BO LAB CHAY TRONG {dt:.1f}s TREN CPU "
          f"(logits gia, compute gan nhu bang 0 - dung nguyen tac "
          f"tiet kiem cua khoa hoc: chung minh cong thuc dung "
          f"tren do choi truoc, roi moi dot GPU cho model that).")


if __name__ == "__main__":
    main()
