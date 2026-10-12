#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Day 02 - Bai 2 (+3): HUAN LUYEN MLP DU DOAN TOKEN KE TIEP + LOG GRADIENT NORM
===============================================================================
Muc dich:
    1. Huan luyen mot MLP cuc nho de "du doan token ke tiep" - phien ban
       thu nho cua cai ma MOI LLM deu lam.
    2. LOG GRADIENT NORM moi epoch: day la "may do huyet ap" cua qua trinh
       huan luyen. Doc duoc no = bat duoc loi train bang con so.
    3. (Buoc 4) So sanh SGD vs AdamW: giu nguyen seed & khoi tao, chi doi
       optimizer -> xem ai hoi tu nhanh hon tren bai toan nho nay.
    4. (Bai 3 - Green) Do thoi gian/epoch; neu co GPU thi thu AMP
       (mixed precision) bang flag --amp.

Cach chay:
    python train_mlp.py                  # chay ca SGD + AdamW, ve plot, in bang so sanh
    python train_mlp.py --optim adamw    # chi chay AdamW
    python train_mlp.py --epochs 100      # doi so epoch
    python train_mlp.py --amp            # bat mixed precision (can GPU/CUDA)

Dau ra:
    - plot_sgd.png / plot_adamw.png : 2 panel (loss | gradient norm log-scale)
    - Bang so sanh: optimizer - epoch dat accuracy 95% - loss cuoi - nhan xet
"""

import argparse
import time

import torch
import torch.nn as nn

# Ve plot tren may khong co man hinh (server): dung backend 'Agg' de chi LUU file,
# khong can mo cua so. Tren may ca nhan co the doi thanh plt.show().
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

# ---------------------------------------------------------------------------
# 0. THAM SO BAI TOAN
# ---------------------------------------------------------------------------
SEED = 0          # seed co dinh -> ket qua lap lai duoc (khoa hoc)
VOCAB = 4         # bo tu vung: {0, 1, 2, 3}
CONTEXT = 3       # nhin 3 token lien tiep de doan token thu 4
N_TOKENS = 2000   # do dai chuoi du lieu do choi
HIDDEN = 32       # so neuron lop an
EPOCHS = 200      # so vong train mac dinh
LR = 1e-2         # learning rate
TARGET_ACC = 0.95  # nguong accuracy coi la "hoc duoc"


# ---------------------------------------------------------------------------
# 1. DU LIEU DO CHOI: chuoi tuan hoan 0 1 2 3 0 1 2 3 ...
# ---------------------------------------------------------------------------
def make_data():
    """Sinh cap (x, y): x = 3 token lien tiep, y = token ke tiep.

    Vi du: chuoi ..., 1, 2, 3, 0, ... -> x = [1,2,3], y = 0.

    Bieu dien x: ONE-HOT. Moi token (0..3) -> vector 4 chieu co duy nhat
    1 vi tri = 1, con lai = 0. Ghep 3 token -> vector 12 chieu.
    Vi sao one-hot ma khong dua so 0..3 truc tiep? Vi so 0..3 mang y nghia
    THU TU gia tao ("3 > 0") trong khi day chi la ky hieu roi rac.
    One-hot noi voi model: "cac token binh dang, khong co thu tu".
    """
    # Chuoi tuan hoan: [0,1,2,3,0,1,2,3,...] dai N_TOKENS
    seq = torch.arange(N_TOKENS) % VOCAB

    # One-hot: (N_TOKENS, 4). eye(4)[k] = hang thu k cua ma tran don vi.
    onehot = torch.eye(VOCAB)[seq]

    # Xep cua so truot: mau thu i lay onehot[i:i+3] ghep phang -> (12,)
    # y[i] = seq[i+3] (token dung ngay sau cua so).
    n = N_TOKENS - CONTEXT
    # LUU Y off-by-one: unfold(size=3, step=1) tren 2000 phan tu cho ra
    # 2000-3+1 = 1998 cua so, nhung y = seq[3:] chi co 1997 phan tu
    # (cua so cuoi cung [1997,1998,1999] khong co "token ke tiep").
    # -> Chi lay n cua so DAU TIEN de X va y khop nhau.
    X = onehot.unfold(0, CONTEXT, 1)[:n]  # (n, 4, 3): n cua so, moi cua so 3 token x 4 chieu
    X = X.permute(0, 2, 1).reshape(n, -1)     # (n, 12): ghep 3 token thanh 1 vector
    y = seq[CONTEXT:]                         # (n,): nhan la LONG tensor (yeu cau cua CrossEntropyLoss)

    # Kiem tra nhanh "hop dong shape" - thoi quen vang khi debug (bai sang).
    assert X.shape == (n, CONTEXT * VOCAB), X.shape
    assert y.shape == (n,), y.shape
    return X, y


# ---------------------------------------------------------------------------
# 2. MO HINH: MLP 12 -> 32 -> 4
# ---------------------------------------------------------------------------
class MLP(nn.Module):
    """Mang 2 lop tuyen tinh + tanh o giua, dau ra 4 logit (ung voi 4 token)."""

    def __init__(self):
        super().__init__()
        # Linear(12 -> 32): moi neuron an la mot "dac trung" hoc duoc tu context.
        self.fc1 = nn.Linear(CONTEXT * VOCAB, HIDDEN)
        # Linear(32 -> 4): bien dac trung thanh diem so (logit) cho 4 token.
        self.fc2 = nn.Linear(HIDDEN, VOCAB)

    def forward(self, x):
        h = torch.tanh(self.fc1(x))  # tanh: phi tuyen, dua gia tri ve (-1, 1)
        return self.fc2(h)           # tra ve LOGIT (chua softmax - CrossEntropyLoss tu lo)


# ---------------------------------------------------------------------------
# 3. HAM TIEN ICH
# ---------------------------------------------------------------------------
def total_grad_norm(model):
    """Tinh TOTAL gradient norm = sqrt( sum ||g_p||^2 ) tren moi param.

    QUY TAC VANG (bay so 2 trong bai): phai do norm TRUOC khi clip_grad_norm_!
    Neu do SAU khi clip, plot luc nao cung dep (bi kep o max_norm) va ta
    mat hoan toan tin hieu chan doan that cua qua trinh train.
    """
    total_sq = 0.0
    # torch.no_grad(): day chi la phep DO, khong can autograd theo doi.
    with torch.no_grad():
        for p in model.parameters():
            if p.grad is not None:
                # .detach(): tach tensor khoi graph (phong khi p.grad van gan graph).
                total_sq += p.grad.detach().pow(2).sum().item()
    return total_sq ** 0.5


@torch.no_grad()  # decorator: ham nay khong can tinh gradient -> nhanh, nhe bo nho
def accuracy(model, X, y):
    """Ti le du doan dung tren toan bo tap du lieu."""
    logits = model(X)
    pred = logits.argmax(dim=1)      # chon token co logit cao nhat moi mau
    return (pred == y).float().mean().item()


# ---------------------------------------------------------------------------
# 4. VONG LAP HUAN LUYEN (core cua Bai 2)
# ---------------------------------------------------------------------------
def train(optim_name, X, y, epochs=EPOCHS, lr=LR, use_amp=False):
    """Huan luyen MLP voi optimizer chi dinh. Tra ve dict ket qua + lich su.

    Trinh tu chuan cua 1 epoch (hoc thuoc long):
        zero_grad -> forward -> loss -> backward -> [do norm] -> [clip] -> step
    """
    # Dat lai seed TRUOC khi tao model -> SGD va AdamW XUAT PHAT GIONG HET NHAU
    # (so sanh cong bang o Buoc 4 chi co y nghia khi diem xuat phat giong nhau).
    torch.manual_seed(SEED)
    model = MLP()
    criterion = nn.CrossEntropyLoss()  # vua softmax vua negative log-likelihood

    if optim_name == "sgd":
        # SGD thuan: buoc di = -lr * gradient (huong doc nhat, do dai co dinh theo lr).
        opt = torch.optim.SGD(model.parameters(), lr=lr)
    elif optim_name == "adamw":
        # AdamW: moi param co learning rate RIENG tu thich nghi theo lich su
        # gradient (moment m, v) + weight decay tach rieng (ban "W").
        opt = torch.optim.AdamW(model.parameters(), lr=lr)
    else:
        raise ValueError(f"optimizer la: {optim_name}")

    # AMP (mixed precision, Bai 3): chi co y nghia tren GPU.
    # autocast: tu dong dung float16 cho phep tinh on dinh, giu float32 cho phan nhay cam.
    # GradScaler: phong to loss truoc backward de gradient nho khong bi "troit"
    # ve 0 o float16 (underflow), roi thu nho lai truoc optimizer.step().
    scaler = torch.amp.GradScaler("cuda") if (use_amp and torch.cuda.is_available()) else None
    if use_amp and not torch.cuda.is_available():
        print("  [Green] Khong thay CUDA -> bo qua AMP, chay float32 tren CPU.")

    losses, grad_norms = [], []
    epoch_95 = None  # epoch dau tien dat accuracy >= 95%
    t0 = time.perf_counter()

    for epoch in range(epochs):
        # Buoc 1: XOA gradient cu. QUEN dong nay -> gradient CONG DON qua cac
        # epoch -> norm phinh to "bi an", loss giam vai epoch roi vot (bay so 1).
        opt.zero_grad()

        # Buoc 2+3: forward -> loss. Dung autocast neu bat AMP.
        if scaler is not None:
            with torch.amp.autocast("cuda"):
                loss = criterion(model(X), y)
        else:
            loss = criterion(model(X), y)

        # Buoc 4: backward - chain rule tu dong, dien gradient vao p.grad.
        if scaler is not None:
            scaler.scale(loss).backward()
        else:
            loss.backward()

        # Buoc 5: DO gradient norm TRUOC khi clip (quy tac vang!).
        # (Neu dung AMP: unscale truoc de norm la gia tri that, khong phai ban phong to.)
        if scaler is not None:
            scaler.unscale_(opt)
        gnorm = total_grad_norm(model)
        grad_norms.append(gnorm)

        # BONUS chan doan nhanh ngay trong luc train:
        if gnorm == 0.0:
            print(f"  [CANH BAO] epoch {epoch}: grad norm = 0 (gradient CHET - model ngung hoc!)")
        elif gnorm > 1e3:
            print(f"  [CANH BAO] epoch {epoch}: grad norm = {gnorm:.1e} (sap NO - can giam lr / clip)")

        # Buoc 6: CLIP gradient - "phanh khan cap" chong gradient no.
        # max_norm=1.0: neu total norm > 1 thi thu nho tat ca gradient theo ti le
        # de norm con dung 1 (giu NGUYEN huong, chi giam do dai buoc).
        # LUU Y: scaler.unscale_(opt) chi duoc goi DUNG 1 LAN (da goi o Buoc 5
        # truoc khi do norm). Go goi lan 2 se chia gradient cho scale them lan
        # nua -> sai. Vi vay o day chi clip, khong unscale lai.
        # -> Muon thi nghiem tac dung cua clip: comment dong nay lai va chay lai.
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

        # Buoc 7: optimizer buoc di 1 buoc (cap nhat trong so).
        if scaler is not None:
            scaler.step(opt)
            scaler.update()
        else:
            opt.step()

        losses.append(loss.item())

        # Ghi nhan epoch dau tien cham nguong 95% (thong ke cho bang so sanh).
        if epoch_95 is None and accuracy(model, X, y) >= TARGET_ACC:
            epoch_95 = epoch

    train_time = time.perf_counter() - t0
    final_acc = accuracy(model, X, y)
    return {
        "optim": optim_name,
        "losses": losses,
        "grad_norms": grad_norms,
        "epoch_95": epoch_95,
        "final_loss": losses[-1],
        "final_acc": final_acc,
        "train_time": train_time,
    }


# ---------------------------------------------------------------------------
# 5. VE PLOT: loss + gradient norm (log-scale) theo epoch
# ---------------------------------------------------------------------------
def save_plot(result, path):
    """Ve 2 panel: (a) loss theo epoch, (b) gradient norm (truc y log) theo epoch.

    Cach doc plot (hoc thuoc 3 "nhip tim"):
      - Norm giam dan ve 0 trong khi loss van cao -> vanishing gradient /
        hoc qua cham -> tang lr, kiem tra init.
      - Norm vot len roi loss NaN -> exploding -> giam lr, bat clip.
      - Norm dao dong deu, loss giam muot -> KHOE (nhip tim binh thuong).
    """
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    epochs = range(len(result["losses"]))

    axes[0].plot(epochs, result["losses"])
    axes[0].set_title(f"Loss theo epoch ({result['optim']})")
    axes[0].set_xlabel("epoch")
    axes[0].set_ylabel("CrossEntropy loss")
    axes[0].grid(True, alpha=0.3)

    axes[1].plot(epochs, result["grad_norms"])
    axes[1].set_yscale("log")  # LOG-SCALE: norm bien thien nhieu bac do, thang
                               # tuyen tinh se "bep" het chi tiet vung nho.
    axes[1].set_title(f"Gradient norm theo epoch - log scale ({result['optim']})")
    axes[1].set_xlabel("epoch")
    axes[1].set_ylabel("total grad norm (log)")
    axes[1].grid(True, alpha=0.3, which="both")

    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)
    print(f"  Da luu plot: {path}")


# ---------------------------------------------------------------------------
# 6. MAIN: chay thi nghiem + in bang so sanh
# ---------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser(description="Day02 - Huan luyen MLP du doan token ke tiep")
    ap.add_argument("--epochs", type=int, default=EPOCHS, help="so epoch (mac dinh 200)")
    ap.add_argument("--optim", choices=["sgd", "adamw", "both"], default="both",
                    help="optimizer can chay (mac dinh: both de so sanh)")
    ap.add_argument("--amp", action="store_true",
                    help="bat mixed precision - AMP (can GPU/CUDA, Bai 3 Green checkpoint)")
    args = ap.parse_args()

    print("=" * 64)
    print("Bai 2 - Huan luyen MLP du doan token ke tiep + log gradient norm")
    print("=" * 64)

    X, y = make_data()
    print(f"Du lieu: X{X.shape} (one-hot), y{y.shape}, {len(y)} mau, vocab={VOCAB}")

    optims = ["sgd", "adamw"] if args.optim == "both" else [args.optim]
    results = []
    for name in optims:
        print(f"\n--- Train voi {name.upper()} (lr={LR}, epochs={args.epochs}) ---")
        res = train(name, X, y, epochs=args.epochs, use_amp=args.amp)
        results.append(res)
        plot_path = f"plot_{name}.png"
        save_plot(res, plot_path)
        print(f"  loss cuoi = {res['final_loss']:.4f} | accuracy = {res['final_acc']*100:.1f}% "
              f"| {res['train_time']/args.epochs*1000:.1f} ms/epoch")

    # ---- Buoc 4: bang so sanh SGD vs AdamW (4 dong nhu bai yeu cau) ----
    if len(results) == 2:
        print("\n" + "=" * 64)
        print("Buoc 4 - So sanh SGD vs AdamW (cung seed, cung khoi tao)")
        print("=" * 64)
        print(f"{'optimizer':<10} | {'epoch dat acc 95%':<17} | {'loss cuoi':<10} | nhan xet")
        print("-" * 64)
        for r in results:
            e95 = str(r["epoch_95"]) if r["epoch_95"] is not None else ">200 (chua dat)"
            note = ("hoi tu nhanh (adaptive lr)" if r["optim"] == "adamw"
                    else "cham hon (lr co dinh)")
            print(f"{r['optim']:<10} | {e95:<17} | {r['final_loss']:<10.4f} | {note}")
        print("\nNhan xet mau: AdamW thuong thang tren bai nho nho adaptive lr giup\n"
              "'leo' nhanh qua cac vung loss doc khac nhau; SGD lr co dinh de i\n"
              "o vung phang. Nhung AdamW ton gap doi bo nho optimizer state (m, v).")

    # ---- Checklist tu dong ----
    print("\n--- Tu kiem tra ---")
    ok = all(r["final_acc"] >= TARGET_ACC for r in results)
    print(f"  [{'OK' if ok else 'CHUA DAT'}] accuracy >= 95% tren tap toy")
    print(f"  [{'OK' if all(len(r['grad_norms']) == args.epochs for r in results) else 'LOI'}] "
          "da log gradient norm moi epoch (TRUOC khi clip)")
    print("  [OK] plot 2 panel loss + grad norm (log-scale) da luu file PNG")
    if not ok:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
