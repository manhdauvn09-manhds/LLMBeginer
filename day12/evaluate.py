#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Eval Lab - pipeline danh gia mini xay tu dau (Day 12 - buoi toi).

5 phan (A->E), 7 cho TODO da dien day du:
    A. EM & F1: metric tra loi dung/sai, kem test tu viet.
    B. Perplexity tu log-prob trung binh (so that tu Day 11).
    C. Bootstrap CI: dung tin chenh lech 1-2% khi CI chong lan.
    D. Regression set: 30 prompt model DA lam dung ("dap thu phanh").
    E. Decontamination scan + canary + scorecard (kem energy metric).

Cach chay:
    python evaluate.py        # khong can GPU, chi can stdlib

Dau ra: regression_set.jsonl (30 prompt).
"""

import json
import math
import random

print("=" * 64)
print("Phan A - EM & F1")
print("=" * 64)


def normalize(s):
    """Viet thuong, gom khoang trang."""
    return " ".join(s.lower().strip().split())


def exact_match(pred, gold):
    """1 neu khop tuyet doi sau normalize, nguoc lai 0."""
    return int(normalize(pred) == normalize(gold))


def f1(pred, gold):
    """F1 tren tap tu (do gan dung)."""
    p, g = normalize(pred).split(), normalize(gold).split()
    # [TRAINER] Ca pred va gold deu rong = khop hoan toan -> F1 = 1.0
    # (chuan SQuAD, nhat quan voi exact_match("", "") == 1). Thieu dong
    # nay thi truong hop bien tra 0.0, mau thuan voi EM cua chinh no.
    if not p and not g:
        return 1.0
    common = sum(min(p.count(w), g.count(w)) for w in set(g))
    if common == 0:
        return 0.0
    prec, rec = common / len(p), common / len(g)
    return 2 * prec * rec / (prec + rec)


# TODO 1 - them cap (pred, gold) cua CHINH MINH + tu tinh tay truoc.
# Cap cua toi: ("Ha Noi la thu do", "ha noi").
#   normalize: pred -> "ha noi la thu do" (5 tu), gold -> "ha noi" (2 tu).
#   EM = 0 (hai chuoi khac nhau).
#   common = 2 ("ha", "noi"); prec = 2/5 = 0.4; rec = 2/2 = 1.0.
#   F1 = 2*0.4*1.0/(0.4+1.0) = 0.8/1.4 = 0.5714 -> 0.57.
qa_tests = [
    ("Thu do Ha Noi", "ha noi"),
    ("ha noi", "ha noi"),
    ("Paris", "paris"),
    ("Ha Noi la thu do", "ha noi"),  # cap cua toi (TODO 1)
]

for pred, gold in qa_tests:
    print(f"pred={pred!r:24} gold={gold!r:12} "
          f"EM={exact_match(pred, gold)} F1={f1(pred, gold):.2f}")

# Cam nhan (TODO 1): EM=0 du chi lech 1 tu -> voi dap an ngan/that,
# bao ca EM (khop tuyet doi) LAN F1 (do gan dung). Chi bao EM don le
# de hieu lam model "sai hoan toan" trong khi no gan dung.

print()
print("=" * 64)
print("Phan B - Perplexity tu log-probabilities")
print("=" * 64)

# TODO 2 - dung SO THAT tu Day 11 (train_grid.py, cross-entropy = -logprob).
#   Model "cu" = S1 (N=112k): loss 3.8126 -> avg_logprob = -3.8126.
#   Model "moi" = L3 (N=2.7M): loss 2.3870 -> avg_logprob = -2.3870.
MODEL_A_avg_logprob = -3.8126   # mo hinh "cu" (S1, Day 11)
MODEL_B_avg_logprob = -2.3870   # mo hinh "moi" cua ban (L3, Day 11)


def perplexity(avg_logprob):
    """Tinh perplexity tu log-prob trung binh.

    PPL = exp(-trung binh log p(w)): mo hinh cang "chac chan" vao tu dung
    thi PPL cang thap.
    """
    return math.exp(-avg_logprob)


print(f"PPL model cu: {perplexity(MODEL_A_avg_logprob):.1f}")
print(f"PPL model moi: {perplexity(MODEL_B_avg_logprob):.1f}")

# TODO 3 - PPL thap hon KHONG dong nghia tra loi cau hoi user tot hon.
# Ly do: PPL chi do kha nang du doan token tiep theo tren mot corpus
# cu the - no khong do instruction-following, tinh dung su that, hay
# kha nang lam dung nhiem vu. Model co the PPL thap nho hoc thuoc long
# corpus eval (leakage!) ma van tra loi te. Luon di kem metric nhiem vu
# (EM/F1/accuracy) - PPL mot minh la "cam giac", khong phai nang luc.

print()
print("=" * 64)
print("Phan C - Bootstrap CI: dung tin chenh lech 1-2%")
print("=" * 64)

random.seed(42)  # cung seed -> cung ket qua (reproducible)

# Ket qua mau: 1 = dung, 0 = sai. Model moi dat 82%, cu 80%.
old = [1] * 400 + [0] * 100
new = [1] * 410 + [0] * 90
random.shuffle(old)
random.shuffle(new)


def bootstrap_ci(scores, n_boot=1000, alpha=0.05):
    """Uoc luong khoang tin cay bang bootstrap co hoan lai.

    Tra ve (mean, bien duoi, bien tren) cua khoang tin cay (1-alpha).
    """
    # [TRAINER] scores rong -> sum(scores)/n chia cho 0 (ZeroDivisionError
    # kho hieu). Nem ValueError ro rang ngay tu dau de nguoi goi biet
    # phai truyen danh sach diem khong rong.
    if not scores:
        raise ValueError("bootstrap_ci: scores khong duoc rong")
    means = []
    n = len(scores)
    for _ in range(n_boot):
        sample = [random.choice(scores) for _ in range(n)]
        means.append(sum(sample) / n)
    means.sort()
    lo = means[int((alpha / 2) * n_boot)]
    hi = means[int((1 - alpha / 2) * n_boot)]
    return sum(scores) / n, lo, hi


acc_old, lo_old, hi_old = bootstrap_ci(old)
acc_new, lo_new, hi_new = bootstrap_ci(new)
print(f"Cu:  {acc_old:.0%}  CI=[{lo_old:.0%}, {hi_old:.0%}]")
print(f"Moi: {acc_new:.0%} CI=[{lo_new:.0%}, {hi_new:.0%}]")

# TODO 4 - CI chong lan -> "chua du bang chung" (khong duoc tuyen bo
# "moi tot hon"). Quyet dinh dung: thu them du lieu eval hoac chap
# nhan hoa. Chi khi CI tach roi moi ket luan "moi tot hon that".
overlap = not (hi_old < lo_new or hi_new < lo_old)
conclusion = ("chua du bang chung 'moi tot hon' (CI chong lan)"
              if overlap else "moi tot hon that (CI tach roi)")
print("Ket luan:", conclusion)

print()
print("=" * 64)
print("Phan D - Regression set: 'dap thu phanh' cua model")
print("=" * 64)

# TODO 5 - 30 prompt model DA lam dung: 10 kien thuc + 10 tieng Viet +
# 10 format. Moi dong 1 JSON {"prompt", "expected"}.
regression = [
    # 10 kien thuc.
    {"prompt": "Thu do cua Viet Nam la gi?", "expected": "Ha Noi"},
    {"prompt": "15 + 27 = ?", "expected": "42"},
    {"prompt": "7 x 8 = ?", "expected": "56"},
    {"prompt": "Thu do cua Nhat Ban la gi?", "expected": "Tokyo"},
    {"prompt": "100 - 37 = ?", "expected": "63"},
    {"prompt": "Hanh tinh gan Mat Troi nhat ten gi?",
     "expected": "Sao Thuy"},
    {"prompt": "12 x 12 = ?", "expected": "144"},
    {"prompt": "2^10 bang bao nhieu?", "expected": "1024"},
    {"prompt": "Mot nam nhuan co bao nhieu ngay?", "expected": "366"},
    {"prompt": "Dai duong lon nhat the gioi la gi?",
     "expected": "Thai Binh Duong"},
    # 10 tieng Viet (dich, chinh ta).
    {"prompt": "Dich sang tieng Viet: 'hello'", "expected": "xin chao"},
    {"prompt": "Dich sang tieng Anh: 'con meo'", "expected": "cat"},
    {"prompt": "Tu trai nghia voi 'nhanh' la gi?", "expected": "cham"},
    {"prompt": "Viet dung chinh ta: 'sach se' hay 'sach se'?",
     "expected": "sach se"},
    {"prompt": "So 5 viet bang chu tieng Viet", "expected": "nam"},
    {"prompt": "Dem so tu trong cau 'Toi yeu Viet Nam'", "expected": "4"},
    {"prompt": "Tu 'vui ve' co may tieng?", "expected": "2"},
    {"prompt": "Dich sang tieng Viet: 'thank you'", "expected": "cam on"},
    {"prompt": "Chu cai dau cua ten rieng co viet hoa khong? (co/khong)",
     "expected": "co"},
    {"prompt": "'mat troi' dich sang tieng Anh", "expected": "sun"},
    # 10 format (in hoa, do dai, sap xep).
    {"prompt": "Viet tu 'perplexity' bang chu in hoa",
     "expected": "PERPLEXITY"},
    {"prompt": "Viet tu 'PYTHON' bang chu thuong", "expected": "python"},
    {"prompt": "Tu 'hello' co bao nhieu ky tu?", "expected": "5"},
    {"prompt": "Viet so 100 bang chu (tieng Viet)", "expected": "mot tram"},
    {"prompt": "Sap xep tang dan cac so 3, 1, 2 (cach nhau boi dau phay)",
     "expected": "1, 2, 3"},
    {"prompt": "Viet ngay 1 thang 1 nam 2026 dang YYYY-MM-DD",
     "expected": "2026-01-01"},
    {"prompt": "Tra loi bang so: 9 + 10 = ?", "expected": "19"},
    {"prompt": "Viet 'dung' 3 lan, cach nhau boi dau phay",
     "expected": "dung, dung, dung"},
    {"prompt": "1 gio co bao nhieu phut?", "expected": "60"},
    {"prompt": "Dem ky tu trong 'abc' (khong tinh khoang trang)",
     "expected": "3"},
]

assert len(regression) == 30, f"thieu prompt: {len(regression)}/30"
with open("regression_set.jsonl", "w", encoding="utf-8") as f:
    for item in regression:
        f.write(json.dumps(item, ensure_ascii=False) + "\n")
print(f"Da ghi {len(regression)} prompt vao regression_set.jsonl")

# TODO 6 - Vi sao regression set PHAI gom prompt model DA lam dung?
# Vi nhiem vu cua no la phat hien HOI QUY (dung -> sai) sau moi thay doi
# model - nhu "dap thu phanh" truoc khi xuat xuong. Prompt kho/chua lam
# duoc thuoc ve task set (do nang luc). Tron prompt kho vao regression
# set thi moi lan chay deu do, mat tac dung canh bao.

print()
print("=" * 64)
print("Phan E - Leakage scan + canary + scorecard")
print("=" * 64)

# --- E1: Decontamination scan ---


def ngrams(text, n=8):
    """Tap cac n-gram (chuoi n tu lien tiep) cua text."""
    toks = text.lower().split()
    return {" ".join(toks[i:i + n]) for i in range(len(toks) - n + 1)}


def find_leakage(train_corpus, eval_set, n=8):
    """Tra ve [(idx_eval, ngram)] cac n-gram cua eval lot trong train."""
    train_ng = set()
    for doc in train_corpus:
        train_ng |= ngrams(doc, n)
    leaks = []
    for i, q in enumerate(eval_set):
        for ng in ngrams(q, n):
            if ng in train_ng:
                leaks.append((i, ng))
                break  # moi cau eval chi can 1 bang chung ro ri
    return leaks


train_corpus = [
    "mo hinh ngon ngu hoc tu van ban tieng viet ve lich su dia ly "
    "va khoa hoc pho thong trong nhieu linh vuc khac nhau",
    "du lieu train gom sach giao khoa va bao chi da duoc lam sach "
    "loai bo thong tin ca nhan truoc khi dua vao huan luyen",
]
eval_set = [
    "thu do cua viet nam la gi",
    "giai thich hien tuong nhat thuc xay ra khi nao trong nam",
]

# TODO 7 - quet moi n-gram cua eval_set trong train_corpus.
leaks = find_leakage(train_corpus, eval_set, n=8)
if leaks:
    for i, ng in leaks:
        print(f"LEAKAGE: eval[{i}] trung 8-gram voi train: '{ng}'")
else:
    print("Decontamination scan: OK (khong n-gram eval nao lot trong "
          "train)")

# Tu kiem detector: cai san 1 leak de chung minh no bat duoc.
_self_test = find_leakage(["khoa hoc du lieu la mot linh vuc rat thu vi"],
                          ["linh vuc rat thu vi va day trien vong"], n=4)
assert _self_test, "detector hong: khong bat duoc leak cai san!"
print("Tu kiem detector: OK (bat duoc leak cai san voi n=4)")

CANARY = "canary-9f3k-llm-eval-ngay12"
for doc in train_corpus:
    assert CANARY not in doc, "RO RI: canary xuat hien trong du lieu train!"
print("Canary check: OK (khong co ro ri chuoi bi mat)")

# --- E2: Scorecard + energy metric ---
print("\n=== SCORECARD ===")
scorecard = [
    ("Perplexity (thap tot)", "45.3", "10.9"),
    ("EM tren QA set (cao tot)", "61%", "66%"),
    ("F1 tren QA set (cao tot)", "0.74", "0.79"),
    ("Accuracy/task set (cao tot)", "80% ±4%", "82% ±4%"),
]
for name, old_v, new_v in scorecard:
    print(f"{name:28} | cu: {old_v:10} | moi: {new_v}")
# Energy: accuracy / watt-hour (Green checkpoint).
#   cu: 5 Wh/1000 req -> 80/5 = 16.0 %/Wh ; moi: 8 Wh -> 82/8 = 10.25 %/Wh
eff_old, eff_new = 80 / 5, 82 / 8
print(f"Energy efficiency (acc/Wh): cu {eff_old:.2f} | moi {eff_new:.2f} "
      "<- model moi KEM hieu qua nang luong!")
print("Bai hoc: doc scorecard phai doc ca cot energy, khong chi accuracy.")
print("=" * 64)
