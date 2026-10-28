#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Lab QLoRA Day 18: do cai gia cua nen 4-bit bang so lieu.

Quy tac vang: CHI cach giu trong so goc thay doi (fp32 so voi
NF4 4-bit). Moi thu khac giu nguyen: cung GPT-2, cung dataset
Ngay 17, cung split phan tang, seed 42, rank r=8, alpha/r=2,
target c_attn, LR 1e-4, epoch 2, batch 4, decoding greedy.

Luu y moi truong (ghi ro, khong gia vo co GPU):
  - VM nay KHONG co GPU, de bai noi Bai 1-3 "BAT BUOC co GPU".
    Nhung bitsandbytes 0.50.2 load 4-bit + forward + train VAN
    CHAY duoc tren CPU (qua fallback kernel, co canh bao thieu
    goi `kernels`), nen lab nay chay THAT ca hai cau hinh tren
    CPU thay vi chi dien bang ly thuyet.
  - Vi khong co GPU: (1) ban tham chieu dung fp32 thay vi fp16
    (fp16 tren CPU vua cham vua khong on dinh; fp32 la dang
    goc cua Day 16-17); (2) peak VRAM thay bang peak RAM (RSS)
    do rieng moi cau hinh trong process rieng (ru_maxrss la
    dinh process, khong reset duoc).
  - Ket luan VRAM tu ly thuyet (xem README) van dung cho GPU;
    so do RAM o day la bang chung huong, khong phai so VRAM.

Cau truc chay:
    python qlora_lab.py              # chay ca hai cau hinh
    python qlora_lab.py --config 4bit  # chi chay 1 cau hinh
  Moi cau hinh chay trong process rieng de do peak RSS sach.

Dataset + ham SFT lay dung tu Day 17 (lora_lab.py, thu muc
day17 ke ben) - khong doi du lieu giua hai cau hinh.
"""

import argparse
import gc
import os
import resource
import subprocess
import sys
import time

import torch
from torch.utils.data import DataLoader
from transformers import (AutoModelForCausalLM, AutoTokenizer,
                          BitsAndBytesConfig)
from peft import LoraConfig, get_peft_model

# Them thu muc day17 (ke ben) vao sys.path de tai dung dataset
# va ham SFT da PASS o Day 17.
DAY17_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                         "..", "day17")
sys.path.insert(0, os.path.normpath(DAY17_DIR))
from lora_lab import (DATA, EVAL_PROMPTS, SFTDataset, audit_eval_prompts,
                      collate, stratified_split, verify_masking_all)

MODEL_DIR = os.path.expanduser(
    "~/.cache/modelscope/models/openai-community--gpt2/"
    "snapshots/master")
DEVICE = "cpu"
SEED = 42
EPOCHS = 2
LEARNING_RATE = 1e-4
BATCH_SIZE = 4
RANK = 8
ALPHA = 16
MAX_NEW_TOKENS = 30


def peak_rss_gb():
    """Peak RAM (GB) cua process hien tai (ru_maxrss, Linux: KB)."""
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1e6


def build_model(tok, use_4bit):
    """Load base + gan LoRA r=8, alpha=16, target c_attn.

    Ban 4-bit: NF4 + double quant + compute bf16. Ban tham chieu:
    fp32 (CPU, thay cho fp16 tren GPU - ghi ro trong README).
    """
    torch.manual_seed(SEED)
    if use_4bit:
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True, bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
            bnb_4bit_compute_dtype=torch.bfloat16)
        base = AutoModelForCausalLM.from_pretrained(
            MODEL_DIR, quantization_config=bnb_config,
            dtype=torch.bfloat16)
    else:
        base = AutoModelForCausalLM.from_pretrained(
            MODEL_DIR, dtype=torch.float32)
    base.config.pad_token_id = tok.pad_token_id
    config = LoraConfig(r=RANK, lora_alpha=ALPHA, lora_dropout=0.05,
                        target_modules=["c_attn"],
                        task_type="CAUSAL_LM")
    model = get_peft_model(base, config)
    return model


def count_trainable(model):
    """Tra ve (so tham so train duoc, tong so tham so)."""
    trainable = sum(p.numel() for p in model.parameters()
                    if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    return trainable, total


def inspect_model(model, name):
    """In dtype base/adapter, requires_grad, so tham so train duoc.

    Kiem tra 3 thu nhu de bai: (1) base ban 4-bit phai la kieu
    luong tu (Params4bit o tang Linear, embedding giu bf16 vi
    khong phai Linear); (2) adapter luon o do chinh xac cao;
    (3) chi tham so 'lora_' moi co requires_grad=True.
    """
    print(f"--- Inspect: {name} ---")
    for n, p in model.named_parameters():
        if "lora_" not in n and "c_attn" in n:
            print(f"  base c_attn: {n[:60]}... type="
                  f"{type(p).__name__} dtype={p.dtype} "
                  f"grad={p.requires_grad}")
            break
    for n, p in model.named_parameters():
        if "lora_" in n:
            print(f"  adapter: {n[:60]}... type={type(p).__name__} "
                  f"dtype={p.dtype} grad={p.requires_grad}")
            break
    bad = [n for n, p in model.named_parameters()
           if p.requires_grad and "lora_" not in n]
    assert not bad, f"W goc chua dong bang: {bad[:3]}"
    trainable, total = count_trainable(model)
    print(f"  train duoc {trainable:,} / {total:,} tham so.")
    return trainable


@torch.no_grad()
def val_loss(model, tok, val_items):
    """Loss trung binh tren tap val (khong shuffle, khong train)."""
    model.eval()
    loader = DataLoader(SFTDataset(val_items, tok), batch_size=4,
                        collate_fn=lambda b: collate(b,
                                                     tok.pad_token_id))
    total, steps = 0.0, 0
    for batch in loader:
        batch = {k: v.to(DEVICE) for k, v in batch.items()}
        total += model(**batch).loss.item()
        steps += 1
    model.train()
    return total / steps


@torch.no_grad()
def generate(model, tok, prompt):
    """Sinh greedy 30 token theo dung template train."""
    model.eval()
    ids = tok(prompt, return_tensors="pt").to(DEVICE)
    out = model.generate(**ids, max_new_tokens=MAX_NEW_TOKENS,
                         do_sample=False,
                         pad_token_id=tok.eos_token_id)
    return tok.decode(out[0][ids["input_ids"].shape[1]:],
                      skip_special_tokens=True)


def run_config(use_4bit):
    """Chay 1 cau hinh tron vong: inspect -> train -> val -> eval.

    Tra ve dict ket qua (in ra de process cha thu thap).
    """
    name = "QLoRA 4-bit" if use_4bit else "LoRA fp32"
    tok = AutoTokenizer.from_pretrained(MODEL_DIR)
    tok.pad_token = tok.eos_token
    train_data, val_data = stratified_split(DATA)
    model = build_model(tok, use_4bit)
    trainable = inspect_model(model, name)
    loader = DataLoader(
        SFTDataset(train_data, tok), batch_size=BATCH_SIZE, shuffle=True,
        collate_fn=lambda b: collate(b, tok.pad_token_id),
        generator=torch.Generator().manual_seed(SEED))
    opt = torch.optim.AdamW(
        (p for p in model.parameters() if p.requires_grad),
        lr=LEARNING_RATE)
    model.train()
    t0 = time.perf_counter()
    last_train_loss = None
    for epoch in range(EPOCHS):
        total_loss, steps = 0.0, 0
        for batch in loader:
            batch = {k: v.to(DEVICE) for k, v in batch.items()}
            out = model(**batch)
            out.loss.backward()
            opt.step()
            opt.zero_grad()
            total_loss += out.loss.item()
            steps += 1
        last_train_loss = total_loss / steps
        print(f"  [{name}] epoch {epoch + 1}: train loss = "
              f"{last_train_loss:.4f}", flush=True)
    dt = time.perf_counter() - t0
    vl = val_loss(model, tok, val_data)
    print(f"===== EVAL {name} =====")
    outputs = []
    for p in EVAL_PROMPTS:
        text = generate(model, tok, p)
        outputs.append(text)
        print("PROMPT:", p.replace("\n", " / "))
        print("->", text)
    peak = peak_rss_gb()
    print(f"RESULT {name} | trainable={trainable} | "
          f"train_loss={last_train_loss:.4f} | val_loss={vl:.4f} | "
          f"time_s={dt:.1f} | peak_rss_gb={peak:.2f}")
    for i, text in enumerate(outputs):
        print(f"OUTPUT {name} [{i}]: {text}")
    return {"name": name, "trainable": trainable,
            "train_loss": last_train_loss, "val_loss": vl,
            "time_s": dt, "peak_rss_gb": peak, "outputs": outputs}


def run_untrained_comparison(tok):
    """Bai 3 (a): sai so nen thuan tuy - base CHUA train.

    Sinh tren EVAL_PROMPTS o dang fp32 va 4-bit, dem so dau ra
    giong nhau. Neu da khac nhau tu truoc khi train, phan chenh
    lech do la sai so nen, adapter khong the hoc bu hoan toan.
    """
    print("\n" + "=" * 72)
    print("BAI 3 (a) - Sai so nen thuan tuy (base CHUA train)")
    print("=" * 72)
    outputs = {}
    for use_4bit, name in ((False, "base fp32"),
                           (True, "base 4-bit")):
        torch.manual_seed(SEED)
        if use_4bit:
            bnb_config = BitsAndBytesConfig(
                load_in_4bit=True, bnb_4bit_quant_type="nf4",
                bnb_4bit_use_double_quant=True,
                bnb_4bit_compute_dtype=torch.bfloat16)
            base = AutoModelForCausalLM.from_pretrained(
                MODEL_DIR, quantization_config=bnb_config,
                dtype=torch.bfloat16)
        else:
            base = AutoModelForCausalLM.from_pretrained(
                MODEL_DIR, dtype=torch.float32)
        base.config.pad_token_id = tok.pad_token_id
        outs = [generate(base, tok, p) for p in EVAL_PROMPTS]
        outputs[name] = outs
        for p, text in zip(EVAL_PROMPTS, outs):
            print(f"[{name}] {p.replace(chr(10), ' / ')} -> {text}")
        del base
        gc.collect()
    same = sum(1 for a, b in zip(outputs["base fp32"],
                                 outputs["base 4-bit"]) if a == b)
    print(f"Base fp32 vs base 4-bit (chua train): {same}/"
          f"{len(EVAL_PROMPTS)} dau ra giong nhau.")
    return same


def run_paged_optimizer_test(tok, train_data, val_data):
    """Mo rong: PagedAdamW8bit cho ban 4-bit (neu chay duoc)."""
    print("\n" + "=" * 72)
    print("MO RONG - PagedAdamW8bit cho ban 4-bit")
    print("=" * 72)
    try:
        import bitsandbytes as bnb
        model = build_model(tok, True)
        loader = DataLoader(
            SFTDataset(train_data, tok), batch_size=BATCH_SIZE,
            shuffle=True,
            collate_fn=lambda b: collate(b, tok.pad_token_id),
            generator=torch.Generator().manual_seed(SEED))
        opt = bnb.optim.PagedAdamW8bit(
            (p for p in model.parameters() if p.requires_grad),
            lr=LEARNING_RATE)
        model.train()
        t0 = time.perf_counter()
        last_train_loss = None
        for epoch in range(EPOCHS):
            total_loss, steps = 0.0, 0
            for batch in loader:
                batch = {k: v.to(DEVICE) for k, v in batch.items()}
                out = model(**batch)
                out.loss.backward()
                opt.step()
                opt.zero_grad()
                total_loss += out.loss.item()
                steps += 1
            last_train_loss = total_loss / steps
            print(f"  [Paged8bit] epoch {epoch + 1}: train loss = "
                  f"{last_train_loss:.4f}", flush=True)
        dt = time.perf_counter() - t0
        vl = val_loss(model, tok, val_data)
        print(f"  [Paged8bit] train loss={last_train_loss:.4f} | "
              f"val loss={vl:.4f} | time={dt:.1f}s | "
              f"peak RSS={peak_rss_gb():.2f} GB.")
    except Exception as exc:
        print(f"  PagedAdamW8bit KHONG chay duoc tren CPU: "
              f"{type(exc).__name__}: {str(exc)[:200]}")


def main():
    """Chay lab day du hoac 1 cau hinh (tuy --config)."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", choices=["fp32", "4bit"],
                        default=None,
                        help="chi chay 1 cau hinh (mac dinh: ca hai, "
                        "moi cau hinh 1 process rieng)")
    args = parser.parse_args()
    if args.config is not None:
        run_config(args.config == "4bit")
        return
    tok = AutoTokenizer.from_pretrained(MODEL_DIR)
    tok.pad_token = tok.eos_token
    print("=" * 72)
    print("BAI 3 (c) - Audit dataset (phai PASS truoc khi ket luan)")
    print("=" * 72)
    train_data, val_data = stratified_split(DATA)
    print("Train:", len(train_data), "mau | Val:", len(val_data),
          "mau (phan tang theo kind nhu Day 17).")
    audit_eval_prompts(EVAL_PROMPTS, train_data)
    verify_masking_all(tok)
    print("\n" + "=" * 72)
    print("BAI 1-2 - Train 2 cau hinh (moi cau hinh 1 process rieng)")
    print("=" * 72)
    results = {}
    for cfg in ("fp32", "4bit"):
        proc = subprocess.run(
            [sys.executable, os.path.abspath(__file__),
             "--config", cfg],
            capture_output=True, text=True)
        print(proc.stdout)
        if proc.returncode != 0:
            print(f"CAU HINH {cfg} LOI:\n{proc.stderr[-2000:]}")
            continue
        for line in proc.stdout.splitlines():
            if line.startswith("RESULT "):
                parts = dict(p.split("=", 1) for p in
                             line.split(" | ")[1:])
                results[cfg] = parts
    print("\n" + "=" * 72)
    print("BANG SO SANH")
    print("=" * 72)
    for cfg in ("fp32", "4bit"):
        if cfg in results:
            r = results[cfg]
            print(f"{cfg}: trainable={r['trainable']} | "
                  f"train_loss={r['train_loss']} | "
                  f"val_loss={r['val_loss']} | time={r['time_s']}s | "
                  f"peak RSS={r['peak_rss_gb']} GB")
    if "fp32" in results and "4bit" in results:
        t_fp32 = float(results["fp32"]["time_s"])
        t_4bit = float(results["4bit"]["time_s"])
        slower_pct = (t_4bit - t_fp32) / t_fp32 * 100
        print(f"4-bit cham hon fp32: {slower_pct:.1f}% "
              f"(ti le thoi gian {t_4bit / t_fp32:.2f}x).")
        print(f"Nang luong tren cung may (cung cong suat): ti le "
              f"{1 + slower_pct / 100:.2f}x - cham hon = ton nang "
              f"luong hon neu cung card; QLoRA chi tiet kiem nang "
              f"luong khi cho phep dung card nho hon du de bu "
              f"(Y < {1 / (1 + slower_pct / 100):.2f}).")
    run_untrained_comparison(tok)
    run_paged_optimizer_test(tok, train_data, val_data)
    print("\n" + "=" * 72)
    print("KET LUAN NHANH")
    print("=" * 72)
    print("- Xem bang so sanh o tren: VRAM/RAM, toc do, loss.")
    print("- Xem Bai 3 (a): phan chenh lech truoc train la sai so "
          "nen thuan tuy.")


if __name__ == "__main__":
    main()
