#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Bo sung Day 16: train MASKING 10 epoch de kiem chung gia thuyet.

Ket qua ban 2 epoch: loss giam (3.32 -> 2.39) nhung model chi
ECHO lai prompt (vd: "u do cua Viet Nam la gi?") thay vi tra loi
"Ha Noi.". Gia thuyet: 2 epoch x 11 mau la QUA IT cho model
tieng Anh hoc dap an tieng Viet - model moi hoc duoc hanh vi de
nhat la copy token tu prompt.

Script nay train them 10 epoch (cung du lieu, cung LR, cung
seed) + do val loss moi epoch de xem: (a) model co nho duoc dap
an khong, (b) val loss co bat dau tang (overfit) khong.

Neu 10 epoch cho dap an dung -> gia thuyet dung, 2 epoch chi la
chua du. Neu van echo -> co van de sau hon, can dieu tra.
"""

import os
import time

import torch
from torch.utils.data import DataLoader

from sft_lab import (BATCH_SIZE, DATA, LEARNING_RATE, MODEL_DIR,
                     PROMPTS, SEED, SFTDataset, collate,
                     encode_example, evaluate)

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
EPOCHS_LONG = 10


@torch.no_grad()
def val_loss(model, tok, val_items):
    """Loss trung binh tren tap val (labels da masking san)."""
    model.eval()
    loader = DataLoader(SFTDataset(val_items), batch_size=4,
                        collate_fn=lambda b: collate(b,
                                                     tok.pad_token_id))
    total, steps = 0.0, 0
    for batch in loader:
        batch = {k: v.to(DEVICE) for k, v in batch.items()}
        total += model(**batch).loss.item()
        steps += 1
    model.train()
    return total / steps


def main():
    """Train 10 epoch, in train/val loss moi epoch, sinh 5 prompt."""
    from transformers import AutoModelForCausalLM, AutoTokenizer

    print(f"torch {torch.__version__} | device: {DEVICE}")
    tok = AutoTokenizer.from_pretrained(MODEL_DIR)
    tok.pad_token = tok.eos_token
    n_val = max(1, round(len(DATA) * 0.2))
    train_data, val_data = DATA[:-n_val], DATA[-n_val:]
    train_items = [encode_example(d, tok, mask_prompt=True)
                   for d in train_data]
    val_items = [encode_example(d, tok, mask_prompt=True)
                 for d in val_data]
    torch.manual_seed(SEED)
    model = AutoModelForCausalLM.from_pretrained(MODEL_DIR).to(DEVICE)
    model.config.pad_token_id = tok.pad_token_id
    loader = DataLoader(
        SFTDataset(train_items), batch_size=BATCH_SIZE, shuffle=True,
        collate_fn=lambda b: collate(b, tok.pad_token_id),
        generator=torch.Generator().manual_seed(SEED))
    opt = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE)
    model.train()
    t0 = time.perf_counter()
    for epoch in range(EPOCHS_LONG):
        total_loss, steps = 0.0, 0
        for batch in loader:
            batch = {k: v.to(DEVICE) for k, v in batch.items()}
            out = model(**batch)
            out.loss.backward()
            opt.step()
            opt.zero_grad()
            total_loss += out.loss.item()
            steps += 1
        vl = val_loss(model, tok, val_items)
        print(f"Epoch {epoch + 1}: train loss = "
              f"{total_loss / steps:.4f} | val loss = {vl:.4f}")
    print(f"Train xong trong {time.perf_counter() - t0:.1f}s "
          f"({EPOCHS_LONG} epoch, {len(train_data)} mau).")
    evaluate(model, tok, "SAU 10 EPOCH (masking BAT)")


if __name__ == "__main__":
    main()
