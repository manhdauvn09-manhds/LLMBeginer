#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Bo sung Bai 4: sinh lai voi prompt TIENG VIET CO DAU.

Ly do: ban dau script dung prompt khong dau (theo quy uoc comment
khong dau) nen model hieu sai - greedy tu choi, sampling lac sang
'trang web'. Chay lai voi prompt goc co dau cua de bai de kiem
chung: chat luong co tot hon khong, va 3 tinh chat (greedy on
dinh, sampling da dang, prompt tran te hon) co con dung khong.

Prompt goc cua de bai:
  system: "Ban la tro ly huu ich, tra loi ngan gon."
  user: "Viet mot cau gioi thieu ve tri tue nhan tao."
(duoi day viet co dau day du, vi day la DU LIEU chay model, khong
phai comment code).
"""

import os
import time

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

MODEL = os.path.expanduser(
    "~/.cache/modelscope/models/Qwen--Qwen2.5-0.5B-Instruct/"
    "snapshots/master")
MAX_NEW_TOKENS = 30


def gen(model, tok, prompt, **kwargs):
    """Sinh 1 lan, do thoi gian bang perf_counter."""
    inputs = tok(prompt, return_tensors="pt").to(model.device)
    n_in = inputs["input_ids"].shape[1]
    t0 = time.perf_counter()
    out = model.generate(**inputs, max_new_tokens=MAX_NEW_TOKENS,
                         **kwargs)
    dt = time.perf_counter() - t0
    return tok.decode(out[0][n_in:], skip_special_tokens=True), dt


def main():
    """Load model 1 lan, sinh 3 kieu voi prompt co dau."""
    tok = AutoTokenizer.from_pretrained(MODEL)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL, dtype=torch.float32)
    messages = [
        {"role": "system",
         "content": "Bạn là trợ lý hữu ích, trả lời ngắn gọn."},
        {"role": "user",
         "content": "Viết một câu giới thiệu về trí tuệ nhân tạo."},
    ]
    prompt = tok.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True)
    print("=== TEMPLATE + GREEDY (co dau) ===")
    text, dt = gen(model, tok, prompt, do_sample=False)
    print(f"  ({dt:.1f}s): {text}")
    print("=== TEMPLATE + SAMPLING temp=0.8 (co dau) ===")
    text, dt = gen(model, tok, prompt, do_sample=True,
                   temperature=0.8, top_p=0.9)
    print(f"  ({dt:.1f}s): {text}")
    print("=== PROMPT TRAN + GREEDY (co dau, de so sanh) ===")
    text, dt = gen(model, tok, "Viết một câu giới thiệu về "
                   "trí tuệ nhân tạo.", do_sample=False)
    print(f"  ({dt:.1f}s): {text}")


if __name__ == "__main__":
    main()
