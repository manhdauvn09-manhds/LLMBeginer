#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Baseline CPU cho mini-project 3 (chi chay bang venv Day 15).

Load Qwen2.5-0.5B-Instruct co san trong cache ModelScope cua
may, sinh cau tra loi greedy cho mp3_eval.jsonl qua chat
template, ghi predictions_baseline_qwen05.jsonl, roi cham
bang eval_predictions() cua miniproject3_lab.

Muc dich: kiem chung ong eval chay tron vong tren output
model THAT. Day KHONG phai baseline tuan 4 cho model 1.5B
(model khac, phan cung khac) - baseline 1.5B phai chay tren
GPU tuan sau, cung bo eval v1.

Cach chay:
    ~/workspace/venvs/day15/bin/python baseline_cpu.py
"""

import json
import os
import time

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from miniproject3_lab import eval_predictions

HERE = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR = os.path.expanduser(
    "~/.cache/modelscope/models/Qwen--Qwen2.5-0.5B-Instruct/"
    "snapshots/master")
EVAL_PATH = os.path.join(HERE, "mp3_eval.jsonl")
PRED_PATH = os.path.join(HERE, "predictions_baseline_qwen05.jsonl")
MAX_NEW_TOKENS = 80
SYSTEM = ("Bạn là trợ lý trả lời câu hỏi về khóa học LLM 30 "
          "ngày. Trả lời ngắn gọn bằng tiếng Việt, chỉ dùng "
          "kiến thức trong tài liệu khóa học. Nếu câu hỏi nằm "
          "ngoài tài liệu hoặc thông tin không tồn tại, hãy "
          "nói không biết, không bịa thêm.")


@torch.no_grad()
def answer(model, tok, instruction):
    """Sinh 1 cau tra loi greedy qua chat template."""
    messages = [{"role": "system", "content": SYSTEM},
                {"role": "user", "content": instruction}]
    text = tok.apply_chat_template(messages, tokenize=False,
                                   add_generation_prompt=True)
    ids = tok(text, return_tensors="pt")
    out = model.generate(**ids, max_new_tokens=MAX_NEW_TOKENS,
                         do_sample=False,
                         pad_token_id=tok.eos_token_id)
    new = out[0][ids["input_ids"].shape[1]:]
    return tok.decode(new, skip_special_tokens=True).strip()


def main():
    """Load model, sinh predictions, cham bang eval v1."""
    t0 = time.perf_counter()
    tok = AutoTokenizer.from_pretrained(MODEL_DIR)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_DIR, dtype=torch.float32)
    model.eval()
    print(f"Load model xong sau {time.perf_counter() - t0:.1f}s.")
    rows = [json.loads(line) for line in
            open(EVAL_PATH, encoding="utf-8") if line.strip()]
    with open(PRED_PATH, "w", encoding="utf-8") as f:
        for row in rows:
            text = answer(model, tok, row["instruction"])
            f.write(json.dumps({"id": row["id"], "output": text},
                               ensure_ascii=False) + "\n")
            print(f"[{row['id']}] {row['instruction'][:60]}...")
            print(f"  -> {text[:160]}")
    dt = time.perf_counter() - t0
    print(f"\nSinh {len(rows)} predictions trong {dt:.1f}s "
          f"(~{dt / len(rows):.1f}s/cau, CPU).")
    print("=" * 72)
    eval_predictions(PRED_PATH, EVAL_PATH)


if __name__ == "__main__":
    main()
