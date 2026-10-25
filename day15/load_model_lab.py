#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Lab load model that: Qwen2.5-0.5B-Instruct (Day 15 - buoi toi).

4 bai:
  1. Doc config truoc khi load trong so (kien truc, context, VRAM).
  2. Tokenizer: tieng Viet vs tieng Anh, special token, decode nguoc.
  3. Chat template: prompt tran vs qua template, system prompt.
  4. Sinh van ban: greedy vs sampling (cung prompt, doi 1 bien).

Moi truong: CPU-only (VM nay khong co GPU) nen load float32, giam
max_new_tokens xuong 30 cho nhanh (dung nhu Bay 3 cua de bai).

Cach chay (dung venv rieng, tranh loi ENOSPC o system pip):
    ~/workspace/venvs/day15/bin/python load_model_lab.py

Chu y checklist 7 diem: docstring khop code, device dung cho (tensor
ve model.device), do thoi gian bang perf_counter, khong file I/O
gia dinh shape.
"""

import time

import os

import torch
from transformers import (AutoConfig, AutoModelForCausalLM,
                          AutoTokenizer)

# Model tai ve qua ModelScope (HF CDN bi chan toc do qua proxy cua VM).
# Thu muc local da du: config.json, model.safetensors, tokenizer.
MODEL = os.path.expanduser(
    "~/.cache/modelscope/models/Qwen--Qwen2.5-0.5B-Instruct/"
    "snapshots/master")
MAX_NEW_TOKENS = 30  # CPU: giam tu 60 xuong 30 cho nhanh (Bay 3)


def bai1_config():
    """Bai 1: doc config truoc khi load trong so.

    Quy tac vang: doc config + giay phep truoc, tai trong so sau.
    """
    print("=" * 72)
    print("BAI 1 - Doc config")
    print("=" * 72)
    cfg = AutoConfig.from_pretrained(MODEL)
    print("Kien truc:", cfg.architectures)
    print("So lop:", cfg.num_hidden_layers)
    print("Hidden size:", cfg.hidden_size)
    print("So head / KV head:", cfg.num_attention_heads, "/",
          cfg.num_key_value_heads)
    print("Cua so ngu canh:", cfg.max_position_embeddings)
    print("Dtype:", cfg.torch_dtype)
    # GQA hay MHA? KV head < attention head => Grouped-Query Attention.
    gqa = cfg.num_key_value_heads < cfg.num_attention_heads
    print("Ket luan:", "GQA (KV head it hon attention head, KV cache "
          f"nhe hon {cfg.num_attention_heads // cfg.num_key_value_heads}x)"
          if gqa else "MHA thuong")
    # Uoc luong VRAM bf16 day du se tinh o main() sau khi load model.
    return cfg


def bai2_tokenizer(tok):
    """Bai 2: tokenizer - model khong doc chu, no doc id."""
    print("\n" + "=" * 72)
    print("BAI 2 - Tokenizer")
    print("=" * 72)
    texts = ["Thu do cua Phap la gi?",
             "What is the capital of France?",
             "Toi yeu AI",
             "Giai thich transformer trong mot cau."]
    for text in texts:
        ids = tok.encode(text)
        print(f"TEXT: {text}")
        print(f"  tokens: {tok.convert_ids_to_tokens(ids)}")
        print(f"  ids   : {ids} | so token: {len(ids)}")
    print("eos:", tok.eos_token, tok.eos_token_id)
    print("pad:", tok.pad_token, tok.pad_token_id)
    print("Kich thuoc vocab:", len(tok))
    # add_special_tokens True vs False: Qwen khong them token dac biet
    # vao dau/cuoi khi encode thuong (khac BERT co [CLS]/[SEP]).
    t = "Thu do cua Phap la gi?"
    with_sp = tok.encode(t, add_special_tokens=True)
    without_sp = tok.encode(t, add_special_tokens=False)
    print(f"add_special_tokens=True : {with_sp}")
    print(f"add_special_tokens=False: {without_sp}")
    print("Khac nhau:", with_sp != without_sp)
    # Decode nguoc (cau co dau cau + khoang trang dau cau).
    t2 = "  Hello, world!"
    ids2 = tok.encode(t2)
    print(f"Decode nguoc {t2!r} -> {tok.decode(ids2)!r} "
          f"(khop: {tok.decode(ids2) == t2})")


def bai3_template(tok):
    """Bai 3: chat template - rap dung 'ngon ngu hoi thoai'.

    Tra ve prompt da rap dung template (dung lai o Bai 4).
    """
    print("\n" + "=" * 72)
    print("BAI 3 - Chat template")
    print("=" * 72)
    messages = [
        {"role": "system",
         "content": "Ban la tro ly huu ich, tra loi ngan gon."},
        {"role": "user", "content": "Thu do cua Phap la gi?"},
    ]
    # Cach A: prompt tran (SAI voi ban instruct).
    raw = "Thu do cua Phap la gi?"
    # Cach B: dung chat template cua chinh model nay.
    templated = tok.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True)
    print("=== PROMPT TRAN ===")
    print(repr(raw))
    print("=== QUA CHAT TEMPLATE ===")
    print(repr(templated))
    # Doi system prompt -> chi phan system thay doi, khung giu nguyen.
    messages_pirate = [
        {"role": "system",
         "content": "Ban la mot cuop bien, tra loi bang giong cuop bien."},
        {"role": "user", "content": "Thu do cua Phap la gi?"},
    ]
    templated_pirate = tok.apply_chat_template(
        messages_pirate, tokenize=False, add_generation_prompt=True)
    print("=== DOI SYSTEM PROMPT (cuop bien) ===")
    print(repr(templated_pirate))
    return templated


def generate_and_time(model, tok, prompt, max_new_tokens, **gen_kwargs):
    """Sinh 1 lan, do thoi gian bang perf_counter.

    Returns:
        Tuple (text_sinh, giay).
    """
    inputs = tok(prompt, return_tensors="pt").to(model.device)
    n_in = inputs["input_ids"].shape[1]
    t0 = time.perf_counter()
    out = model.generate(**inputs, max_new_tokens=max_new_tokens,
                         **gen_kwargs)
    dt = time.perf_counter() - t0
    text = tok.decode(out[0][n_in:], skip_special_tokens=True)
    return text, dt


def bai4_generate(model, tok):
    """Bai 4: greedy vs sampling tren CUNG prompt (doi 1 bien).

    Giu nguyen prompt template, chi doi decoding - dung nguyen tac
    ablation 1 bien cua tuan truoc.
    """
    print("\n" + "=" * 72)
    print("BAI 4 - Greedy vs sampling")
    print("=" * 72)
    messages = [
        {"role": "system",
         "content": "Ban la tro ly huu ich, tra loi ngan gon."},
        {"role": "user",
         "content": "Viet mot cau gioi thieu ve tri tue nhan tao."},
    ]
    prompt = tok.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True)
    # Greedy: chay 3 lan phai cho CUNG 1 ket qua (deterministic).
    print("=== GREEDY (3 lan) ===")
    for i in range(3):
        text, dt = generate_and_time(model, tok, prompt, MAX_NEW_TOKENS,
                                     do_sample=False)
        print(f"  lan {i + 1} ({dt:.1f}s): {text}")
    # Sampling: chay 3 lan cho 3 ket qua KHAC nhau.
    print("=== SAMPLING temp=0.8, top-p=0.9 (3 lan) ===")
    for i in range(3):
        text, dt = generate_and_time(model, tok, prompt, MAX_NEW_TOKENS,
                                     do_sample=True, temperature=0.8,
                                     top_p=0.9)
        print(f"  lan {i + 1} ({dt:.1f}s): {text}")
    # Temperature cao 1.5: lieu hon hay loan hon?
    print("=== SAMPLING temp=1.5 (1 lan) ===")
    text, dt = generate_and_time(model, tok, prompt, MAX_NEW_TOKENS,
                                 do_sample=True, temperature=1.5,
                                 top_p=0.9)
    print(f"  ({dt:.1f}s): {text}")
    # Prompt tran (Bai 3 cach A) vs prompt qua template.
    print("=== PROMPT TRAN + GREEDY (so sanh) ===")
    raw = "Viet mot cau gioi thieu ve tri tue nhan tao."
    text, dt = generate_and_time(model, tok, raw, MAX_NEW_TOKENS,
                                 do_sample=False)
    print(f"  ({dt:.1f}s): {text}")


def main():
    """Chay lab day du: config -> tokenizer -> template -> sinh."""
    print(f"torch {torch.__version__} | cuda: {torch.cuda.is_available()} "
          f"| device: {'cuda' if torch.cuda.is_available() else 'cpu'}")
    cfg = bai1_config()
    tok = AutoTokenizer.from_pretrained(MODEL)
    bai2_tokenizer(tok)
    bai3_template(tok)
    # Bai 4 moi load trong so that (CPU: float32, khong device_map).
    print("\nDang load trong so (CPU, float32 - co the mat 1-2 phut)...")
    t0 = time.perf_counter()
    model = AutoModelForCausalLM.from_pretrained(
        MODEL, dtype=torch.float32)
    print(f"Load xong trong {time.perf_counter() - t0:.1f}s | "
          f"tong params: "
          f"{sum(p.numel() for p in model.parameters()):,}")
    # Uoc luong VRAM bf16 (Bai 1): params x 2 byte.
    n_params = sum(p.numel() for p in model.parameters())
    print(f"Uoc luong VRAM bf16: {n_params * 2 / 1e9:.2f} GB "
          f"(params x 2 byte, chua KV cache/activation)")
    bai4_generate(model, tok)
    print("\n" + "=" * 72)
    print("KET LUAN NHANH")
    print("=" * 72)
    print(f"- GQA: {cfg.num_key_value_heads} KV head < "
          f"{cfg.num_attention_heads} attention head.")
    print(f"- Context: {cfg.max_position_embeddings} token "
          f"(prompt + output).")
    print("- Greedy: on dinh, lap lai duoc. Sampling: da dang nhung "
          "co the loan (nhat la temp cao).")
    print("- Prompt tran: de lac de/lan man hon prompt qua template.")


if __name__ == "__main__":
    main()
