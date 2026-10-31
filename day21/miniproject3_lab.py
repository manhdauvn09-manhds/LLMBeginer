#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Mini-project 3 - Lab ngay 21: on tap tuan 3 + khoi dong.

Tac vu: tro ly hoi dap tren tai lieu khoa hoc LLM 30 ngay
(tuan 3: ngay 15-20). Toan bo phan 1 chay bang Python thuan,
khong can GPU, khong can API key: python miniproject3_lab.py

Buoc 0: kiem tra dataset + tinh ngan sach LoRA/QLoRA.
Buoc 1: dataset v1 (mp3_train.jsonl 30 mau, mp3_eval.jsonl
    17 mau, du 4 truc, contamination = 0).
Buoc 2: chot cau hinh - doc config.json THAT cua Qwen2.5-1.5B,
    tinh ca cong thuc khung cua de bai VA ban chinh xac theo
    GQA (k/v nho hon q/o vi KV head it hon).
Buoc 3: gate release viet trong mp3_plan.md TRUOC khi train.
Phan 2 (GPU): baseline CPU bang Qwen2.5-0.5B co san tren may
    (baseline_cpu.py, chay rieng bang venv) - chi de kiem
    chung ong eval, KHONG phai baseline tuan 4 cho model 1.5B.
"""

import json
import os

HERE = os.path.dirname(os.path.abspath(__file__))
TRAIN_PATH = os.path.join(HERE, "mp3_train.jsonl")
EVAL_PATH = os.path.join(HERE, "mp3_eval.jsonl")
CONFIG_15B = os.path.join(HERE, "qwen2.5-1.5b_config.json")
PRED_PATH = os.path.join(HERE, "predictions_baseline_qwen05.jsonl")

# Cau hinh Qwen2.5-0.5B doc tu config.json that trong cache
# ModelScope (may nay, ngay 15) - phuong an du phong chay CPU.
CONFIG_05B = {"hidden_size": 896, "num_hidden_layers": 24,
              "num_attention_heads": 14, "num_key_value_heads": 2,
              "intermediate_size": 4864, "vocab_size": 151936,
              "tie_word_embeddings": True}


def normalize(text):
    """Chuan hoa de so trung: lower + gom khoang trang."""
    import re
    return re.sub(r"\s+", " ", text.strip().lower())


def load_jsonl(path, need_response=True):
    """Doc JSONL phong thu: bao loi ro so dong neu sai.

    Raises:
        FileNotFoundError: neu thieu file.
        ValueError: neu dong khong phai JSON, thieu truong,
            hoac instruction/response rong.
    """
    rows = []
    try:
        f = open(path, encoding="utf-8")
    except FileNotFoundError:
        raise FileNotFoundError(f"Khong thay file: {path}")
    with f:
        for i, line in enumerate(f):
            if not line.strip():
                continue
            try:
                r = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"{path} dong {i}: JSON loi "
                                 f"({exc}).")
            if "instruction" not in r or "response" not in r:
                raise ValueError(f"{path} dong {i}: thieu "
                                 f"instruction/response.")
            if not r["instruction"].strip() or \
                    not r["response"].strip():
                raise ValueError(f"{path} dong {i}: rong.")
            rows.append(r)
    return rows


def check_dataset(train_path, eval_path):
    """Kiem tra: du truong, khong trung, eval khong lan train.

    Contamination = eval co instruction (sau chuan hoa) trung
    train. Tra ve (train, eval).
    """
    train = load_jsonl(train_path)
    ev = load_jsonl(eval_path)
    train_q = {normalize(r["instruction"]) for r in train}
    leaked = [r["instruction"] for r in ev
              if normalize(r["instruction"]) in train_q]
    dup = len(ev) - len({normalize(r["instruction"])
                         for r in ev})
    print(f"train: {len(train)} mau | eval: {len(ev)} mau")
    print(f"eval trung train (contamination): {len(leaked)} "
          f"{leaked[:3]}")
    print(f"eval trung nhau: {dup}")
    return train, ev


def lora_budget(d_model, n_layers, r, target_modules, alpha=16):
    """So tham so LoRA theo cong thuc khung cua de bai.

    Moi module muc tieu: A (r x d) + B (d x r) = 2*r*d.
    Gia dinh moi module VUONG d x d - chi dung khi q/k/v/o
    cung kich thuoc; voi GQA thi k/v nho hon, xem ban exact.
    """
    per_layer = 2 * r * d_model * len(target_modules)
    total = per_layer * n_layers
    print(f"LoRA r={r}, alpha={alpha}, target={target_modules}:")
    print(f"  Tham so train (cong thuc khung): {total:,} "
          f"(~{total / 1e6:.2f}M)")
    return total


def module_shapes(config):
    """Kich thuoc (in, out) that cua tung module attention.

    GQA: q/o la d x d, k/v la d x (kv_heads * head_dim) -
    nho hon vi KV head it hon query head. head_dim = d / heads.
    """
    d = config["hidden_size"]
    heads = config["num_attention_heads"]
    kv_heads = config["num_key_value_heads"]
    head_dim = d // heads
    kv_dim = kv_heads * head_dim
    return {"q_proj": (d, d), "k_proj": (d, kv_dim),
            "v_proj": (d, kv_dim), "o_proj": (d, d)}


def lora_budget_exact(config, r, target_modules, alpha=16):
    """So tham so LoRA chinh xac theo kich thuoc that (GQA).

    Moi module: r * (in + out). Tra ve tong so tham so.
    """
    shapes = module_shapes(config)
    per_layer = sum(r * (shapes[m][0] + shapes[m][1])
                    for m in target_modules)
    total = per_layer * config["num_hidden_layers"]
    print(f"LoRA r={r}, alpha={alpha}, target={target_modules}:")
    print(f"  Tham so train (chinh xac GQA): {total:,} "
          f"(~{total / 1e6:.2f}M)")
    return total


def estimate_base_params(config):
    """Uoc luong tong tham so base tu config (khong can tai).

    Embedding (tied) + moi layer: attention (q/k/v/o + 2 norm)
    + MLP (gate/up/down). So sanh voi con so cong bo cua model.
    """
    d = config["hidden_size"]
    layers = config["num_hidden_layers"]
    inter = config["intermediate_size"]
    shapes = module_shapes(config)
    attn = sum(shapes[m][0] * shapes[m][1]
               for m in shapes) + 2 * d
    mlp = d * inter * 3
    embed = config["vocab_size"] * d
    total = embed + layers * (attn + mlp)
    return total


def vram_estimate(base_params, adapter_params):
    """Uoc luong VRAM QLoRA (GB): base 4-bit + adapter bf16 +
    AdamW (8 byte/params adapter) + activation (uoc luong).

    Activation la uoc luong tho (batch 4, seq 512, co gradient
    checkpointing): ~1-2 GB. Con so tong la KHOANG, phai do
    that khi train - dung suy tu so bit (bai hoc tuan 3).
    """
    base_gb = base_params * 0.5 / 1e9
    scales_gb = base_params * 0.125 / 1e9
    adapter_gb = adapter_params * 2 / 1e9
    optim_gb = adapter_params * 8 / 1e9
    act_gb = 1.5
    total = base_gb + scales_gb + adapter_gb + optim_gb + act_gb
    print(f"  VRAM uoc luong: base 4-bit {base_gb:.2f} GB + "
          f"scale {scales_gb:.2f} GB + adapter bf16 "
          f"{adapter_gb:.3f} GB + AdamW {optim_gb:.3f} GB + "
          f"activation ~{act_gb:.1f} GB = ~{total:.2f} GB.")
    return total


def eval_predictions(pred_path, gold_path):
    """Cham nhanh: predictions co id + output, gold co id +
    must_contain/must_not_contain (+ axis tuy chon).

    Predictions thieu (khong co id) tinh la truot va bao so
    luong. Tra ve (ok_total, by_axis).
    """
    preds = {}
    for line in open(pred_path, encoding="utf-8"):
        if line.strip():
            row = json.loads(line)
            preds[row["id"]] = row["output"].lower()
    golds = load_jsonl(gold_path)
    by_axis = {}
    ok_total = 0
    missing = 0
    for g in golds:
        out = preds.get(g["id"])
        if out is None:
            missing += 1
            ok = False
        else:
            ok = all(k.lower() in out
                     for k in g.get("must_contain", [])) and \
                all(k.lower() not in out
                    for k in g.get("must_not_contain", []))
        ok_total += ok
        by_axis.setdefault(g.get("axis", "general"),
                           []).append(ok)
    print(f"Tong: {ok_total}/{len(golds)} dat "
          f"(thieu predictions: {missing})")
    for axis, vals in by_axis.items():
        print(f"  {axis}: {sum(vals)}/{len(vals)} dat")
    return ok_total, by_axis


def main():
    """Chay lab day du: mau khung -> ngan sach that -> dataset
    -> eval baseline (neu co predictions)."""
    print("=" * 72)
    print("BUOC 0 - Kiem tra cong thuc khung (vi du de bai)")
    print("=" * 72)
    total = lora_budget(d_model=2048, n_layers=24, r=16,
                        target_modules=["q_proj", "v_proj"])
    assert total == 3145728, f"Sai cong thuc: {total}!"
    print("Khop so de bai: 3.145.728 tham so.")

    print("\n" + "=" * 72)
    print("BUOC 2 - Ngan sach that cho Qwen2.5-1.5B-Instruct")
    print("=" * 72)
    with open(CONFIG_15B, encoding="utf-8") as f:
        config = json.load(f)
    print(f"Config that: hidden={config['hidden_size']}, "
          f"layers={config['num_hidden_layers']}, "
          f"heads={config['num_attention_heads']}, KV heads="
          f"{config['num_key_value_heads']}, inter="
          f"{config['intermediate_size']}, vocab="
          f"{config['vocab_size']}, tied="
          f"{config['tie_word_embeddings']}.")
    base = estimate_base_params(config)
    print(f"Tong tham so base uoc luong: {base:,} (~{base / 1e9:.2f}B).")
    print("Cau hinh (a) r=8, target q_proj+v_proj:")
    lora_budget(d_model=config["hidden_size"],
                n_layers=config["num_hidden_layers"], r=8,
                target_modules=["q_proj", "v_proj"])
    adapter_a = lora_budget_exact(config, 8, ["q_proj", "v_proj"])
    print("Cau hinh (b) r=16, target q+k+v+o:")
    lora_budget(d_model=config["hidden_size"],
                n_layers=config["num_hidden_layers"], r=16,
                target_modules=["q_proj", "k_proj", "v_proj",
                                "o_proj"])
    adapter_b = lora_budget_exact(
        config, 16, ["q_proj", "k_proj", "v_proj", "o_proj"])
    print("VRAM QLoRA cau hinh (b) (chinh xac GQA):")
    vram_estimate(base, adapter_b)
    print("Luu y: cong thuc khung tinh THUA ~26% vi gia dinh "
          "k/v vuong d x d, trong khi GQA k/v chi rong "
          "256 chieu. Dung so chinh xac khi chot ngan sach.")
    print("\nPhuong an du phong Qwen2.5-0.5B (chay CPU, local):")
    adapter_05 = lora_budget_exact(CONFIG_05B, 16,
                                   ["q_proj", "k_proj", "v_proj",
                                    "o_proj"])
    vram_estimate(estimate_base_params(CONFIG_05B), adapter_05)

    print("\n" + "=" * 72)
    print("BUOC 1 - Kiem tra dataset v1")
    print("=" * 72)
    check_dataset(TRAIN_PATH, EVAL_PATH)

    print("\n" + "=" * 72)
    print("EVAL BASELINE (neu co predictions)")
    print("=" * 72)
    if os.path.exists(PRED_PATH):
        eval_predictions(PRED_PATH, EVAL_PATH)
    else:
        print("Chua co predictions - chay baseline_cpu.py (venv) "
              "truoc, roi chay lai script nay.")


if __name__ == "__main__":
    main()
