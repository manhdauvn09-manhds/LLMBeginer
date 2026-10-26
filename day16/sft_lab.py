#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Lab SFT model nho GPT-2 (Day 16 - buoi toi).

3 bai + ablation (mo rong):
  1. Xay du lieu SFT, encode_example() voi masking (-100 cho prompt),
     assert kiem chung masking PASS moi duoc train.
  2. Do padding: ti le token that / token dem, uoc luong packing.
  3. Train 2 epoch, LR 5e-5, split train/val 20%, sinh truoc-sau
     tren cung 5 prompt co dinh (cung template, greedy, 30 token).
  Ablation: train them 1 ban KHONG masking (labels = input_ids),
  moi thu khac giu nguyen (cung seed, du lieu, epoch, LR) - chi
  doi 1 bien duy nhat la masking.

Moi truong: CPU-only, GPT-2 124M, float32. Template co dinh cho
ca train va sinh (doi template giua chung la doi luon bai kiem
tra):
    Cau hoi: {instruction}\nTra loi: {output}<|endoftext|>

Du lieu: 4 mau cua de bai + 10 mau tu viet, tron 4 dang: hoi dap
ngan, tom tat 1 cau, email 2-3 cau, dich 1 cau. DU LIEU VIET CO
DAU DAY DU (bai hoc Day 15: tieng Viet khong dau lam model hieu
sai) - day la du lieu train, khong phai comment code.

Cach chay (dung venv Day 15):
    ~/workspace/venvs/day15/bin/python sft_lab.py
"""

import os
import time

import torch
from torch.utils.data import DataLoader, Dataset
from transformers import AutoModelForCausalLM, AutoTokenizer

MODEL_DIR = os.path.expanduser(
    "~/.cache/modelscope/models/openai-community--gpt2/"
    "snapshots/master")
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
MAX_NEW_TOKENS = 30
EPOCHS = 2
LEARNING_RATE = 5e-5
BATCH_SIZE = 4
SEED = 42

# 4 mau cua de bai + 10 mau tu viet (tron 4 dang nhiem vu).
DATA = [
    # 4 mau cua de bai.
    {"instruction": "Thủ đô của Việt Nam là gì?",
     "output": "Hà Nội."},
    {"instruction": "Thủ đô của Nhật Bản là gì?",
     "output": "Tokyo."},
    {"instruction": "Tóm tắt bằng một câu: Mèo là loài vật nuôi "
     "phổ biến, hiền lành và thích ngủ nhiều.",
     "output": "Mèo là loài vật nuôi hiền lành, thích ngủ nhiều."},
    {"instruction": "Viết một câu chào buổi sáng lịch sự.",
     "output": "Chào buổi sáng, chúc bạn một ngày làm việc hiệu quả."},
    # 10 mau tu viet - hoi dap ngan.
    {"instruction": "Thủ đô của Pháp là gì?",
     "output": "Paris."},
    {"instruction": "Một tuần có mấy ngày?",
     "output": "Một tuần có 7 ngày."},
    {"instruction": "Nước sôi ở bao nhiêu độ C?",
     "output": "Nước sôi ở 100 độ C."},
    # Tom tat 1 cau.
    {"instruction": "Tóm tắt bằng một câu: Chó là loài vật trung thành, "
     "thông minh và là bạn thân của con người.",
     "output": "Chó là loài vật trung thành và thông minh."},
    {"instruction": "Tóm tắt bằng một câu: Sách giúp con người mở rộng "
     "kiến thức, rèn tư duy và nuôi dưỡng tâm hồn.",
     "output": "Sách giúp mở rộng kiến thức và rèn tư duy."},
    # Email 2-3 cau.
    {"instruction": "Viết email xin nghỉ phép 1 ngày gửi quản lý.",
     "output": "Chào anh, em xin phép nghỉ làm ngày mai vì việc gia "
     "đình. Em sẽ bàn giao công việc cho đồng nghiệp trước khi "
     "nghỉ. Cảm ơn anh đã thông cảm."},
    {"instruction": "Viết email cảm ơn khách hàng sau buổi họp.",
     "output": "Cảm ơn quý khách đã dành thời gian họp với chúng tôi "
     "hôm nay. Chúng tôi sẽ gửi báo giá chi tiết trong tuần này. "
     "Rất mong được hợp tác lâu dài."},
    # Dich 1 cau.
    {"instruction": "Dịch sang tiếng Anh: Tôi yêu Việt Nam.",
     "output": "I love Vietnam."},
    {"instruction": "Dịch sang tiếng Anh: Cảm ơn bạn rất nhiều.",
     "output": "Thank you very much."},
    {"instruction": "Dịch sang tiếng Việt: Good morning.",
     "output": "Chào buổi sáng."},
]

# 5 prompt co dinh: 3 cua de bai + 2 tu viet, dung template train.
PROMPTS = [
    "Câu hỏi: Thủ đô của Việt Nam là gì?\nTrả lời: ",
    "Câu hỏi: Viết một câu chào buổi sáng lịch sự.\nTrả lời: ",
    "Câu hỏi: Tóm tắt bằng một câu: Mèo là loài vật nuôi phổ biến, "
    "hiền lành và thích ngủ nhiều.\nTrả lời: ",
    "Câu hỏi: Thủ đô của Pháp là gì?\nTrả lời: ",
    "Câu hỏi: Dịch sang tiếng Anh: Cảm ơn bạn rất nhiều.\nTrả lời: ",
]


def format_prompt(instruction):
    """Rap prompt theo template co dinh cua lab.

    Template nay dung cho CA train VA sinh - khong doi giua
    chung, ke ca dau xuong dong va token ket thuc.
    """
    return f"Câu hỏi: {instruction}\nTrả lời: "


def encode_example(item, tok, mask_prompt=True):
    """Ma hoa 1 mau SFT thanh (input_ids, labels).

    Prompt va dap an token hoa RIENG RE, noi lai thanh input_ids.
    Neu mask_prompt=True: labels che prompt bang -100, chi tinh
    loss tren dap an (chuan SFT). Neu False (ablation): labels =
    input_ids, tinh loss ca prompt.

    Args:
        item: dict {"instruction": ..., "output": ...}.
        tok: tokenizer GPT-2 (pad = eos).
        mask_prompt: True = che prompt, False = khong che.

    Returns:
        Tuple (input_ids, labels) la 2 list int.

    Raises:
        ValueError: neu instruction hoac output rong (dap an rong
            se lam labels toan -100, loss thanh NaN).
    """
    if not item["instruction"] or not item["output"]:
        raise ValueError("instruction va output khong duoc rong.")
    prompt = format_prompt(item["instruction"])
    answer = item["output"] + tok.eos_token
    prompt_ids = tok(prompt, add_special_tokens=False)["input_ids"]
    answer_ids = tok(answer, add_special_tokens=False)["input_ids"]
    input_ids = prompt_ids + answer_ids
    if mask_prompt:
        labels = [-100] * len(prompt_ids) + answer_ids
    else:
        labels = list(input_ids)
    return input_ids, labels


def verify_masking(tok):
    """Kiem chung masking: PASS het moi duoc sang Bai 2.

    Assert: moi label o vi tri prompt la -100, label o vi tri
    dap an bang dung token that.
    """
    ids, labels = encode_example(DATA[0], tok, mask_prompt=True)
    prompt_len = len(tok(format_prompt(DATA[0]["instruction"]),
                         add_special_tokens=False)["input_ids"])
    assert all(l == -100 for l in labels[:prompt_len]), \
        "Prompt chua duoc che het!"
    assert labels[prompt_len:] == ids[prompt_len:], \
        "Label dap an phai bang token that!"
    n_loss = sum(1 for l in labels if l != -100)
    print(f"Masking OK. Mau 0 co {len(ids)} token, trong do "
          f"{n_loss} token tinh loss.")
    print(tok.decode(ids))
    # Edge case: dap an rong phai bao loi ro rang, khong NaN am tham.
    try:
        encode_example({"instruction": "Hoi?", "output": ""}, tok)
        print("CANH BAO: output rong khong bi chan!")
    except ValueError as exc:
        print(f"Edge case output rong: chan dung ({exc})")


def measure_padding(tok):
    """Bai 2: do ti le token that / token dem cua toan bo DATA."""
    encoded = [encode_example(d, tok) for d in DATA]
    max_len = max(len(ids) for ids, _ in encoded)
    total = len(encoded) * max_len
    real = sum(len(ids) for ids, _ in encoded)
    print(f"Mau dai nhat: {max_len} token | "
          f"Tong token (ke ca padding): {total}")
    print(f"Token that: {real} ({real / total:.0%}) | "
          f"Token dem lang phi: {total - real} "
          f"({1 - real / total:.0%})")
    # Uoc luong packing: xep cac mau noi tiep cho day sequence 128
    # token, chi thua 1 doan padding cuoi cung (< 128 token).
    n_seq = (real + 127) // 128
    packed_total = n_seq * 128
    print(f"Uoc luong packing (seq 128): {n_seq} sequence, "
          f"token dem chi con {packed_total - real} "
          f"({(packed_total - real) / packed_total:.1%}) "
          f"thay vi {1 - real / total:.0%}.")
    return encoded


class SFTDataset(Dataset):
    """Dataset SFT: encode san toan bo mau luc khoi tao."""

    def __init__(self, items):
        """Args: items la list (input_ids, labels) da encode."""
        self.items = items

    def __len__(self):
        """Tra ve so mau."""
        return len(self.items)

    def __getitem__(self, i):
        """Tra ve dict input_ids/labels cua mau thu i."""
        ids, labels = self.items[i]
        return {"input_ids": ids, "labels": labels}


def collate(batch, pad_id):
    """Dem batch ve cung do dai: pad token + nhan -100.

    Attention mask = 1 cho token that, 0 cho padding.
    """
    max_len = max(len(b["input_ids"]) for b in batch)
    input_ids, labels, attn = [], [], []
    for b in batch:
        pad = max_len - len(b["input_ids"])
        input_ids.append(b["input_ids"] + [pad_id] * pad)
        labels.append(b["labels"] + [-100] * pad)
        attn.append([1] * len(b["input_ids"]) + [0] * pad)
    return {"input_ids": torch.tensor(input_ids),
            "labels": torch.tensor(labels),
            "attention_mask": torch.tensor(attn)}


def train_model(tok, train_items, tag):
    """Train GPT-2 tu dau (load lai trong so goc) va tra ve model.

    Seed lai truoc moi lan train de 2 ban (masking bat/tat) co
    cung thu tu shuffle va dropout - chi khac 1 bien masking
    (labels da encode san trong train_items truoc khi goi ham).
    """
    torch.manual_seed(SEED)
    model = AutoModelForCausalLM.from_pretrained(MODEL_DIR)
    model.to(DEVICE)
    model.config.pad_token_id = tok.pad_token_id
    loader = DataLoader(
        SFTDataset(train_items), batch_size=BATCH_SIZE, shuffle=True,
        collate_fn=lambda b: collate(b, tok.pad_token_id),
        generator=torch.Generator().manual_seed(SEED))
    opt = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE)
    model.train()
    t0 = time.perf_counter()
    epoch_losses = []
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
        avg = total_loss / steps
        epoch_losses.append(avg)
        print(f"  [{tag}] Epoch {epoch + 1}: train loss = {avg:.4f}")
    dt = time.perf_counter() - t0
    print(f"  [{tag}] Train xong trong {dt:.1f}s "
          f"({EPOCHS} epoch, {len(train_items)} mau).")
    return model, epoch_losses


@torch.no_grad()
def generate(model, tok, prompt):
    """Sinh greedy 30 token theo dung template train."""
    ids = tok(prompt, return_tensors="pt").to(DEVICE)
    out = model.generate(**ids, max_new_tokens=MAX_NEW_TOKENS,
                         do_sample=False,
                         pad_token_id=tok.eos_token_id)
    return tok.decode(out[0][ids["input_ids"].shape[1]:],
                      skip_special_tokens=True)


def evaluate(model, tok, title):
    """Sinh tren 5 prompt co dinh, in ket qua de so sanh."""
    print(f"===== {title} =====")
    model.eval()
    for p in PROMPTS:
        print("PROMPT:", p.replace("\n", " / "))
        print("->", generate(model, tok, p))


def main():
    """Chay lab day du: masking -> padding -> train -> ablation."""
    print(f"torch {torch.__version__} | device: {DEVICE}")
    tok = AutoTokenizer.from_pretrained(MODEL_DIR)
    tok.pad_token = tok.eos_token
    print("\n" + "=" * 72)
    print("BAI 1 - Masking")
    print("=" * 72)
    verify_masking(tok)
    print("\n" + "=" * 72)
    print("BAI 2 - Padding")
    print("=" * 72)
    measure_padding(tok)
    # Split train/val: 20% val (3/14 mau cuoi), 11 mau train.
    n_val = max(1, round(len(DATA) * 0.2))
    train_data, val_data = DATA[:-n_val], DATA[-n_val:]
    print(f"\nSplit: {len(train_data)} train / {len(val_data)} val "
          f"(val = {n_val} mau cuoi: "
          f"{[d['instruction'][:30] for d in val_data]}).")
    print("\n" + "=" * 72)
    print("BAI 3 - Truoc khi SFT (base model)")
    print("=" * 72)
    base = AutoModelForCausalLM.from_pretrained(MODEL_DIR).to(DEVICE)
    base.config.pad_token_id = tok.pad_token_id
    evaluate(base, tok, "TRUOC KHI SFT")
    del base
    print("\n" + "=" * 72)
    print("BAI 3 - Train (masking BAT, chuan SFT)")
    print("=" * 72)
    train_items = [encode_example(d, tok, mask_prompt=True)
                   for d in train_data]
    model_masked, _ = train_model(tok, train_items, "masking-BAT")
    evaluate(model_masked, tok, "SAU KHI SFT (masking BAT)")
    del model_masked
    print("\n" + "=" * 72)
    print("ABLATION - Train (masking TAT, chi doi 1 bien)")
    print("=" * 72)
    train_items_nomask = [encode_example(d, tok, mask_prompt=False)
                          for d in train_data]
    model_nomask, _ = train_model(tok, train_items_nomask,
                                  "masking-TAT")
    evaluate(model_nomask, tok, "SAU KHI TRAIN (masking TAT)")
    print("\n" + "=" * 72)
    print("KET LUAN NHANH")
    print("=" * 72)
    print("- Masking BAT: loss chi tinh tren dap an, model hoc "
          "'dap an roi dung'.")
    print("- Masking TAT: loss tinh ca prompt, model de sinh tiep "
          "cau hoi / lap khuon mau.")
    print("- Template train va sinh giong het nhau (cung ham "
          "format_prompt).")


if __name__ == "__main__":
    main()
