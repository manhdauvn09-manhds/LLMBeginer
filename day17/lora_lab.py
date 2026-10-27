#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Lab ablation LoRA r = 4 / 8 / 16 (Day 17 - buoi toi).

Bai 0: ra dataset truoc khi ablation - stratified split theo kind,
    audit prompt danh gia (phai chua tung thay), masking -100 tren
    MOI mau.
Bai 1: gan LoRA (target c_attn), dem tham so train duoc theo rank,
    assert W goc bi dong bang.
Bai 2: ablation r = 4 / 8 / 16 - CHI rank thay doi, moi thu khac
    giu nguyen: cung dataset, split, seed, epoch, LR, decoding.
    (Quy tac "2 model bang tham so" cua tuan truoc KHONG ap dung
    o day vi rank chinh la bien doc lap.)
Bai 3: merge adapter cua rank tot nhat (theo val loss), kiem tra
    dau ra truoc/sau merge giong nhau, luu adapter, do dung luong.
Mo rong: ablation target modules (c_attn vs c_attn + c_proj) o
    rank tot nhat - ablation RIENG, khong gop vao bang rank.

Moi truong: CPU-only, GPT-2 124M, float32. Template co dinh:
    Cau hoi: {instruction}\nTra loi: {output}<|endoftext|>
Du lieu: 32 mau, 8 mau moi kind (qa, summary, email, translate),
    gom 14 mau Day 16 + 18 mau moi. DU LIEU VIET CO DAU DAY DU
    (bai hoc Day 15: tieng Viet khong dau lam model hieu sai).

Cach chay (dung venv Day 15):
    ~/workspace/venvs/day15/bin/python lora_lab.py
"""

import gc
import os
import random
import time
from collections import Counter

import torch
from torch.utils.data import DataLoader, Dataset
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import LoraConfig, get_peft_model

MODEL_DIR = os.path.expanduser(
    "~/.cache/modelscope/models/openai-community--gpt2/"
    "snapshots/master")
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
SEED = 42
EPOCHS = 2
LEARNING_RATE = 1e-4
BATCH_SIZE = 4
MAX_NEW_TOKENS = 30
ADAPTER_DIR = os.path.expanduser("~/workspace/daily/day17/adapter_best")

# 32 mau: 8 moi kind. Kind: qa, summary, email, translate.
DATA = [
    # qa (5 mau Day 16 + 3 moi).
    {"kind": "qa", "instruction": "Thủ đô của Việt Nam là gì?",
     "output": "Hà Nội."},
    {"kind": "qa", "instruction": "Thủ đô của Nhật Bản là gì?",
     "output": "Tokyo."},
    {"kind": "qa", "instruction": "Thủ đô của Pháp là gì?",
     "output": "Paris."},
    {"kind": "qa", "instruction": "Một tuần có mấy ngày?",
     "output": "Một tuần có 7 ngày."},
    {"kind": "qa", "instruction": "Nước sôi ở bao nhiêu độ C?",
     "output": "Nước sôi ở 100 độ C."},
    {"kind": "qa", "instruction": "Thủ đô của Ý là gì?",
     "output": "Rome."},
    {"kind": "qa", "instruction": "Một năm có mấy tháng?",
     "output": "Một năm có 12 tháng."},
    {"kind": "qa", "instruction": "Thủ đô của Anh là gì?",
     "output": "London."},
    # summary (3 mau Day 16 + 5 moi).
    {"kind": "summary", "instruction": "Tóm tắt bằng một câu: Mèo là loài "
     "vật nuôi phổ biến, hiền lành và thích ngủ nhiều.",
     "output": "Mèo là loài vật nuôi hiền lành, thích ngủ nhiều."},
    {"kind": "summary", "instruction": "Tóm tắt bằng một câu: Chó là loài "
     "vật trung thành, thông minh và là bạn thân của con người.",
     "output": "Chó là loài vật trung thành và thông minh."},
    {"kind": "summary", "instruction": "Tóm tắt bằng một câu: Sách giúp con "
     "người mở rộng kiến thức, rèn tư duy và nuôi dưỡng tâm hồn.",
     "output": "Sách giúp mở rộng kiến thức và rèn tư duy."},
    {"kind": "summary", "instruction": "Tóm tắt bằng một câu: Hoa sen là "
     "quốc hoa của Việt Nam, tượng trưng cho sự thanh khiết và "
     "vươn lên từ bùn lầy.",
     "output": "Hoa sen tượng trưng cho sự thanh khiết và vươn lên."},
    {"kind": "summary", "instruction": "Tóm tắt bằng một câu: Biển là nơi "
     "sinh sống của nhiều loài sinh vật, điều hòa khí hậu và mang "
     "lại nguồn lợi thủy sản.",
     "output": "Biển điều hòa khí hậu và mang lại nguồn lợi thủy sản."},
    {"kind": "summary", "instruction": "Tóm tắt bằng một câu: Máy tính giúp "
     "con người xử lý thông tin nhanh, lưu trữ dữ liệu lớn và kết "
     "nối với nhau qua mạng.",
     "output": "Máy tính giúp xử lý thông tin nhanh và kết nối qua mạng."},
    {"kind": "summary", "instruction": "Tóm tắt bằng một câu: Thể dục mỗi "
     "ngày giúp cơ thể khỏe mạnh, tinh thần sảng khoái và phòng "
     "tránh nhiều bệnh tật.",
     "output": "Thể dục mỗi ngày giúp cơ thể khỏe mạnh và phòng bệnh."},
    {"kind": "summary", "instruction": "Tóm tắt bằng một câu: Du lịch giúp "
     "con người khám phá văn hóa mới, mở rộng tầm nhìn và thư giãn "
     "sau thời gian làm việc.",
     "output": "Du lịch giúp khám phá văn hóa mới và thư giãn."},
    # email (3 mau Day 16 + 5 moi).
    {"kind": "email", "instruction": "Viết một câu chào buổi sáng lịch sự.",
     "output": "Chào buổi sáng, chúc bạn một ngày làm việc hiệu quả."},
    {"kind": "email", "instruction": "Viết email xin nghỉ phép 1 ngày gửi "
     "quản lý.",
     "output": "Chào anh, em xin phép nghỉ làm ngày mai vì việc gia đình. "
     "Em sẽ bàn giao công việc cho đồng nghiệp trước khi nghỉ. "
     "Cảm ơn anh đã thông cảm."},
    {"kind": "email", "instruction": "Viết email cảm ơn khách hàng sau "
     "buổi họp.",
     "output": "Cảm ơn quý khách đã dành thời gian họp với chúng tôi hôm nay. "
     "Chúng tôi sẽ gửi báo giá chi tiết trong tuần này. Rất mong "
     "được hợp tác lâu dài."},
    {"kind": "email", "instruction": "Viết email hẹn gặp đối tác vào tuần "
     "tới.",
     "output": "Chào anh, tôi muốn hẹn gặp anh vào tuần tới để trao đổi về "
     "hợp tác. Anh cho tôi biết thời gian phù hợp nhé. Cảm ơn anh."},
    {"kind": "email", "instruction": "Viết email thông báo họp nhóm vào "
     "sáng thứ Hai.",
     "output": "Chào mọi người, nhóm sẽ họp vào sáng thứ Hai để chốt kế hoạch "
     "tuần. Mọi người chuẩn bị báo cáo tiến độ trước buổi họp. "
     "Cảm ơn mọi người."},
    {"kind": "email", "instruction": "Viết email chúc mừng sinh nhật đồng "
     "nghiệp.",
     "output": "Chúc mừng sinh nhật bạn! Chúc bạn luôn khỏe mạnh, vui vẻ và "
     "gặt hái nhiều thành công trong công việc. Cả nhóm sẽ tổ chức "
     "một buổi tiệc nhỏ vào chiều nay."},
    {"kind": "email", "instruction": "Viết email xác nhận đã nhận được đơn "
     "hàng.",
     "output": "Cảm ơn quý khách đã đặt hàng. Chúng tôi đã nhận được đơn hàng "
     "và sẽ xử lý trong 24 giờ tới. Quý khách sẽ nhận được thông "
     "báo khi hàng được giao đi."},
    {"kind": "email", "instruction": "Viết email xin lỗi vì trả lời muộn.",
     "output": "Xin lỗi anh vì tôi trả lời muộn do công việc bận. Tôi đã xem "
     "kỹ yêu cầu của anh và sẽ gửi phản hồi chi tiết ngay hôm nay. "
     "Cảm ơn anh đã thông cảm."},
    # translate (3 mau Day 16 + 5 moi).
    {"kind": "translate", "instruction": "Dịch sang tiếng Anh: Tôi yêu Việt "
     "Nam.",
     "output": "I love Vietnam."},
    {"kind": "translate", "instruction": "Dịch sang tiếng Anh: Cảm ơn bạn rất "
     "nhiều.",
     "output": "Thank you very much."},
    {"kind": "translate", "instruction": "Dịch sang tiếng Việt: Good morning.",
     "output": "Chào buổi sáng."},
    {"kind": "translate", "instruction": "Dịch sang tiếng Anh: Tôi đang học "
     "tiếng Nhật.",
     "output": "I am learning Japanese."},
    {"kind": "translate", "instruction": "Dịch sang tiếng Anh: Hẹn gặp lại "
     "ngày mai.",
     "output": "See you tomorrow."},
    {"kind": "translate",
     "instruction": "Dịch sang tiếng Việt: I like coffee.",
     "output": "Tôi thích cà phê."},
    {"kind": "translate", "instruction": "Dịch sang tiếng Việt: How are you?",
     "output": "Bạn khỏe không?"},
    {"kind": "translate", "instruction": "Dịch sang tiếng Anh: Chúc bạn một "
     "ngày tốt lành.",
     "output": "Have a nice day."},
]

# 6 prompt danh gia: CHUA TUNG THAY trong train (kiem chung o Bai 0).
EVAL_PROMPTS = [
    "Câu hỏi: Thủ đô của Hàn Quốc là gì?\nTrả lời: ",
    "Câu hỏi: Thủ đô của Đức là gì?\nTrả lời: ",
    "Câu hỏi: Tóm tắt bằng một câu: Hoa hồng là loài hoa đẹp, có "
    "hương thơm và thường được tặng trong các dịp lễ.\nTrả lời: ",
    "Câu hỏi: Viết email xin lỗi khách hàng vì giao hàng trễ."
    "\nTrả lời: ",
    "Câu hỏi: Dịch sang tiếng Anh: Hôm nay trời đẹp.\nTrả lời: ",
    "Câu hỏi: Dịch sang tiếng Việt: Thank you for your help."
    "\nTrả lời: ",
]


def format_prompt(instruction):
    """Rap prompt theo template co dinh cua lab."""
    return f"Câu hỏi: {instruction}\nTrả lời: "


def encode_example(item, tok):
    """Ma hoa 1 mau SFT: labels che prompt bang -100.

    Raises:
        ValueError: neu instruction hoac output rong.
    """
    if not item["instruction"] or not item["output"]:
        raise ValueError("instruction va output khong duoc rong.")
    prompt = format_prompt(item["instruction"])
    answer = item["output"] + tok.eos_token
    prompt_ids = tok(prompt, add_special_tokens=False)["input_ids"]
    answer_ids = tok(answer, add_special_tokens=False)["input_ids"]
    input_ids = prompt_ids + answer_ids
    labels = [-100] * len(prompt_ids) + answer_ids
    return input_ids, labels


def stratified_split(data, seed=SEED, val_frac=0.2):
    """Chia train/val phan tang theo kind (80/20 moi kind).

    Xao tron co seed co dinh trong tung kind. Kind co < 5 mau
    thi khong lay val (tranh val chi co 1 mau dai dien kem).
    """
    rng = random.Random(seed)
    by_kind = {}
    for item in data:
        by_kind.setdefault(item["kind"], []).append(item)
    train, val = [], []
    for kind, items in by_kind.items():
        items = items[:]
        rng.shuffle(items)
        if len(items) >= 5:
            n_val = max(1, int(len(items) * val_frac))
        else:
            n_val = 0
        val.extend(items[:n_val])
        train.extend(items[n_val:])
    return train, val


def audit_eval_prompts(eval_prompts, train):
    """Prompt danh gia phai chua tung thay trong train.

    Kiem tra: khong prompt nao chua nguyen van 1 instruction
    cua train (neu khong, diem cao chi la hoc thuoc).
    """
    train_instructions = {d["instruction"] for d in train}
    for p in eval_prompts:
        for ins in train_instructions:
            assert ins not in p, f"Prompt danh gia trung train: {ins}"
    print(f"Eval audit OK: {len(eval_prompts)} prompt deu chua tung "
          f"thay trong train.")


def verify_masking_all(tok):
    """Kiem chung masking -100 tren MOI mau (khong chi mau 0)."""
    for i, item in enumerate(DATA):
        ids, labels = encode_example(item, tok)
        prompt_len = len(tok(format_prompt(item["instruction"]),
                             add_special_tokens=False)["input_ids"])
        assert all(l == -100 for l in labels[:prompt_len]), \
            f"Mau {i} chua che het prompt!"
        assert labels[prompt_len:] == ids[prompt_len:], \
            f"Mau {i} label dap an sai!"
    print(f"Masking OK tren tat ca {len(DATA)} mau.")


class SFTDataset(Dataset):
    """Dataset SFT: encode san toan bo mau luc khoi tao."""

    def __init__(self, items, tok):
        """Args: items la list dict mau, tok la tokenizer."""
        self.items = [encode_example(d, tok) for d in items]

    def __len__(self):
        """Tra ve so mau."""
        return len(self.items)

    def __getitem__(self, i):
        """Tra ve dict input_ids/labels cua mau thu i."""
        ids, labels = self.items[i]
        return {"input_ids": ids, "labels": labels}


def collate(batch, pad_id):
    """Dem batch ve cung do dai: pad token + nhan -100."""
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


def count_trainable(model):
    """Tra ve (so tham so train duoc, tong so tham so)."""
    trainable = sum(p.numel() for p in model.parameters()
                    if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    return trainable, total


def build_lora_model(tok, rank, target_modules):
    """Load GPT-2 moi tinh, gan LoRA, assert W goc bi dong bang.

    Ti le alpha/r giu co dinh = 2 (alpha = 2r) trong toan bo lab.
    """
    torch.manual_seed(SEED)
    base = AutoModelForCausalLM.from_pretrained(MODEL_DIR).to(DEVICE)
    base.config.pad_token_id = tok.pad_token_id
    config = LoraConfig(r=rank, lora_alpha=2 * rank, lora_dropout=0.05,
                        target_modules=target_modules,
                        task_type="CAUSAL_LM")
    model = get_peft_model(base, config)
    trainable, total = count_trainable(model)
    print(f"  r={rank} {target_modules}: train duoc {trainable:,} / "
          f"{total:,} tham so ({trainable / total:.2%})")
    # Assert: chi tham so LoRA (ten chua 'lora_') duoc train.
    bad = [n for n, p in model.named_parameters()
           if p.requires_grad and "lora_" not in n]
    assert not bad, f"W goc chua dong bang: {bad[:3]}"
    return model


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


def eval_all(model, tok, title):
    """Sinh tren EVAL_PROMPTS, tra ve list dau ra (de doi chieu)."""
    print(f"===== {title} =====")
    outputs = []
    for p in EVAL_PROMPTS:
        text = generate(model, tok, p)
        outputs.append(text)
        print("PROMPT:", p.replace("\n", " / "))
        print("->", text)
    return outputs


def train_one_rank(tok, rank, train_data, val_data, target_modules):
    """Train 1 rank, tra ve dict (model, train_loss, val_loss, giay).

    Moi thu co dinh ngoai rank: seed, epoch, LR, batch, decoding.
    """
    model = build_lora_model(tok, rank, target_modules)
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
        print(f"  r={rank} epoch {epoch + 1}: train loss = "
              f"{last_train_loss:.4f}")
    dt = time.perf_counter() - t0
    vl = val_loss(model, tok, val_data)
    print(f"  r={rank}: val loss cuoi = {vl:.4f} | train trong "
          f"{dt:.1f}s.")
    return {"model": model, "train_loss": last_train_loss,
            "val_loss": vl, "seconds": dt}


def adapter_size_mb(model, save_dir):
    """Luu adapter va tra ve dung luong (MB) cua thu muc."""
    model.save_pretrained(save_dir)
    total = 0
    for root, _, files in os.walk(save_dir):
        for f in files:
            total += os.path.getsize(os.path.join(root, f))
    return total / 1e6


def main():
    """Chay lab day du: Bai 0 -> Bai 1 -> Bai 2 -> Bai 3 -> mo rong."""
    print(f"torch {torch.__version__} | device: {DEVICE}")
    tok = AutoTokenizer.from_pretrained(MODEL_DIR)
    tok.pad_token = tok.eos_token
    print("\n" + "=" * 72)
    print("BAI 0 - Ra dataset truoc khi ablation")
    print("=" * 72)
    train_data, val_data = stratified_split(DATA)
    print("Train:", len(train_data),
          Counter(d["kind"] for d in train_data))
    print("Val:  ", len(val_data), Counter(d["kind"] for d in val_data))
    assert set(d["kind"] for d in val_data) == \
        set(d["kind"] for d in train_data), \
        "Val thieu kind so voi train!"
    audit_eval_prompts(EVAL_PROMPTS, train_data)
    verify_masking_all(tok)
    print("\n" + "=" * 72)
    print("BAI 1 - Dem tham so theo rank (chua train)")
    print("=" * 72)
    print("Cong thuc: r x (768 + 2304) x 12 lop = r x 36864.")
    for rank in (4, 8, 16):
        model = build_lora_model(tok, rank, ["c_attn"])
        del model
        gc.collect()
    print("\n" + "=" * 72)
    print("BAI 2 - Ablation r = 4 / 8 / 16 (chi doi rank)")
    print("=" * 72)
    print(f"Co dinh: seed={SEED}, epoch={EPOCHS}, lr={LEARNING_RATE}, "
          f"batch={BATCH_SIZE}, alpha/r=2, target=c_attn, greedy.")
    results = {}
    for rank in (4, 8, 16):
        print(f"--- Rank {rank} ---")
        res = train_one_rank(tok, rank, train_data, val_data,
                             ["c_attn"])
        trainable, _ = count_trainable(res["model"])
        res["trainable"] = trainable
        res["eval_outputs"] = eval_all(
            res["model"], tok, f"EVAL r={rank} (c_attn)")
        results[rank] = res
    print("\n" + "=" * 72)
    print("BANG ABLATION")
    print("=" * 72)
    for rank, res in results.items():
        print(f"r={rank}: trainable={res['trainable']:,} | "
              f"train loss={res['train_loss']:.4f} | "
              f"val loss={res['val_loss']:.4f} | "
              f"{res['seconds']:.1f}s")
    best_rank = min(results, key=lambda r: results[r]["val_loss"])
    print(f"Rank tot nhat theo val loss: r={best_rank}.")
    print("\n" + "=" * 72)
    print(f"BAI 3 - Merge adapter r={best_rank} va kiem tra")
    print("=" * 72)
    best_model = results[best_rank]["model"]
    size_mb = adapter_size_mb(best_model, ADAPTER_DIR)
    print(f"Dung luong adapter r={best_rank}: {size_mb:.2f} MB "
          f"(so voi ~500 MB cua GPT-2 day du).")
    merged = best_model.merge_and_unload()
    merged_outputs = eval_all(merged, tok,
                              f"EVAL r={best_rank} SAU MERGE")
    same = sum(1 for a, b in zip(results[best_rank]["eval_outputs"],
                                 merged_outputs) if a == b)
    print(f"Merge check: {same}/{len(EVAL_PROMPTS)} dau ra giong "
          f"het truoc/sau merge.")
    print("\n" + "=" * 72)
    print("MO RONG - Ablation target modules (c_attn + c_proj)")
    print("=" * 72)
    print("Chi doi target modules, giu nguyen rank tot nhat, seed, "
          "epoch, LR, dataset.")
    res_ext = train_one_rank(tok, best_rank, train_data, val_data,
                             ["c_attn", "c_proj"])
    trainable_ext, _ = count_trainable(res_ext["model"])
    print(f"r={best_rank} c_attn+c_proj: trainable={trainable_ext:,} | "
          f"train loss={res_ext['train_loss']:.4f} | "
          f"val loss={res_ext['val_loss']:.4f} | "
          f"{res_ext['seconds']:.1f}s (so voi c_attn: "
          f"val loss={results[best_rank]['val_loss']:.4f}).")
    eval_all(res_ext["model"], tok,
             f"EVAL r={best_rank} (c_attn+c_proj)")
    print("\n" + "=" * 72)
    print("KET LUAN NHANH")
    print("=" * 72)
    print(f"- Rank tot nhat (val loss): r={best_rank}.")
    print("- Merge khong doi dau ra (hoac chi lech do lam tron).")
    print("- Adapter chi vai MB so voi 500 MB model day du.")


if __name__ == "__main__":
    main()
