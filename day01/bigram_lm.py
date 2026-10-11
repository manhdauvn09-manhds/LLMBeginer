# bigram_lm.py — Bigram Language Model (char-level), Ngày 1 buổi tối
import argparse
import math
import sys
import time
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F

torch.manual_seed(1337)
device = 'cuda' if torch.cuda.is_available() else 'cpu'

# ---------- Dữ liệu ----------
text = open('corpus.txt', encoding='utf-8').read()
chars = sorted(set(text))
vocab_size = len(chars)
stoi = {ch: i for i, ch in enumerate(chars)}
itos = {i: ch for i, ch in enumerate(chars)}
encode = lambda s: [stoi[c] for c in s]
decode = lambda l: ''.join([itos[i] for i in l])

data = torch.tensor(encode(text), dtype=torch.long)
n = int(0.9 * len(data))
train_data, val_data = data[:n], data[n:]
print(f"vocab={vocab_size}, train tokens={len(train_data)}, val tokens={len(val_data)}")

batch_size, block_size = 32, 16   # block_size = độ dài context (bigram dùng 1, nhưng giữ 16 để Bước 4)

def get_batch(split):
    # TODO (Bước 2): trả về x, y trên đúng device.
    # x: (batch_size, block_size) token đầu vào
    # y: (batch_size, block_size) token mục tiêu = x dịch phải 1 vị trí
    """Bước 2: trả về x, y có shape (batch_size, block_size)."""
    if split not in ("train", "val"):
        raise ValueError("split phải là 'train' hoặc 'val'.")
    data = train_data if split == "train" else val_data
    if len(data) <= block_size:
        raise ValueError(f"Tập {split} cần ít nhất {block_size + 1} token.")

    # high bị loại trừ: i lớn nhất là len(data) - block_size - 1.
    starts = torch.randint(len(data) - block_size, (batch_size,))
    x = torch.stack([data[i:i + block_size] for i in starts.tolist()])
    y = torch.stack([data[i + 1:i + block_size + 1] for i in starts.tolist()])
    return x.to(device), y.to(device)

# ---------- Mô hình ----------
class BigramLM(nn.Module):
    def __init__(self, vocab_size):
        super().__init__()
        # TODO (Bước 3): tạo bảng embedding vocab_size x vocab_size
        self.token_embedding_table = nn.Embedding(vocab_size, vocab_size)

    def forward(self, idx, targets=None):
        # TODO (Bước 3): logits = embedding(idx); nếu có targets thì tính loss
        logits = self.token_embedding_table(idx)  # (B, T, V)
        loss = None
        if targets is not None:
            if targets.shape != idx.shape:
                raise ValueError("targets phải có cùng shape với idx.")
            # Cross entropy nhận logits thô, không softmax trước.
            loss = F.cross_entropy(
                logits.reshape(-1, logits.size(-1)),
                targets.reshape(-1),
            )
        return logits, loss

    def generate(self, idx, max_new_tokens, temperature=1.0):
        # TODO (Bước 5): sinh token mới bằng sampling với temperature
        if not math.isfinite(temperature) or temperature <= 0:
            raise ValueError("temperature phải hữu hạn và > 0.")
        if max_new_tokens < 0:
            raise ValueError("max_new_tokens phải >= 0.")
        if idx.ndim != 2 or idx.size(1) == 0:
            raise ValueError("idx phải có shape (B, T), với T >= 1.")

        for _ in range(max_new_tokens):
            # Bigram chỉ phụ thuộc token cuối cùng, không cần cả lịch sử.
            logits, _ = self(idx[:, -1:])
            logits = logits[:, -1, :] / temperature  # (B, V)
            probabilities = F.softmax(logits, dim=-1)
            next_token = torch.multinomial(probabilities, num_samples=1)
            idx = torch.cat((idx, next_token), dim=1)
        return idx

@torch.no_grad()
def estimate_val_loss(model, eval_iters=20):
    """Đánh giá trung bình nhiều batch, không tính gradient."""
    if eval_iters <= 0:
        raise ValueError("eval_iters phải > 0.")
    was_training = model.training
    model.eval()
    try:
        losses = []
        for _ in range(eval_iters):
            x, y = get_batch("val")
            _, loss = model(x, y)
            losses.append(loss.item())
        return sum(losses) / len(losses)
    finally:
        model.train(was_training)


def peak_ram_mib():
    """Peak RAM của tiến trình, tính bằng MiB, do hệ điều hành ghi nhận."""
    if sys.platform == "win32":
        import psutil
        return psutil.Process().memory_info().peak_wset / (1024 ** 2)
    import resource
    peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    # macOS trả bytes; Linux trả KiB.
    return peak / (1024 ** 2 if sys.platform == "darwin" else 1024)


def synchronize_device():
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def main():
    global batch_size, block_size, device, train_data, val_data

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input", type=Path, help="File văn bản UTF-8.")
    parser.add_argument("--steps", type=int, default=5000)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--block-size", type=int, default=32)
    parser.add_argument("--eval-interval", type=int, default=200)
    parser.add_argument("--eval-iters", type=int, default=20)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    args = parser.parse_args()
    for name in ("steps", "batch_size", "block_size", "eval_interval", "eval_iters"):
        if getattr(args, name) <= 0:
            parser.error(f"--{name.replace('_', '-')} phải > 0.")

    torch.manual_seed(1337)
    batch_size, block_size = args.batch_size, args.block_size
    selected_device = args.device
    if selected_device == "auto":
        selected_device = "cuda" if torch.cuda.is_available() else "cpu"
    if selected_device == "cuda" and not torch.cuda.is_available():
        parser.error("CUDA không khả dụng; hãy dùng --device cpu.")
    device = torch.device(selected_device)

    # Phần dữ liệu bổ sung do không xuất hiện trong ảnh.
    try:
        text = args.input.read_text(encoding="utf-8")
    except (OSError, UnicodeError) as exc:
        parser.error(f"Không đọc được file UTF-8: {exc}")
    chars = sorted(set(text))
    stoi = {ch: i for i, ch in enumerate(chars)}
    vocab_size = len(chars)
    data = torch.tensor([stoi[ch] for ch in text], dtype=torch.long)
    split_index = int(0.9 * len(data))
    train_data, val_data = data[:split_index], data[split_index:]
    if min(len(train_data), len(val_data)) <= block_size:
        parser.error(
            f"Sau chia 90/10, mỗi tập cần ít nhất {block_size + 1} token. "
            "Hãy thêm văn bản hoặc giảm --block-size."
        )

    def decode(ids):
        return "".join(chars[i] for i in ids)

    model = BigramLM(vocab_size).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    print(f"device={device}, vocab_size={vocab_size}")

    # Bước 4: huấn luyện; mỗi 200 bước in train loss và đo val loss.
    model.train()
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    synchronize_device()
    started_at = time.perf_counter()
    running_loss = 0.0
    window_steps = 0
    best_val = float("inf")          # <-- thêm dòng này
    patience, no_improve = 5, 0   # thêm cạnh best_val = float("inf")
    for step in range(1, args.steps + 1):
        x, y = get_batch("train")
        _, loss = model(x, y)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
        window_steps += 1

        if step % args.eval_interval == 0 or step == args.steps:
            val_loss = estimate_val_loss(model, args.eval_iters)
            if val_loss < best_val:                          # <-- thêm từ đây
                best_val = val_loss
                torch.save(model.state_dict(), "best_bigram.pt")
                print(f"  ★ lưu checkpoint tốt nhất (val {best_val:.4f})")
                no_improve = 0  # có cải thiện -> đếm lại từ đầu
            else:
                no_improve += 1
                if no_improve >= patience:
                    print(f"Dừng sớm ở step {step} (val không cải thiện {patience} lần liên tiếp)")
                    break
            print(                                          # <-- đến đây
                f"step {step:4d}/{args.steps} | "
                f"train loss {running_loss / window_steps:.4f} | "
                f"val loss {val_loss:.4f}"
            )
            running_loss = 0.0
            window_steps = 0

    synchronize_device()
    elapsed_seconds = time.perf_counter() - started_at
    steps_done = step  # số step chạy thật (kể cả khi dừng sớm)
    model.load_state_dict(torch.load("best_bigram.pt", weights_only=True))  # <-- thêm dòng này
    training_peak_ram = peak_ram_mib()                       # <-- thêm lại
    peak_vram = (                                            # <-- thêm lại
        torch.cuda.max_memory_allocated(device) / (1024 ** 2)
        if device.type == "cuda" else None
    )

    # Bước 6: sinh 200 token mới với mỗi temperature được yêu cầu.
    model.eval()
    for temperature in (0.7, 1.0, 1.3):
        context = torch.zeros((1, 1), dtype=torch.long, device=device)
        generated = model.generate(context, 200, temperature=temperature)
        print(f"\n--- temperature={temperature} ---")
        print(decode(generated[0].tolist()))  # Gồm 1 token đầu + 200 token mới.

    # Bước 7: thống kê, chụp trước phần sinh văn bản.
    print("\n--- Green checkpoint ---")
    print(f"Thời gian vòng huấn luyện (gồm validation/log): {elapsed_seconds:.2f} s")
    print(f"Số bước optimizer: {steps_done}")
    print(f"Peak RAM tiến trình từ khởi động đến hết train: {training_peak_ram:.2f} MiB")
    if peak_vram is not None:
        print(f"Peak VRAM tensor do PyTorch cấp phát khi train: {peak_vram:.2f} MiB")


if __name__ == "__main__":
    main()