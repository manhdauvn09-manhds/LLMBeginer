# -*- coding: utf-8 -*-
# bigram_lm.py — Bigram Language Model ở mức ký tự (char-level), viết bằng PyTorch.
# Bài thực hành tối Ngày 1, khóa "Trở Thành AI Engineer".
#
# Ý tưởng trong một câu: dạy máy một trò duy nhất — nhìn MỘT ký tự,
# đoán ký tự TIẾP THEO — bằng cách học thuộc một "bảng điểm" V×V, trong đó
# V là số ký tự khác nhau trong văn bản.
#
# Chạy thử:
#   python bigram_lm.py corpus.txt --steps 5000 --batch-size 32 --block-size 16
#
# Luồng chương trình (7 bước):
#   1. Đọc văn bản, biến chữ thành số (encode)            -> def main
#   2. Bốc batch ngẫu nhiên: x = đề bài, y = đáp án      -> def get_batch
#   3. Model = bảng điểm V×V, tra cứu là ra dự đoán       -> class BigramLM
#   4. Lặp: đoán -> chấm điểm (loss) -> sửa bảng điểm     -> def main
#   5. Sinh văn bản mới bằng "tung xúc xắc có gian lận"   -> BigramLM.generate
#   6. Temperature: núm vặn "độ liều" khi sáng tạo        -> def main
#   7. Báo cáo thời gian / RAM (green checkpoint)         -> def main

import argparse   # đọc tham số dòng lệnh: --steps, --batch-size, ...
import math       # kiểm tra temperature có phải số hợp lệ không
import sys        # biết đang chạy trên Windows/Linux/macOS (để đo RAM đúng cách)
import time       # bấm giờ huấn luyện (green checkpoint)

from pathlib import Path  # đọc file văn bản theo cách hiện đại, an toàn

import torch
import torch.nn as nn
import torch.nn.functional as F

# "Gieo hạt giống" cho bộ số ngẫu nhiên: mọi lần chạy, các số ngẫu nhiên
# (vị trí batch bốc được, kết quả tung xúc xắc...) đều giống hệt nhau.
# Nhờ vậy thí nghiệm lặp lại được, dễ so sánh và dễ gỡ lỗi.
torch.manual_seed(1337)

# Thiết bị tính toán mặc định: có GPU (cuda) thì dùng cho nhanh,
# không có thì dùng CPU. main() cho phép ghi đè bằng --device.
device = "cuda" if torch.cuda.is_available() else "cpu"

# Các biến toàn cục dưới đây do main() khởi tạo từ tham số dòng lệnh.
# Để chúng ở đây (thay vì đọc file ngay khi import) để ai đó
# "import bigram_lm" từ file khác không bị chạy ké cả chương trình.
train_data = None   # 90% dữ liệu: "sách giáo khoa" để học
val_data = None     # 10% dữ liệu: "đề thi thử" để kiểm tra học vẹt
batch_size = 32     # mỗi bước học 32 đoạn văn một lúc
block_size = 32     # mỗi đoạn dài bao nhiêu ký tự


def get_batch(split):
    """Bốc một batch ngẫu nhiên, trả về (x, y) đã nằm trên đúng device.

    Quy ước của học máy ở đây rất gọn:
      - x[b] : "đề bài"  — một đoạn block_size ký tự đưa cho model đoán.
      - y[b] : "đáp án"  — chính đoạn đó nhưng DỊCH PHẢI 1 vị trí:
               đáp án của ký tự x[b,t] là ký tự thật đứng ngay sau nó.
    Không cần ai ra đề: đáp án nằm sẵn trong văn bản, chỉ việc dịch phải 1 ô.
    """
    if split not in ("train", "val"):
        raise ValueError("split phải là 'train' hoặc 'val'.")
    # Dùng tên split_data để không đè lên biến toàn cục (tránh nhầm lẫn).
    split_data = train_data if split == "train" else val_data
    if len(split_data) <= block_size:
        raise ValueError(f"Tập {split} cần ít nhất {block_size + 1} token.")

    # Bốc batch_size vị trí bắt đầu ngẫu nhiên.
    # high của randint bị LOẠI TRỪ, nên start lớn nhất = len - block_size - 1,
    # đảm bảo y (lấy tới start+block_size) không bao giờ tràn khỏi mảng.
    starts = torch.randint(len(split_data) - block_size, (batch_size,))

    # Kỹ thuật "broadcast indexing": từ mỗi start, lấy block_size token liên tiếp
    # trong MỘT phép toán duy nhất (chạy bằng C, nhanh hơn vòng lặp Python nhiều).
    #   starts[:, None] : (B,1)  +  arange(block_size) : (T,)  ->  (B,T) chỉ số.
    offsets = torch.arange(block_size)
    x = split_data[starts[:, None] + offsets]      # (B, T): đề bài
    y = split_data[starts[:, None] + offsets + 1]  # (B, T): đáp án = x dịch phải 1
    return x.to(device), y.to(device)


class BigramLM(nn.Module):
    """Toàn bộ "bộ não": một bảng điểm hình vuông V×V.

    - Hàng i  = ký tự hiện tại.
    - Cột j  = điểm số (logit) cho khả năng "ký tự j đứng ngay sau ký tự i".
    - Ô nào điểm cao nghĩa là "cặp này hay đi với nhau trong văn bản thật".

    Ban đầu mọi ô là số ngẫu nhiên; huấn luyện sẽ sửa dần các ô cho khôn ra.
    """

    def __init__(self, vocab_size):
        super().__init__()
        # nn.Embedding ở đây được "tái sử dụng" một cách khéo léo:
        # thay vì nhúng từ thành vector ngữ nghĩa, ta nhúng mỗi ký tự
        # thành MỘT HÀNG ĐIỂM dài V — chính là bảng điểm V×V cần tìm.
        self.token_embedding_table = nn.Embedding(vocab_size, vocab_size)

    def forward(self, idx, targets=None):
        """Một lượt "suy nghĩ": đưa chữ vào -> nhận hàng điểm số.

        idx     : (B, T) các ký tự đầu vào (dạng số).
        targets : (B, T) đáp án thật (nếu có) để chấm điểm.
        Trả về (logits, loss):
          - logits: (B, T, V) — với mỗi vị trí, điểm số của mọi ký tự kế tiếp.
          - loss  : cross-entropy, "điểm phạt" khi đoán sai (None nếu không có targets).
        """
        # Tra bảng: mỗi ký tự -> đúng hàng điểm của nó. Không có phép màu nào khác:
        # bigram KHÔNG nhìn các ký tự trước đó, chỉ nhìn ký tự hiện tại.
        logits = self.token_embedding_table(idx)  # (B, T, V)
        loss = None
        if targets is not None:
            if targets.shape != idx.shape:
                raise ValueError("targets phải có cùng shape với idx.")
            # Cross-entropy nhận logits THÔ (chưa qua softmax) — nó tự xử lý
            # bên trong một cách ổn định số học. Đừng softmax trước rồi mới đưa vào!
            # reshape (-1, V): gộp mọi vị trí của mọi đoạn thành một danh sách dự đoán.
            loss = F.cross_entropy(
                logits.reshape(-1, logits.size(-1)),
                targets.reshape(-1),
            )
        return logits, loss

    def generate(self, idx, max_new_tokens, temperature=1.0):
        """Viết văn bản mới, mỗi vòng "đẻ" thêm đúng 1 ký tự.

        Vì là bigram nên mỗi vòng CHỈ NHÌN ký tự cuối cùng (idx[:, -1:]) —
        cả đoạn văn đã viết trước đó bị "quên" sạch. Đó chính là
        "trí nhớ cá vàng" mà tài liệu giải thích đã nói.
        """
        if not math.isfinite(temperature) or temperature <= 0:
            raise ValueError("temperature phải hữu hạn và > 0.")
        if max_new_tokens < 0:
            raise ValueError("max_new_tokens phải >= 0.")
        if idx.ndim != 2 or idx.size(1) == 0:
            raise ValueError("idx phải có shape (B, T), với T >= 1.")

        for _ in range(max_new_tokens):
            # 1. Tra bảng điểm của ký tự cuối cùng. Kết quả đã có shape (B, 1, V)
            #    nên lấy luôn [:, 0, :] cho gọn.
            logits, _ = self(idx[:, -1:])
            logits = logits[:, 0, :] / temperature  # (B, V)

            # 2. Softmax: biến điểm số thành xác suất (%). Tổng cả hàng = 100%.
            probabilities = F.softmax(logits, dim=-1)

            # 3. "Tung xúc xắc có gian lận": bốc ngẫu nhiên theo ĐÚNG tỉ lệ xác suất.
            #    Vì sao không chọn luôn ký tự điểm cao nhất? Vì khi đó văn bản sẽ
            #    lặp đi lặp lại một cách nhàm chán ("the the the..."). Tung xúc xắc
            #    giúp văn bản đa dạng mà vẫn "nghe có vẻ đúng": ký tự điểm cao trúng
            #    nhiều, ký tự điểm thấp thỉnh thoảng trúng.
            next_token = torch.multinomial(probabilities, num_samples=1)  # (B, 1)

            # 4. Dán ký tự mới vào cuối chuỗi, vòng sau đoán tiếp từ nó.
            idx = torch.cat((idx, next_token), dim=1)
        return idx


@torch.no_grad()
def estimate_val_loss(model, eval_iters=20):
    """Ước lượng loss trên tập val: trung bình nhiều batch, KHÔNG tính gradient.

    @torch.no_grad(): tắt hoàn toàn việc ghi nhận gradient -> chạy nhanh hơn
    và tốn ít RAM hơn. Bắt buộc khi đánh giá, vì ta không sửa model lúc này.
    Khối try/finally đảm bảo model được trả về đúng chế độ train ban đầu,
    kể cả khi giữa chừng có lỗi.
    """
    if eval_iters <= 0:
        raise ValueError("eval_iters phải > 0.")
    was_training = model.training
    model.eval()  # chuyển sang chế độ đánh giá (ở model này không khác gì train,
                  # nhưng là thói quen bắt buộc với các model có dropout/batchnorm)
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
    """Peak RAM của cả tiến trình (tính bằng MiB), do hệ điều hành ghi nhận.

    - Windows: dùng psutil (peak_wset).
    - Linux/macOS: dùng resource.getrusage (lưu ý đơn vị khác nhau:
      macOS trả bytes, Linux trả KiB).
    """
    if sys.platform == "win32":
        import psutil  # import lười: chỉ cần khi thật sự đo trên Windows
        return psutil.Process().memory_info().peak_wset / (1024 ** 2)
    import resource
    peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return peak / (1024 ** 2 if sys.platform == "darwin" else 1024)


def synchronize_device():
    """Trên GPU, các phép tính chạy BẤT ĐỒNG BỘ: lệnh bấm giờ ở Python có thể
    chạy trước khi GPU tính xong -> đo thời gian sẽ sai. Hàm này bắt CPU
    đứng chờ GPU làm xong hết rồi mới đo tiếp. Trên CPU thì không cần."""
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def main():
    # Các biến toàn cục (train_data, val_data, ...) được gán lại trong hàm này,
    # nên phải khai báo global để Python không tạo biến cục bộ mới.
    global batch_size, block_size, device, train_data, val_data

    # ---- Đọc tham số dòng lệnh ----
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input", type=Path, help="File văn bản UTF-8.")
    parser.add_argument("--steps", type=int, default=5000)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--block-size", type=int, default=32)
    parser.add_argument("--eval-interval", type=int, default=200)
    parser.add_argument("--eval-iters", type=int, default=20)
    parser.add_argument(
        "--patience", type=int, default=5,
        help="Dừng sớm nếu val loss không cải thiện sau bấy nhiêu lần eval liên tiếp.",
    )
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    args = parser.parse_args()
    for name in ("steps", "batch_size", "block_size",
                 "eval_interval", "eval_iters", "patience"):
        if getattr(args, name) <= 0:
            parser.error(f"--{name.replace('_', '-')} phải > 0.")

    # Kiểm tra sớm thư viện đo RAM trên Windows: thiếu thì báo NGAY từ đầu,
    # chứ đừng để train xong 5000 steps mới crash ở bước in báo cáo.
    if sys.platform == "win32":
        try:
            import psutil  # noqa: F401
        except ImportError:
            parser.error("Thiếu thư viện psutil — cài bằng: pip install psutil")

    # Gieo lại hạt giống để mỗi lần chạy từ dòng lệnh đều lặp lại được.
    torch.manual_seed(1337)
    batch_size, block_size = args.batch_size, args.block_size
    selected_device = args.device
    if selected_device == "auto":
        selected_device = "cuda" if torch.cuda.is_available() else "cpu"
    if selected_device == "cuda" and not torch.cuda.is_available():
        parser.error("CUDA không khả dụng; hãy dùng --device cpu.")
    device = torch.device(selected_device)

    # ---- Bước 1: đọc văn bản, biến chữ thành số (encode) ----
    try:
        text = args.input.read_text(encoding="utf-8")
    except (OSError, UnicodeError) as exc:
        parser.error(f"Không đọc được file UTF-8: {exc}")
    # "Danh bạ ký tự": liệt kê mọi ký tự khác nhau, xếp theo thứ tự, đánh số từ 0.
    chars = sorted(set(text))
    stoi = {ch: i for i, ch in enumerate(chars)}  # chữ -> số (encode)
    vocab_size = len(chars)
    # Cả cuốn sách giờ chỉ còn là một dãy số dài. Mọi bước sau (học, đoán,
    # sáng tạo) đều diễn ra trên dãy số này; cuối cùng mới dịch ngược thành chữ.
    data = torch.tensor([stoi[ch] for ch in text], dtype=torch.long)
    # Chia 90/10: 90% làm "sách giáo khoa", 10% làm "đề thi thử".
    # Đề thi thử giúp phát hiện "học vẹt": thuộc sách nhưng thi thử điểm kém.
    split_index = int(0.9 * len(data))
    train_data, val_data = data[:split_index], data[split_index:]
    if min(len(train_data), len(val_data)) <= block_size:
        parser.error(
            f"Sau chia 90/10, mỗi tập cần ít nhất {block_size + 1} token. "
            "Hãy thêm văn bản hoặc giảm --block-size."
        )

    def decode(ids):
        """Dịch ngược dãy số thành chữ để người đọc (chỉ dùng khi in kết quả)."""
        return "".join(chars[i] for i in ids)

    # ---- Bước 3: dựng model + bộ tối ưu ----
    model = BigramLM(vocab_size).to(device)
    # AdamW: "thợ sửa" khéo tay — điều chỉnh từng ô trong bảng điểm một chút
    # mỗi bước, dựa trên hướng giảm lỗi. lr=1e-3: mỗi lần sửa một chút thôi,
    # sửa mạnh quá dễ "phá" những gì đã học được.
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    print(f"device={device}, vocab_size={vocab_size}")

    # ---- Bước 4: huấn luyện — đoán -> chấm điểm -> sửa, lặp nhiều vòng ----
    model.train()
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    synchronize_device()  # bấm giờ chính xác: chờ GPU làm xong việc cũ
    started_at = time.perf_counter()
    running_loss = 0.0   # cộng dồn loss để lấy trung bình mỗi 200 bước
    window_steps = 0
    best_val = float("inf")      # val loss tốt nhất từng thấy (để lưu checkpoint)
    patience, no_improve = args.patience, 0  # bộ đếm cho early stopping
    for step in range(1, args.steps + 1):
        x, y = get_batch("train")   # 1. bốc đề ngẫu nhiên
        _, loss = model(x, y)       # 2. đoán + chấm điểm (trả về loss)
        optimizer.zero_grad(set_to_none=True)  # 3a. xóa gradient cũ (set_to_none: nhanh + nhẹ RAM hơn)
        loss.backward()             # 3b. lan truyền ngược: "truy tìm thủ phạm" —
                                    #     tính xem mỗi ô trong bảng phải sửa bao nhiêu thì lỗi giảm
        optimizer.step()            # 3c. AdamW sửa các ô đó một chút
        running_loss += loss.item()
        window_steps += 1

        if step % args.eval_interval == 0 or step == args.steps:
            val_loss = estimate_val_loss(model, args.eval_iters)
            if val_loss < best_val:
                # Có cải thiện: lưu lại "bản tốt nhất" và ĐẾM LẠI từ đầu.
                # (Quên dòng reset này là bug kinh điển: bộ đếm sẽ cộng dồn
                #  cả những lần không liên tiếp, dừng sớm oan.)
                best_val = val_loss
                torch.save(model.state_dict(), "best_bigram.pt")
                print(f"  ★ lưu checkpoint tốt nhất (val {best_val:.4f})")
                no_improve = 0
            else:
                # Không cải thiện: tăng bộ đếm; đủ "patience" lần LIÊN TIẾP
                # thì dừng — train tiếp chỉ tốn điện mà val không tốt lên.
                no_improve += 1
                if no_improve >= patience:
                    print(f"Dừng sớm ở step {step} "
                          f"(val không cải thiện {patience} lần liên tiếp)")
                    break
            print(
                f"step {step:4d}/{args.steps} | "
                f"train loss {running_loss / window_steps:.4f} | "
                f"val loss {val_loss:.4f}"
            )
            running_loss = 0.0
            window_steps = 0

    # step giữ giá trị của vòng lặp cuối cùng (kể cả khi break sớm) —
    # dùng nó để báo cáo TRUNG THỰC số bước đã chạy, thay vì in args.steps.
    steps_done = step
    # Nạp lại bản tốt nhất: model cuối cùng có thể đã "học vẹt" thêm,
    # bản ở đáy chữ U (val thấp nhất) mới là bản đáng dùng để sinh chữ.
    model.load_state_dict(torch.load("best_bigram.pt", weights_only=True))

    synchronize_device()
    elapsed_seconds = time.perf_counter() - started_at
    training_peak_ram = peak_ram_mib()
    peak_vram = (
        torch.cuda.max_memory_allocated(device) / (1024 ** 2)
        if device.type == "cuda" else None
    )

    # ---- Bước 5+6: sinh văn bản với 3 "tính cách" temperature ----
    # Công thức: P(ký tự tiếp theo) = softmax(điểm / T)
    #   - T nhỏ (0.7): khoảng cách điểm bị PHÓNG ĐẠI -> ký tự đầu bảng gần như
    #     luôn thắng -> văn bản an toàn, khuôn mẫu.
    #   - T = 1.0   : giữ nguyên những gì đã học.
    #   - T lớn (1.3): mọi ký tự bị CÀO BẰNG nhau -> văn bản táo bạo, đôi khi kỳ quặc.
    model.eval()
    for temperature in (0.7, 1.0, 1.3):
        # Bắt đầu từ token 0 (ký tự nhỏ nhất sau khi sort, thường là xuống dòng
        # hoặc dấu cách) rồi để model tự "đẻ" tiếp 200 ký tự.
        context = torch.zeros((1, 1), dtype=torch.long, device=device)
        generated = model.generate(context, 200, temperature=temperature)
        print(f"\n--- temperature={temperature} ---")
        print(decode(generated[0].tolist()))  # gồm 1 token mồi + 200 token mới

    # ---- Bước 7: green checkpoint — "hóa đơn năng lượng" của buổi train ----
    print("\n--- Green checkpoint ---")
    print(f"Thời gian vòng huấn luyện (gồm validation/log): {elapsed_seconds:.2f} s")
    print(f"Số bước optimizer: {steps_done}")
    print(f"Peak RAM tiến trình từ khởi động đến hết train: {training_peak_ram:.2f} MiB")
    if peak_vram is not None:
        print(f"Peak VRAM tensor do PyTorch cấp phát khi train: {peak_vram:.2f} MiB")


if __name__ == "__main__":
    main()
