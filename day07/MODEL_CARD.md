# MODEL_CARD.md - TinyGPT (Day 07 mini-project)
1. Kien truc: char-level Transformer 4 layer x 4 head, n_embd=128, MLP 4x + GELU, pre-norm, causal mask.
2. Du lieu: input.txt, vocab 65 ky tu, tach 90/10 train/val.
3. Sieu tham so: block_size=64, batch=32, AdamW lr=0.0003 wd=0.01, 2000 iters, seed=1337.
4. Tham so: 816,640 (~0.82M).
5. Loss: train 4.23 -> 1.72, val 4.24 -> 1.87 (sau 2000 iters, 9.5 phut CPU).
6. Sample tot nhat (T=0.7): " to man once will on liftse of ster's\nAnd him be my tham the be make her weethere\nTishall 'dwings, that and ev"
7. Sample 'dien' nhat (T=1.3): "lt!'s not\nTo willson liftse yourser's\nAnd himabud velt my water; bop his weem, 'twaus a ch'd.\n\nSICILgA:\nAy, eD"
8. Gioi han da biet: char-level + context 64 nen van ban con sai chinh ta; model nho (~1M) khong hieu nghia sau; chi nho pattern ngan.
9. Xanh: 7,162 tokens/s; train gap doi iters ~ gap doi dien nang (chi phi tuyen tinh theo so buoc).
