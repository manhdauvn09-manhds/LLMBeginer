# DATA CARD - Pipeline du lieu huan luyen LLM (Day 08)

- Nguon: raw_docs.jsonl (du lieu tong hop bang make_dirty_data.py).
- License: du lieu tong hop cho muc dich hoc tap.
- Ngay tao: 2026-10-18.
- Quyet dinh normalize: NFC + xoa HTML + gom space; KHONG lowercase (chu hoa mang thong tin: 'Apple' cong ty vs 'apple').

## Thong ke
- Train: 132 docs | Val: 15 docs.
- Shard: 4 file (toi da 50 docs/shard).

## Cac buoc loc (so mau BI LOAI moi buoc)
- Che PII: 30 thuc the (email-> [EMAIL], SDT -> [PHONE], so 9-12 chu so -> [ID_NUMBER]).
- Quality: loai 15 (qua ngan 4, qua dai 3, ky tu lap 3, khac tieng Viet 5, rong 0).
- Exact dedup (SHA-256): loai 20.
- Near dedup (Jaccard 5-gram > 0.8): loai 15.
- Chong ro ri eval (nguyen van hoac Jaccard 8-gram > 0.9): loai 3.

## Green checkpoint
- Token huu ich / cong suc: 270.1 token sach cho moi doc bi loai.
- Vi sao dedup tiet kiem dien: doc trung lap la buoc gradient vo ich (model hoc lai dieu da biet); loai som 1 doc trung = bot di nhieu buoc train dot dien vo nghia khi chay that.
