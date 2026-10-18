#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Pipeline du lieu huan luyen LLM (Day 08 - buoi toi).

Chay: ingest -> normalize -> scrub PII -> quality filter -> exact dedup
-> near dedup -> contamination check -> split -> shard.

Dau vao: raw_docs.jsonl (200 docs, tao boi make_dirty_data.py),
         test_set.jsonl (10 cau hoi).
Dau ra: train-*.jsonl / val-*.jsonl (shard, toi da 50 docs/shard),
        manifest.json (SHA-256 tung shard + bao cao tung buoc),
        data_card.md (tai lieu mo ta du lieu).

Cach chay:
    python prepare_data.py

Ky vong:
    In bang bao cao so mau bi loai o TUNG BUOC (khong bao gio chi bao
    so mau con lai). Chay 2 lan cho manifest hash giong het nhau
    (deterministic).

Chi dung thu vien chuan Python.
"""

import hashlib
import json
import os
import re
import unicodedata
from collections import Counter
from datetime import date

# ---------------------------------------------------------------------------
# Cau hinh pipeline (moi nguong dat o day, khong hardcode trong ham)
# ---------------------------------------------------------------------------
RAW_PATH = "raw_docs.jsonl"
TEST_PATH = "test_set.jsonl"
SHARD_SIZE = 50       # so doc toi da moi shard
TRAIN_RATIO = 0.9     # 90% train / 10% val
MIN_WORDS = 30        # doc it hon -> qua ngan
MAX_WORDS = 2000      # doc nhieu hon -> qua dai
REPEAT_RATIO = 0.3    # ky tu pho bien nhat > 30% do dai -> rac
VI_DIACRITICS = set(  # tap ky tu tieng Viet co dau de nhan dien ngon ngu
    "àáạảãâầấậẩẫăằắặẳẵèéẹẻẽêềếệểễìíịỉĩòóọỏõôồốộổỗơờớợởỡ"
    "ùúụủũưừứựửữỳýỵỷỹđĐ")
LANG_MIN_WORDS = 20   # doc dai hon moi kiem tra ngon ngu
LANG_MIN_RATIO = 0.05  # ti le ky tu co dau toi thieu
NEAR_DUP_JACCARD = 0.8   # Jaccard 5-gram > 0.8 -> gan trung
CONTAM_JACCARD = 0.9     # Jaccard 8-gram > 0.9 voi cau test -> o nhiem

# Regex che PII (thu tu quan trong: phone truoc id vi SDT cung la so).
EMAIL_RE = re.compile(r"[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}")
PHONE_RE = re.compile(r"\b0\d{9}\b")      # SDT VN: 0 + 9 so (10 so)
ID_RE = re.compile(r"\b\d{9,12}\b")       # day so 9-12 chu so (CMND/CCCD)
HTML_RE = re.compile(r"<[^>]+>")
WS_RE = re.compile(r"\s+")
# Dau hieu "day la code, khong phai van ban tieng Anh" (mien tru language).
CODE_MARKS = ("{", "}", ";", "def ", "import ", "class ", "();")


# ---------------------------------------------------------------------------
# Buoc 1 - normalize(text): chuan hoa van ban
# ---------------------------------------------------------------------------
def normalize(text):
    """Chuan hoa van ban ve dang nhat quan.

    1. NFC: gop dau tieng Viet ve dang chuan (e + dau -> e co dau san).
       Neu khong, "ệ" kieu to hop va kieu dung san la 2 chuoi khac nhau
       du exact dedup that bai mot cach vo duyen.
    2. Xoa HTML tags (du lieu web luon lan the <p>, <div>...).
    3. Gom khoang trang thua (ke ca xuong dong) thanh 1 space, strip.

    QUYET DINH (ghi vao data_card): KHONG lowercase - chu hoa mang thong
    tin ("Apple" cong ty vs "apple" trai cay). Bay 1: normalize qua tay
    lam mat thong tin ma khong giai thich duoc vi sao.
    """
    text = unicodedata.normalize("NFC", text)
    text = HTML_RE.sub(" ", text)
    text = WS_RE.sub(" ", text).strip()
    return text


# ---------------------------------------------------------------------------
# Buoc 2 - scrub_pii(text): che thong tin nhay cam
# ---------------------------------------------------------------------------
def scrub_pii(text):
    """Thay email/SDT/so dinh danh bang token che, dem so luot che.

    Dung subn() de vua thay vua dem trong 1 luot. Thu tu: email ->
    phone -> id (neu chay id truoc, SDT 10 so se bi nuot thanh [ID]).

    Returns:
        Tuple (text_da_che, so_thuc_the_bi_che).
    """
    count = 0
    text, n = EMAIL_RE.subn("[EMAIL]", text)
    count += n
    text, n = PHONE_RE.subn("[PHONE]", text)
    count += n
    text, n = ID_RE.subn("[ID_NUMBER]", text)
    count += n
    return text, count


# ---------------------------------------------------------------------------
# Buoc 3 - quality_filter(doc): loc chat luong
# ---------------------------------------------------------------------------
def quality_filter(text):
    """Kiem tra chat luong doc.

    Thu tu kiem tra (tu tong quat den chi tiet): rong -> qua dai ->
    ky tu/tu lap -> qua ngan -> ngon ngu. Moi doc roi vao ly do CHINH
    XAC nhat (vd "a"*500 la repetitive chu khong phai too_short).

    Returns:
        Tuple (giu_lai: bool, ly_do_loai: str | None). Ly do loai phuc vu
        bao cao - pipeline tot phai noi duoc no loai cai gi.
    """
    if not text:
        return False, "empty"
    words = text.split()
    if len(words) > MAX_WORDS:
        return False, "too_long"
    # Ky tu/tu lap: "aaaaa..." co 1 ky tu chiem gan 100% do dai; "xyz "*200
    # co 1 tu chiem 100%. Van ban thuong: space ~15-20%, tu pho bien
    # nhat ("cua", "va") chi ~3-5% -> nguong 30% an toan.
    top_char_ratio = Counter(text).most_common(1)[0][1] / len(text)
    # Ti le tu lap chi y nghia khi doc du dai (doc 2 tu nhu "Xin chao."
    # thi moi tu da chiem 50% -> bao sai). Nguong 10 tu la du an toan.
    top_word_ratio = (Counter(words).most_common(1)[0][1] / len(words)
                      if len(words) >= 10 else 0)
    if top_char_ratio > REPEAT_RATIO or top_word_ratio > REPEAT_RATIO:
        return False, "repetitive"
    if len(words) < MIN_WORDS:
        return False, "too_short"
    # Language ID don gian: dem ky tu tieng Viet co dau. Doc dai ma gan
    # nhu khong co dau -> khong phai tieng Viet -> loai (tru khi la code).
    if len(words) > LANG_MIN_WORDS:
        diac = sum(1 for ch in text if ch in VI_DIACRITICS)
        marks = sum(1 for m in CODE_MARKS if m in text)
        if diac / len(text) < LANG_MIN_RATIO and marks < 2:
            return False, "non_vietnamese"
    return True, None


# ---------------------------------------------------------------------------
# Buoc 4 - dedup(docs): loai trung lap 2 tang
# ---------------------------------------------------------------------------
def shingles(text, k=5):
    """Tap k-gram ky tu (shingle) de so sanh gan dung.

    Bo space truoc khi cat: "a b" va "ab" cho shingle giong nhau, bot
    nhay cam voi format thua.
    """
    t = text.replace(" ", "")
    return {t[i:i + k] for i in range(max(1, len(t) - k + 1))}


def jaccard(a, b):
    """Do tuong dong Jaccard = |A giao B| / |A hop B|, trong [0, 1]."""
    return len(a & b) / max(1, len(a | b))


def content_hash(text):
    """SHA-256 hex cua noi dung (dinh danh doc)."""
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def dedup(docs):
    """Loai trung lap, giu ban xuat hien dau tien (thu tu deterministic).

    Tang 1 - exact: bam SHA-256 toan van ban (da normalize) -> trung hash
    la loai, O(1) moi doc.
    Tang 2 - near: Jaccard 5-gram > 0.8 voi doc da giu -> loai. Can thiet
    vi chi 1 ky tu khac la SHA-256 da khac han (Q2): bao chi copy lai chi
    khac tieu de thi exact dedup bo tay.

    GHI CHU MO RONG: so sanh tuan tu la O(n^2) - voi 200 docs thi OK
    (40k cap), nhung voi 1M docs la khong tuong (10^12 cap). Thuc te dung
    MinHash (nen tap shingle thanh vector ngan) + LSH (chi so sanh cac
    doc cung "xô") de dua ve gan O(n).

    Args:
        docs: list dict {"id", "text"} (text da normalize + scrub).

    Returns:
        Tuple (docs_giu_lai, so_exact_loai, so_near_loai).
    """
    # Sort theo hash TRUOC khi dedup: thu tu duyet khong phu thuoc
    # PYTHONHASHSEED cua phien chay -> deterministic (Bay 3).
    docs = sorted(docs, key=lambda d: content_hash(d["text"]))

    seen_hash, kept, kept_shingles = set(), [], []
    n_exact = 0
    for d in docs:
        h = content_hash(d["text"])
        if h in seen_hash:
            n_exact += 1
            continue
        seen_hash.add(h)
        kept.append(d)
        kept_shingles.append(shingles(d["text"]))

    # Near dedup tren tap da sach exact (it doc hon -> it cap so sanh hon).
    final, final_shingles = [], []
    n_near = 0
    for d, sh in zip(kept, kept_shingles):
        if any(jaccard(sh, ks) > NEAR_DUP_JACCARD for ks in final_shingles):
            n_near += 1
            continue
        final.append(d)
        final_shingles.append(sh)
    return final, n_exact, n_near


# ---------------------------------------------------------------------------
# Buoc 5 - check_contamination(docs, test_set): chong ro ri eval
# ---------------------------------------------------------------------------
def check_contamination(docs, test_set):
    """Loai doc train chua dap an / cau hoi cua test set.

    2 dau hieu (gap 1 trong 2 la loai):
      1. Chua NGUYEN VAN cau test (sau normalize, so sanh chuoi con).
      2. Jaccard 8-gram > 0.9 voi cau test (ban gan nhu y he cau test).

    THU TU CUC KY QUAN TRONG: chay TRUOC khi split train/val - vi val
    cung la eval, ro ri vao val cung la gian lan. Day la loi dao duc khoa
    hoc chu khong chi la loi ky thuat: diem eval ao -> tuong model gioi.

    Returns:
        Tuple (docs_sach, so_doc_o_nhiem).
    """
    norm_q = [(q["id"], normalize(q["question"])) for q in test_set]
    q_shingles = [(qid, shingles(q, k=8)) for qid, q in norm_q]
    clean, n_contam = [], 0
    for d in docs:
        t = d["text"]  # da normalize tu buoc ingest
        hit = any(q in t for _, q in norm_q)
        if not hit:
            d_sh = shingles(t, k=8)
            hit = any(jaccard(d_sh, qs) > CONTAM_JACCARD
                      for _, qs in q_shingles)
        if hit:
            n_contam += 1
            continue
        clean.append(d)
    return clean, n_contam


# ---------------------------------------------------------------------------
# Buoc 6 - split_and_shard(docs): chia tap + shard + manifest + data card
# ---------------------------------------------------------------------------
def split_and_shard(docs, report):
    """Chia train/val deterministic, ghi shard, manifest, data card.

    Split: sort theo SHA-256 noi dung, 90% dau -> train, 10% cuoi -> val.
    Vi sao khong random.shuffle? Vi seed quen set la mat tai lap; hash-sort
    thi may nao chay cung giong het nhau (Q3).

    Returns:
        Dict thong ke cuoi (train/val docs, tokens, thoi gian...).
    """
    docs = sorted(docs, key=lambda d: content_hash(d["text"]))
    n_train = int(len(docs) * TRAIN_RATIO)
    train_docs, val_docs = docs[:n_train], docs[n_train:]

    os.makedirs("shards", exist_ok=True)
    shards = []
    for split, sdocs in (("train", train_docs), ("val", val_docs)):
        for i in range(0, len(sdocs), SHARD_SIZE):
            part = sdocs[i:i + SHARD_SIZE]
            path = f"shards/{split}-{i // SHARD_SIZE:05d}.jsonl"
            with open(path, "w", encoding="utf-8") as f:
                for d in part:
                    f.write(json.dumps({"id": d["id"], "text": d["text"]},
                                       ensure_ascii=False) + "\n")
            with open(path, "rb") as f:
                digest = hashlib.sha256(f.read()).hexdigest()
            n_tok = sum(len(d["text"].split()) for d in part)
            shards.append({"path": path, "sha256": digest,
                           "num_docs": len(part), "num_tokens": n_tok})

    manifest = {
        "created_at": date.today().isoformat(),
        "source": RAW_PATH,
        "license": "Du lieu tong hop cho muc dich hoc tap (khong dung that)",
        "stage_report": report,
        "shards": shards,
    }
    with open("manifest.json", "w", encoding="utf-8") as f:
        json.dump(manifest, f, ensure_ascii=False, indent=2)

    write_data_card(report, train_docs, val_docs, shards)
    return {"n_train": len(train_docs), "n_val": len(val_docs),
            "tok_train": sum(s["num_tokens"] for s in shards
                             if s["path"].startswith("shards/train")),
            "tok_val": sum(s["num_tokens"] for s in shards
                           if s["path"].startswith("shards/val"))}


def write_data_card(report, train_docs, val_docs, shards):
    """Sinh data_card.md: tai lieu mo ta du lieu (provenance)."""
    # Counter chi co key khi co doc bi loai vi ly do do -> dung .get().
    q = report["quality"]
    n_short = q.get("too_short", 0)
    n_long = q.get("too_long", 0)
    n_rep = q.get("repetitive", 0)
    n_lang = q.get("non_vietnamese", 0)
    n_empty = q.get("empty", 0)
    lines = [
        "# DATA CARD - Pipeline du lieu huan luyen LLM (Day 08)",
        "",
        f"- Nguon: {RAW_PATH} (du lieu tong hop bang make_dirty_data.py).",
        "- License: du lieu tong hop cho muc dich hoc tap.",
        f"- Ngay tao: {date.today().isoformat()}.",
        "- Quyet dinh normalize: NFC + xoa HTML + gom space; KHONG "
        "lowercase (chu hoa mang thong tin: 'Apple' cong ty vs 'apple').",
        "",
        "## Thong ke",
        f"- Train: {len(train_docs)} docs | Val: {len(val_docs)} docs.",
        f"- Shard: {len(shards)} file (toi da {SHARD_SIZE} docs/shard).",
        "",
        "## Cac buoc loc (so mau BI LOAI moi buoc)",
        f"- Che PII: {report['pii_redactions']} thuc the "
        "(email-> [EMAIL], SDT -> [PHONE], so 9-12 chu so -> [ID_NUMBER]).",
        f"- Quality: loai {sum(q.values())} "
        f"(qua ngan {n_short}, qua dai {n_long}, "
        f"ky tu lap {n_rep}, khac tieng Viet {n_lang}, "
        f"rong {n_empty}).",
        f"- Exact dedup (SHA-256): loai {report['exact_dedup']}.",
        f"- Near dedup (Jaccard 5-gram > {NEAR_DUP_JACCARD}): "
        f"loai {report['near_dedup']}.",
        f"- Chong ro ri eval (nguyen van hoac Jaccard 8-gram > "
        f"{CONTAM_JACCARD}): loai {report['contamination']}.",
        "",
        "## Green checkpoint",
        f"- Token huu ich / cong suc: {report['green_ratio']:.1f} token sach "
        "cho moi doc bi loai.",
        "- Vi sao dedup tiet kiem dien: doc trung lap la buoc gradient vo "
        "ich (model hoc lai dieu da biet); loai som 1 doc trung = bot di "
        "nhieu buoc train dot dien vo nghia khi chay that.",
    ]
    with open("data_card.md", "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")


# ---------------------------------------------------------------------------
# Main: chay full pipeline + in bang bao cao
# ---------------------------------------------------------------------------
def main():
    """Ingest -> loc -> dedup -> contamination -> split -> shard."""
    with open(RAW_PATH, encoding="utf-8") as f:
        raw = [json.loads(line) for line in f if line.strip()]
    with open(TEST_PATH, encoding="utf-8") as f:
        test_set = [json.loads(line) for line in f if line.strip()]
    n_ingest = len(raw)

    # Normalize + scrub PII cho toan bo (PII duoc CHE chu khong bi loai).
    docs, pii_count = [], 0
    for d in raw:
        text, n = scrub_pii(normalize(d["text"]))
        pii_count += n
        docs.append({"id": d["id"], "text": text})

    # Quality filter (gi ly do de bao cao).
    kept, q_drop = [], Counter()
    for d in docs:
        ok, reason = quality_filter(d["text"])
        if ok:
            kept.append(d)
        else:
            q_drop[reason] += 1

    # Dedup 2 tang (deterministic).
    kept, n_exact, n_near = dedup(kept)

    # Chong ro ri eval (TRUOC split).
    kept, n_contam = check_contamination(kept, test_set)

    # Bao cao so mau BI LOAI tung buoc (khong bao gio chi bao so con lai).
    n_drop = (sum(q_drop.values()) + n_exact + n_near + n_contam)
    n_final_tokens = sum(len(d["text"].split()) for d in kept)
    report = {
        "ingest": n_ingest,
        "pii_redactions": pii_count,
        "quality": dict(q_drop),
        "exact_dedup": n_exact,
        "near_dedup": n_near,
        "contamination": n_contam,
        "final": len(kept),
        # Green checkpoint 1: token huu ich / cong suc.
        "green_ratio": n_final_tokens / max(1, n_drop),
    }

    stats = split_and_shard(kept, report)

    print("=" * 64)
    print("Bao cao pipeline (so mau BI LOAI tung buoc)")
    print("=" * 64)
    print(f"ingest: {n_ingest} "
          f"-> quality: -{sum(q_drop.values())} {dict(q_drop)} "
          f"-> exact_dedup: -{n_exact} "
          f"-> near_dedup: -{n_near} "
          f"-> contamination: -{n_contam} "
          f"-> final: {len(kept)} "
          f"(train {stats['n_train']} / val {stats['n_val']})")
    print(f"PII da che: {pii_count} thuc the "
          f"(doc van duoc giu lai, chi che thong tin nhay cam)")
    print(f"Token cuoi: train {stats['tok_train']:,} + "
          f"val {stats['tok_val']:,} = "
          f"{stats['tok_train'] + stats['tok_val']:,}")
    print(f"Green: {report['green_ratio']:.1f} token sach / moi doc bi loai")
    print("Da ghi: shards/, manifest.json, data_card.md")
    print("=" * 64)


if __name__ == "__main__":
    main()
