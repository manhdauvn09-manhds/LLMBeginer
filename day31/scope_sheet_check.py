"""Tu kiem tra thesis scope sheet Day 31.

Doc file thesis_scope_sheet.md theo checklist cua
Day31_practice.html: du 6 muc, moi muc 2-4 dong, N da chot,
P va Y ghi "chot sau baseline" (khong bia so), du 5 cau
tra loi phan bien, moi cau 2-4 dong.

Chay: python scope_sheet_check.py (exit 0 = dat, exit != 0
= hong, kem thong bao loi cu the).
"""

import pathlib
import re
import sys

SHEET = pathlib.Path(__file__).parent / "thesis_scope_sheet.md"


def doc_section(text, so_muc):
    """Cat lay than van ban cua 1 muc theo so thu tu.

    Vi du: doc_section(text, 1) tra ve than cua muc "## 1. ...".
    """
    mau = re.compile(r"^## %d\. .*$" % so_muc, re.M)
    vi_tri = [m.start() for m in mau.finditer(text)]
    assert len(vi_tri) == 1, "muc %d xuat hien %d lan, can dung 1" % (
        so_muc, len(vi_tri))
    dau = vi_tri[0]
    # Tim dong tieu de ke tiep (## ...) de cat phan cuoi muc.
    sau = re.compile(r"^#+ ", re.M).search(text, dau + 1)
    cuoi = sau.start() if sau else len(text)
    than = text[dau:cuoi].split("\n", 1)[1]
    dong = [d.strip() for d in than.splitlines() if d.strip()]
    return [d for d in dong if d != "---"]


def doc_cau_tra_loi(text, so_cau):
    """Cat lay than van ban cua 1 cau tra loi Q1-Q5."""
    mau = re.compile(r"^## Q%d — .*$" % so_cau, re.M)
    vi_tri = [m.start() for m in mau.finditer(text)]
    assert len(vi_tri) == 1, "Q%d xuat hien %d lan, can dung 1" % (
        so_cau, len(vi_tri))
    dau = vi_tri[0]
    sau = re.compile(r"^## ", re.M).search(text, dau + 1)
    cuoi = sau.start() if sau else len(text)
    than = text[dau:cuoi].split("\n", 1)[1]
    return [d.strip() for d in than.splitlines() if d.strip()]


def main():
    """Chay toan bo kiem tra, in ket qua tung muc."""
    text = SHEET.read_text(encoding="utf-8")
    loi = []

    # 1. Du 6 muc, moi muc 2-4 dong (khong tinh dong tieu de).
    for so_muc in range(1, 7):
        try:
            dong = doc_section(text, so_muc)
        except AssertionError as exc:
            loi.append(str(exc))
            continue
        assert 2 <= len(dong) <= 4, (
            "muc %d co %d dong, can 2-4 dong" % (so_muc, len(dong)))
        print("Muc %d: %d dong - OK" % (so_muc, len(dong)))

    # 2. N da chot = 10M, co ly do Chinchilla (20 token/tham so).
    assert "N = 10M" in text, "chua chot N = 10M trong sheet"
    assert "20 token/tham so" in text, "thieu ly do Chinchilla cho N"
    print("N = 10M + ly do Chinchilla - OK")

    # 3. P va Y ghi "chot sau baseline", khong bia so cu the.
    assert text.count("chot sau baseline") >= 2, (
        "can it nhat 2 cho ghi 'chot sau baseline' (muc 3 va 5)")
    assert "[P]" in text and "[Y]" in text, (
        "muc 5 phai de [P] va [Y] dang cho, khong dien so")
    assert "quyet dinh cua anh" not in text, (
        "con placeholder quyet dinh chua dien trong sheet")
    print("P/Y: chot sau baseline, khong bia so - OK")

    # 4. Du 5 cau tra loi phan bien, moi cau 2-4 dong.
    for so_cau in range(1, 6):
        try:
            dong = doc_cau_tra_loi(text, so_cau)
        except AssertionError as exc:
            loi.append(str(exc))
            continue
        assert 2 <= len(dong) <= 4, (
            "Q%d co %d dong, can 2-4 dong" % (so_cau, len(dong)))
        print("Q%d: %d dong - OK" % (so_cau, len(dong)))

    # 5. Ba thu co dinh cho eval (muc 4 + Q5): cung tokenizer,
    # cung tap held-out, cung cach tinh.
    for cum in ("cung tokenizer", "cung tap held-out",
                "cung cach tinh"):
        assert cum in text, "thieu cum '%s' trong sheet" % cum
    print("3 thu co dinh cho eval - OK")

    if loi:
        print("LOI: " + "; ".join(loi), file=sys.stderr)
        return 1
    print("Tat ca kiem tra DAT: 6 muc + 5 cau tra loi.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
