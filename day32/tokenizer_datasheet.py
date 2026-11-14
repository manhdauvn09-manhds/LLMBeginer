"""Lab Day 32 — Tokenizer BPE + Datasheet Wikipedia-vi (code kiem chung).

Hai bai code ngan, moi con so trong datasheet deu tinh bang code nay,
khong tinh nham:

- Bai 1: ngan sach vocab — embedding an bao nhieu tham so trong
  ngan sach ~10M voi d_model = 256 (vocab 8k / 16k / 20.480).
- Bai 2: chuan hoa NFC — chung minh cung mot chu co dau co the
  ma hoa thanh hai day ky tu khac nhau neu khong chuan hoa.

Chay: ~/workspace/venvs/day15/bin/python tokenizer_datasheet.py
Tu kiem tra phai FAIL bang assert neu so sai (khong chi in ra).
"""

import unicodedata


# Ngan sach tham so cua Tiny LM, chot o scope sheet Day 31.
BUDGET_PARAMS = 10_000_000
D_MODEL = 256

# Kich thuoc dump Wikipedia-tieng Viet, ban 01/11/2026 (so da kiem
# chung trong de bai, don vi byte, file .bz2).
DUMP_BYTES = 1_165_020_800


def embedding_params(vocab, d_model, budget=BUDGET_PARAMS):
    """Tra ve (so tham so embedding, ti trong % trong ngan sach).

    Cong thuc: moi token trong vocab co mot vector embedding
    dai d_model, nen so tham so = vocab * d_model.
    Vi du: vocab 8.000, d_model 256 -> 2.048.000 tham so.

    Args:
        vocab: kich thuoc vocab (so nguyen > 0).
        d_model: chieu rong vector embedding (so nguyen > 0).
        budget: tong ngan sach tham so (so nguyen > 0).

    Returns:
        Tuple (params, share): params la so nguyen, share la ti
        trong % (float) cua embedding trong ngan sach.

    Raises:
        ValueError: neu bat ky tham so nao khong phai so nguyen
            duong (ke ca kieu float, chuoi, hoac so <= 0).
    """
    for name, value in (("vocab", vocab), ("d_model", d_model),
                        ("budget", budget)):
        if isinstance(value, bool) or not isinstance(value, int):
            raise ValueError(name + " phai la so nguyen, nhan duoc "
                             + repr(value))
        if value <= 0:
            raise ValueError(name + " phai > 0, nhan duoc "
                             + str(value))
    params = vocab * d_model
    share = 100.0 * params / budget
    return params, share


def bytes_to_gib(n_bytes):
    """Doi so byte sang GiB (1 GiB = 1024 ** 3 byte).

    Vi du: 1.165.020.800 byte -> khoang 1,085 GiB (khong phai
    1,11 GiB nhu trong de ghi — xem ghi chu o ham main).

    Args:
        n_bytes: so byte (so nguyen >= 0).

    Returns:
        So GiB (float).

    Raises:
        ValueError: neu n_bytes khong phai so nguyen >= 0.
    """
    if isinstance(n_bytes, bool) or not isinstance(n_bytes, int):
        raise ValueError("n_bytes phai la so nguyen, nhan duoc "
                         + repr(n_bytes))
    if n_bytes < 0:
        raise ValueError("n_bytes phai >= 0, nhan duoc "
                         + str(n_bytes))
    return n_bytes / 1024 ** 3


def nfc_demo(word):
    """Tra ve thong tin NFC/NFD cua mot tu de chung minh can NFC.

    Cung mot chu hien thi giong nhau nhung co the ma hoa bang hai
    cach: dang dung san (NFC, 1 ky tu cho 1 chu co dau) va dang
    dau to hop (NFD, chu goc + tung dau rieng). Neu khong chuan
    hoa NFC truoc khi train tokenizer, cung mot chu se thanh hai
    token khac nhau, lang phi vocab.

    Args:
        word: tu tieng Viet o dang dung san (vi du "truong").

    Returns:
        Dict voi cac khoa: nfc (str), nfd (str), len_nfc (int),
        len_nfd (int), diff (int, = len_nfd - len_nfc),
        nfc_codes (list cac ma hex), nfd_codes (list cac ma hex).

    Raises:
        ValueError: neu word khong phai chuoi hoac chuoi rong.
    """
    if not isinstance(word, str) or word == "":
        raise ValueError("word phai la chuoi khong rong, nhan duoc "
                         + repr(word))
    nfc = unicodedata.normalize("NFC", word)
    nfd = unicodedata.normalize("NFD", word)
    return {
        "nfc": nfc,
        "nfd": nfd,
        "len_nfc": len(nfc),
        "len_nfd": len(nfd),
        "diff": len(nfd) - len(nfc),
        "nfc_codes": [hex(ord(c)) for c in nfc],
        "nfd_codes": [hex(ord(c)) for c in nfd],
    }


def _self_check():
    """Tu kiem tra: FAIL bang assert neu bat ky so nao sai."""
    # Bai 1: bang ngan sach vocab (d_model = 256, ngan sach 10M).
    # 8.000 * 256 = 2.048.000 = 20,48% ; 16.000 * 256 = 4.096.000
    # = 40,96% ; 20.480 * 256 = 5.242.880 = 52,4288%.
    assert embedding_params(8000, 256) == (2_048_000, 20.48)
    assert embedding_params(16000, 256) == (4_096_000, 40.96)
    params_20480, share_20480 = embedding_params(20480, 256)
    assert params_20480 == 5_242_880
    assert abs(share_20480 - 52.4288) < 1e-9
    # Phan transformer con lai sau khi tru embedding.
    assert BUDGET_PARAMS - 2_048_000 == 7_952_000
    assert BUDGET_PARAMS - 4_096_000 == 5_904_000
    assert BUDGET_PARAMS - 5_242_880 == 4_757_120
    # Vocab 16k lay them dung 2.048.000 tham so so voi vocab 8k.
    p8, _ = embedding_params(8000, 256)
    p16, _ = embedding_params(16000, 256)
    assert p16 - p8 == 2_048_000

    # Guard: tham so sai kieu hoac <= 0 phai ValueError, khong
    # duoc chet bang TypeError mo hoac tra so vo nghia.
    for bad in (0, -1, 8000.5, "8000", None, True):
        try:
            embedding_params(bad, 256)
        except ValueError:
            pass
        else:
            raise AssertionError("vocab=" + repr(bad)
                                 + " phai ValueError")
    for bad in (0, -256, 256.0, "256"):
        try:
            embedding_params(8000, bad)
        except ValueError:
            pass
        else:
            raise AssertionError("d_model=" + repr(bad)
                                 + " phai ValueError")
    try:
        embedding_params(8000, 256, budget=0)
    except ValueError:
        pass
    else:
        raise AssertionError("budget=0 phai ValueError")

    # Kich thuoc dump: 1.165.020.800 byte = 1.085 GiB (lam tron
    # 3 chu so thap phan). De bai ghi "~1,11 GiB" la lam tron sai
    # (lay 1.110,9 MiB chia cho 1.000 thay vi 1.024); so dung de
    # ghi vao datasheet la 1,085 GiB.
    gib = bytes_to_gib(DUMP_BYTES)
    assert abs(gib - 1.085) < 0.001, gib
    assert abs(bytes_to_gib(1024 ** 3) - 1.0) < 1e-12
    assert bytes_to_gib(0) == 0.0
    try:
        bytes_to_gib(-1)
    except ValueError:
        pass
    else:
        raise AssertionError("n_bytes=-1 phai ValueError")

    # Bai 2: "truong" — NFC 6 ky tu, NFD 9 ky tu (u -> u + dau
    # moc, o -> o + dau moc + dau huyen), dai hon 3 ky tu.
    demo = nfc_demo("trường")
    assert demo["len_nfc"] == 6, demo
    assert demo["len_nfd"] == 9, demo
    assert demo["diff"] == 3, demo
    # "e" — NFC 1 ky tu, NFD 3 ky tu (e + dau cham duoi + dau mu),
    # dai hon dung 2 ky tu: vi du gon nhat cho cau 5.
    demo_e = nfc_demo("ệ")
    assert demo_e["len_nfc"] == 1, demo_e
    assert demo_e["len_nfd"] == 3, demo_e
    assert demo_e["diff"] == 2, demo_e
    # Round-trip: chuan hoa NFC lai dang NFD phai tra ve dang goc.
    for word in ("trường", "ệ", "Nguyễn", "ữ"):
        info = nfc_demo(word)
        assert unicodedata.normalize("NFC", info["nfd"]) \
            == info["nfc"]
        assert unicodedata.normalize("NFD", info["nfc"]) \
            == info["nfd"]
    try:
        nfc_demo("")
    except ValueError:
        pass
    else:
        raise AssertionError("word rong phai ValueError")
    try:
        nfc_demo(123)
    except ValueError:
        pass
    else:
        raise AssertionError("word=123 phai ValueError")


def main():
    """In bang ket qua Bai 1 + Bai 2, roi chay tu kiem tra."""
    print("Bai 1 — Ngan sach vocab (d_model = 256, ngan sach 10M)")
    for vocab in [8000, 16000, 20480]:
        params, share = embedding_params(vocab, D_MODEL)
        rest = BUDGET_PARAMS - params
        print(f"vocab {vocab}: {params:,} tham so = {share:.1f}% "
              f"ngan sach 10M, transformer con lai {rest:,}")
    print()
    print("Kich thuoc dump Wikipedia-vi: "
          f"{DUMP_BYTES:,} byte = {bytes_to_gib(DUMP_BYTES):.3f} GiB")
    print()
    print("Bai 2 — NFC bang mat")
    for word in ("trường", "ệ"):
        info = nfc_demo(word)
        print(f"tu {info['nfc']!r}: NFC len {info['len_nfc']}, "
              f"NFD len {info['len_nfd']} (dai hon "
              f"{info['diff']} ky tu)")
        print(f"  NFC: {info['nfc_codes']}")
        print(f"  NFD: {info['nfd_codes']}")
    print()
    _self_check()
    print("Tat ca kiem tra PASS.")


if __name__ == "__main__":
    main()
