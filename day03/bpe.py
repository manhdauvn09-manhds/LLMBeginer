#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Day 03 - Buoi toi: TU VIET BPE & DO FERTILITY
=============================================
Muc dich:
    Bai 1: Cam nhan bang tay rang byte != ky tu (UTF-8).
    Bai 2: Viet thuat toan BPE (Byte Pair Encoding) tu scratch bang Python
           thuan - khong dung thu vien - tren corpus ti hon "low low lower",
           tai hien dung 2 merge dau: (l,o) roi (lo,w).
    Bai 3: Train BPE tren corpus tieng Anh nho tu viet, roi do
           fertility (so token / so tu) tren cau tieng Viet vs tieng Anh,
           kem thi nghiem them: khoang cach Viet-Anh thay doi ra sao khi
           tang so merge (30 -> 200).

Cach chay:
    python bpe.py
    # Ky vong: in byte_info, 2 merge dau (l,o) -> (lo,w),
    #         encode("lowest", merges[:2]) -> ('low','e','s','t','</w>'),
    #         bang fertility Viet vs Anh + bang ti le theo so merge
    #         (merge cang nhieu, tieng Viet cang ton gap ~3x tieng Anh).

Giu lai file nay - tuan sau (Ngay 6, GPT-2) se quay lai dung no.

Cach doc file (roadmap cho nguoi moi):
    1. bai1(): UTF-8 - tai sao byte != ky tu, va "thue byte" tieng Viet phai tra.
    2. to_vocab() -> get_stats() -> merge_vocab() -> train_bpe(): 4 ham core
       cua BPE, doc theo dung thu tu nay vi ham sau dung ham truoc.
    3. encode(): dung day merges da hoc de tach 1 tu moi (ke ca tu chua gap).
    4. bai2(): chay thu BPE tren corpus "low low lower" (co assert tu kiem tra).
    5. bai3(): do fertility Viet vs Anh + thi nghiem tang so merge.
"""

# [TRAINER] 2026-10-13: bo sung comment chi tiet toan file theo yeu cau user.
# Tat ca comment giai thich them trong dot nay deu nham muc dich "de hieu"
# (vi du trace tay tung buoc) - logic code giu nguyen 100%, khong doi hanh vi.
# Chi tiet them gi o tung ham xem bao cao truoc/sau trong chat.

# ---------------------------------------------------------------------------
# BAI 1 - So tan tay UTF-8: byte != ky tu
# ---------------------------------------------------------------------------
def byte_info(ch):
    """In code point (dang U+xxxx) va day byte UTF-8 (dang hex) cua 1 ky tu.

    - ord(ch): code point - "so can cuoc" cua ky tu trong bang Unicode.
      Vi du: 'A' -> 65 (U+0041), 'ệ' -> 7879 (U+1EC7).
    - ch.encode("utf-8"): ma hoa ky tu thanh DAY BYTE theo chuan UTF-8.
      UTF-8 la bien do dai: ky tu ASCII can 1 byte, Latin mo rong 2 byte,
      chu Viet co dau thuong 3 byte, emoji 4 byte.
    """
    cp = ord(ch)
    bs = ch.encode("utf-8")
    # f"{b:02X}": dinh dang 1 byte thanh 2 ky tu HEX in hoa.
    #   02 = luon du 2 ky tu, thieu thi them so 0 o truoc (vd byte 10 -> "0A").
    #   X  = he co so 16, chu in hoa (vd byte 225 -> "E1").
    # Vi du: 'ệ' -> 3 byte E1 BB 87 (kiem tra: neu may ban in ra khac 3 byte
    # nay thi file dang luu sai encoding, xem ghi chu o bai1).
    hexs = " ".join(f"{b:02X}" for b in bs)
    print(f"  {ch!r}: U+{cp:04X}, {len(bs)} byte -> {hexs}")


def byte_ratio(s):
    """Tra ve cap (so ky tu, so byte) cua chuoi s.

    LUU Y: len(s) dem theo KY TU Python (code point), con
    len(s.encode("utf-8")) dem theo BYTE. Voi tieng Anh 2 con so bang nhau;
    voi tieng Viet, byte nhieu hon ky tu vi chu co dau ton 2-3 byte/ky tu -
    do la "thue byte" tieng Viet phai tra TRUOC KHI tokenizer lam bat cu
    viec gi.
    """
    return len(s), len(s.encode("utf-8"))


def bai1():
    print("=" * 64)
    print("Bai 1 - UTF-8: byte != ky tu")
    print("=" * 64)
    print("Code point va byte UTF-8 cua tung ky tu:")
    # Chon 4 dai dien cho 4 "co" byte cua UTF-8: 1 byte (ASCII), 2 byte
    # (Latin mo rong), 3 byte (Viet co dau), 4 byte (emoji). Khi doc output,
    # de y cot "so byte" - do chinh la trong tam cua bai nay: cung la 1 ky tu
    # nhung "gia byte" khac nhau, va tieng Viet toan phai tra gia 2-3 byte.
    for ch in ["A", "é", "ệ", "🤖"]:
        byte_info(ch)
    print("  (Ky vong: 1 / 2 / 3 / 4 byte. Neu 'ệ' khong ra E1 BB 87 thi file\n"
          "   dang luu sai encoding - kiem tra editor luu UTF-8.)")

    en = "i love artificial intelligence"
    vi = "tôi yêu trí tuệ nhân tạo"
    en_chars, en_bytes = byte_ratio(en)
    vi_chars, vi_bytes = byte_ratio(vi)
    print("\n'Thue byte' - cung y nghia, cung 5 tu:")
    print(f"  EN: {en_chars} ky tu, {en_bytes} byte")
    print(f"  VI: {vi_chars} ky tu, {vi_bytes} byte")
    print(f"  => Cau tieng Viet IT KY TU HON ({vi_chars} < {en_chars}) nhung lai "
          f"NHIEU BYTE HON ({vi_bytes} > {en_bytes}):")
    print( "     moi chu co dau 'e, o, a...' ton 2-3 byte. Day la 'thue' phai tra")
    print( "     ngay tu muc byte, truoc khi tokenizer lam bat cu viec gi.")


# ---------------------------------------------------------------------------
# BAI 2 - Viet BPE tu scratch
# ---------------------------------------------------------------------------
# Ky tu ket thuc tu: "hang rao" de merge khong "tran" qua ranh gioi tu.
# (Xem cau tu kiem tra 1.)
END = "</w>"


def to_vocab(corpus):
    """Bien corpus thanh vocab: moi tu -> tuple cac ky tu + END o cuoi.

    Vi du: "low" -> ('l', 'o', 'w', '</w>').
    Tai sao them END? De cap (w, l) cuoi tu nay + dau tu kia khong bao gio
    duoc dem/merge thanh token lai vo nghia nhu 'wl'.
    """
    # Vi du chay tung buoc voi corpus "low low lower":
    #   corpus.split() -> ["low", "low", "lower"]        (tach theo khoang trang)
    #   "low" -> tuple("low") = ('l','o','w') -> them END -> ('l','o','w','</w>')
    # Ket qua: [('l','o','w','</w>'), ('l','o','w','</w>'),
    #           ('l','o','w','e','r','</w>')]
    # Chu y 2 diem:
    #   (1) Tu "low" lap lai 2 lan -> vocab co 2 phan tu giong het nhau.
    #       Tan suat lap lai nay QUAN TRONG: BPE dem cap tren TOAN vocab nen
    #       tu cang pho bien thi cap cua no cang duoc cong diem nhieu.
    #   (2) Tai sao tuple ma khong phai list? Vi tuple la bat bien (immutable)
    #       nen an toan khi dem, khong so bi sua nham giua chung.
    return [tuple(word) + (END,) for word in corpus.split()]


def get_stats(vocab):
    """Dem tan suat moi cap token KE NHAU trong toan bo vocab.

    Tra ve dict {(token_a, token_b): so lan xuat hien}.
    Voi corpus "low low lower" phai ra:
      (l,o):3, (o,w):3, (w,</w>):2, (w,e):1, (e,r):1, (r,</w>):1.
    """
    # [TRAINER] 2026-10-13: docstring cu thieu (w,</w>):2 va (r,</w>):1
    # (chi ghi "cac cap chua </w> cung duoc dem" trong ngoac don). Docstring
    # vi du phai khop 100% output that cua code - nguoi doc se copy-paste
    # de kiem tra.
    # Vi du dem tay (vocab rut gon con 2 tu de de nhin):
    #   Tu ('l','o','w','</w>'):         cap (l,o), (o,w), (w,</w>)
    #   Tu ('l','o','w','e','r','</w>'): cap (l,o), (o,w), (w,e), (e,r), (r,</w>)
    #   Cong don: (l,o):2, (o,w):2, (w,</w>):1, (w,e):1, (e,r):1, (r,</w>):1
    # Voi du 3 tu "low low lower": tu "low" thu 2 gop them (l,o)+1, (o,w)+1,
    # (w,</w>)+1 -> (l,o):3, (o,w):3, (w,</w>):2. Hai cap dau HOA 3-3 diem,
    # merge dau tien se chon (l,o) nho thu tu chen dict (xem Bay 1 o train_bpe).
    pairs = {}
    for word in vocab:
        # Duyet tung cap ke nhau trong 1 tu: (word[0],word[1]), (word[1],word[2]), ...
        for i in range(len(word) - 1):
            p = (word[i], word[i + 1])
            pairs[p] = pairs.get(p, 0) + 1
    return pairs


def merge_vocab(pair, vocab):
    """Gop moi lan xuat hien cua cap (a, b) thanh token moi 'a+b'.

    Quet tu trai sang phai, KHONG de cac merge chong lan nhau:
    gap (a,b) -> ghi 'ab', nhay 2 buoc; khong gap -> giu nguyen, nhay 1 buoc.
    Tra ve vocab MOI (khong sua vocab cu).
    """
    a, b = pair
    # Vi du trace TAY tung buoc: word = ('l','o','w','</w>'), pair = ('l','o'):
    #   i=0: word[0]=='l' va word[1]=='o' -> KHOP -> ghi 'lo', nhay i=2
    #   i=2: word[2]=='w' != 'l'          -> khong khop -> giu 'w', i=3
    #   i=3: word[3]=='</w>'              -> giu nguyen, i=4 -> het vong
    #   Ket qua: ('lo', 'w', '</w>')
    # Tai sao khi khop phai "nhay 2 buoc" (i += 2)? De 2 token vua gop khong
    # bi dem lai vao cap ke tiep. Moi luot merge chi gop DUNG 1 cap (a,b);
    # token 'lo' vua tao ra se doi den luot merge sau (vd merge ('lo','w'))
    # chu khong duoc gop tiep ngay trong luot nay. Dieu kien
    # "i < len(word) - 1" la de word[i+1] khong bao gio vuot qua cuoi tuple.
    out = []
    for word in vocab:
        new_word, i = [], 0
        while i < len(word):
            if i < len(word) - 1 and word[i] == a and word[i + 1] == b:
                new_word.append(a + b)
                i += 2  # da "an" 2 token -> nhay qua ca cap
            else:
                new_word.append(word[i])
                i += 1
        out.append(tuple(new_word))
    return out


def train_bpe(corpus, n_merges=30, verbose=True):
    """Vong lap train BPE: lap di lap lai (dem cap -> merge cap pho bien nhat).

    Moi vong:
      1. get_stats: dem cap ke nhau hien tai.
      2. Chon cap tan suat CAO NHAT (best).
      3. merge_vocab: gop cap do khap vocab.
      4. Luu cap vao list `merges` (thu tu nay QUAN TRONG khi encode).
    Dung khi het cap de merge (corpus ti hon can rat nhanh - binh thuong).

    LUU Y (Bay 1): neu 2 cap hoa tan suat, max() tra ve cap gap DAU TIEN
    co gia tri max (theo thu tu chen dict). Hai nguoi chay co the ra 2 day
    merge khac nhau ma deu "dung" - thu vien that dung tie-break de on dinh.
    """
    # Buoc 0 - khoi tao: vocab la list cac tuple ky tu (xem to_vocab),
    # merges la list rong, se ghi lai lich su cac cap da gop THEO THU TU.
    # Thu tu nay cuc ky quan trong: luc encode phai ap dung merges dung
    # thu tu da hoc thi ket qua moi khop (xem encode).
    vocab = to_vocab(corpus)
    merges = []
    # Moi vong lap = dung 1 merge duy nhat: cap pho bien nhat TAI THOI DIEM
    # hien tai (sau khi vocab da bi bien doi boi cac merge truoc). Vi du voi
    # corpus "low low lower":
    #   Vong 1: (l,o) 3 diem (hoa voi (o,w) nhung thang nho thu tu dict)
    #           -> vocab: [('lo','w','</w>'), ..., ('lo','w','e','r','</w>')]
    #   Vong 2: dem lai tren vocab MOI -> (lo,w) 3 diem -> gop tiep
    #           -> tu "low" chi con ('low','</w>'): 1 token noi dung!
    # Cu the, cang nhieu merge thi tu cang it manh vun.
    for step in range(n_merges):
        stats = get_stats(vocab)
        if not stats:
            if verbose:
                print(f"  (het cap de merge sau {step} vong - dung)")
            break
        best = max(stats, key=stats.get)  # cap pho bien nhat
        merges.append(best)
        vocab = merge_vocab(best, vocab)
        if verbose:
            print(f"  merge {step + 1:>2}: {best} -> {vocab[0]}")
    return vocab, merges


def encode(word, merges):
    """Encode 1 tu (ke ca tu CHUA TUNG GAP) bang day merges da hoc.

    - Bat dau tu tuple ky tu cua tu + END (GIONG HET tien xu ly luc train -
      quen END la Bay 3: cac merge chua END se khong bao gio khop).
    - Ap dung LAN LUOT tung merge theo dung thu tu da hoc.
    Ket qua: tu quen -> it token (da hoc san), tu la -> bi che thanh cac
    manh subword da hoc. Day la suc manh that cua subword (Bay 2:
    code nay merge theo KY TU de de nhin; BPE that merge theo BYTE).
    """
    # [TRAINER] Nhan xet hieu nang 2026-10-13 (KHONG doi code - muc dich su
    # pham): moi encode() quet lai TOAN BO day merges, do phuc tap O(M*T)
    # voi M = so merge, T = so token. Corpus ti hon thi khong sao, nhung
    # day la ly do thi nghiem 200 merge chay cham dan. Tokenizer that
    # khac phuc bang cache (vd functools.lru_cache tren encode, doi merges
    # sang tuple cho hash duoc) hoac bien dich merges thanh trie. Bai tap
    # tu lam them: cache encode() roi do lai thoi gian vong thi nghiem.
    # Vi du trace TAY tung buoc: encode("lowest", merges=[('l','o'),('lo','w')])
    #   Khoi tao: ('l','o','w','e','s','t','</w>')  <- nho +END giong luc train!
    #   Merge 1 ('l','o'):  -> ('lo','w','e','s','t','</w>')
    #   Merge 2 ('lo','w'): -> ('low','e','s','t','</w>')
    #   Ket qua: tu la "lowest" bi che thanh 4 manh da hoc + END.
    # So sanh: encode("low", ...) -> ('low','</w>') chi 1 token noi dung, vi
    # "low" nam nguyen trong corpus train. Day la suc manh that cua subword:
    # tu moi khong bi bo roi ma duoc ghep tu manh quen thuoc (word-level ma
    # gap tu la thi chi con cach danh dau <unk>).
    tokens = tuple(word) + (END,)
    for pair in merges:
        tokens = merge_vocab(pair, [tokens])[0]
    return tokens


def bai2():
    print("\n" + "=" * 64)
    print("Bai 2 - Viet BPE tu scratch (corpus: 'low low lower')")
    print("=" * 64)

    print("\nBuoc A - to_vocab:")
    vocab = to_vocab("low low lower")
    print(f"  {vocab}")

    print("\nBuoc B - get_stats (tan suat cap ke nhau):")
    stats = get_stats(vocab)
    # In theo tan suat giam dan cho de doc.
    for p, c in sorted(stats.items(), key=lambda kv: -kv[1]):
        print(f"  {p}: {c}")
    # Tu kiem tra: "low" x2 gop 2 diem cho (l,o) va (o,w); "lower" gop them
    # 1 diem cho moi cap -> ca 2 cap deu = 3. Neu assert nay fail, loi nam o
    # get_stats (dem sai), khong phai o merge - thu in stats ra so voi vi du
    # trong docstring cua get_stats de debug.
    assert stats[("l", "o")] == 3 and stats[("o", "w")] == 3, "get_stats sai!"

    print("\nBuoc C - vong lap merge:")
    vocab, merges = train_bpe("low low lower", n_merges=30, verbose=True)

    # Kiem tra 2 merge dau phai dung nhu de bai yeu cau.
    # Merge 1 PHAI la (l,o) chu khong phai (o,w) du hoa 3-3 diem: nho quy tac
    # tie-break cua max() - cap nao duoc chen vao dict TRUOC thi thang
    # (xem Bay 1 trong train_bpe). Merge 2 PHAI la (lo,w): sau khi gop (l,o),
    # dem lai tren vocab moi thi cap (lo,w) xuat hien 3 lan, cao nhat.
    assert merges[0] == ("l", "o"), f"merge 1 sai: {merges[0]}"
    assert merges[1] == ("lo", "w"), f"merge 2 sai: {merges[1]}"
    print(f"\n  => 2 merge dau: {merges[0]} -> {merges[1]} (KHOP de bai)")

    print("\nBuoc D - encode tu moi (chi dung 2 merge dau de test):")
    r1 = encode("lowest", merges[:2])
    r2 = encode("low", merges[:2])
    print(f"  encode('lowest') -> {'|'.join(r1)}")
    print(f"  encode('low')    -> {'|'.join(r2)}")
    assert r1 == ("low", "e", 's', 't', END), f"encode('lowest') sai: {r1}"
    assert r2 == ("low", END), f"encode('low') sai: {r2}"
    print("  => 'lowest' (tu la) bi che thanh manh da hoc; 'low' (tu quen) "
          "chi ton 1 token.")


# ---------------------------------------------------------------------------
# BAI 3 - Do fertility: tieng Viet vs tieng Anh
# ---------------------------------------------------------------------------
# Corpus train: 12 cau tieng Anh don gian ve AI (viet tay, chu thuong,
# khong dau cau de don gian).
EN_CORPUS = """\
artificial intelligence is changing the world
machine learning models learn from data
neural networks have many layers
deep learning needs a lot of data
language models predict the next token
training a model takes time and energy
data is the fuel of ai systems
tokens are pieces of words
attention helps models focus on context
open models are free for everyone to use
ai can write code and answer questions
the future of ai is bright and exciting"""

# 5 cau test tieng Anh: CHI DUNG tu da xuat hien trong corpus train (de phan anh
# dung fertility cua tokenizer tren ngon ngu no duoc train - tu la tieng Anh
# khac nao tu la tieng Viet: deu bi che vun nhu nhau).
EN_TEST = [
    "ai models learn from data",
    "neural networks predict the next token",
    "training a model needs time and energy",
    "language models can write code",
    "the future of ai is bright",
]

# 5 cau test tieng Viet, cung chu de, do dai tuong duong (6-7 am tiet/cau).
VI_TEST = [
    "mô hình ai học dữ liệu",
    "mạng nơ ron đoán token tiếp",
    "huấn luyện mô hình cần thời gian",
    "mô hình ngôn ngữ viết được code",
    "tương lai của ai rất sáng",
]


def fertility(sentence, merges):
    """Fertility = so token / so "tu" cua 1 cau.

    - Moi "tu" (tach bang split()) duoc encode rieng (kem END nhu luc train).
    - Token "</w>" don le khong dem vao token noi dung (no chi la dau
      phan cach); token nhu 'w</w>' van dem la 1 token noi dung.
    - Fertility cao = cau bi che vun -> ton token.

    LUU Y TINH TE: tieng Viet viet cach nhau theo AM TIET ("mô hình" = 2
    don vi khi split), khong phai theo tu. Nen con so duoi la fertility
    tren am tiet; tinh tren TU that thi tieng Viet con cao hon nua.
    """
    words = sentence.split()
    # [TRAINER] Guard 2026-10-13: cau rong -> split() ra [], chia cho
    # len(words)=0 se gay ZeroDivisionError. Tra ve 0 thay vi crash.
    if not words:
        return 0.0, 0, 0
    toks = []
    for w in words:
        toks.extend(encode(w, merges))
    # Loc bo token END don le: no chi la "hang rao" phan cach tu, khong phai
    # noi dung thuc. NHUNG token dinh kem END nhu 'w</w>' (chu w + hang rao
    # dinh lien) van duoc tinh la 1 token noi dung - vi no mang thong tin
    # chu 'w'. Ham tra ve tuple 3 gia tri de tien in bang dep o bai3:
    # (fertility, so token noi dung, so tu).
    content = [t for t in toks if t != END]
    return len(content) / len(words), len(content), len(words)


def bai3():
    print("\n" + "=" * 64)
    print("Bai 3 - Do fertility: tieng Viet vs tieng Anh")
    print("=" * 64)

    # [TRAINER] Toi uu 2026-10-13: chi train 1 LAN (200 merge) roi cat lat
    # merges[:30], [:100] khi can, thay vi train rieng 4 lan. Hop le nho
    # tinh prefix-consistent cua BPE: merge thu k chi phu thuoc vao k-1
    # merge truoc no, nen 30 merge dau cua lan train 200 TRUNG KHIT voi
    # lan train rieng 30 merge (cung seed logic, cung tie-break). Ket qua
    # so hoc khong doi, tiet kiem ~2/3 thoi gian train o thi nghiem duoi.
    print("\nTrain BPE 1 lan duy nhat (200 merge) tren corpus tieng Anh nho...")
    _, merges200 = train_bpe(EN_CORPUS, n_merges=200, verbose=False)
    merges = merges200[:30]
    print(f"  Hoc duoc {len(merges200)} merge (dung {len(merges)} dau cho bang "
          f"fertility). 5 merge dau: {merges[:5]}")

    print("\nFertility (so token / so tu) - merges hoc tu tieng Anh:")
    print(f"  {'cau':<45} {'token/tu':>9}  fertility")
    print("  " + "-" * 66)
    en_f, vi_f = [], []
    for s in EN_TEST:
        f, nt, nw = fertility(s, merges)
        en_f.append(f)
        print(f"  EN {s:<42} {nt:>2}/{nw:<2}     {f:.2f}")
    for s in VI_TEST:
        f, nt, nw = fertility(s, merges)
        vi_f.append(f)
        print(f"  VI {s:<42} {nt:>2}/{nw:<2}     {f:.2f}")

    avg_en = sum(en_f) / len(en_f)
    avg_vi = sum(vi_f) / len(vi_f)
    print("  " + "-" * 66)
    print(f"  Trung binh EN: {avg_en:.2f} | Trung binh VI: {avg_vi:.2f} "
          f"=> tieng Viet ton gap {avg_vi / avg_en:.1f} lan token")

    # --- Thi nghiem them: tang so merge thi khoang cach thay doi ra sao? ---
    # Y tuong: 30 merge la tokenizer "ti hon" - no nen tieng Anh cung kem
    # (chua nen tot). Tang merge ~ cho tokenizer "gioi tieng Anh" dan len,
    # tien gan ve tokenizer that (50k merge). Xem khoang cach Viet-Anh?
    print("\nThi nghiem them - khoang cach Viet/Anh theo so merge:")
    print(f"  {'merges':>6} | {'EN':>5} | {'VI':>5} | ti le VI/EN")
    print("  " + "-" * 38)
    # Chon 3 moc de thay RO xu huong (dung doan truoc khi chay roi doi chieu):
    #   30  = tokenizer "ti hon": chua nen tot ca tieng Anh -> EN van cao.
    #   100 = kha hon: EN giam ro, VI van vun.
    #   200 = gan nhat voi tokenizer that (50k merge): EN tien ve ~1.0,
    #         VI giu ~3.0 -> ti le VI/EN hien ro ~3x.
    # Du doan: cot EN giam dan, cot VI it giam, cot ti le TANG dan.
    for n in (30, 100, 200):
        # [TRAINER] Dung lat cat merges200[:n] thay vi train lai (xem ghi chu
        # train-1-lan o tren) - ket qua so hoc giong het train rieng.
        m = merges200[:n]
        e = sum(fertility(s, m)[0] for s in EN_TEST) / len(EN_TEST)
        v = sum(fertility(s, m)[0] for s in VI_TEST) / len(VI_TEST)
        print(f"  {len(m):>6} | {e:>5.2f} | {v:>5.2f} | {v / e:.1f}x")

    print("\nNhan xet (goi y - hay viet lai bang loi cua ban khi nop bai):")
    print("  O 30 merge, khoang cach Viet-Anh chua ro: tokenizer ti hon nen")
    print("  tieng Anh CUNG kem (chua nen tot), tieng Viet cung bi che vun.")
    print("  Cang tang merge (tokenizer cang 'gioi tieng Anh', tien gan ve")
    print("  tokenizer that 50k merge), tieng Anh cang tien ve ~1 token/tu")
    print("  trong khi tieng Viet giu ~3 token/am tiet -> gap ~3x, khop con so")
    print("  2.5-3.5x cua bai sang. Nghia la:")
    print("  (a) Cung mot y, prompt tieng Viet ton token gap ~3 lan -> tra tien")
    print("      API theo token thi user Viet tra dat hon cho cung luong y tuong.")
    print("  (b) Context window 'nhin' duoc it y tuong tieng Viet hon vi token bi")
    print("      che vun - day la ly do phai do fertility truoc khi chon tokenizer")
    print("      cho san pham tieng Viet (va vi sao can tokenizer hieu tieng Viet).")


# ---------------------------------------------------------------------------
# MAIN
# ---------------------------------------------------------------------------
def main():
    bai1()
    bai2()
    bai3()
    print("\n" + "=" * 64)
    print("XONG - Checklist: 2 merge dau (l,o)->(lo,w) OK; encode OK;")
    print("        bang fertility Viet vs Anh OK.")
    print("=" * 64)


if __name__ == "__main__":
    main()
