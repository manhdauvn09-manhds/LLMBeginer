# THESIS SCOPE SHEET — Tiny Vietnamese LM (Ngay 31, 10/11/2026)

## 1. Ten de tai + van de

Tiny Vietnamese LM — mo hinh ngon ngu nho, chay duoc tren may thuong, tiet kiem nang luong cho tieng Viet.
Van de: cac mo hinh tieng Viet tot hien nay deu lon, nang, ton dien; mot mo hinh nho co viet duoc tieng Viet mach lac khong — va cai gia phai tra la bao nhieu?

## 2. Ba cau hoi nghien cuu (RQ1–RQ3)

- RQ1 — Kich thuoc vs chat luong: voi ngan sach 10–50M tham so, mo hinh dat held-out perplexity bao nhieu tren Wikipedia tieng Viet, va duong cong PPL theo kich thuoc doc toi dau (do o 10M, 25M, 50M)?
- RQ2 — Tokenizer: BPE 8k vs 16k (deu chuan hoa NFC) thay doi PPL va do dai chuoi trung binh bao nhieu tren cung mot mo hinh 10M?
- RQ3 — Nang luong: so voi baseline lon gap 4 lan tham so (40M), mo hinh 10M tiet kiem bao nhieu nang luong cho moi 1.000 token sinh ra, voi muc tang PPL la bao nhieu?

## 3. Gia thuyet kiem chung duoc dau tien + dieu kien bac bo

Mo hinh 10M tham so, train ~200M token Wikipedia-vi (khoang 1 epoch), dat PPL thap hon baseline n-gram tren cung tap held-out Wikipedia-vi.
Bac bo neu: PPL >= baseline n-gram, hoac sau khi chot tran P (Day 38–44, sau khi do baseline) ma PPL vuot tran P.
Tran P cu the: chot sau baseline — khong dien so cam tinh.

## 4. Du lieu + tokenizer + eval (co dinh)

Du lieu chinh Wikipedia tieng Viet; du phong mau CulturaX-vi; OSCAR-2301 da loai (tam dinh chi tren Hugging Face).
Tokenizer BPE 8k–16k + chuan hoa NFC (8k vs 16k la RQ2, khong phai lua chon cam tinh).
Eval: held-out perplexity tren tap held-out Wikipedia-vi tach tu dau (Day 34); cung tokenizer, cung cach tinh cho moi lan chay; do xen ke A/B, lay trung vi; nang luong tinh tren 1.000 token.

## 5. Tieu chi thanh cong

Mo hinh <= 10M tham so dat PPL < [P] tren held-out Wikipedia-vi, trong do P chot sau khi do baseline (Day 38–44); tiet kiem >= [Y]% nang luong/token so voi baseline 40M, trong do Y chot sau baseline.
Ca P va Y hien ghi "chot sau baseline" — khong bia so.

## 6. Timeline 31–60 + viec ngay mai (Day 32)

31–37 doc paper · 38–44 de cuong + baseline · 45–53 thi nghiem A/B · 54–60 viet bao cao + bao ve.
Ngay mai (Day 32, 11/11): chot nguon du lieu + viet datasheet (nguon, license, kich thuoc, 3 quy tac loc kem ly do) + 2 paper ve tokenizer tieng Viet.

---

# 5 CAU HOI PHAN BIEN — TRA LOI (muc 4, Day31_practice.html)

## Q1 — Chinchilla tren tieng Viet

Luat Chinchilla duoc fit tren tieng Anh, no gia dinh ngam ve luong thong tin moi token mang.
Tieng Viet khac: tu ngan, nhieu tu ghep, dau thanh lam tang da dang ky tu, ti le token/tu cua BPE khac tieng Anh — nen cung 1 token tieng Viet co the mang it/nhieu thong tin hon 1 token tieng Anh.
Vi vay ti le toi uu ~20 token/tham so co the lech; chi tin sau khi kiem chung bang pilot 10M, khong lay luat tieng Anh lam chan ly cho tieng Viet.

## Q2 — Chon N

Chon N = 10M: corpus Wikipedia-vi uoc ~200M token, theo Chinchilla ~20 token/tham so thi 10M la co can bang voi dung 1 epoch — khong can bien minh them.
Neu chon lon hon (25M/50M) phai bao ve bang multi-epoch (hieu qua giam dan sau ~4 epoch, theo literature) hoac bo sung CulturaX-vi (rui ro lech phan phoi so voi Wikipedia).
Gioi han cua ca hai cach bao ve: multi-epoch hoc lap lai cung du lieu, du lieu bo sung doi phan phoi train — ca hai deu co the lam PPL tren held-out Wikipedia-vi khong cai thien nhu ky vong.

## Q3 — "Mach lac" do bang gi

PPL chi do kha nang doan tu tiep theo, khong do ngu phap, mach lac chu de hay lap tu — mot model PPL thap van co the sinh cau lap vo nghia.
Do them: cham mau sinh boi nguoi ban ngu (hoac LLM manh lam giam khao voi rubric co dinh: ngu phap, bam chu de, khong lap) tren cung bo prompt co dinh, kem ti le lap n-gram.
PPL mot minh la chua du vi no khong phan biet "de doan" voi "viet hay": van ban lap lai de doan nen PPL thap nhung doc khong mach lac.

## Q4 — Wikipedia vs TinyStories

Wikipedia la van ban that: tu vung chuyen nganh rong, cau dai phuc tap, nhieu thuc the rieng, nhieu the loai va thanh ghi trang trong.
TinyStories la truyen tong hop: von tu ~1.500 tu, cau ngan, mot giong van ke chuyen, noi dung don gian.
Dieu do de doa ket luan "model nho + data don gian van mach lac": mach lac tren truyen don gian khong dam bao mach lac tren van ban bach khoa phuc tap — model co the can dung luong lon hon de xu ly do da dang nay.

## Q5 — Bay eval

Doi tap held-out giua hai lan chay de "so dep hon" la cherry-picking: so PPL khong con so duoc voi nhau, ket luan "tot hon" la gia.
Ba thu phai co dinh de hai con so PPL so duoc: (1) cung tokenizer — cung vocab va merges; (2) cung tap held-out — cung cach tach va tien xu ly; (3) cung cach tinh — cung do dai ngu canh, cung cach tinh trung binh theo trong so token, cung xu ly BOS/EOS.
