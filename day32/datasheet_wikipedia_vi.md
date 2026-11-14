# DATASHEET DU LIEU — Wikipedia tieng Viet (Day 32, 14/11/2026)

De tai: Tiny Vietnamese LM — RQ2: BPE 8k vs 16k.
Nguon so: moi con so deu tinh bang code `tokenizer_datasheet.py`
(chay bang venv day15, exit 0, tat ca assert PASS). So nao chua do
thi ghi ro "chua do", khong bia.

## PHAN A — DATASHEET 7 MUC

### 1. Nguon + URL + ngay dump/ban

- Nguon chinh: Wikipedia tieng Viet.
- URL dump: `dumps.wikimedia.org/viwiki/latest/`
- File: `viwiki-latest-pages-articles.xml.bz2`, ban 01/11/2026.
- Nguon du phong: mau CulturaX-vi (xem muc 6).

### 2. Giay phep tung nguon

- Wikipedia: van ban duoc cap phep CC BY-SA 4.0 (kem GFDL khong
  ghi phien ban) — da tra cuu trang Wikipedia:Copyrights ngay
  14/11/2026. Nghia la: duoc tai xuong, sua doi, train mo hinh,
  nhung san pham phai sinh phai ghi nguon va chia se lai cung
  giay phep tuong tu.
- CulturaX-vi (du phong): theo dieu khoan Hugging Face, dataset
  gated — can dong y dieu khoan truoc khi tai, khong tai toan bo,
  chi lay mau vai tram MB khi can da dang.

### 3. Kich thuoc

- File nen: 1.165.020.800 byte = 1,085 GiB (1,165 GB thap phan),
  tinh bang code `bytes_to_gib`.
- Luu y: de bai ghi "~1,11 GiB" la lam tron sai (lay 1.110,9 MiB
  chia cho 1.000 thay vi 1.024). So dung de ghi vao datasheet la
  1,085 GiB.
- Kich thuoc sau giai nen: chua do — do that o Day 34 roi bo sung.
- So token sau loc: chua do — do that o Day 33–34. Gia dinh
  "~200M token" dung sau quyet dinh N ~ 10M van la gia dinh,
  chua duoc kiem chung (kiem chung o Day 33).

### 4. Cach loc du kien — 3 quy tac, moi quy tac kem ly do

Thu tu loc co dinh: NFC -> dedup -> bo dong rac. Thu tu nay cung
bi dong bang: khong doi thu tu khi thay ket qua xau.

- Quy tac 1 — Chuan hoa NFC truoc moi buoc khac.
  Ly do: cung mot chu co dau co the ma hoa thanh hai day ky tu
  khac nhau (vi du "truong": NFC 6 ky tu, NFD 9 ky tu — code Bai 2
  da chung minh). Khong chuan hoa thi tokenizer hoc hai token cho
  cung mot chu, lang phi vocab va do fertility sai.
- Quy tac 2 — Dedup theo hash bai viet, roi dedup theo hash dong.
  Ly do: Wikipedia co bai trung lap (ban dich may nhan ban, trang
  dinh huong, template lap lai). Du lieu trung lap day model hoc
  thuoc long thay vi khat quat, dong thoi lam phinh so token ao
  (anh huong truc tiep den tinh toan Chinchilla o Day 33).
- Quy tac 3 — Bo dong qua ngan (< 20 ky tu sau khi bo khoang
  trang) va dong co > 30% ky tu la (ky tu khong thuoc bang chu
  Latin + dau tieng Viet + chu so + dau cau co ban).
  Ly do: dong ngan thuong la muc luc, chu thich, tieu de muc —
  khong day ngon ngu; dong nhieu ky tu la thuong la ma wiki con
  sot, spam link, hoac van ban ngoai ngu lot vao.
  Nguong 20 ky tu / 30% la quyet dinh hom nay; Day 33 do phan phoi
  do dai dong that va ghi lai ti le dong bi loai — chi ghi so,
  khong sua nguong.

### 5. Cach tach held-out — DONG BANG

- Tach mot lan duy nhat, truoc khi train bat ky mo hinh nao.
- Tach theo bai viet (article-level), khong tach theo dong:
  cac dong cua cung mot bai khong duoc nam o ca hai phia train
  va held-out (tranh ro ri noi dung).
- Seed co dinh 42, ti le 1% so bai viet vao held-out.
- Sau khi tach, danh sach file held-out duoc dong bang: khong doi
  tap, khong doi seed, khong doi ti le giua cac lan chay.
- Day 34 chi duoc thay doi so merge cua BPE (8k vs 16k). Moi thay
  doi cach loc hay cach tach sau khi thay ket qua deu la doi luat
  sau khi xem dap an — ket qua se mat tinh so sanh cong bang.
- Chi tiet thuc thi (ten file danh sach, so bai cu the): ghi bo
  sung khi tach that o Day 33–34.

### 6. Trang thai truy cap tung nguon tai thoi diem chot (14/11/2026)

- Wikipedia-vi: song, dump 01/11/2026 tai duoc.
- CulturaX-vi: gated tren Hugging Face — can dong y dieu khoan,
  chi lay mau khi can.
- OSCAR-2301: tam dinh chi truy cap — DA LOAI, khong dung.

### 7. Rui ro da biet + cach xu ly

- Bai viet chat luong khong deu, co the lan ban dich may.
  Xu ly: loc theo Quy tac 2 + 3 o muc 4, kem kiem tra mau bang
  tay truoc khi train — doc ngau nhien ~50 bai, ghi ti le bai
  loi (dich may, cau vo nghia) vao nhat ky Day 33. Neu ti le loi
  cao thi bo sung buoc loc heuristic (vi du: cau lap mau dich
  may) — nhung chi bo sung truoc khi chay Day 34, khong sua
  giua chung.
- Van ban bach khoa thien van phong trang trong, it hoi thoai
  va ngon ngu doi thuong — PPL tren held-out Wikipedia co the
  khong phan anh het kha nang viet tu nhien. Ghi nhan la gioi
  han, khong sua tap eval de "dep so".
- So token thuc te co the lech xa gia dinh ~200M token. Neu lech
  xa, N ~ 10M phai tinh lai theo Chinchilla (Day 33) — do la
  ly do Day 33 do corpus that truoc khi chot P.

## PHAN B — TRA LOI 5 CAU HOI DINH HUONG

### Cau 1 — BPE 8k vs 16k khac nhau o dau, gia cua 16k la bao nhieu?

Voi cung corpus, vocab 16k chia moi tu thanh it manh hon nen chuoi
token ngan hon; vocab 8k chia manh hon, chuoi dai hon. Gia cua 16k
voi d_model = 256: embedding 16.000 x 256 = 4.096.000 tham so
(~41,0% ngan sach 10M), hon vocab 8k dung 2.048.000 tham so
(2.048.000, ~20,5%). Chon 16k nghia la lay 2M tham so tu phan
transformer (chi con ~5,9M thay vi ~8,0M) de tra cho embedding.

### Cau 2 — Rust 2021 lam tren BERT encoder, cai gi be duoc sang decoder LM?

Be duoc: nguyen ly "tokenizer chuyen cho mot ngon ngu giup mo hinh
hieu ngon ngu do tot hon" — vi hieu ung nay nam o dau vao (cach
chia token), khong phu thuoc kien truc encoder hay decoder.
Khong be duoc: muc cai thien cu the theo tac vu (Rust do bang
fine-tune classification/NER tren BERT; decoder LM do bang PPL
va sinh van ban) va ket luan ve pretraining MLM — chung khong
noi gi ve kha nang sinh van ban tu hoi cua decoder. Day 34 phai
do lai bang PPL chu khong lay so cua Rust lam ket luan.

### Cau 3 — Fertility thap chua chac mo hinh tot hon, Day 34 do them gi?

Do them held-out perplexity tren cung mot tap held-out da dong
bang o muc 5 — do moi la eval chinh cua de tai (Ali 2024 chung
minh fertility khong phai luc nao cung du bao chat luong
downstream). Kem theo do dai chuoi trung binh (token/cau) de thay
chi phi suy luan thuc te cua moi tokenizer.

### Cau 4 — Vi sao Tiny LM khong copy vocab 20.480 cua PhoGPT?

PhoGPT la mo hinh ti tham so, d_model lon, vocab 20.480 chi la
phan nho trong tong so. Voi d_model = 256 cua Tiny LM, vocab
20.480 an 20.480 x 256 = 5.242.880 tham so (~52,4% ngan sach
10M), chi con ~4,76M cho toan bo transformer — khong du cho
mo hinh hoc duoc gi dang ke. Copy con so ma khong copy quy mo
mo hinh la sai.

### Cau 5 — NFC giai quyet van de gi cua tieng Viet? Vi du cu the?

Tieng Viet co dau to hop: cung mot chu hien thi giong nhau nhung
ma hoa thanh hai day byte khac nhau — dang dung san (NFC) va dang
dau to hop (NFD). Vi du "truong": NFC 6 ky tu
(0x74, 0x72, 0x1b0, 0x1edd, 0x6e, 0x67), NFD 9 ky tu
(0x75, 0x31b, 0x6f, 0x31b, 0x300, ...). Gon nhat: chu "e" —
NFC 1 ky tu (0x1ec7), NFD 3 ky tu (0x65, 0x323, 0x302).
Khong chuan hoa NFC, tokenizer thay hai token khac nhau cho cung
mot chu.

## PHAN 5 — TRA LOI 4 CAU HOI PHAN BIEN (muc 5 cua de)

### Phan bien 1 — "16k cho chuoi ngan hon, train nhanh hon, sao khong chon luon 16k?"

Ngoai con so 41%, 16k co van de ve tan suat: voi corpus ~200M
token, moi token trong vocab 16k trung binh duoc gap it lan hon
trong vocab 8k, nen embedding cua token hiem hoc kem chac hon.
8k giu moi token "duoc gap" nhieu lan hon, hoc vung hon tren du
lieu it. Chuoi ngan hon chi tiet kiem chut thoi gian train, khong
doi chat luong sinh — khong dang tra 2M tham so neu PPL khong
cai thien ro.

### Phan bien 2 — "Bang chung Rust 2021 khong ap dung cho decoder LM cua ban"?

Dap: diem dung vung cua Rust la hieu ung nam o dau vao — tokenizer
khop ngon ngu giam fertility va giam so token phai xu ly, loi nay
co cho ca encoder lan decoder vi cung la dau vao. Diem phai thua
nhan la gioi han: Rust do tren tac vu hieu cua BERT, chua chung
minh tokenizer don ngu cai thien sinh van ban tu hoi hay PPL cua
decoder. Ket luan cho decoder hien la suy luan, se kiem chung
bang PPL o Day 34 — neu PPL khong cai thien, se thua nhan gioi
han do trong bao cao.

### Phan bien 3 — "Wikipedia chat luong khong deu, sao khong cho xin bo du lieu sach hon?"

Vi cho la mat thoi gian ma khong co ngay chac chan duoc quyen;
ca de tai dung yen trong luc cho, mat nhip Day 33–34. Wikipedia
da co san, du lon, giay phep ro rang (CC BY-SA 4.0), va rui ro
chat luong xu ly duoc bang loc + kiem tra mau (muc 7). Bo du lieu
"sach hon" cung khong dam bao khop phan phoi Wikipedia — ma eval
cua de tai van do tren Wikipedia.

### Phan bien 4 — "Day 34, 16k tot hon 8k chi 0,2 PPL — chot cai nao?"

Tieu chi de xuat truoc khi do: 0,2 PPL tren thang ~7–8 la khoang
2,5% tuong doi — qua nho so voi cai gia 2M tham so (20% ngan
sach). Chi chot 16k neu cai thien ro rang, vi du >= 0,5 PPL
(khoang 6–7%) hoac PPL tot hon kem chuoi ngan hon du bu chi phi
suy luan. Neu chi 0,2 thi chot 8k, danh tham so cho transformer.
Day la tieu chi de xuat hom nay, se chot cung sau khi do baseline.

## KET QUA CHAY CODE (da kiem chung)

- vocab 8.000: 2.048.000 tham so = 20,5% ngan sach 10M,
  transformer con lai 7.952.000.
- vocab 16.000: 4.096.000 tham so = 41,0% ngan sach 10M,
  transformer con lai 5.904.000.
- vocab 20.480: 5.242.880 tham so = 52,4% ngan sach 10M,
  transformer con lai 4.757.120.
- Dump Wikipedia-vi: 1.165.020.800 byte = 1,085 GiB.
- "truong": NFC 6 ky tu, NFD 9 ky tu (dai hon 3).
- "e": NFC 1 ky tu, NFD 3 ky tu (dai hon 2).

## QUY TAC DONG BANG (ghi de Day 34 khoi tranh cai lai)

Thu duy nhat thay doi khi so sanh 8k vs 16k o Day 34 la so merge
cua BPE. Corpus, cach loc (muc 4), cach tach held-out (muc 5),
d_model = 256 va moi sieu tham so khac deu giu nguyen.

## PHAN D — SCOPE SHEET DAY 31

Da xong tu Day 31 (`day31/thesis_scope_sheet.md`, commit 39aeed2):
N ~ 10M la con so toi uu Chinchilla tren corpus ~200M token;
P (nguong PPL) va Y (nguong tiet kiem nang luong) ghi "chot sau
baseline", khong bia so. Hom nay khong can dien them.

## DANH SACH "CHUA DO" (bo sung o Day 33–34)

- Kich thuoc dump sau giai nen.
- So bai viet, so token tho sau loc NFC + dedup.
- Ti le dong bi loai boi Quy tac 3 (de kiem tra nguong 20 ky tu / 30%).
- Danh sach file held-out cu the (sau khi tach that, seed 42).
- Fertility + held-out PPL cua BPE 8k vs 16k (Day 34).
