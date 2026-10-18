#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Sinh du lieu THUC HANH CO Y BAN cho pipeline Ngay 8.

Tao raw_docs.jsonl (200 docs) voi cac "vet ban" co chu dich de pipeline
co cai ma loc, va test_set.jsonl (10 cau hoi) de kiem tra ro ri eval.

Phan bo 200 docs (seed co dinh -> tai lap duoc):
    20 docs TRUNG CHINH XAC (copy nguyen van)
    15 docs GAN TRUNG (doi 1-2 tu / them cau mo dau)
    10 docs chua PII (email, SDT, so dinh danh gia)
    10 docs rac (qua ngan, qua dai, ky tu lap)
     5 docs KHAC NGON NGU (tieng Anh thuan)
     3 docs O NHIEM EVAL (chua nguyen van 3 cau hoi test)
    137 docs sach tieng Viet
Cong: 20+15+10+10+5+3+137 = 200.

Truong "kind" chi de NGUOI doc kiem chung pipeline (pipeline that
khong duoc phep doc no). Pipeline chi duoc dung id + text.

Cach chay:
    python make_dirty_data.py
"""

import json
import random

RNG = random.Random(42)  # seed co dinh: sinh lan nao cung giong nhau

# ---------------------------------------------------------------------------
# Kho cau tieng Viet sach (100 cau, 5 chu de) - moi doc = 5 cau ghep lai.
# Du lieu co dau day du de bo loc ngon ngu dem duoc ky tu co dau.
# ---------------------------------------------------------------------------
SENTENCES = [
    # Tri tue nhan tao / hoc may (20 cau)
    "Trí tuệ nhân tạo đang thay đổi cách chúng ta làm việc mỗi ngày.",
    "Mô hình ngôn ngữ lớn học được tri thức từ hàng tỷ trang văn bản.",
    "Học sâu cần rất nhiều dữ liệu sạch để đạt hiệu quả cao.",
    "Kỹ sư AI phải hiểu cả toán học lẫn kỹ năng lập trình thực tế.",
    "Mạng nơ-ron gồm nhiều lớp biến đổi dữ liệu từ đơn giản đến phức tạp.",
    "Quá trình huấn luyện mô hình tiêu tốn lượng điện năng đáng kể.",
    "Dữ liệu huấn luyện kém chất lượng khiến mô hình đưa ra câu trả lời sai.",
    "Việc đánh giá mô hình cần tập kiểm tra hoàn toàn tách biệt.",
    "Rò rỉ dữ liệu kiểm tra làm cho điểm số trở nên ảo và gây hiểu lầm.",
    "Loại bỏ dữ liệu trùng lặp giúp tiết kiệm chi phí huấn luyện.",
    "Vector embedding biến từ ngữ thành các con số mà máy tính hiểu được.",
    "Cơ chế chú ý giúp mô hình tập trung vào phần quan trọng của câu.",
    "Mô hình càng lớn thì càng cần nhiều bộ nhớ và thời gian huấn luyện.",
    "Tinh chỉnh mô hình có sẵn thường rẻ hơn huấn luyện từ đầu.",
    "Đạo đức trong AI là vấn đề mà cả ngành công nghiệp đang tranh luận.",
    "Dữ liệu cá nhân trong tập huấn luyện phải được che giấu cẩn thận.",
    "Mô hình ngôn ngữ đôi khi bịa ra thông tin trông rất thuyết phục.",
    "Người dùng nên kiểm chứng lại câu trả lời quan trọng của mô hình.",
    "Nghiên cứu AI tiến bộ nhanh đến mức khó theo kịp mỗi tuần.",
    "Tương lai của AI phụ thuộc vào cách chúng ta quản lý nó hôm nay.",
    # Lap trinh Python (20 cau)
    "Python là ngôn ngữ lập trình phổ biến cho người mới bắt đầu.",
    "Cú pháp của Python gần gũi với ngôn ngữ tự nhiên nên dễ đọc.",
    "Thư viện chuẩn của Python cung cấp sẵn rất nhiều công cụ hữu ích.",
    "Biểu thức chính quy giúp tìm kiếm và thay thế văn bản hiệu quả.",
    "Khi đọc file văn bản, bạn nên chỉ rõ mã hóa UTF-8.",
    "Hàm băm SHA-256 tạo ra dấu vân tay duy nhất cho mỗi chuỗi.",
    "Sắp xếp danh sách trước khi xử lý giúp kết quả ổn định "
    "giữa các lần chạy.",
    "Kiểm thử đơn vị phát hiện lỗi sớm trước khi chúng gây hậu quả lớn.",
    "Vòng lặp for trong Python duyệt qua từng phần tử một cách đơn giản.",
    "Từ điển trong Python lưu trữ cặp khóa và giá trị rất linh hoạt.",
    "Xử lý ngoại lệ giúp chương trình không bị sập khi gặp lỗi.",
    "Viết chú thích rõ ràng giúp người khác hiểu code của bạn nhanh hơn.",
    "Tái cấu trúc code thường xuyên giữ cho dự án gọn gàng và dễ bảo trì.",
    "Môi trường ảo giúp tách biệt các thư viện của từng dự án.",
    "Trình quản lý gói pip giúp cài đặt thư viện chỉ bằng một dòng lệnh.",
    "Ghi log đầy đủ giúp việc gỡ lỗi trở nên dễ dàng hơn nhiều.",
    "Đọc tài liệu chính thức là thói quen tốt của mọi lập trình viên.",
    "Code review giúp cả nhóm học hỏi lẫn nhau và bắt lỗi sớm.",
    "Tự động hóa các tác vụ lặp đi lặp lại tiết kiệm rất nhiều thời gian.",
    "Sao lưu code lên git giúp bạn không bao giờ mất công sức đã bỏ ra.",
    # Am thuc / ca phe Viet Nam (20 cau)
    "Cà phê sữa đá là thức uống quen thuộc của người Việt mỗi sáng.",
    "Phở bò Hà Nội nổi tiếng với nước dùng trong và thơm mùi quế hồi.",
    "Bánh mì Việt Nam được nhiều tạp chí quốc tế bình chọn là món "
    "ăn đường phố ngon nhất.",
    "Cơm tấm sườn bì chả là đặc sản không thể bỏ qua khi đến Sài Gòn.",
    "Trà đá vỉa hè là nét văn hóa bình dị nhưng đầy sức sống.",
    "Bún chả Hà Nội ăn kèm nước mắm chua ngọt và rau sống tươi ngon.",
    "Cà phê trứng có lớp kem béo ngậy phủ trên cà phê đen đậm đà.",
    "Bánh xèo vàng giòn cuốn với rau sống chấm nước mắm rất hấp dẫn.",
    "Chè thái với đủ loại thạch và nước cốt dừa ngọt mát ngày hè.",
    "Mì Quảng đặc trưng với sợi mì vàng và đậu phộng rang giòn.",
    "Hủ tiếu Nam Vang có nước lèo ngọt thanh từ xương hầm.",
    "Bánh cuốn Thanh Trì mỏng tang ăn với chả quế thơm lừng.",
    "Nem rán giòn rụm là món khai vị được nhiều người yêu thích.",
    "Lẩu thái chua cay thích hợp cho những buổi tụ họp bạn bè.",
    "Gỏi cuốn tôm thịt chấm tương đậu phộng béo ngậy khó quên.",
    "Bánh tráng trộn với đủ vị chua cay mặn ngọt rất kích thích.",
    "Sinh tố bơ béo mịn là lựa chọn bổ dưỡng cho buổi sáng.",
    "Cà phê cốt dừa kết hợp vị đắng và béo một cách hài hòa.",
    "Bún bò Huế với nước dùng đậm đà và sả ớt cay nồng.",
    "Cháo lòng nóng hổi là món ăn sáng ấm bụng ngày se lạnh.",
    # Du lich / doi song (20 cau)
    "Hội An lung linh với hàng trăm chiếc đèn lồng đủ màu sắc.",
    "Vịnh Hạ Long được UNESCO công nhận là di sản thiên nhiên thế giới.",
    "Đà Lạt mát mẻ quanh năm với những đồi thông xanh ngát.",
    "Phố cổ Hà Nội nhộn nhịp với ba mươi sáu phố phường truyền thống.",
    "Cố đô Huế trầm mặc với lăng tẩm và đền đài cổ kính.",
    "Phú Quốc có những bãi biển cát trắng và nước biển trong xanh.",
    "Sa Pa mùa đông đôi khi có tuyết phủ trắng xóa đỉnh Fansipan.",
    "Chợ nổi Cái Răng tấp nập ghe thuyền từ sáng sớm tinh mơ.",
    "Hang Sơn Đoòng là hang động tự nhiên lớn nhất thế giới.",
    "Cù Lao Chàm thích hợp cho những ai yêu thích lặn biển ngắm san hô.",
    "Buổi sáng dạo quanh hồ Hoàn Kiếm ngắm người tập thể dục rất thư thái.",
    "Chuyến đi miền Tây cho bạn trải nghiệm miệt vườn sông nước.",
    "Leo núi vào cuối tuần là cách rèn luyện sức khỏe tuyệt vời.",
    "Ngắm hoàng hôn trên biển là khoảnh khắc bình yên khó quên.",
    "Du lịch bụi giúp bạn khám phá những điều không có trong sách hướng dẫn.",
    "Chuẩn bị kỹ hành lý giúp chuyến đi trở nên suôn sẻ hơn nhiều.",
    "Ẩm thực địa phương luôn là điểm nhấn đáng nhớ của mỗi chuyến đi.",
    "Chụp ảnh lưu giữ kỷ niệm nhưng đừng quên tận hưởng khoảnh khắc.",
    "Đi du lịch cùng gia đình tạo ra những kỷ niệm gắn kết bền lâu.",
    "Mỗi vùng đất đều có câu chuyện riêng đang chờ bạn khám phá.",
    # Hoc tap / lam viec (20 cau)
    "Học ngoại ngữ mỗi ngày một ít sẽ tiến bộ hơn học dồn vào cuối tuần.",
    "Đặt mục tiêu rõ ràng giúp bạn duy trì động lực lâu dài.",
    "Ghi chép bằng tay giúp não bộ ghi nhớ kiến thức tốt hơn.",
    "Ôn tập ngắt quãng là phương pháp được khoa học chứng minh hiệu quả.",
    "Dậy sớm cho bạn thêm thời gian yên tĩnh để tập trung làm việc.",
    "Làm việc nhóm tốt đòi hỏi kỹ năng lắng nghe và tôn trọng lẫn nhau.",
    "Thuyết trình tự tin đến từ sự chuẩn bị kỹ lưỡng và luyện tập nhiều lần.",
    "Quản lý thời gian hiệu quả bắt đầu từ việc biết nói không đúng lúc.",
    "Đọc sách mỗi tối giúp mở rộng vốn hiểu biết và vốn từ vựng.",
    "Kiên trì theo đuổi mục tiêu quan trọng hơn tài năng bẩm sinh.",
    "Thất bại là bài học quý giá nếu bạn chịu rút kinh nghiệm.",
    "Sức khỏe tinh thần cũng quan trọng như sức khỏe thể chất.",
    "Tập thể dục đều đặn giúp đầu óc minh mẫn và làm việc hiệu quả hơn.",
    "Ngủ đủ giấc là nền tảng cho một ngày làm việc năng suất.",
    "Cân bằng giữa công việc và cuộc sống giúp bạn đi đường dài.",
    "Học hỏi từ đồng nghiệp giỏi là cách thăng tiến nhanh nhất.",
    "Đừng ngại đặt câu hỏi vì không ai biết hết mọi thứ.",
    "Phản hồi mang tính xây dựng giúp cả nhóm cùng tiến bộ.",
    "Tự động viên bản thân vượt qua những ngày làm việc khó khăn.",
    "Thành công là quá trình tích lũy những nỗ lực nhỏ mỗi ngày.",
]

# Cau tieng Anh cho 5 docs khac ngon ngu (khong dau).
EN_SENTENCES = [
    "Machine learning models learn patterns from large amounts of data.",
    "Clean data is often more valuable than a bigger model.",
    "Engineers should always validate their models on unseen data.",
    "Removing duplicates saves both time and energy during training.",
    "Attention mechanisms help models focus on relevant information.",
    "A good data pipeline reports everything it removes.",
    "Evaluation leakage makes benchmark scores misleading.",
    "Deterministic processing ensures reproducible results.",
    "Hash functions create unique fingerprints for text.",
    "Quality filters remove documents that are too short or too long.",
    "Language detection keeps the training corpus consistent.",
    "Near-duplicate detection uses shingles and Jaccard similarity.",
    "Contamination checks protect the integrity of evaluation.",
    "Sharding splits data into manageable files for training.",
    "Manifests record hashes for every data shard produced.",
    "Data cards document sources, licenses, and filtering steps.",
    "Energy efficiency matters at every stage of the pipeline.",
    "Small models can still learn useful patterns from clean data.",
    "Reproducibility is a cornerstone of serious machine learning.",
    "Documentation is as important as the code itself.",
    "Version control keeps track of every change made.",
    "Automated tests catch bugs before they cause damage.",
    "Clear comments help others understand your code faster.",
    "Regular refactoring keeps projects maintainable.",
    "Backup your work frequently to avoid losing progress.",
]

# Bo test set: 10 cau hoi. 3 cau dau se bi "ro ri" nguyen van vao data.
TEST_SET = [
    {"id": "t01", "question": "Thủ đô của Việt Nam là thành phố nào?",
     "answer": "Hà Nội."},
    {"id": "t02",
     "question": "Hàm băm SHA-256 tạo ra chuỗi dài bao nhiêu bit?",
     "answer": "256 bit."},
    {"id": "t03",
     "question": "Vì sao phải loại bỏ dữ liệu trùng lặp trước khi "
     "huấn luyện mô hình?",
     "answer": "Vì dữ liệu trùng lặp khiến mô hình học đi học lại "
     "điều đã biết, tốn điện và dễ ghi nhớ máy móc."},
    {"id": "t04",
     "question": "Kỹ thuật nào giúp mô hình tập trung vào phần "
     "quan trọng của câu?",
     "answer": "Cơ chế chú ý (attention)."},
    {"id": "t05", "question": "Món ăn nào nổi tiếng nhất của Hà Nội?",
     "answer": "Phở bò."},
    {"id": "t06", "question": "UTF-8 là gì?",
     "answer": "Là chuẩn mã hóa ký tự phổ biến, tương thích ngược với ASCII."},
    {"id": "t07", "question": "Ôn tập ngắt quãng có tác dụng gì?",
     "answer": "Giúp ghi nhớ lâu hơn so với học dồn một lúc."},
    {"id": "t08",
     "question": "Vịnh Hạ Long được UNESCO công nhận vào năm nào?",
     "answer": "Năm 1994."},
    {"id": "t09", "question": "Biểu thức chính quy dùng để làm gì?",
     "answer": "Tìm kiếm và thay thế văn bản theo mẫu."},
    {"id": "t10", "question": "Cà phê trứng có nguồn gốc từ đâu?",
     "answer": "Hà Nội, Việt Nam."},
]

# Tu dong nghia don gian de tao ban "gan trung" (doi 1-2 tu).
SYNONYMS = {
    "rất": "cực kỳ", "tốt": "tuyệt vời", "nhanh": "tốc độ",
    "mỗi ngày": "hằng ngày", "quan trọng": "thiết yếu",
}


def make_doc(sentences):
    """Ghep 5 cau thanh 1 doc (chuoi van ban)."""
    return " ".join(sentences)


def main():
    """Sinh raw_docs.jsonl (200 docs) + test_set.jsonl."""
    docs = []
    idx = 1

    def add(text, kind):
        """Them 1 doc voi id tang dan."""
        nonlocal idx
        docs.append({"id": f"doc-{idx:04d}", "text": text, "kind": kind})
        idx += 1
        return text

    # 1. Docs sach: 137 docs, moi doc 7 cau lay ngau nhien (seed co dinh).
    # Dung 7 cau/doc de 2 docs sach hiem khi chia se du cau tro thanh
    # "gan trung" oan (can Jaccard > 0.8).
    clean_texts = []
    for _ in range(137):
        t = make_doc(RNG.sample(SENTENCES, 7))
        clean_texts.append(t)
        add(t, "clean")

    # 2-3. Lay 35 docs sach RIENG BIET cho 2 loai bien the (khong overlap):
    # neu 1 doc vua bi copy exact vua bi sua near thi so lieu kho giai
    # thich (gia dinh nao bi loai truoc?). Tach rieng -> moi quan he
    # 1-1 ro rang: 20 cap exact, 15 cap near.
    variant_pool = RNG.sample(clean_texts, 35)

    # 2. Trung chinh xac: copy nguyen van 20 docs sach.
    for t in variant_pool[:20]:
        add(t, "exact_dup")

    # 3. Gan trung: doi 1-2 tu hoac them cau mo dau (15 docs).
    for t in variant_pool[20:]:
        t2 = t
        if RNG.random() < 0.5:
            for old, new in RNG.sample(list(SYNONYMS.items()), 2):
                # Chi nhan khi tu can thay THUC SU co trong doc; neu
                # khong, replace() la no-op -> vo tinh tao exact dup!
                if old in t2:
                    t2 = t2.replace(old, new, 1)
            if t2 == t:  # khong doi duoc tu nao -> dung cach prepend
                t2 = "Sau đây là bài viết đáng đọc. " + t
        else:
            t2 = "Sau đây là bài viết đáng đọc. " + t
        assert t2 != t, "near_dup phai khac ban goc"
        add(t2, "near_dup")

    # 4. PII: doc DOC LAP (cau moi, khong copy tu clean) + chen PII.
    # Ly do: neu copy tu clean doc, sau khi scrub no se gan nhu y he ban
    # goc -> near-dedup loai oan. Doc that co PII la doc doc lap.
    pii_pool = [
        "nguyen.van.a@example.com", "tran.thi.b123@mail.vn",
        "le.c.test@company.com.vn",
    ]
    phone_pool = ["0912345678", "0987654321", "0334567890"]
    id_pool = ["123456789", "123456789012", "987654321098"]
    for _ in range(10):
        t = make_doc(RNG.sample(SENTENCES, 7))
        t += (f" Liên hệ: {RNG.choice(pii_pool)}, "
              f"SĐT: {RNG.choice(phone_pool)}, "
              f"CCCD: {RNG.choice(id_pool)}.")
        add(t, "pii")

    # 5. Rac: qua ngan / qua dai / ky tu lap (10 docs).
    for _ in range(4):
        add("Xin chào.", "junk_short")            # 2 tu -> qua ngan
    for _ in range(3):
        add("từ vựng " * 2500, "junk_long")       # 5000 tu -> qua dai
    for s in ["a" * 500, "ab" * 300, "xyz " * 200]:
        add(s, "junk_repeat")                     # ky tu lap -> rac

    # 6. Khac ngon ngu: tieng Anh thuan (5 docs, 7 cau nhu doc sach).
    for _ in range(5):
        add(make_doc(RNG.sample(EN_SENTENCES, 7)), "non_vietnamese")

    # 7. O nhiem eval: 3 docs chua nguyen van 3 cau hoi test.
    for q in TEST_SET[:3]:
        t = make_doc(RNG.sample(SENTENCES, 2))
        t += f" Câu hỏi ôn tập: {q['question']} "
        t += make_doc(RNG.sample(SENTENCES, 2))
        add(t, "contaminated")

    assert len(docs) == 200, f"phai du 200 docs, dang co {len(docs)}"

    # Tron thu tu (seed co dinh) de doc trung khong nam canh nhau.
    RNG.shuffle(docs)
    # Danh lai id theo thu tu sau khi tron.
    for i, d in enumerate(docs, 1):
        d["id"] = f"doc-{i:04d}"

    with open("raw_docs.jsonl", "w", encoding="utf-8") as f:
        for d in docs:
            f.write(json.dumps(d, ensure_ascii=False) + "\n")

    with open("test_set.jsonl", "w", encoding="utf-8") as f:
        for q in TEST_SET:
            f.write(json.dumps(q, ensure_ascii=False) + "\n")

    # Bao cao so luong tung loai de doi chieu voi pipeline.
    from collections import Counter
    print("Da sinh raw_docs.jsonl (200 docs):")
    for kind, n in sorted(Counter(d["kind"] for d in docs).items()):
        print(f"  {kind:>15}: {n}")
    print("Da sinh test_set.jsonl (10 cau hoi).")


if __name__ == "__main__":
    main()
