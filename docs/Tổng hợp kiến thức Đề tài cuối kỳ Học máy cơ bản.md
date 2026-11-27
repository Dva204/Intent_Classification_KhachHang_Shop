# Tổng hợp kiến thức: Đề tài cuối kỳ Học máy cơ bản

**Đề tài:** Phân loại ý định (intent classification) câu hỏi của khách hàng shop online bằng tiếng Việt

**Thuật toán:** Naive Bayes, SVM (bắt buộc đủ 2) và Decision Tree (tuỳ chọn)

---

## 1. Yêu cầu của đề thi

- Huấn luyện mô hình giải quyết bài toán **phân lớp hoặc hồi quy**.
- Dùng **ít nhất 2 thuật toán** đã học trong học phần Học máy cơ bản (L3 Decision Trees, L4 Naive Bayes, L5 SVMs, L6 Neural Network). Có thể dùng thêm thuật toán khác.
- Tiêu chí chấm điểm: **đóng góp trong việc phân tích, xử lý đặc trưng, xử lý dữ liệu và phát triển mô hình.**
- Mỗi nhóm chuẩn bị **1 máy chứa code backend** của mô hình.
- **Không được dùng comment trong code.**

**Cách đề tài này đáp ứng yêu cầu**

| Yêu cầu | Đáp ứng |
| --- | --- |
| Phân lớp | Phân loại câu thành 5 nhóm ý định |
| Ít nhất 2 thuật toán | Naive Bayes + SVM (thêm Decision Tree nếu muốn) |
| Xử lý dữ liệu, đặc trưng | Làm sạch văn bản, thử BoW / TF-IDF / bigram |
| Backend trên 1 máy | Chương trình nhỏ (hoặc API) nhận câu và trả về nhóm |
| Không comment | Đặt tên hàm, biến rõ nghĩa, rà lại trước khi nộp |

---

## 2. Bài toán

Khách nhắn một câu, mô hình đoán câu đó thuộc nhóm nào. Chỉ nên làm **5 nhóm** để vừa sức:

| Nhóm | Ví dụ |
| --- | --- |
| Hỏi giá | "Áo này bao nhiêu vậy shop?" |
| Hỏi tình trạng đơn | "Đơn mình bao giờ tới?" |
| Đổi / trả hàng | "Mình muốn trả lại cái quần" |
| Hỏi size / còn hàng | "Còn size M không?" |
| Khác (chào, cảm ơn...) | "Cảm ơn shop" |

**Vì sao đơn giản?**

- Máy không đọc được chữ, nên ta chuyển câu thành số bằng **TF-IDF**.
- Sau đó đưa số vào thuật toán như bài toán dữ liệu dạng bảng.
- Chỉ cần `scikit-learn`, không cần GPU hay NLP sâu.

Đề tài này nằm trong hướng chatbot chăm sóc khách hàng, nên làm xong có thể tái sử dụng cho đồ án lớn hơn sau này.

---

## 3. Dữ liệu

- Mỗi nhóm khoảng **100 câu**, tổng khoảng **500 câu** (nhiều hơn càng tốt nếu còn thời gian).
- Cách làm nhanh: mỗi thành viên viết khoảng 25 câu mỗi nhóm, hoặc nhờ AI sinh rồi cả nhóm đọc lại, sửa.
- Cố ý thêm câu **viết tắt, sai chính tả, không dấu** ("ko", "dc", "sp", "ship", "bao nhieu") cho giống thực tế và có nội dung để phân tích trong báo cáo.
- Lưu thành file CSV có 2 cột: `cau_hoi`, `nhan`.
- Kiểm tra số câu mỗi nhóm có đều nhau không (cân bằng nhãn).

---

## 4. Quy trình thực hiện

### Bước 1. Xử lý dữ liệu

- Chuyển chữ thường, bỏ dấu câu và ký tự thừa.
- Thay các từ viết tắt phổ biến bằng từ đầy đủ ("ko" thành "không").
- Chia dữ liệu **80% để học, 20% để kiểm tra**, dùng `stratify` để mỗi nhóm có tỉ lệ như nhau ở hai phần.

### Bước 2. Xử lý đặc trưng

- Thử 2 cách biến câu thành số: **Bag-of-Words** (đếm từ) và **TF-IDF**.
- Thử thêm **bigram** (cặp từ liền nhau), vì "trả hàng" mang nghĩa khác "trả" và "hàng" đứng riêng.
- Ghi lại cách nào cho kết quả tốt hơn.

### Bước 3. Huấn luyện và so sánh

- Train Naive Bayes và SVM trên các cách đặc trưng ở bước 2.
- Lập bảng so sánh accuracy của từng tổ hợp.
- Chỉnh tham số `C` của SVM bằng GridSearch (thử 0.1, 1, 10).

### Bước 4. Đánh giá

- Dùng accuracy, precision, recall, F1 theo từng nhóm.
- Vẽ **ma trận nhầm lẫn** (confusion matrix).
- Xem các câu dự đoán sai và giải thích lý do. Ví dụ: "Mình muốn trả hàng vì bị lỗi" dễ bị nhầm giữa "đổi trả" và "khiếu nại".

### Bước 5. Demo

- Viết chương trình nhỏ nhận một câu và in ra nhóm dự đoán cùng độ tin cậy, chạy trên máy của nhóm.
- **Không để comment trong code.**

---

## 5. Kiến thức nền cần hiểu

### 5.1. TF-IDF

TF-IDF cho mỗi từ trong câu một trọng số:

- **TF (term frequency):** từ xuất hiện càng nhiều trong câu thì điểm càng cao.
- **IDF (inverse document frequency):** từ xuất hiện trong càng ít câu của cả tập dữ liệu thì điểm càng cao. Các từ phổ biến như "shop", "ơi" bị giảm trọng số.

Kết quả: mỗi câu trở thành một vector số. Mỗi từ là một chiều, nên số chiều rất lớn (hàng trăm đến hàng nghìn).

### 5.2. Naive Bayes

- **Ý tưởng:** tính xác suất câu thuộc từng nhóm dựa trên các từ xuất hiện trong câu, rồi chọn nhóm có xác suất cao nhất.
- **Giả định "ngây thơ":** các từ độc lập với nhau. Giả định này không hoàn toàn đúng nhưng vẫn cho kết quả tốt với văn bản.
- **Ưu điểm:** rất nhanh, đơn giản, chạy tốt với dữ liệu ít. Phù hợp làm **baseline**.
- **Nhược điểm:** bỏ qua mối quan hệ giữa các từ.

### 5.3. SVM (Support Vector Machine)

**Ý tưởng cốt lõi:** tìm một đường ranh giới chia các nhóm ra khỏi nhau, và chọn đường **an toàn nhất**, tức nằm xa cả hai nhóm nhất có thể.

```
  ●  ●                    ▲  ▲
    ●   ●      |         ▲   ▲
  ●   ●        |           ▲
    ●          |         ▲  ▲
          đường ranh giới
```

Vì sao chọn đường xa nhất? Nếu một điểm mới hơi lệch, đường nằm sát một nhóm dễ phân loại sai, còn đường nằm giữa vẫn đúng. Mô hình vì vậy **ổn định và tổng quát tốt hơn**.

**Các khái niệm cần nhớ**

| Khái niệm | Ý nghĩa |
| --- | --- |
| **Margin (lề)** | Khoảng cách từ đường ranh giới đến điểm gần nhất của mỗi nhóm. SVM cố làm margin lớn nhất. |
| **Support vectors** | Các điểm sát đường ranh giới nhất. Chỉ các điểm này quyết định vị trí đường ranh giới. |
| **Hyperplane (siêu phẳng)** | Đường ranh giới. 2 chiều là đường thẳng, nhiều chiều gọi chung là siêu phẳng. |

**Tham số C (khi dữ liệu không chia sạch được)**

| C | Ý nghĩa | Hệ quả |
| --- | --- | --- |
| C nhỏ | Cho phép sai vài điểm, ưu tiên margin rộng | "Dễ tính", ít overfit |
| C lớn | Cố không sai điểm nào, margin hẹp lại | "Khắt khe", dễ overfit |

**Kernel**

- **Linear kernel:** chia bằng đường thẳng. Đơn giản, nhanh.
- **RBF kernel:** chia bằng đường cong, mềm dẻo hơn nhưng chậm hơn và cần chỉnh thêm tham số.
- **Với văn bản TF-IDF, linear kernel thường là đủ tốt** vì số chiều rất lớn nên dữ liệu thường đã chia được bằng đường thẳng. Đó là lý do dùng `LinearSVC`.

**Chia nhiều nhóm:** SVM gốc chỉ chia 2 nhóm. Với 5 nhóm, scikit-learn dùng cách **một-với-tất-cả (one-vs-rest)**: huấn luyện 5 bộ phân loại, mỗi bộ trả lời "câu này có thuộc nhóm X không?", rồi xếp vào nhóm có điểm cao nhất. Bạn không cần tự code.

**Vì sao SVM thường tốt với văn bản**

- Hoạt động tốt khi số chiều rất lớn.
- Hoạt động tốt kể cả khi dữ liệu ít.
- Chỉ phụ thuộc vào support vectors nên ít bị nhiễu từ các điểm "dễ".

**Nhược điểm:** khó giải thích hơn Decision Tree, chậm với dữ liệu rất lớn (không ảnh hưởng bài thi này).

### 5.4. Decision Tree (tuỳ chọn)

- **Ý tưởng:** hỏi liên tiếp các câu hỏi có/không về đặc trưng (ví dụ "câu có chứa từ 'size' không?") để đi đến nhóm cuối.
- **Ưu điểm:** dễ giải thích, vẽ được hình cây, xem được từ nào quan trọng.
- **Nhược điểm với văn bản:** TF-IDF tạo rất nhiều cột nên cây dễ **overfit** (điểm train rất cao, điểm test thấp). Thường kết quả **thấp hơn SVM**, và đó là một phát hiện tốt để viết vào báo cáo.
- **Cách giảm overfit:** giới hạn `max_depth` (5, 10, 20), đặt `min_samples_leaf` (2, 5), giảm số đặc trưng bằng `max_features` của TF-IDF (500 hoặc 1000).

### 5.5. So sánh nhanh 3 thuật toán

|  | Naive Bayes | SVM | Decision Tree |
| --- | --- | --- | --- |
| Ý tưởng | Tính xác suất từng nhóm theo từ xuất hiện | Tìm đường chia có margin lớn nhất | Hỏi liên tiếp các câu có/không |
| Tốc độ | Rất nhanh | Nhanh (linear) | Nhanh |
| Dễ giải thích | Trung bình | Khó | Dễ |
| Với văn bản | Tốt cho baseline | Thường tốt nhất | Dễ overfit |

---

## 6. Có cần dùng Decision Tree không?

**Không bắt buộc.** Đề chỉ yêu cầu ít nhất 2 thuật toán, nên Naive Bayes + SVM đã đủ.

**Nên bỏ nếu:** nhóm ít thời gian, muốn báo cáo gọn, hoặc muốn hiểu chắc 2 thuật toán hơn là dàn trải ra 3.

**Nên thêm nếu:** cả nhóm đã hiểu bài L3, vì code tốn rất ít công (thêm 1 dòng) và Decision Tree cho tài liệu để viết phần phân tích đặc trưng (top từ quan trọng, hình cây).

### Ba cách kết hợp Decision Tree

1. **Làm mô hình thứ 3 để so sánh** (dễ nhất, nên làm nếu dùng).
2. **Dùng để giải thích mô hình:** in top từ quan trọng bằng `feature_importances_`, vẽ cây bằng `plot_tree` (giới hạn độ sâu 3–4).
3. **Ghép nhiều mô hình** (nâng cao, không bắt buộc): `VotingClassifier` cho các mô hình bỏ phiếu, hoặc Random Forest (chỉ nhắc như phần mở rộng nếu môn chưa dạy).

---

## 7. Code mẫu (không có comment)

```python
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import Pipeline
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import LinearSVC
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.metrics import classification_report, accuracy_score
import matplotlib.pyplot as plt

data = pd.read_csv("data.csv")
x_train, x_test, y_train, y_test = train_test_split(
    data["cau_hoi"], data["nhan"], test_size=0.2, stratify=data["nhan"], random_state=42
)

def build_pipeline(classifier):
    return Pipeline([
        ("tfidf", TfidfVectorizer(ngram_range=(1, 2), max_features=1000)),
        ("clf", classifier),
    ])

models = {
    "naive_bayes": build_pipeline(MultinomialNB()),
    "svm": build_pipeline(LinearSVC()),
    "decision_tree": build_pipeline(DecisionTreeClassifier(max_depth=10, min_samples_leaf=2, random_state=42)),
}

for name, model in models.items():
    model.fit(x_train, y_train)
    predictions = model.predict(x_test)
    print(name, accuracy_score(y_test, predictions))
    print(classification_report(y_test, predictions))

tree_pipeline = models["decision_tree"]
vocabulary = tree_pipeline.named_steps["tfidf"].get_feature_names_out()
importances = tree_pipeline.named_steps["clf"].feature_importances_
top_features = sorted(zip(vocabulary, importances), key=lambda item: item[1], reverse=True)[:15]
for word, score in top_features:
    print(word, round(score, 4))

plt.figure(figsize=(18, 8))
plot_tree(
    tree_pipeline.named_steps["clf"],
    feature_names=vocabulary,
    class_names=tree_pipeline.named_steps["clf"].classes_,
    max_depth=3,
    filled=True,
)
plt.savefig("decision_tree.png", dpi=150)
```

**Ghép mô hình bằng Voting (nâng cao, không bắt buộc)**

```python
from sklearn.ensemble import VotingClassifier
from sklearn.calibration import CalibratedClassifierCV

voting_model = Pipeline([
    ("tfidf", TfidfVectorizer(ngram_range=(1, 2), max_features=1000)),
    ("clf", VotingClassifier(
        estimators=[
            ("nb", MultinomialNB()),
            ("svm", CalibratedClassifierCV(LinearSVC())),
            ("dt", DecisionTreeClassifier(max_depth=10, min_samples_leaf=2, random_state=42)),
        ],
        voting="soft",
    )),
])
voting_model.fit(x_train, y_train)
print(accuracy_score(y_test, voting_model.predict(x_test)))
```

**Lưu ý về comment:** các đoạn code trên không có dòng `#` nào. Trước khi nộp, hãy tìm trong toàn bộ file xem còn ký tự `#` dùng làm comment hay không (cả dòng đơn lẻ lẫn cuối dòng). Chuỗi dạng docstring ba dấu nháy cũng nên bỏ để chắc chắn.

---

## 8. Khung báo cáo gợi ý

1. Giới thiệu bài toán và ứng dụng
2. Dữ liệu (cách thu thập, số lượng, ví dụ, phân bố nhãn)
3. Xử lý dữ liệu và đặc trưng (BoW, TF-IDF, bigram, kết quả so sánh)
4. Các thuật toán sử dụng (giải thích ngắn gọn bằng lời của nhóm)
5. Kết quả và so sánh (bảng accuracy, confusion matrix)
6. Phân tích lỗi và hướng cải thiện
7. Demo

**Nếu dùng Decision Tree, thêm vào báo cáo:**

- Nhận xét so sánh với hai mô hình còn lại (thường SVM tốt nhất, Decision Tree dễ overfit hơn).
- So sánh `max_depth` = 3, 10 và không giới hạn để cho thấy hiệu quả của việc giới hạn độ sâu.
- Top từ quan trọng và hình cây để giải thích mô hình "nhìn" vào đâu.

---

## 9. Kế hoạch thời gian ước lượng

| Việc | Thời gian |
| --- | --- |
| Tạo dữ liệu | 1 ngày |
| Xử lý + huấn luyện + đánh giá | 1–2 ngày |
| Demo + báo cáo | 1–2 ngày |

Nếu thấy nặng, có thể giảm xuống **3 nhóm** và bỏ bước thử bigram mà vẫn đạt yêu cầu đề.

---

## 10. Phương án dự phòng

- **Dự đoán giá nhà / giá xe cũ (hồi quy):** dùng dữ liệu bảng có sẵn trên Kaggle. Dễ làm nhưng ít đặc sắc.
- **Phân loại bình luận tích cực / tiêu cực (sentiment):** cùng kỹ thuật với đề tài chính nhưng chỉ có 2–3 nhãn nên nhẹ hơn.

---

## 11. Câu hỏi có thể bị hỏi khi vấn đáp

**Vì sao dùng TF-IDF thay vì chỉ đếm từ?** TF-IDF giảm trọng số các từ xuất hiện ở hầu hết các câu (ít thông tin) và tăng trọng số các từ đặc trưng cho từng nhóm.

**SVM hoạt động thế nào?**

> SVM tìm một siêu phẳng phân chia các lớp sao cho khoảng cách (margin) từ siêu phẳng đến các điểm gần nhất của mỗi lớp là lớn nhất. Các điểm gần nhất này gọi là support vectors. Tham số C điều chỉnh mức độ chấp nhận sai sót để tránh overfit. Với dữ liệu văn bản vector hoá bằng TF-IDF, nhóm dùng linear kernel vì không gian đặc trưng có số chiều lớn nên thường đã phân tách tuyến tính được.

**Vì sao Naive Bayes gọi là "ngây thơ"?** Vì giả định các từ trong câu độc lập với nhau, điều này không đúng hoàn toàn trong thực tế.

**Overfit là gì, nhóm xử lý thế nào?** Mô hình học thuộc dữ liệu huấn luyện nên điểm train cao nhưng điểm test thấp. Nhóm xử lý bằng chia tập train/test, chỉnh tham số (C của SVM, `max_depth` của cây) và giảm số đặc trưng.

**Vì sao không chỉ nhìn accuracy?** Nếu các nhóm không cân bằng, accuracy có thể cao mà mô hình vẫn sai nhiều ở nhóm nhỏ, nên cần xem thêm precision, recall, F1 và confusion matrix.

**Mô hình nhầm ở đâu và vì sao?** Hãy chuẩn bị sẵn 2–3 câu dự đoán sai cụ thể kèm lý do (ví dụ câu chứa từ khoá của hai nhóm cùng lúc).

**Ba điều quan trọng nhất cần nhớ về SVM:** margin lớn nhất, support vectors, và C điều chỉnh mức độ khắt khe.

---

## 12. Danh sách việc cần làm

- [ ] Chốt 5 nhóm ý định
- [ ] Tạo file `data.csv` (khoảng 500 câu, có câu viết tắt / không dấu)
- [ ] Xử lý dữ liệu và chia train/test
- [ ] Thử BoW, TF-IDF, bigram và ghi kết quả
- [ ] Train Naive Bayes và SVM, chỉnh `C`
- [ ] (Tuỳ chọn) Thêm Decision Tree, top từ quan trọng, hình cây
- [ ] Đánh giá: F1, confusion matrix, phân tích câu sai
- [ ] Viết chương trình demo chạy trên máy nhóm
- [ ] Rà lại code, xoá hết comment
- [ ] Viết báo cáo và luyện trả lời câu hỏi vấn đáp
