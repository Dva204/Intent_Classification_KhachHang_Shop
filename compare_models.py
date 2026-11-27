#1. Import thư viện
import pandas as pd 

from preprocess import clean_text

from sklearn.model_selection import train_test_split, GridSearchCV #GridSearchCV để tìm tham số c tốt nhất cho SVM
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer #CountVectorizer để làm Bag-of-words
from sklearn.pipeline import Pipeline
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, classification_report, f1_score #f1_score : ghi thêm marco f1 vào bảng so sánh
from sklearn.svm import LinearSVC, SVC #thuật toán SVM
from sklearn.tree import DecisionTreeClassifier #thuật toán DecesionTree
from sklearn.base import clone #để nhân bản vectorizer/classifier sạch trước mỗi lần train trong vòng lặp


#2. Đọc dữ liệu và tiền xử lý

df = pd.read_csv("data.csv")

#làm sạch cột câu hỏi
X = df["cau_hoi"].fillna("").apply(clean_text)

#lấy nhãn
y = df["nhan"]

print("Tổng số dữ liệu :", len(df))
print("Tổng số nhãn : ")
print(y.value_counts())

#3. Chia tập train/test (80/20)

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    stratify=y,
    random_state=42
)

print("\n===== KIEM TRA DU LIEU =====")
print("Train:", len(X_train))
print("Test:", len(X_test))
print("\nPhan bo nhan trong tap Test:")
print(y_test.value_counts())

#4. Khai báo 4 phương pháp biểu diễn văn bản

vectorizers = {
    "BoW_unigram": CountVectorizer(ngram_range=(1, 1)),
    "BoW_bigram": CountVectorizer(ngram_range=(1, 2)),
    "TFIDF_unigram": TfidfVectorizer(ngram_range=(1, 1)),
    "TFIDF_bigram": TfidfVectorizer(ngram_range=(1, 2))
}

#5. Khai báo 4 thuật toán phân loại

classifiers = {
    "Naive Bayes": MultinomialNB(),
    "SVM_Linear": LinearSVC(random_state=42),
    "SVM_RBF": SVC(kernel="rbf", random_state=42),
    "Decision_Tree": DecisionTreeClassifier(
        max_depth=10,
        min_samples_leaf=2,
        random_state=42,
        criterion="entropy"
    )
}

#6. So sánh 16 tổ hợp mô hình

results = []
for vec_name, vec in vectorizers.items():
    for clf_name, clf in classifiers.items():
        
        print(f"\nĐang huấn luyện: {vec_name} + {clf_name}")

        #Tạo Pipline
        model = Pipeline([
            ("vec", clone(vec)),
            ("clf", clone(clf))
        ])

        #Huấn luyện
        model.fit(X_train, y_train)

        #Dự đoán trên tập train
        y_train_pred = model.predict(X_train)

        #Dự đoán trên tập test
        y_test_pred = model.predict(X_test)

        #Tính các chỉ số
        train_acc = accuracy_score(y_train, y_train_pred)

        test_acc = accuracy_score(y_test, y_test_pred)

        test_f1 = f1_score(
            y_test,
            y_test_pred,
            average="macro"
        )

        #Lưu kết quả
        results.append({
            "Vectorizer": vec_name,
            "Classifier": clf_name,
            "Train_Acc": train_acc,
            "Test_Acc": test_acc,
            "Macro_F1": test_f1
        })

#Chuyển kết quả thành DataFrame
result_df = pd.DataFrame(results)

#Sắp xếp theo Test Accuracy giảm dần
result_df = result_df.sort_values(
    by="Test_Acc",
    ascending=False
).reset_index(drop=True)

print("\n==========================================")
print("       BANG SO SANH 16 TO HOP MO HINH")
print("==========================================")
print(result_df.to_string(index=False, float_format="%.4f"))


#7. Tối ưu tham số C cho SVM bằng GridSearchCV

print("\n==========================================")
print("          TOI UU MO HINH SVM")
print("==========================================")

#Tạo Pipeline SVM với TF-IDF Bigram
svm_pipeline = Pipeline([
    ("tfidf", TfidfVectorizer(ngram_range=(1, 2))),
    ("clf", LinearSVC(random_state=42))
])

#Khai báo các giá trị C cần thử
param_grid = {
    "clf__C": [0.1, 1.0, 10.0]
}

#Khởi tạo GridSearchCV
grid = GridSearchCV(
    svm_pipeline,
    param_grid,
    cv=5,
    scoring="accuracy"
)

grid.fit(X_train, y_train)

#Tìm tham số tốt nhất trên tập train
print("\nTham số tốt nhất: ")
print(grid.best_params_)

#Điểm trung bình 5-fold Cross Validation
print("\nBest CV Score: ")
print(grid.best_score_)

#Đánh giá mô hình tốt nhất trên tập test
best_svm = grid.best_estimator_

y_svm_pred = best_svm.predict(X_test)

svm_test_acc = accuracy_score(y_test, y_svm_pred)

svm_test_f1 = f1_score(
    y_test, 
    y_svm_pred,
    average='macro'
)

print("\nAccuracy tren tap Test:")
print(svm_test_acc)

print("\nMacro F1 tren tap Test:")
print(svm_test_f1)

print("\n===== HOAN THANH THU NGHIEM =====")