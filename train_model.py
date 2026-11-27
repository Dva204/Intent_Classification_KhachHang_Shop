import pandas as pd
import matplotlib.pyplot as plt
import joblib

from preprocess import clean_text

from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.pipeline import Pipeline
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, classification_report, f1_score, confusion_matrix, ConfusionMatrixDisplay
from sklearn.svm import LinearSVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.base import clone

from sklearn.calibration import CalibratedClassifierCV

df = pd.read_csv("data.csv")

X = df["cau_hoi"].fillna("").apply(clean_text)

y = df["nhan"]

print("Tổng số dữ liệu :", len(df))
print("Số lượng nhãn :")
print(y.value_counts())

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    stratify=y,
    random_state=42
)

print("\n===== KIỂM TRA DỮ LIỆU =====")
print("So luong Train:", len(X_train))
print("So luong Test:", len(X_test))

model = Pipeline([
    ("tfidf", TfidfVectorizer(ngram_range=(1, 2))),
    ("clf", CalibratedClassifierCV(LinearSVC(C=1.0, random_state=42), cv=5))
])

model.fit(X_train, y_train)

y_pred = model.predict(X_test)

df_ket_qua = pd.DataFrame({
    "Mẫu": [f"X{i+1}" for i in range(len(y_test))],
    "Phân lớp trong thực tế": y_test.values,
    "Phân lớp sử dụng mô hình": y_pred
})

print("\n===== BẢNG KẾT QUẢ DỰ ĐOÁN TỪNG MẪU =====")
print(df_ket_qua.to_string(index=False))

labels = sorted(y.unique())
rows = []

for label in labels:
    tp = ((y_test == label) & (y_pred == label)).sum()
    fp = ((y_test != label) & (y_pred == label)).sum()
    fn = ((y_test == label) & (y_pred != label)).sum()
    tn = ((y_test != label) & (y_pred != label)).sum()

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0

    rows.append({
        "Nhãn": label,
        "TP": tp,
        "FP": fp,
        "FN": fn,
        "TN": tn,
        "Precision": round(precision, 2),
        "Recall": round(recall, 2),
        "F1-score": round(f1, 2),
        "Support": tp + fn
    })

df_metrics = pd.DataFrame(rows)
print("\n===== BẢNG CHI TIẾT TP, FP, FN =====")
print(df_metrics.to_string(index=False))

accuracy = accuracy_score(y_test, y_pred)

print("\n===== KẾT QUẢ BASELINE =====")
print("Accuracy:", accuracy)

print("\nClassification Report:")
print(classification_report(y_test, y_pred))

cm = confusion_matrix(
    y_test,
    y_pred,
    labels=model.classes_
)

disp = ConfusionMatrixDisplay(
    confusion_matrix=cm,
    display_labels=model.classes_
)

disp.plot(
    cmap="Blues",
    xticks_rotation=25
)

plt.title("Confusion Matrix - TF-IDF Bigram + LinearSVC")
plt.tight_layout()

plt.savefig("confusion_matrix.png", dpi=150)
plt.close()

print("\nĐã lưu ma trận nhầm lẫn : confusion_matrix.png")

print("\n========== CAC CA DỰ ĐOÁN SAI==========")

original_questions = df.loc[X_test.index, "cau_hoi"]

wrong_count = 0

for question, true_label, pred_label in zip(original_questions, y_test, y_pred):
    if pred_label != true_label:
        wrong_count += 1

        print(f"\nLỗi thứ {wrong_count}:")
        print(f"Câu hỏi: {question}")
        print(f"Nhãn thật: {true_label}")
        print(f"Máy đoán: {pred_label}")

if wrong_count == 0:
    print("Không có câu nào được dự đoán sai")

print(f"\nTổng số câu dự đoán sai: {wrong_count}/{len(y_test)}")

joblib.dump(model, "intent_model.joblib")

print("\nĐã lưu mô hình: intent_model.joblib")
print("\n========== HOAN THANH CHANG 4 ==========")