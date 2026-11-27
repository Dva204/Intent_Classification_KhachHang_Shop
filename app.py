import re
import os
import joblib
import pandas as pd
import streamlit as st
from sklearn.base import clone
from sklearn.calibration import CalibratedClassifierCV
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline
from sklearn.svm import LinearSVC, SVC
from sklearn.tree import DecisionTreeClassifier

from preprocess import ABBREVIATIONS, clean_text

st.set_page_config(
    page_title="Phân Loại Ý Định Khách Hàng Shop Online",
    page_icon="🛍️",
    layout="wide",
)

LABEL_NAMES = {
    "hoi_gia": "Hỏi giá",
    "tinh_trang_don": "Hỏi tình trạng đơn",
    "doi_tra": "Đổi / trả hàng",
    "size_con_hang": "Hỏi size / còn hàng",
    "khac": "Khác (chào hỏi, cảm ơn...)",
}


@st.cache_data
def load_dataset():
    df = pd.read_csv("data.csv")
    df["cau_hoi_sach"] = df["cau_hoi"].fillna("").apply(clean_text)
    df["so_tu"] = df["cau_hoi"].astype(str).str.split().str.len()
    df["khong_dau"] = df["cau_hoi"].astype(str).apply(
        lambda text: not any(ord(ch) > 127 for ch in text)
    )
    return df


@st.cache_resource
def load_saved_model():
    return joblib.load("intent_model.joblib")


def split_dataset(df):
    return train_test_split(
        df["cau_hoi_sach"],
        df["nhan"],
        test_size=0.2,
        stratify=df["nhan"],
        random_state=42,
    )


def trace_preprocessing_steps(raw_text):
    step_lower = str(raw_text).lower()
    step_order = re.sub(r"\b[a-z]{2,3}\d{4,6}\b", "ma_don", step_lower)
    step_punct = re.sub(r"[^\w\s]", " ", step_order)
    words = step_punct.split()
    for idx, word in enumerate(words):
        if word == "m":
            if idx == 0 or words[idx - 1] != "size":
                words[idx] = "mình"
        else:
            words[idx] = ABBREVIATIONS.get(word, word)
    step_abbrev = " ".join(words)
    return [
        ("0. Câu hỏi gốc", raw_text),
        ("1. Chuyển chữ thường (.lower)", step_lower),
        ("2. Chuẩn hóa mã đơn -> ma_don", step_order),
        ("3. Xóa dấu câu & ký tự đặc biệt", step_punct),
        ("4. Chuẩn hóa viết tắt (giữ ngữ cảnh size m)", step_abbrev),
    ]


@st.cache_data
def run_model_comparison(df):
    x_train, x_test, y_train, y_test = split_dataset(df)

    vectorizers = {
        "BoW_unigram": CountVectorizer(ngram_range=(1, 1)),
        "BoW_bigram": CountVectorizer(ngram_range=(1, 2)),
        "TFIDF_unigram": TfidfVectorizer(ngram_range=(1, 1)),
        "TFIDF_bigram": TfidfVectorizer(ngram_range=(1, 2)),
    }

    classifiers = {
        "Naive Bayes": MultinomialNB(),
        "SVM_Linear": LinearSVC(random_state=42),
        "SVM_RBF": SVC(kernel="rbf", random_state=42),
        "Decision_Tree": DecisionTreeClassifier(
            max_depth=10,
            min_samples_leaf=2,
            random_state=42,
            criterion="entropy",
        ),
    }

    rows = []
    for vec_name, vec in vectorizers.items():
        for clf_name, clf in classifiers.items():
            pipe = Pipeline([("vec", clone(vec)), ("clf", clone(clf))])
            pipe.fit(x_train, y_train)
            train_pred = pipe.predict(x_train)
            test_pred = pipe.predict(x_test)
            rows.append(
                {
                    "Vectorizer": vec_name,
                    "Classifier": clf_name,
                    "Train_Acc": round(accuracy_score(y_train, train_pred), 4),
                    "Test_Acc": round(accuracy_score(y_test, test_pred), 4),
                    "Macro_F1": round(f1_score(y_test, test_pred, average="macro"), 4),
                }
            )

    result_df = (
        pd.DataFrame(rows)
        .sort_values(by="Test_Acc", ascending=False)
        .reset_index(drop=True)
    )

    svm_pipeline = Pipeline(
        [
            ("tfidf", TfidfVectorizer(ngram_range=(1, 2))),
            ("clf", LinearSVC(random_state=42)),
        ]
    )
    grid = GridSearchCV(
        svm_pipeline,
        {"clf__C": [0.1, 1.0, 10.0]},
        cv=5,
        scoring="accuracy",
    )
    grid.fit(x_train, y_train)
    best_pred = grid.best_estimator_.predict(x_test)

    grid_summary = {
        "best_params": grid.best_params_,
        "best_cv_score": round(float(grid.best_score_), 4),
        "test_acc": round(float(accuracy_score(y_test, best_pred)), 4),
        "test_f1": round(float(f1_score(y_test, best_pred, average="macro")), 4),
    }
    return result_df, grid_summary


@st.cache_data
def evaluate_best_model(df):
    x_train, x_test, y_train, y_test = split_dataset(df)
    model = Pipeline(
        [
            ("tfidf", TfidfVectorizer(ngram_range=(1, 2))),
            ("clf", CalibratedClassifierCV(LinearSVC(C=1.0, random_state=42), cv=5)),
        ]
    )
    model.fit(x_train, y_train)
    y_pred = model.predict(x_test)

    labels = sorted(df["nhan"].unique())
    metric_rows = []
    for label in labels:
        tp = int(((y_test == label) & (y_pred == label)).sum())
        fp = int(((y_test != label) & (y_pred == label)).sum())
        fn = int(((y_test == label) & (y_pred != label)).sum())
        tn = int(((y_test != label) & (y_pred != label)).sum())
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = (
            2 * precision * recall / (precision + recall)
            if (precision + recall) > 0
            else 0.0
        )
        metric_rows.append(
            {
                "Nhãn": label,
                "Tên ý định": LABEL_NAMES.get(label, label),
                "TP": tp,
                "FP": fp,
                "FN": fn,
                "TN": tn,
                "Precision": round(precision, 2),
                "Recall": round(recall, 2),
                "F1-score": round(f1, 2),
                "Support": tp + fn,
            }
        )

    wrong_rows = []
    original_questions = df.loc[x_test.index, "cau_hoi"]
    for question, true_label, pred_label in zip(original_questions, y_test, y_pred):
        if pred_label != true_label:
            wrong_rows.append(
                {
                    "Câu hỏi gốc": question,
                    "Sau tiền xử lý": clean_text(question),
                    "Nhãn thật": true_label,
                    "Máy đoán": pred_label,
                }
            )

    summary = {
        "accuracy": accuracy_score(y_test, y_pred),
        "macro_f1": f1_score(y_test, y_pred, average="macro"),
        "correct": int((y_test == y_pred).sum()),
        "total": len(y_test),
    }
    return summary, pd.DataFrame(metric_rows), pd.DataFrame(wrong_rows)


df = load_dataset()

st.title("🛍️ Phân Loại Ý Định Câu Hỏi Khách Hàng Shop Online")
st.caption(
    "Quy trình Học máy cơ bản hoàn chỉnh: Dữ liệu → Tiền xử lý → So sánh mô hình → Đánh giá → Demo dự đoán"
)

tab1, tab2, tab3, tab4, tab5 = st.tabs(
    [
        "1. Dữ liệu (data.csv)",
        "2. Tiền xử lý (preprocess.py)",
        "3. So sánh 16 mô hình",
        "4. Đánh giá & Lỗi",
        "5. Demo dự đoán",
    ]
)

with tab1:
    st.subheader("Tổng quan bộ dữ liệu")
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Tổng số câu", len(df))
    col2.metric("Số nhóm ý định", df["nhan"].nunique())
    col3.metric("Tỉ lệ câu không dấu", f"{df['khong_dau'].mean() * 100:.1f}%")
    col4.metric("Độ dài TB (số từ)", f"{df['so_tu'].mean():.2f}")

    st.markdown("**Phân bố số câu theo từng nhãn (cân bằng tuyệt đối 100 câu / nhãn)**")
    label_counts = df["nhan"].value_counts().rename_axis("Nhãn").reset_index(name="Số câu")
    st.bar_chart(label_counts.set_index("Nhãn"))

    st.markdown("**Xem mẫu dữ liệu theo nhãn**")
    selected_label = st.selectbox(
        "Chọn nhãn để lọc:",
        ["Tất cả"] + sorted(df["nhan"].unique().tolist()),
    )
    view_df = df if selected_label == "Tất cả" else df[df["nhan"] == selected_label]
    st.dataframe(
        view_df[["cau_hoi", "nhan", "so_tu", "khong_dau"]].head(30),
        use_container_width=True,
        hide_index=True,
    )

with tab2:
    st.subheader("Mô phỏng từng bước tiền xử lý văn bản")
    preset_samples = [
        "Cho minh hoi tinh trang don SP32365",
        "dép này còn size M k ạ?",
        "MD921822 giao den dau roi b?",
        "mik muốn hỏi giá sản phẩm này bao nhiu",
        "mac ko vua doi sz dc ko",
    ]
    sample_choice = st.selectbox("Chọn nhanh câu mẫu:", preset_samples)
    custom_input = st.text_input("Hoặc tự nhập câu hỏi cần kiểm tra:", value=sample_choice)

    steps = trace_preprocessing_steps(custom_input)
    step_df = pd.DataFrame(steps, columns=["Bước xử lý", "Kết quả"])
    st.table(step_df)

    with st.expander("Xem từ điển chuẩn hóa viết tắt (ABBREVIATIONS)"):
        abbrev_df = pd.DataFrame(
            list(ABBREVIATIONS.items()),
            columns=["Từ viết tắt", "Từ chuẩn hóa"],
        )
        st.dataframe(abbrev_df, use_container_width=True, hide_index=True)

with tab3:
    st.subheader("So sánh 16 tổ hợp (4 đặc trưng × 4 thuật toán)")
    st.write(
        "Chia dữ liệu **80% Train (400 câu) / 20% Test (100 câu)** với `stratify=y`, `random_state=42`."
    )

    result_df, grid_summary = run_model_comparison(df)

    st.dataframe(result_df, use_container_width=True, hide_index=True)

    st.markdown("**So sánh Test Accuracy cao nhất của từng thuật toán**")
    best_per_clf = (
        result_df.groupby("Classifier")[["Train_Acc", "Test_Acc", "Macro_F1"]]
        .max()
        .sort_values(by="Test_Acc", ascending=False)
    )
    st.bar_chart(best_per_clf[["Train_Acc", "Test_Acc"]], stack=False)

    st.markdown("**Tối ưu tham số `C` cho `TF-IDF Bigram + LinearSVC` bằng GridSearchCV (5-fold)**")
    g1, g2, g3, g4 = st.columns(4)
    g1.metric("Tham số tốt nhất", str(grid_summary["best_params"]))
    g2.metric("Best CV Score (Train)", f"{grid_summary['best_cv_score'] * 100:.2f}%")
    g3.metric("Test Accuracy", f"{grid_summary['test_acc'] * 100:.2f}%")
    g4.metric("Test Macro F1", f"{grid_summary['test_f1'] * 100:.2f}%")

with tab4:
    st.subheader("Đánh giá mô hình tốt nhất: TF-IDF Bigram + Calibrated LinearSVC")
    summary, metrics_df, wrong_df = evaluate_best_model(df)

    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Test Accuracy", f"{summary['accuracy'] * 100:.2f}%")
    m2.metric("Macro F1-score", f"{summary['macro_f1'] * 100:.2f}%")
    m3.metric("Đoán đúng", f"{summary['correct']} / {summary['total']}")
    m4.metric("Đoán sai", f"{summary['total'] - summary['correct']} / {summary['total']}")

    st.markdown("**Bảng chi tiết TP, FP, FN, Precision, Recall, F1 theo từng lớp**")
    st.dataframe(metrics_df, use_container_width=True, hide_index=True)

    col_left, col_right = st.columns([1.1, 1])
    with col_left:
        st.markdown("**Ma trận nhầm lẫn (Confusion Matrix)**")
        if os.path.exists("confusion_matrix.png"):
            st.image("confusion_matrix.png", use_container_width=True)
        else:
            st.warning("Chưa tìm thấy file confusion_matrix.png. Hãy chạy train_model.py trước.")
    with col_right:
        st.markdown("**Các câu dự đoán sai trên tập Test**")
        if len(wrong_df) > 0:
            st.dataframe(wrong_df, use_container_width=True, hide_index=True)
        else:
            st.success("Không có câu nào bị dự đoán sai!")

with tab5:
    st.subheader("Demo dự đoán ý định trực tiếp (load từ intent_model.joblib)")

    quick_queries = [
        "ao nay bn z shop",
        "don AB12345 giao toi dau r",
        "mac ko vua doi sz dc ko",
        "con size L mau den ko",
        "cam on shop nha",
    ]
    st.markdown("**Bấm nhanh câu mẫu để thử:**")
    btn_cols = st.columns(len(quick_queries))
    if "demo_query" not in st.session_state:
        st.session_state["demo_query"] = quick_queries[0]

    for col, q in zip(btn_cols, quick_queries):
        if col.button(q, use_container_width=True):
            st.session_state["demo_query"] = q

    user_question = st.text_input(
        "Nhập câu hỏi của khách hàng:",
        key="demo_query",
    )

    if user_question.strip():
        saved_model = load_saved_model()
        cleaned = clean_text(user_question)
        probs = saved_model.predict_proba([cleaned])[0]
        best_idx = probs.argmax()
        pred_label = saved_model.classes_[best_idx]
        confidence = float(probs[best_idx])

        st.info(f"**Câu sau tiền xử lý:** `{cleaned}`")

        r1, r2 = st.columns(2)
        r1.metric("Ý định dự đoán", LABEL_NAMES.get(pred_label, pred_label))
        r2.metric("Độ tin cậy", f"{confidence * 100:.2f}%")

        if confidence < 0.5:
            st.warning("Độ tin cậy thấp (< 50%), nên chuyển đến nhân viên tư vấn kiểm tra lại.")

        prob_df = pd.DataFrame(
            {
                "Ý định": [LABEL_NAMES.get(c, c) for c in saved_model.classes_],
                "Xác suất (%)": [round(float(p) * 100, 2) for p in probs],
            }
        ).sort_values(by="Xác suất (%)", ascending=False)

        st.markdown("**Phân bố xác suất của cả 5 nhóm ý định (%)**")
        st.bar_chart(prob_df.set_index("Ý định"))
        st.dataframe(prob_df, use_container_width=True, hide_index=True)

