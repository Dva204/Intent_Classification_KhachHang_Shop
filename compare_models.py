import pandas as pd 

from preprocess import clean_text

from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.pipeline import Pipeline
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, classification_report, f1_score
from sklearn.svm import LinearSVC, SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.base import clone

df = pd.read_csv("data.csv")

X = df["cau_hoi"].fillna("").apply(clean_text)

y = df["nhan"]

print("Tổng số dữ liệu :", len(df))
print("Tổng số nhãn : ")
print(y.value_counts())

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

vectorizers = {
    "BoW_unigram": CountVectorizer(ngram_range=(1, 1)),
    "BoW_bigram": CountVectorizer(ngram_range=(1, 2)),
    "TFIDF_unigram": TfidfVectorizer(ngram_range=(1, 1)),
    "TFIDF_bigram": TfidfVectorizer(ngram_range=(1, 2))
}

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

results = []
for vec_name, vec in vectorizers.items():
    for clf_name, clf in classifiers.items():
        
        print(f"\nĐang huấn luyện: {vec_name} + {clf_name}")

        model = Pipeline([
            ("vec", clone(vec)),
            ("clf", clone(clf))
        ])

        model.fit(X_train, y_train)

        y_train_pred = model.predict(X_train)

        y_test_pred = model.predict(X_test)

        train_acc = accuracy_score(y_train, y_train_pred)

        test_acc = accuracy_score(y_test, y_test_pred)

        test_f1 = f1_score(
            y_test,
            y_test_pred,
            average="macro"
        )

        results.append({
            "Vectorizer": vec_name,
            "Classifier": clf_name,
            "Train_Acc": train_acc,
            "Test_Acc": test_acc,
            "Macro_F1": test_f1
        })

result_df = pd.DataFrame(results)

result_df = result_df.sort_values(
    by="Test_Acc",
    ascending=False
).reset_index(drop=True)

print("\n==========================================")
print("       BANG SO SANH 16 TO HOP MO HINH")
print("==========================================")
print(result_df.to_string(index=False, float_format="%.4f"))

print("\n==========================================")
print("          TOI UU MO HINH SVM")
print("==========================================")

svm_pipeline = Pipeline([
    ("tfidf", TfidfVectorizer(ngram_range=(1, 2))),
    ("clf", LinearSVC(random_state=42))
])

param_grid = {
    "clf__C": [0.1, 1.0, 10.0]
}

grid = GridSearchCV(
    svm_pipeline,
    param_grid,
    cv=5,
    scoring="accuracy"
)

grid.fit(X_train, y_train)

print("\nTham số tốt nhất: ")
print(grid.best_params_)

print("\nBest CV Score: ")
print(grid.best_score_)

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