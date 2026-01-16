# train_model.py
import json
import os

import joblib
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score
from sklearn.naive_bayes import BernoulliNB


TRAIN_CSV = "data/Training.csv"
TEST_CSV = "data/Testing.csv"
OUT_DIR = "artifacts"
LABEL_COL = "prognosis"


def train_and_eval():
    os.makedirs(OUT_DIR, exist_ok=True)

    # Load training data
    train_df = pd.read_csv(TRAIN_CSV)
    y_train = train_df[LABEL_COL].astype(str).str.strip()
    X_train = train_df.drop(columns=[LABEL_COL])
    symptom_cols = list(X_train.columns)

    # Train
    model = BernoulliNB()
    model.fit(X_train.values, y_train.values)

    # Eval (nếu có test)
    metrics = {}
    if os.path.exists(TEST_CSV):
        test_df = pd.read_csv(TEST_CSV)
        y_test = test_df[LABEL_COL].astype(str).str.strip()
        X_test = test_df.drop(columns=[LABEL_COL])

        # đảm bảo đúng thứ tự/cùng cột như train
        X_test = X_test.reindex(columns=symptom_cols, fill_value=0)

        y_pred = model.predict(X_test.values)
        metrics = {
            "accuracy": float(accuracy_score(y_test, y_pred)),
            "f1_macro": float(f1_score(y_test, y_pred, average="macro"))
        }

    # Save
    model_path = os.path.join(OUT_DIR, "model.pkl")
    meta_path = os.path.join(OUT_DIR, "meta.json")

    joblib.dump(model, model_path)
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(
            {"label_col": LABEL_COL, "symptom_cols": symptom_cols, "metrics": metrics},
            f,
            ensure_ascii=False,
            indent=2
        )

    print(f"Đã lưu mô hình vào {model_path}")
    if metrics:
        print("Kết quả trên tập kiểm thử:", json.dumps(metrics, ensure_ascii=False, indent=2))
    else:
        print("Không tìm thấy Testing.csv -> bỏ qua đánh giá.")


if __name__ == "__main__":
    train_and_eval()
