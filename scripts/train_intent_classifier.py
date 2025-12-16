"""
Train a lightweight intent classifier on top of embeddings.
Starts with logistic regression; swap in a fine-tuned transformer if needed.
"""

import joblib
import pandas as pd
from sentence_transformers import SentenceTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split


def main(data_path: str, model_path: str = "intent_model_v1.pkl"):
    df = pd.read_csv(data_path)
    model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
    embeddings = model.encode(df["text"].tolist(), convert_to_numpy=True)

    X_train, X_test, y_train, y_test = train_test_split(
        embeddings, df["label"], test_size=0.2, random_state=42
    )
    clf = LogisticRegression(max_iter=1000)
    clf.fit(X_train, y_train)
    preds = clf.predict(X_test)
    print(classification_report(y_test, preds))
    joblib.dump({"classifier": clf, "labels": clf.classes_}, model_path)
    print(f"Saved intent model to {model_path}")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--data", required=True, help="CSV with columns: text,label")
    parser.add_argument("--output", default="intent_model_v1.pkl")
    args = parser.parse_args()
    main(args.data, args.output)
