"""
Train a lightweight intent classifier on top of sentence embeddings.

Input: data/golden_dataset_v1.jsonl with fields:
{
  "id": "...",
  "conversation": {"subject": "...", "messages": [{"body": "..."}]},
  "labels": {"intent": "..."}
}
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import joblib
import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split


def load_golden_dataset(path: Path):
    texts = []
    intents = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            row = json.loads(line)
            labels = row.get("labels", {})
            intent = labels.get("intent")
            if not intent:
                continue
            conv = row.get("conversation", {})
            subject = conv.get("subject") or ""
            bodies = [m.get("body") or "" for m in conv.get("messages", [])]
            body = "\n".join(bodies)
            text = f"{subject}\n\n{body}".strip()
            if not text:
                continue
            texts.append(text)
            intents.append(intent)
    return texts, intents


def main(dataset_path: str, output_dir: str) -> None:
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)

    texts, intents = load_golden_dataset(Path(dataset_path))
    model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
    embeddings = model.encode(
        texts,
        convert_to_numpy=True,
        normalize_embeddings=True,
    )

    X_train, X_test, y_train, y_test = train_test_split(
        embeddings,
        np.array(intents),
        test_size=0.2,
        random_state=42,
        stratify=intents,
    )
    clf = LogisticRegression(max_iter=1000, n_jobs=-1, class_weight="balanced")
    clf.fit(X_train, y_train)
    preds = clf.predict(X_test)
    print(classification_report(y_test, preds))

    bundle = {"classifier": clf, "labels": clf.classes_}
    model_path = output / "intent_model_v1.pkl"
    joblib.dump(bundle, model_path)
    print(f"Saved intent model to {model_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dataset",
        default="data/golden_dataset_v1.jsonl",
        help="Path to golden dataset JSONL.",
    )
    parser.add_argument(
        "--output-dir",
        default="models",
        help="Directory to write the trained model bundle.",
    )
    args = parser.parse_args()
    main(args.dataset, args.output_dir)
