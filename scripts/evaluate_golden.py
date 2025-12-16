"""
Evaluate current model stack on the golden dataset.

This script:
1. Loads a JSONL golden dataset with ground-truth labels.
2. Runs embedding + taxonomy + intent classifier.
3. Computes accuracy and confusion matrix.
4. Exits with non-zero status if accuracy < threshold.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path
from typing import Tuple

import numpy as np

from src.config import get_settings
from src.services.ai_service import EmbeddingService, IntentClassifier
from src.services.taxonomy import Taxonomy


def load_golden(path: Path) -> Tuple[list[str], list[str], list[str]]:
    texts: list[str] = []
    level1: list[str] = []
    intents: list[str] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            row = json.loads(line)
            labels = row.get("labels", {})
            l1 = labels.get("level1")
            intent = labels.get("intent")
            if not l1 or not intent:
                continue
            conv = row.get("conversation", {})
            subject = conv.get("subject") or ""
            bodies = [m.get("body") or "" for m in conv.get("messages", [])]
            text = f"{subject}\n\n" + "\n".join(bodies)
            texts.append(text.strip())
            level1.append(l1)
            intents.append(intent)
    return texts, level1, intents


def main(data_path: str, threshold: float) -> None:
    settings = get_settings()
    dataset = Path(data_path)
    texts, true_l1, true_intent = load_golden(dataset)

    embedder = EmbeddingService()
    taxonomy = Taxonomy(settings.taxonomy_path)
    intent_clf = IntentClassifier(settings.intent_model_path, embedder)

    embeddings = embedder.embed(texts)

    pred_l1: list[str] = []
    pred_intent: list[str] = []
    for emb, text in zip(embeddings, texts):
        matches = taxonomy.match_levels(np.asarray(emb))
        label_hierarchy = matches[0][0] if matches else "Other > Uncategorized"
        l1 = label_hierarchy.split(" > ")[0]
        intent_label, _ = intent_clf.predict(text)
        pred_l1.append(l1)
        pred_intent.append(intent_label)

    correct = sum(
        1 for t, p in zip(true_l1, pred_l1) if t == p
    ) and sum(1 for t, p in zip(true_intent, pred_intent) if t == p)
    total = len(true_l1)
    accuracy = correct / total if total else 0.0

    print(f"Total samples: {total}")
    print(f"Accuracy: {accuracy:.3f}")

    conf = Counter((t, p) for t, p in zip(true_intent, pred_intent))
    print("Confusion (intent):")
    for (t, p), c in conf.most_common():
        print(f"{t} -> {p}: {c}")

    if accuracy < threshold:
        print(
            f"Accuracy {accuracy:.3f} below threshold {threshold:.3f}; failing.",
            file=sys.stderr,
        )
        sys.exit(1)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--data",
        required=True,
        help="Path to golden dataset JSONL.",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=0.8,
        help="Minimum acceptable accuracy before failing.",
    )
    args = parser.parse_args()
    main(args.data, args.threshold)


