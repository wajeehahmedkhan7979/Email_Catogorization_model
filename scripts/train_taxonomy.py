"""
Train taxonomy centroids from cleaned conversation data.

The script expects an input JSONL file where each line contains:
{
  "conversation_id": "...",
  "text": "...",
  "level1": "...",
  "level2": "..."
}
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path
from typing import Dict, Tuple

import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity


def load_conversations(path: Path) -> Tuple[np.ndarray, list[str], list[str]]:
    texts = []
    labels = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            row = json.loads(line)
            text = row.get("text") or ""
            level1 = row.get("level1") or "Other"
            level2 = row.get("level2") or "Other"
            if not text.strip():
                continue
            texts.append(text)
            labels.append(f"{level1} > {level2}")
    return np.array(texts), labels, sorted(set(labels))


def compute_centroids(
    embeddings: np.ndarray,
    labels: list[str],
) -> Dict[str, np.ndarray]:
    grouped: Dict[str, list[np.ndarray]] = defaultdict(list)
    for vec, label in zip(embeddings, labels):
        grouped[label].append(vec)
    centroids: Dict[str, np.ndarray] = {}
    for label, vecs in grouped.items():
        stack = np.vstack(vecs)
        centroids[label] = stack.mean(axis=0)
    return centroids


def main(input_path: str, taxonomy_json: str, centroids_out: str) -> None:
    input_file = Path(input_path)
    texts, labels, unique_labels = load_conversations(input_file)

    model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
    embeddings = model.encode(
        texts.tolist(),
        convert_to_numpy=True,
        normalize_embeddings=True,
    )

    centroids_map = compute_centroids(embeddings, labels)
    ordered_labels = list(centroids_map.keys())
    centroid_matrix = np.vstack([centroids_map[l] for l in ordered_labels])

    # Compute summary similarity stats (each sample vs its centroid).
    sims = []
    for text_vec, label in zip(embeddings, labels):
        centroid = centroids_map[label].reshape(1, -1)
        sim = float(cosine_similarity(text_vec.reshape(1, -1), centroid)[0][0])
        sims.append(sim)

    avg_similarity = float(np.mean(sims)) if sims else 0.0

    np.save(centroids_out, centroid_matrix)

    taxonomy = {
        "version": "v1.0",
        "labels": {
            "level1": sorted({l.split(" > ")[0] for l in ordered_labels}),
            "level2": {},
        },
        "centroids_info": {
            "file": centroids_out,
            "labels": ordered_labels,
        },
        "metrics": {
            "avg_similarity": avg_similarity,
            "num_conversations": len(texts),
            "num_centroids": len(ordered_labels),
        },
    }

    Path(taxonomy_json).write_text(json.dumps(taxonomy, indent=2), encoding="utf-8")
    print(f"Saved taxonomy to {taxonomy_json}")
    print(f"Average similarity to centroid: {avg_similarity:.3f}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input",
        required=True,
        help="Path to JSONL with fields: text, level1, level2",
    )
    parser.add_argument("--taxonomy-json", default="taxonomy_v1.json")
    parser.add_argument("--centroids-out", default="taxonomy_centroids_v1.npy")
    args = parser.parse_args()
    main(args.input, args.taxonomy_json, args.centroids_out)

