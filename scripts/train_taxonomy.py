"""
Offline clustering script placeholder.
Compute embeddings on a sample email corpus, run HDBSCAN, and
export taxonomy JSON. Fill in data loading and persistence
paths as needed.
"""

import json
from pathlib import Path

import hdbscan
import numpy as np
from sentence_transformers import SentenceTransformer


def main(input_path: str, output_path: str = "taxonomy_v1.json"):
    texts = Path(input_path).read_text(encoding="utf-8").splitlines()
    model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
    embeddings = model.encode(texts, convert_to_numpy=True)
    clusterer = hdbscan.HDBSCAN(min_cluster_size=5)
    labels = clusterer.fit_predict(embeddings)

    taxonomy = {
        "version": "v1.0",
        "labels": {"level1": [], "level2": {}, "level3": []},
    }
    # TODO: Map labels to human-approved names. This is a placeholder export.
    taxonomy["metadata"] = {
        "num_clusters": int(labels.max()) + 1,
        "unassigned": int(np.sum(labels == -1)),
    }

    Path(output_path).write_text(json.dumps(taxonomy, indent=2), encoding="utf-8")
    print(f"Saved taxonomy to {output_path}")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input", required=True, help="Path to newline-delimited email texts"
    )
    parser.add_argument("--output", default="taxonomy_v1.json")
    args = parser.parse_args()
    main(args.input, args.output)
