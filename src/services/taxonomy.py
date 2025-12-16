"""
Taxonomy metadata and centroid-based similarity matching.

This module loads a versioned taxonomy JSON and an accompanying NumPy
centroid matrix, then provides helpers to match new embeddings to the
nearest label centroids using cosine similarity.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import numpy as np
from sklearn.metrics.pairwise import cosine_similarity


class Taxonomy:
    """
    Versioned taxonomy with centroid-based similarity lookup.
    """

    def __init__(self, taxonomy_path: str) -> None:
        self.path = Path(taxonomy_path)
        self.data: Dict = json.loads(self.path.read_text(encoding="utf-8"))
        centroids_info = self.data.get("centroids_info", {})
        centroids_file = centroids_info.get("file", "taxonomy_centroids_v1.npy")
        self.centroid_labels: Sequence[str] = centroids_info.get("labels", [])
        self.centroids: np.ndarray = np.load(centroids_file)

        if not self.centroid_labels or len(self.centroid_labels) != len(
            self.centroids
        ):
            raise ValueError(
                "Taxonomy centroid labels and vectors are inconsistent in "
                f"{centroids_file}"
            )

    @property
    def version(self) -> str:
        """
        Version string from the taxonomy JSON.
        """
        return self.data.get("version", "unknown")

    @property
    def is_placeholder(self) -> bool:
        """
        True if the taxonomy appears to be a placeholder (single generic label).
        """
        labels = self.data.get("labels", {})
        return len(labels.get("level1", [])) <= 1

    def match_levels(
        self,
        embedding: np.ndarray,
        top_k: int = 3,
    ) -> List[Tuple[str, float]]:
        """
        Match an embedding to the nearest taxonomy centroids.

        Args:
            embedding: Normalised embedding vector.
            top_k: Number of top matches to return.

        Returns:
            A list of (label_hierarchy, similarity) sorted by similarity
            descending. label_hierarchy is usually "Level1 > Level2".
        """
        if embedding.ndim == 1:
            query = embedding.reshape(1, -1)
        else:
            query = embedding

        sims = cosine_similarity(query, self.centroids)[0]
        order = np.argsort(sims)[::-1][:top_k]
        results: List[Tuple[str, float]] = []
        for idx in order:
            label = str(self.centroid_labels[idx])
            results.append((label, float(sims[idx])))
        return results
