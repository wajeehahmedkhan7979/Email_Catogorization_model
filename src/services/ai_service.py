from typing import Dict, List, Tuple

import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity


class EmbeddingService:
    def __init__(
        self, model_name: str = "sentence-transformers/all-MiniLM-L6-v2"
    ):
        self.model = SentenceTransformer(model_name)

    def embed(self, texts: List[str]) -> np.ndarray:
        return np.asarray(self.model.encode(texts, convert_to_numpy=True))


class SimilarityMatcher:
    def __init__(self, label_vectors: Dict[str, np.ndarray]):
        self.label_vectors = label_vectors

    def match(
        self, vector: np.ndarray, top_k: int = 3
    ) -> List[Tuple[str, float]]:
        labels, vectors = zip(*self.label_vectors.items())
        matrix = np.vstack(vectors)
        sims = cosine_similarity(vector.reshape(1, -1), matrix).flatten()
        top_indices = sims.argsort()[::-1][:top_k]
        return [(labels[i], float(sims[i])) for i in top_indices]
