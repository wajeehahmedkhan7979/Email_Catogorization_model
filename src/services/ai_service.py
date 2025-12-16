"""
Local AI services: embeddings and intent classification.

These classes are designed to be cheap, reliable, and run fully within
your own infrastructure using open-source models.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import List, Tuple

import joblib
import numpy as np
from sentence_transformers import SentenceTransformer

logger = logging.getLogger(__name__)


class EmbeddingService:
    """
    Wrapper around a sentence-transformers model for computing embeddings.

    Embeddings are L2-normalised to make cosine similarity equivalent to
    dot product.
    """

    def __init__(
        self,
        model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
    ) -> None:
        self.model = SentenceTransformer(model_name)

    def embed(self, texts: List[str]) -> List[np.ndarray]:
        """
        Compute normalised embeddings for a list of texts.
        """
        if not texts:
            return []
        vectors = self.model.encode(
            texts,
            convert_to_numpy=True,
            normalize_embeddings=True,
        )
        return [np.asarray(v) for v in vectors]


class IntentClassifier:
    """
    Lightweight intent classifier using a pre-trained embedding model and
    a pickled sklearn classifier + label encoder.
    """

    def __init__(
        self,
        model_path: str,
        embedding_service: EmbeddingService,
    ) -> None:
        self.model_path = Path(model_path)
        self.embedding_service = embedding_service
        try:
            bundle = joblib.load(self.model_path)
            self.classifier = bundle["classifier"]
            self.label_encoder = bundle["labels"]
        except Exception as exc:  # pragma: no cover - defensive logging
            logger.exception(
                "Failed to load intent model from %s: %s",
                model_path,
                exc,
            )
            raise

    def predict(self, text: str) -> Tuple[str, float]:
        """
        Predict an intent label and its probability for the given text.

        Returns:
            (intent_label, probability)
        """
        if not text.strip():
            return "Unknown", 0.0

        embedding_list = self.embedding_service.embed([text])
        if not embedding_list:
            return "Unknown", 0.0

        vector = embedding_list[0].reshape(1, -1)
        proba = self.classifier.predict_proba(vector)[0]
        idx = int(np.argmax(proba))
        label = str(self.label_encoder[idx])
        return label, float(proba[idx])
