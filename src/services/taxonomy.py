import json
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np

from .ai_service import SimilarityMatcher


class Taxonomy:
    def __init__(self, path: str, embeddings: Dict[str, np.ndarray]):
        self.path = Path(path)
        self.data = json.loads(self.path.read_text(encoding="utf-8"))
        self.matcher = SimilarityMatcher(embeddings)

    def match_levels(self, vector: np.ndarray) -> List[Tuple[str, float]]:
        return self.matcher.match(vector, top_k=3)

    @property
    def version(self) -> str:
        return self.data.get("version", "unknown")

