from typing import Dict, List, Optional


class DriftMonitor:
    def __init__(self, baseline: Dict[str, float], window_size: int = 1000):
        self.baseline = baseline
        self.window_size = window_size
        self.values: List[float] = []

    def record(self, similarity: float) -> None:
        self.values.append(similarity)
        if len(self.values) > self.window_size:
            self.values.pop(0)

    def _mean(self) -> Optional[float]:
        if not self.values:
            return None
        return sum(self.values) / len(self.values)

    def _low_ratio(self, threshold: float) -> float:
        if not self.values:
            return 0.0
        return sum(v < threshold for v in self.values) / len(self.values)

    def check(self) -> Optional[str]:
        if len(self.values) < 100:
            return None

        mean = self._mean()
        if mean is None:
            return None

        low_ratio = self._low_ratio(0.7)

        if mean < self.baseline.get("mean_similarity", 0.0) * 0.9:
            return "MEAN_SIMILARITY_DROP"

        if low_ratio > 0.25:
            return "LOW_CONFIDENCE_SPIKE"

        return None

