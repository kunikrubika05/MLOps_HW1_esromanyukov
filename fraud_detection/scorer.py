from pathlib import Path
from typing import Any

import pandas as pd
from catboost import CatBoostClassifier

from fraud_detection.preprocessing import load_state, preprocess


class Scorer:
    def __init__(
        self,
        artifacts: Path = Path('artifacts'),
        threshold: float = 0.98,
    ) -> None:
        if not 0 <= threshold <= 1:
            raise ValueError('Threshold must be between 0 and 1')
        self.model = CatBoostClassifier()
        self.model.load_model(artifacts / 'model.cbm')
        self.state = load_state(artifacts / 'preprocessing.json')
        self.threshold = threshold

    def score(
        self,
        transaction: dict[str, Any],
    ) -> tuple[float, int]:
        return self.score_many([transaction])[0]

    def score_many(
        self,
        transactions: list[dict[str, Any]],
    ) -> list[tuple[float, int]]:
        features = preprocess(pd.DataFrame(transactions), self.state)
        scores = self.model.predict_proba(features, thread_count=1)[:, 1]
        return [(float(score), int(score > self.threshold)) for score in scores]
