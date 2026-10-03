import argparse
import json
import shutil
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from catboost import CatBoostClassifier

from preprocessing import (
    CATEGORICAL_COLUMNS,
    NUMERIC_COLUMNS,
    TIME_COLUMNS,
    base_features,
)


def prepare(
    train_path: Path,
    model_path: Path,
    output: Path,
) -> None:
    train = pd.read_csv(train_path)
    frame = base_features(train)
    categories: dict[str, dict[str, str]] = {}
    for column in CATEGORICAL_COLUMNS:
        counts = (
            train.groupby(column, dropna=False)[['target']]
            .count()
            .sort_values('target', ascending=False)
            .reset_index()
            .reset_index()
        )
        counts['index'] = counts.apply(
            lambda row: np.nan if pd.isna(row[column]) else row['index'],
            axis=1,
        )
        labels = [
            'cat_NAN'
            if pd.isna(index)
            else 'cat_' + str(index)
            if index < 50
            else 'cat_50+'
            for index in counts['index']
        ]
        mapping = dict(zip(counts[column], labels, strict=True))
        categories[column] = {
            str(key): value for key, value in mapping.items() if not pd.isna(key)
        }
        frame[column + '_cat'] = train[column].map(mapping)
    frame['target'] = train['target']
    means = {}
    for column in [*[c + '_cat' for c in CATEGORICAL_COLUMNS], *TIME_COLUMNS]:
        means[column] = {
            str(key): float(value)
            for key, value in frame.groupby(column)['target'].mean().items()
        }
    model = CatBoostClassifier()
    model.load_model(model_path)
    state: dict[str, Any] = {
        'features': model.feature_names_,
        'categories': categories,
        'means': means,
        'numeric_means': {c: float(frame[c].mean()) for c in NUMERIC_COLUMNS},
    }
    output.mkdir(parents=True, exist_ok=True)
    (output / 'preprocessing.json').write_text(json.dumps(state, ensure_ascii=False))
    shutil.copyfile(model_path, output / 'model.cbm')


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument('--train', type=Path, required=True)
    parser.add_argument('--model', type=Path, required=True)
    parser.add_argument('--output', type=Path, default=Path('artifacts'))
    args = parser.parse_args()
    prepare(args.train, args.model, args.output)


if __name__ == '__main__':
    main()
