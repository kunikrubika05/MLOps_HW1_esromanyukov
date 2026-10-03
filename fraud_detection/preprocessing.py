import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

CATEGORICAL_COLUMNS = ['gender', 'merch', 'cat_id', 'one_city', 'us_state', 'jobs']
TIME_COLUMNS = ['hour', 'year', 'month', 'day_of_month', 'day_of_week']
NUMERIC_COLUMNS = ['amount', 'population_city', 'distance']
INPUT_COLUMNS = [
    'transaction_time',
    *CATEGORICAL_COLUMNS,
    'amount',
    'population_city',
    'lat',
    'lon',
    'merchant_lat',
    'merchant_lon',
]


def base_features(
    frame: pd.DataFrame,
) -> pd.DataFrame:
    missing = set(INPUT_COLUMNS) - set(frame.columns)
    if missing:
        raise ValueError(f'Missing columns: {", ".join(sorted(missing))}')
    result = frame[INPUT_COLUMNS].copy()
    dates = pd.to_datetime(result.pop('transaction_time'), errors='raise')
    if dates.isna().any():
        raise ValueError('Missing transaction time')
    for name, values in zip(
        TIME_COLUMNS,
        [
            dates.dt.hour,
            dates.dt.year,
            dates.dt.month,
            dates.dt.day,
            dates.dt.dayofweek,
        ],
        strict=True,
    ):
        result[name] = values
    coordinates = result[['lat', 'lon', 'merchant_lat', 'merchant_lon']].astype(float)
    limits = pd.Series([90, 180, 90, 180], index=coordinates.columns)
    if (coordinates.abs() > limits).any().any():
        raise ValueError('Coordinates outside valid range')
    lat, lon, merchant_lat, merchant_lon = np.radians(coordinates.to_numpy()).T
    haversine = (
        np.sin((merchant_lat - lat) / 2) ** 2
        + np.cos(lat) * np.cos(merchant_lat) * np.sin((merchant_lon - lon) / 2) ** 2
    )
    result['distance'] = 6371.009 * 2 * np.arcsin(np.sqrt(np.clip(haversine, 0, 1)))
    return result.drop(columns=coordinates.columns)


def preprocess(
    frame: pd.DataFrame,
    state: dict[str, Any],
) -> pd.DataFrame:
    result = base_features(frame)
    for column in CATEGORICAL_COLUMNS:
        result[column + '_cat'] = (
            result.pop(column).map(state['categories'][column]).fillna('cat_NAN')
        )
    for column, mapping in state['means'].items():
        result[column + '_mean_enc'] = result[column].astype(str).map(mapping)
    for column in NUMERIC_COLUMNS:
        values = pd.to_numeric(result.pop(column), errors='raise')
        if np.isinf(values).any() or (values.dropna() < 0).any():
            raise ValueError(f'Invalid {column}')
        result[column + '_log'] = np.log1p(
            values.fillna(state['numeric_means'][column])
        )
    for column in [*TIME_COLUMNS, *[c + '_cat' for c in CATEGORICAL_COLUMNS]]:
        result[column] = result[column].astype(str)
    return result[state['features']]


def load_state(
    path: Path,
) -> dict[str, Any]:
    return json.loads(path.read_text())
