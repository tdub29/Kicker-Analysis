from __future__ import annotations

import pandas as pd
from sklearn.linear_model import LinearRegression


def fit_expected_points_model(data: pd.DataFrame) -> LinearRegression:
    features = data[["adjhomeexpatt", "homekick_pct", "homexppg"]].fillna(0)
    target = data["homefppg"].fillna(0)
    model = LinearRegression()
    model.fit(features, target)
    return model


def predict_expected_points(model: LinearRegression, data: pd.DataFrame) -> pd.Series:
    features = data[["adjhomeexpatt", "homekick_pct", "homexppg"]].fillna(0)
    return pd.Series(model.predict(features), index=data.index)
