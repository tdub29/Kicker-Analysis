from __future__ import annotations

import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error


def evaluate_predictions(y_true: pd.Series, y_pred: pd.Series) -> dict[str, float]:
    return {
        "mae": float(mean_absolute_error(y_true, y_pred)),
        "rmse": float(mean_squared_error(y_true, y_pred, squared=False)),
    }
