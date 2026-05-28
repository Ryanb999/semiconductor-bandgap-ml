"""
models.py
---------
Shared training / evaluation utilities used by both the real Materials Project
pipeline (train_models.py) and the self-contained demo (run_demo.py).

We deliberately mirror the model family used by Linton & Aidhy (APL Mach.
Learn. 1, 016109, 2023): Linear Regression (LR), Random Forest Regression
(RFR), and Gradient Boosted Regression (GBR). Reporting R^2 and RMSE matches
the validation criteria used in that work.
"""

from __future__ import annotations

import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_squared_error


RANDOM_STATE = 42


def get_models():
    """Return the dict of models to compare. n_estimators kept modest to avoid
    overfitting, consistent with the reference paper's choice."""
    return {
        "LinearRegression": LinearRegression(),
        "RandomForest": RandomForestRegressor(
            n_estimators=200, random_state=RANDOM_STATE, n_jobs=-1
        ),
        "GradientBoosting": GradientBoostingRegressor(
            n_estimators=200, max_depth=3, learning_rate=0.1,
            random_state=RANDOM_STATE
        ),
    }


def evaluate(y_true, y_pred):
    """Return (R^2, RMSE) for a set of predictions."""
    r2 = r2_score(y_true, y_pred)
    rmse = float(np.sqrt(mean_squared_error(y_true, y_pred)))
    return float(r2), rmse


def train_and_evaluate(X, y, test_size=0.2):
    """Train every model on an 80/20 split and return a results dict plus the
    fitted estimators and the held-out test split (for plotting)."""
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=RANDOM_STATE
    )

    results = {}
    fitted = {}
    predictions = {}
    for name, model in get_models().items():
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        r2, rmse = evaluate(y_test, y_pred)
        results[name] = {"R2": round(r2, 4), "RMSE": round(rmse, 4)}
        fitted[name] = model
        predictions[name] = y_pred

    return {
        "results": results,
        "fitted": fitted,
        "predictions": predictions,
        "y_test": np.asarray(y_test),
        "X_test": np.asarray(X_test),
        "n_train": len(y_train),
        "n_test": len(y_test),
    }
