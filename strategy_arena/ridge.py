"""A small ridge regression (numpy only) with standardization and multi-output support."""
from __future__ import annotations

import numpy as np


class Ridge:
    """Ridge regression: minimizes ||y - Xb - c||^2 + alpha * ||b||^2 on standardized X.

    `y` may be 1-D or 2-D (one column per target); the intercept is not penalized.
    """

    def __init__(self, alpha: float = 1.0):
        self.alpha = alpha

    def fit(self, X, y) -> "Ridge":
        X = np.asarray(X, dtype=float)
        y = np.asarray(y, dtype=float)
        self.mean_ = X.mean(axis=0)
        self.scale_ = X.std(axis=0)
        self.scale_[self.scale_ == 0] = 1.0
        Z = (X - self.mean_) / self.scale_
        self.intercept_ = y.mean(axis=0)
        A = Z.T @ Z + self.alpha * np.eye(Z.shape[1])
        self.coef_ = np.linalg.solve(A, Z.T @ (y - self.intercept_))
        return self

    def standardize(self, X) -> np.ndarray:
        return (np.asarray(X, dtype=float) - self.mean_) / self.scale_

    def predict(self, X) -> np.ndarray:
        return self.standardize(X) @ self.coef_ + self.intercept_


def fit_ridge_cv(X, y, alphas=(0.1, 1.0, 10.0, 100.0, 1000.0), val_frac: float = 0.25, gap: int = 0) -> Ridge:
    """Pick alpha on the most recent `val_frac` of rows (time-ordered), then refit on all rows.

    `gap` rows between train and validation are dropped so overlapping targets cannot leak.
    """
    X = np.asarray(X, dtype=float)
    y = np.asarray(y, dtype=float)
    n = len(X)
    n_val = int(n * val_frac)
    n_train = n - n_val - gap
    if n_val < 5 or n_train < 10:
        return Ridge(alphas[len(alphas) // 2]).fit(X, y)
    best_alpha, best_err = alphas[0], np.inf
    for a in alphas:
        model = Ridge(a).fit(X[:n_train], y[:n_train])
        err = np.mean((model.predict(X[-n_val:]) - y[-n_val:]) ** 2)
        if err < best_err:
            best_alpha, best_err = a, err
    return Ridge(best_alpha).fit(X, y)
