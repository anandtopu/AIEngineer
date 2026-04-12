"""Time-series forecasting baseline with time-based split."""

from __future__ import annotations

import numpy as np
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import mean_absolute_percentage_error, mean_squared_error


def make_synthetic_time_series(n: int = 1000, seed: int = 0):
    """Generate synthetic daily sales with trend + seasonality + noise."""
    rng = np.random.default_rng(seed)

    t = np.arange(n)
    trend = 0.05 * t
    seasonality = 10 * np.sin(2 * np.pi * t / 365.25)  # yearly
    weekly = 5 * np.sin(2 * np.pi * t / 7)  # weekly
    noise = rng.normal(0, 3, size=n)

    y = 100 + trend + seasonality + weekly + noise
    return t, y


def make_features(t: np.ndarray, y: np.ndarray, lags: list[int] = [1, 7, 30]):
    """Create time-series features."""
    n = len(y)
    max_lag = max(lags)

    features = []
    for lag in lags:
        lag_feat = np.full(n, np.nan)
        lag_feat[lag:] = y[:-lag]
        features.append(lag_feat)

    # Rolling mean features
    for window in [7, 30]:
        roll = np.full(n, np.nan)
        for i in range(window, n):
            roll[i] = np.mean(y[i - window : i])
        features.append(roll)

    # Time features
    day_of_week = t % 7
    month = (t // 30) % 12

    features.extend([day_of_week, month])

    X = np.column_stack(features)
    return X


def time_based_split(t: np.ndarray, train_frac: float = 0.8):
    """Split by time (no shuffling!)."""
    split_idx = int(len(t) * train_frac)
    return slice(0, split_idx), slice(split_idx, len(t))


def main():
    t, y = make_synthetic_time_series(n=1000)

    X = make_features(t, y, lags=[1, 7, 30])

    # Remove rows with NaN (from lag features)
    valid = ~np.isnan(X).any(axis=1)
    t_valid = t[valid]
    X_valid = X[valid]
    y_valid = y[valid]

    train_idx, test_idx = time_based_split(t_valid, train_frac=0.8)

    X_train, y_train = X_valid[train_idx], y_valid[train_idx]
    X_test, y_test = X_valid[test_idx], y_valid[test_idx]
    t_test = t_valid[test_idx]

    model = GradientBoostingRegressor(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)

    mape = mean_absolute_percentage_error(y_test, y_pred)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))

    print(f"MAPE={mape:.4f} ({mape*100:.2f}%)")
    print(f"RMSE={rmse:.4f}")

    # Naive baseline: predict last value
    naive_pred = y_test.copy()
    naive_pred[0] = y_train[-1] if len(y_train) > 0 else y_test[0]
    for i in range(1, len(naive_pred)):
        naive_pred[i] = y_test[i - 1]

    mape_naive = mean_absolute_percentage_error(y_test, naive_pred)
    print(f"Naive MAPE={mape_naive:.4f}")
    print(f"Model beats naive: {mape < mape_naive}")


if __name__ == "__main__":
    main()
