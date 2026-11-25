'Benchmarks for day-ahead price forecasting (Lago et al., 2021, "Forecasting day-ahead electricity prices: A review of state-of-the-art algorithms, best practices and an'

import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from sklearn.linear_model import LassoLarsIC

PRICE_LAGS = (1, 2, 3, 7)
EXOG = ("load_fc", "res_fc")
EXOG_LAGS = (0, 1, 7)
RES_COLS = ["solar_fc", "wind_on_fc", "wind_off_fc"]


def load_wide(path: str, max_gap_hours: int = 6) -> dict[str, pd.DataFrame]:
    """Long (date, hour) dataset -> dict of daily wide frames on a gap-free daily index."""
    df = pd.read_parquet(path)

    df["res_fc"] = df[RES_COLS].sum(axis=1, min_count=len(RES_COLS))
    wide = {}
    for col in ("price", *EXOG):
        s = df[col].copy()
        if col != "price":
            s = s.interpolate(limit=max_gap_hours, limit_area="inside")
        w = s.unstack("hour")
        w.index = pd.DatetimeIndex(pd.to_datetime(w.index), name="date")
        wide[col] = w.asfreq("D")
    return wide


def design_matrix(
    wide: dict[str, pd.DataFrame], res_lags=EXOG_LAGS
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """X row for date D uses only information available at 12:00 on D-1.

    res_lags=(1, 7) drops the wind/solar forecast for D itself (leakage check, Week 4).
    """
    P = wide["price"]
    blocks = [P.shift(k).add_prefix(f"p_d{k}_h") for k in PRICE_LAGS]
    for c in EXOG:
        lags = res_lags if c == "res_fc" else EXOG_LAGS
        blocks += [wide[c].shift(k).add_prefix(f"{c}_d{k}_h") for k in lags]
    dow = pd.get_dummies(P.index.dayofweek, prefix="dow").astype(float)
    dow.index = P.index
    return pd.concat(blocks + [dow], axis=1), P


class AsinhScaler:
    """z = asinh((x - median) / MAD). Robust to spikes and handles negative prices."""

    def fit(self, a: np.ndarray):
        self.med = np.median(a, axis=0)
        mad = 1.4826 * np.median(np.abs(a - self.med), axis=0)
        std = np.std(a, axis=0)
        self.mad = np.where(mad > 1e-8, mad, np.where(std > 1e-8, std, 1.0))
        return self

    def transform(self, a):
        return np.arcsinh((a - self.med) / self.mad)

    def inverse(self, z):
        return np.sinh(z) * self.mad + self.med


def naive_forecast(P: pd.DataFrame, dates: pd.DatetimeIndex) -> pd.DataFrame:
    out = {}
    for d in dates:
        lag = 7 if d.dayofweek in (0, 5, 6) else 1
        prev = d - pd.Timedelta(days=lag)
        out[d] = P.loc[prev].to_numpy() if prev in P.index else np.full(24, np.nan)
    return pd.DataFrame.from_dict(out, orient="index", columns=P.columns)


def _fit_window(X: pd.DataFrame, Y: pd.DataFrame, d0: pd.Timestamp, window: int):
    """Fit 24 hourly LEAR models on the `window` days before d0. Returns a predictor."""
    train = slice(d0 - pd.Timedelta(days=window), d0 - pd.Timedelta(days=1))
    Xt, Yt = X.loc[train], Y.loc[train]
    ok = Xt.notna().all(axis=1) & Yt.notna().all(axis=1)
    Xt, Yt = Xt[ok].to_numpy(), Yt[ok].to_numpy()
    n_dummies = sum(c.startswith("dow") for c in X.columns)
    if len(Xt) < Xt.shape[1] + 10:
        return None
    xs = AsinhScaler().fit(Xt[:, :-n_dummies])
    ys = AsinhScaler().fit(Yt)
    Z = np.hstack([xs.transform(Xt[:, :-n_dummies]), Xt[:, -n_dummies:]])
    Zy = ys.transform(Yt)
    models = [
        LassoLarsIC(criterion="aic", max_iter=2500).fit(Z, Zy[:, h]) for h in range(24)
    ]

    def predict(x_row: np.ndarray) -> np.ndarray:
        z = np.hstack([xs.transform(x_row[:-n_dummies]), x_row[-n_dummies:]])[None, :]
        zy = np.array([m.predict(z)[0] for m in models])
        return ys.inverse(zy)

    return predict


def _lear_block(X, Y, days, windows):
    """Recalibrate once at days[0], forecast every day in the block."""
    predictors = [
        p for w in windows if (p := _fit_window(X, Y, days[0], w)) is not None
    ]
    rows = {}
    for d in days:
        x = X.loc[d].to_numpy(dtype=float)
        if not predictors or np.isnan(x).any():
            rows[d] = np.full(24, np.nan)
        else:
            rows[d] = np.mean([p(x) for p in predictors], axis=0)
    return rows


def lear_forecast(
    X, Y, dates, windows=(364, 728, 1092), recal_every=1, n_jobs=-1, verbose=5
):
    blocks = [dates[i : i + recal_every] for i in range(0, len(dates), recal_every)]
    results = Parallel(n_jobs=n_jobs, verbose=verbose)(
        delayed(_lear_block)(X, Y, b, windows) for b in blocks
    )
    rows = {d: r for res in results for d, r in res.items()}
    return pd.DataFrame.from_dict(rows, orient="index", columns=Y.columns)
