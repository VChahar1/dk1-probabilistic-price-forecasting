"""
Week 3: probabilistic day-ahead price forecasts with LightGBM quantile regression.

  * one global model across hours (hour is a feature), one model per quantile
  * target trained in asinh(median/MAD) space; quantiles are mapped back exactly,
    because quantiles are invariant under monotone transforms
  * rolling recalibration on a fixed-length window (default 1092 days, weekly refit)
  * quantile crossing removed by sorting each row
  * conformal calibration (CQR) of the intervals is done afterwards, in evaluate_prob.py
"""
import lightgbm as lgb
import numpy as np
import pandas as pd
from joblib import Parallel, delayed

from src.lear import AsinhScaler

QUANTILES = (0.05, 0.10, 0.25, 0.50, 0.75, 0.90, 0.95)
QCOLS = [f"q{q:.2f}" for q in QUANTILES]
RES_COLS = ["solar_fc", "wind_on_fc", "wind_off_fc"]

PARAMS = dict(n_estimators=600, learning_rate=0.03, num_leaves=31, min_child_samples=50,
              subsample=0.8, subsample_freq=1, colsample_bytree=0.8, reg_lambda=1.0,
              n_jobs=1, verbose=-1)  # fixed, NOT tuned on the test period


RES_D_FEATURES = ["solar_fc", "wind_on_fc", "wind_off_fc", "res_fc", "residual_load_fc",
                  "res_share_fc", "res_fc_dmean", "residual_load_fc_dmean",
                  "residual_load_fc_dmax", "residual_load_fc_dmin", "residual_load_fc_rel",
                  "residual_load_fc_ramp"]


def _lag_one_day(df: pd.DataFrame, cols: list[str]) -> pd.DataFrame:
    """Same-hour values from the previous day, merged on (date, hour)."""
    lag = df[["date", "hour", *cols]].copy()
    lag["date"] += pd.Timedelta(days=1)
    return df.merge(lag.rename(columns={c: f"{c}_d1" for c in cols}), on=["date", "hour"], how="left")


def load_long(path: str, res_d: bool = True, res_override: pd.Series | None = None,
              extra: pd.DataFrame | None = None) -> pd.DataFrame:
    """(date, hour) dataset + extra features that are all known at gate closure on D-1.

    res_d=False (leakage check): every feature built from the wind/solar forecast for D is
    removed and replaced by the forecast that was made for D-1 (published by 18:00 on D-2).
    res_override (pre-gate variant): replaces the ENTSO-E renewables forecast for D with a
    series indexed by (date, hour); the ENTSO-E components are dropped. `extra` adds more
    (date, hour)-indexed columns, e.g. raw weather indices.
    """
    df = pd.read_parquet(path).reset_index()
    df["date"] = pd.to_datetime(df["date"])
    df = df.sort_values(["date", "hour"]).reset_index(drop=True)
    # strict renewables total (missing if any component is missing); LightGBM handles NaN
    df["res_fc"] = df[RES_COLS].sum(axis=1, min_count=len(RES_COLS))
    if res_override is not None:
        key = pd.MultiIndex.from_arrays([df["date"], df["hour"]])
        df["res_fc"] = res_override.reindex(key).to_numpy()
        df = df.drop(columns=RES_COLS)
    if extra is not None:
        df = df.merge(extra.reset_index(), on=["date", "hour"], how="left")
    df["residual_load_fc"] = df["load_fc"] - df["res_fc"]
    df["res_share_fc"] = df["res_fc"] / df["load_fc"]

    # shape of tomorrow: daily aggregates of the D forecasts
    g = df.groupby("date")
    for c in ("residual_load_fc", "res_fc", "load_fc"):
        df[f"{c}_dmean"] = g[c].transform("mean")
    df["residual_load_fc_dmax"] = g["residual_load_fc"].transform("max")
    df["residual_load_fc_dmin"] = g["residual_load_fc"].transform("min")
    # position of this hour within the day's residual-load profile
    df["residual_load_fc_rel"] = df["residual_load_fc"] - df["residual_load_fc_dmean"]
    # hour-to-hour ramps of the forecast (steep ramps -> scarcity / spikes)
    df["residual_load_fc_ramp"] = g["residual_load_fc"].diff()

    # last price of D-1 (hour 23): continuity into the delivery day
    last = df.loc[df["hour"] == 23, ["date", "price"]].copy()
    last["date"] += pd.Timedelta(days=1)
    df = df.merge(last.rename(columns={"price": "price_d1_h23"}), on="date", how="left")
    if not res_d:
        df = _lag_one_day(df, ["res_fc", "res_share_fc", "residual_load_fc", "res_fc_dmean"])
        df = df.drop(columns=RES_D_FEATURES)
    return df.set_index(["date", "hour"]).sort_index()


def feature_cols(df: pd.DataFrame) -> list[str]:
    return [c for c in df.columns if c != "price"] + ["hour_feat"]


def _with_hour(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["hour_feat"] = out.index.get_level_values("hour")
    return out


def _fit_predict_block(df, feats, days, window, return_importance=False):
    d0 = days[0]
    dates = df.index.get_level_values("date")
    tr = (dates >= d0 - pd.Timedelta(days=window)) & (dates <= d0 - pd.Timedelta(days=1))
    train = df[tr & df["price"].notna().to_numpy()]
    test = df[dates.isin(days)]
    if len(train) < 24 * 180 or test.empty:
        return pd.DataFrame(np.nan, index=test.index, columns=QCOLS), None

    ys = AsinhScaler().fit(train["price"].to_numpy())
    z = ys.transform(train["price"].to_numpy())
    preds, importance = [], None
    for q in QUANTILES:
        m = lgb.LGBMRegressor(objective="quantile", alpha=q, **PARAMS)
        m.fit(train[feats], z)
        preds.append(ys.inverse(m.predict(test[feats])))
        if return_importance and q == 0.5:
            importance = pd.Series(m.booster_.feature_importance("gain"), index=feats)
    out = np.sort(np.column_stack(preds), axis=1)  # no quantile crossing
    return pd.DataFrame(out, index=test.index, columns=QCOLS), importance


def lgbm_forecast(df, dates, window=1092, recal_every=7, n_jobs=-1, verbose=5):
    df = _with_hour(df)
    feats = feature_cols(df.drop(columns=["hour_feat"]))
    blocks = [dates[i:i + recal_every] for i in range(0, len(dates), recal_every)]
    res = Parallel(n_jobs=n_jobs, verbose=verbose)(
        delayed(_fit_predict_block)(df, feats, b, window, i == len(blocks) - 1)
        for i, b in enumerate(blocks))
    fc = pd.concat([r[0] for r in res]).sort_index()
    importance = res[-1][1]
    return fc, importance
