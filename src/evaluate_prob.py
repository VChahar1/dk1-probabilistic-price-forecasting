"Week 3 evaluation: LightGBM quantiles vs LEAR, point and probabilistic."

import argparse
import time
from pathlib import Path

import numpy as np
import pandas as pd

from src.lgbm import QCOLS, QUANTILES, lgbm_forecast, load_long

FC_DIR = Path("data/forecasts")
INTERVALS = {50: ("q0.25", "q0.75"), 80: ("q0.10", "q0.90"), 90: ("q0.05", "q0.95")}


def wide_to_long(w: pd.DataFrame, name: str) -> pd.Series:
    w = w.copy()
    w.columns = w.columns.astype(int)
    s = w.stack()
    s.index = s.index.set_names(["date", "hour"])
    return s.rename(name)


def lear_eq(lear: pd.Series, y: pd.Series, cal_days: int) -> pd.DataFrame:
    """LEAR point forecast + empirical quantiles of past LEAR errors (per hour)."""
    err = (y - lear).unstack("hour").asfreq("D")
    point = lear.unstack("hour").asfreq("D")
    out = {}
    for q, col in zip(QUANTILES, QCOLS):
        eq = (
            err.rolling(cal_days, min_periods=int(0.75 * cal_days)).quantile(q).shift(1)
        )
        out[col] = (point + eq).stack()
    df = pd.DataFrame(out)
    df.index = df.index.set_names(["date", "hour"])
    return pd.DataFrame(np.sort(df.to_numpy(), axis=1), index=df.index, columns=QCOLS)


def cqr(fc: pd.DataFrame, y: pd.Series, cal_days: int) -> pd.DataFrame:
    """Conformalize each central interval with scores from the previous cal_days days."""
    out = fc.copy()
    dates = fc.index.get_level_values("date")
    udates = pd.DatetimeIndex(dates.unique()).sort_values()
    for level, (lo, hi) in INTERVALS.items():
        alpha = 1 - level / 100
        score = np.maximum(fc[lo] - y, y - fc[hi])
        daily = score.groupby(level="date").apply(lambda s: s.to_numpy())
        for d in udates:
            past = daily.loc[
                (daily.index >= d - pd.Timedelta(days=cal_days))
                & (daily.index <= d - pd.Timedelta(days=1))
            ]
            s = np.concatenate(past.to_numpy()) if len(past) else np.array([])
            s = s[~np.isnan(s)]
            m = dates == d
            if len(past) < cal_days or len(s) == 0:
                out.loc[m, [lo, hi]] = np.nan
                continue
            n = len(s)
            qhat = np.quantile(s, min(1.0, np.ceil((n + 1) * (1 - alpha)) / n))
            out.loc[m, lo] = fc.loc[m, lo] - qhat
            out.loc[m, hi] = fc.loc[m, hi] + qhat
    return out


def ensemble(*fcs: pd.DataFrame) -> pd.DataFrame:
    """Equal-weight quantile averaging (weights fixed a priori, not tuned on the test set)."""
    idx = fcs[0].dropna().index
    for f in fcs[1:]:
        idx = idx.intersection(f.dropna().index)
    return sum(f.loc[idx] for f in fcs) / len(fcs)


def build_prob_models(
    y: pd.Series, lear: pd.Series, lgbm_q: pd.DataFrame, cal_days: int
):
    """LEAR-EQ, LGBM-CQR and their 50/50 ensemble from raw forecasts."""
    eq = lear_eq(lear, y.reindex(lear.index), cal_days)
    fc_cqr = cqr(lgbm_q, y.loc[lgbm_q.index], cal_days)
    return eq, fc_cqr, ensemble(eq, fc_cqr)


def pinball(y: np.ndarray, fc: pd.DataFrame) -> float:
    losses = []
    for q, c in zip(QUANTILES, QCOLS):
        d = y - fc[c].to_numpy()
        losses.append(np.mean(np.maximum(q * d, (q - 1) * d)))
    return float(np.mean(losses))


def interval_stats(y: np.ndarray, fc: pd.DataFrame) -> dict:
    out = {}
    for level, (lo, hi) in INTERVALS.items():
        l, h = fc[lo].to_numpy(), fc[hi].to_numpy()
        out[f"cov{level}_%"] = 100 * np.mean((y >= l) & (y <= h))
        out[f"width{level}"] = np.mean(h - l)
    return out


def report(y, point: dict[str, pd.Series], prob: dict[str, pd.DataFrame]) -> None:
    common = y.dropna().index
    for s in point.values():
        common = common.intersection(s.dropna().index)
    for f in prob.values():
        common = common.intersection(f.dropna().index)
    d = common.get_level_values("date")
    print(
        f"\nEvaluation: {d.nunique()} days ({d.min().date()} -> {d.max().date()}), "
        f"{len(common):,} hours"
    )
    yv = y.loc[common].to_numpy()

    naive_mae = np.mean(np.abs(point["naive"].loc[common].to_numpy() - yv))
    rows = {}
    for k, s in point.items():
        e = s.loc[common].to_numpy() - yv
        rows[k] = {
            "MAE": np.mean(np.abs(e)),
            "RMSE": np.sqrt(np.mean(e**2)),
            "rMAE": np.mean(np.abs(e)) / naive_mae,
        }
    print("\n== Point forecasts (EUR/MWh) ==")
    print(pd.DataFrame(rows).T.round(3))

    rows = {
        k: {"pinball": pinball(yv, f.loc[common]), **interval_stats(yv, f.loc[common])}
        for k, f in prob.items()
    }
    print("\n== Probabilistic (pinball in EUR/MWh; nominal coverage 50/80/90%) ==")
    print(pd.DataFrame(rows).T.round(2))

    q = pd.PeriodIndex(d, freq="Q")
    rows = {}
    for p in q.unique():
        m = q == p
        rows[str(p)] = {k: pinball(yv[m], f.loc[common][m]) for k, f in prob.items()}
        rows[str(p)].update(
            {
                f"cov90_{k}": interval_stats(yv[m], prob[k].loc[common][m])["cov90_%"]
                for k in ("LGBM", "LGBM-CQR")
            }
        )
    print("\n== Pinball loss and 90% coverage by quarter ==")
    print(pd.DataFrame(rows).T.round(2))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--zone", default="DK_1")
    ap.add_argument("--test-start", default="2024-09-01")
    ap.add_argument("--test-end", default="2026-08-31")
    ap.add_argument("--window", type=int, default=1092)
    ap.add_argument("--recal-every", type=int, default=7)
    ap.add_argument("--cal-days", type=int, default=56)
    ap.add_argument("--n-jobs", type=int, default=-1)
    ap.add_argument(
        "--reuse",
        action="store_true",
        help="reuse saved LightGBM quantiles instead of refitting",
    )
    a = ap.parse_args()

    df = load_long(f"data/processed/{a.zone}_dataset.parquet")
    y = df["price"]
    dates = pd.date_range(a.test_start, a.test_end, freq="D")
    print(
        f"LightGBM: {len(QUANTILES)} quantiles, window {a.window} d, "
        f"refit every {a.recal_every} d, {len(dates)} test days"
    )

    if a.reuse:
        fc, importance = pd.read_parquet(FC_DIR / f"{a.zone}_lgbm_q.parquet"), None
        print("Reusing saved LightGBM quantiles")
    else:
        t0 = time.time()
        fc, importance = lgbm_forecast(df, dates, a.window, a.recal_every, a.n_jobs)
        print(f"LightGBM done in {(time.time() - t0) / 60:.1f} min")
        fc.to_parquet(FC_DIR / f"{a.zone}_lgbm_q.parquet")

    lear = wide_to_long(pd.read_parquet(FC_DIR / f"{a.zone}_lear.parquet"), "lear")
    naive = wide_to_long(pd.read_parquet(FC_DIR / f"{a.zone}_naive.parquet"), "naive")
    y_test = y.loc[fc.index]

    eq, fc_cqr, ens = build_prob_models(y, lear, fc, a.cal_days)
    for name, f in {"lear_eq": eq, "lgbm_cqr": fc_cqr, "ens": ens}.items():
        f.to_parquet(FC_DIR / f"{a.zone}_{name}.parquet")

    report(
        y_test,
        point={
            "naive": naive,
            "lear": lear,
            "lgbm_median": fc["q0.50"],
            "ens_median": ens["q0.50"],
        },
        prob={"LEAR-EQ": eq, "LGBM": fc, "LGBM-CQR": fc_cqr, "ENS": ens},
    )

    if importance is not None:
        print("\n== Top 15 features (median model, last refit, share of gain %) ==")
        print(
            (100 * importance / importance.sum())
            .sort_values(ascending=False)
            .head(15)
            .round(1)
        )
