"""
Rolling-window out-of-sample evaluation of the benchmarks.

Usage:
    python -m src.evaluate --zone DK_1 --test-start 2024-09-01 --test-end 2026-08-31
    # faster first run: recalibrate weekly instead of daily
    python -m src.evaluate --zone DK_1 --recal-every 7

Saves forecasts to data/forecasts/<zone>_<model>.parquet (date x 24) and prints metrics.
"""
import argparse
import time
from pathlib import Path

import numpy as np
import pandas as pd

from src.lear import design_matrix, lear_forecast, load_wide, naive_forecast

FC_DIR = Path("data/forecasts")


def metrics(y: pd.DataFrame, f: pd.DataFrame, naive: pd.DataFrame) -> dict:
    e = (f - y).to_numpy()
    mae = np.nanmean(np.abs(e))
    denom = (np.abs(y) + np.abs(f)).to_numpy()
    smape = np.nanmean(np.where(denom > 0, 2 * np.abs(e) / denom, np.nan)) * 100
    mae_naive = np.nanmean(np.abs((naive - y).to_numpy()))
    return {"MAE": mae, "RMSE": np.sqrt(np.nanmean(e ** 2)),
            "sMAPE_%": smape, "rMAE": mae / mae_naive}


def report(y, forecasts: dict[str, pd.DataFrame]) -> None:
    # evaluate every model on exactly the same days
    common = y.dropna().index
    for f in forecasts.values():
        common = common.intersection(f.dropna().index)
    y = y.loc[common]
    fc = {k: v.loc[common] for k, v in forecasts.items()}
    print(f"\nEvaluation days: {len(common)} ({common.min().date()} -> {common.max().date()})")

    print("\n== Overall (EUR/MWh) ==")
    print(pd.DataFrame({k: metrics(y, v, fc["naive"]) for k, v in fc.items()}).T.round(3))

    # regime split: helps show where the model wins or loses
    periods = common.to_period("Q")
    rows = {}
    for q in periods.unique():
        m = periods == q
        rows[str(q)] = {k: metrics(y[m], v[m], fc["naive"][m])["MAE"] for k, v in fc.items()}
    print("\n== MAE by quarter ==")
    print(pd.DataFrame(rows).T.round(2))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--zone", default="DK_1")
    ap.add_argument("--test-start", default="2024-09-01")
    ap.add_argument("--test-end", default="2026-08-31")
    ap.add_argument("--windows", default="364,728,1092")
    ap.add_argument("--recal-every", type=int, default=1)
    ap.add_argument("--n-jobs", type=int, default=-1)
    a = ap.parse_args()

    wide = load_wide(f"data/processed/{a.zone}_dataset.parquet")
    X, Y = design_matrix(wide)
    dates = pd.date_range(a.test_start, a.test_end, freq="D")
    print(f"Design matrix: {X.shape[0]} days x {X.shape[1]} features; "
          f"test days: {len(dates)}; recalibration every {a.recal_every} day(s)")

    FC_DIR.mkdir(parents=True, exist_ok=True)
    naive = naive_forecast(Y, dates)
    t0 = time.time()
    lear = lear_forecast(X, Y, dates, tuple(int(w) for w in a.windows.split(",")),
                         a.recal_every, a.n_jobs)
    print(f"LEAR done in {(time.time() - t0) / 60:.1f} min")

    for name, f in {"naive": naive, "lear": lear}.items():
        f.columns = f.columns.astype(str)
        f.to_parquet(FC_DIR / f"{a.zone}_{name}.parquet")
        f.columns = f.columns.astype(int)

    report(Y.loc[dates], {"naive": naive, "lear": lear})
