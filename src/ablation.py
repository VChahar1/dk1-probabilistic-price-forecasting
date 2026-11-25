"Ablation: where does the pre-gate improvement come from?"

import argparse
import time

import numpy as np
import pandas as pd

from src.dm import daily_abs_loss, daily_pinball, dm_test
from src.evaluate_prob import (
    FC_DIR,
    INTERVALS,
    build_prob_models,
    pinball,
    wide_to_long,
)
from src.lgbm import lgbm_forecast, load_long
from src.pregate import pregate_inputs


def main(a) -> None:
    path = f"data/processed/{a.zone}_dataset.parquet"
    y, wx, res_pregate, _ = pregate_inputs(a.zone, verbose=False)
    eval_start = pd.Timestamp(a.eval_start)
    dates = pd.date_range(
        eval_start - pd.Timedelta(days=a.cal_days), a.eval_end, freq="D"
    )

    t0 = time.time()
    lgbm_b, imp = lgbm_forecast(
        load_long(path, res_override=res_pregate), dates, a.window, 7, a.n_jobs
    )
    print(f"LightGBM (proxy only) done in {(time.time() - t0) / 60:.1f} min")
    lgbm_b.to_parquet(FC_DIR / f"{a.zone}_lgbm_q_pregate_proxyonly.parquet")

    lear = wide_to_long(
        pd.read_parquet(FC_DIR / f"{a.zone}_lear_pregate.parquet"), "lear"
    )
    variants = {
        "A: proxy + weather": pd.read_parquet(
            FC_DIR / f"{a.zone}_lgbm_q_pregate.parquet"
        ),
        "B: proxy only": lgbm_b,
    }
    ens = {k: build_prob_models(y, lear, v, a.cal_days)[2] for k, v in variants.items()}

    idx = y.dropna().index
    idx = idx[idx.get_level_values("date") >= eval_start]
    for k in variants:
        idx = idx.intersection(variants[k].dropna().index).intersection(
            ens[k].dropna().index
        )
    yv = y.loc[idx]
    dd = idx.get_level_values("date")
    print(f"\nEvaluation: {dd.nunique()} days ({dd.min().date()} -> {dd.max().date()})")

    lo, hi = INTERVALS[90]
    rows = {}
    for k in variants:
        e = ens[k].loc[idx]
        rows[k] = {
            "LGBM MAE": np.mean(np.abs(variants[k].loc[idx, "q0.50"] - yv)),
            "LGBM pinball": pinball(yv.to_numpy(), variants[k].loc[idx]),
            "ENS MAE": np.mean(np.abs(e["q0.50"] - yv)),
            "ENS pinball": pinball(yv.to_numpy(), e),
            "ENS cov90 %": 100 * np.mean((yv >= e[lo]) & (yv <= e[hi])),
        }
    print("\n== Ablation (EUR/MWh) ==")
    print(pd.DataFrame(rows).round(3))

    A, B = variants
    for label, loss in [
        (
            "LGBM median MAE",
            lambda k: daily_abs_loss(yv, variants[k].loc[idx, "q0.50"]),
        ),
        ("ENS pinball", lambda k: daily_pinball(yv, ens[k].loc[idx])),
    ]:
        for x, z in [(A, B), (B, A)]:
            s, p = dm_test(loss(x), loss(z))
            print(f"DM ({label}) H1 '{x}' better than '{z}': stat {s:.2f}, p = {p:.4f}")

    print("\n== Top 10 features, proxy-only LightGBM (last refit, % gain) ==")
    print((100 * imp / imp.sum()).sort_values(ascending=False).head(10).round(1))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--zone", default="DK_1")
    ap.add_argument("--eval-start", default="2025-01-01")
    ap.add_argument("--eval-end", default="2026-08-31")
    ap.add_argument("--window", type=int, default=1092)
    ap.add_argument("--cal-days", type=int, default=56)
    ap.add_argument("--n-jobs", type=int, default=-1)
    main(ap.parse_args())
