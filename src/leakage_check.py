"Week 4: how much do the results depend on the wind/solar forecast for day D?"

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
from src.lear import design_matrix, lear_forecast, load_wide
from src.lgbm import lgbm_forecast, load_long


def main(a) -> None:
    path = f"data/processed/{a.zone}_dataset.parquet"
    dates = pd.date_range(a.test_start, a.test_end, freq="D")
    y = load_long(path)["price"]

    t0 = time.time()
    X, Y = design_matrix(load_wide(path), res_lags=(1, 7))
    lear_w = lear_forecast(X, Y, dates, recal_every=a.lear_recal, n_jobs=a.n_jobs)
    print(f"LEAR (no RES(D)) done in {(time.time() - t0) / 60:.1f} min")
    lear_w.columns = lear_w.columns.astype(str)
    lear_w.to_parquet(FC_DIR / f"{a.zone}_lear_nores.parquet")
    lear_r = wide_to_long(lear_w, "lear")

    t0 = time.time()
    lgbm_r, _ = lgbm_forecast(
        load_long(path, res_d=False), dates, a.window, 7, a.n_jobs
    )
    print(f"LightGBM (no RES(D)) done in {(time.time() - t0) / 60:.1f} min")
    lgbm_r.to_parquet(FC_DIR / f"{a.zone}_lgbm_q_nores.parquet")
    eq_r, cqr_r, ens_r = build_prob_models(y, lear_r, lgbm_r, a.cal_days)
    ens_r.to_parquet(FC_DIR / f"{a.zone}_ens_nores.parquet")

    full = {
        "LEAR": wide_to_long(
            pd.read_parquet(FC_DIR / f"{a.zone}_lear.parquet"), "lear"
        ),
        "LEAR-EQ": pd.read_parquet(FC_DIR / f"{a.zone}_lear_eq.parquet"),
        "LGBM-CQR": pd.read_parquet(FC_DIR / f"{a.zone}_lgbm_cqr.parquet"),
        "ENS": pd.read_parquet(FC_DIR / f"{a.zone}_ens.parquet"),
    }
    full["LGBM median"] = pd.read_parquet(FC_DIR / f"{a.zone}_lgbm_q.parquet")["q0.50"]
    red = {
        "LEAR": lear_r,
        "LEAR-EQ": eq_r,
        "LGBM-CQR": cqr_r,
        "ENS": ens_r,
        "LGBM median": lgbm_r["q0.50"],
    }

    idx = y.dropna().index
    for d in (full, red):
        for v in d.values():
            idx = idx.intersection(v.dropna().index)
    yv = y.loc[idx]
    nd = idx.get_level_values("date").nunique()
    print(f"\nEvaluation: {nd} days, {len(idx):,} hours")

    rows = {}
    for k in ("LEAR", "LGBM median"):
        mf = np.mean(np.abs(full[k].loc[idx] - yv))
        mr = np.mean(np.abs(red[k].loc[idx] - yv))
        rows[f"{k} (MAE)"] = {
            "full": mf,
            "no RES(D)": mr,
            "change_%": 100 * (mr / mf - 1),
        }
    mf = np.mean(np.abs(full["ENS"].loc[idx, "q0.50"] - yv))
    mr = np.mean(np.abs(red["ENS"].loc[idx, "q0.50"] - yv))
    rows["ENS median (MAE)"] = {
        "full": mf,
        "no RES(D)": mr,
        "change_%": 100 * (mr / mf - 1),
    }
    for k in ("LEAR-EQ", "LGBM-CQR", "ENS"):
        pf = pinball(yv.to_numpy(), full[k].loc[idx])
        pr = pinball(yv.to_numpy(), red[k].loc[idx])
        rows[f"{k} (pinball)"] = {
            "full": pf,
            "no RES(D)": pr,
            "change_%": 100 * (pr / pf - 1),
        }
    print("\n== Full vs leakage-free (no wind/solar forecast for D), EUR/MWh ==")
    print(pd.DataFrame(rows).T.round(3))

    lo, hi = INTERVALS[90]
    cov = {
        name: 100
        * np.mean((yv >= m["ENS"].loc[idx, lo]) & (yv <= m["ENS"].loc[idx, hi]))
        for name, m in [("full", full), ("no RES(D)", red)]
    }
    print(
        f"\nENS 90% coverage: full {cov['full']:.1f}%, no RES(D) {cov['no RES(D)']:.1f}%"
    )

    stat, p = dm_test(
        daily_pinball(yv, full["ENS"].loc[idx]), daily_pinball(yv, red["ENS"].loc[idx])
    )
    print(
        f"DM test (pinball), H1 'full ENS better than no-RES(D) ENS': "
        f"stat {stat:.2f}, p = {p:.4f}"
    )
    stat, p = dm_test(
        daily_abs_loss(yv, red["ENS"].loc[idx, "q0.50"]),
        daily_abs_loss(yv, full["LEAR"].loc[idx]),
    )
    print(
        f"DM test (MAE), H1 'no-RES(D) ENS better than full LEAR': "
        f"stat {stat:.2f}, p = {p:.4f}"
    )


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--zone", default="DK_1")
    ap.add_argument("--test-start", default="2024-09-01")
    ap.add_argument("--test-end", default="2026-08-31")
    ap.add_argument("--window", type=int, default=1092)
    ap.add_argument("--lear-recal", type=int, default=1)
    ap.add_argument("--cal-days", type=int, default=56)
    ap.add_argument("--n-jobs", type=int, default=-1)
    main(ap.parse_args())
