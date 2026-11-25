"""
Next step 1: replace the ENTSO-E wind/solar forecast for D with a pre-gate ECMWF-based one
and compare three variants on the same days:

    full       ENTSO-E wind/solar forecast for D   (possibly published after gate closure)
    pre-gate   ECMWF 00 UTC run of D-1 -> renewables proxy   (strictly known at 12:00 D-1)
    no RES(D)  only the ENTSO-E forecast made for D-1          (pessimistic lower bound)

Weather-to-power proxy: each month, a linear regression maps the ECMWF indices
(onshore / offshore wind power-curve index, solar radiation) to the ENTSO-E renewables
forecast, fitted on the previous 365 days only (ending 2 days before the month starts).
Where no proxy exists yet (before ~June 2024), training rows keep the ENTSO-E series,
which is legitimate: leakage only matters for the features of the day being forecast.

Usage (after src.weather --build and the Week 2-4 runs):
    python -m src.pregate --zone DK_1                  # LEAR daily refit (~30 min)
    python -m src.pregate --zone DK_1 --lear-recal 7   # quick (~10 min)
"""
import argparse
import time

import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression

from src.dm import daily_abs_loss, daily_pinball, dm_test
from src.evaluate_prob import FC_DIR, INTERVALS, build_prob_models, pinball, wide_to_long
from src.lear import RES_COLS, design_matrix, lear_forecast, load_wide
from src.lgbm import lgbm_forecast, load_long

WX = ["wind_on_idx", "wind_off_idx", "solar_idx"]


# ------------------------------------------------------------- weather -> power proxy
def proxy_coefs(entsoe_res: pd.Series, wx: pd.DataFrame, fit_days=365, min_days=60) -> pd.DataFrame:
    """Monthly weather-to-power coefficients, each fitted on the previous 365 days only
    (ending 2 days before the month starts). Index: month start; columns: WX + intercept."""
    df = wx.join(entsoe_res.rename("res"), how="left")
    dates = df.index.get_level_values("date")
    rows = {}
    for m0 in pd.date_range(dates.min().to_period("M").to_timestamp(), dates.max(), freq="MS"):
        tr = (dates >= m0 - pd.Timedelta(days=fit_days)) & (dates <= m0 - pd.Timedelta(days=2))
        train = df[tr].dropna(subset=WX + ["res"])
        if train.index.get_level_values("date").nunique() < min_days:
            continue
        model = LinearRegression().fit(train[WX], train["res"])
        rows[m0] = dict(zip(WX, model.coef_), intercept=model.intercept_)
    return pd.DataFrame.from_dict(rows, orient="index")


def month_coefs(index: pd.MultiIndex, coefs: pd.DataFrame) -> pd.DataFrame:
    """Coefficients aligned to a (date, hour) index by calendar month."""
    months = index.get_level_values("date").to_period("M").to_timestamp()
    return coefs.reindex(months).set_axis(index)


def res_proxy(entsoe_res, wx, fit_days=365, min_days=60):
    """Out-of-sample monthly-refit linear map: ECMWF indices -> renewables forecast (MW)."""
    c = month_coefs(wx.index, proxy_coefs(entsoe_res, wx, fit_days, min_days))
    pred = c["intercept"] + sum(c[k] * wx[k] for k in WX)
    return pred.clip(lower=0).rename("res_proxy")


def proxy_report(proxy: pd.Series, entsoe_res: pd.Series) -> None:
    j = pd.concat([proxy, entsoe_res.rename("entsoe")], axis=1).dropna()
    e = j["res_proxy"] - j["entsoe"]
    r2 = 1 - (e ** 2).sum() / ((j["entsoe"] - j["entsoe"].mean()) ** 2).sum()
    d = j.index.get_level_values("date")
    print(f"\nWeather-to-power proxy vs ENTSO-E renewables forecast "
          f"({d.min().date()} -> {d.max().date()}, out-of-sample):")
    print(f"  R2 = {r2:.3f}, MAE = {e.abs().mean():.0f} MW "
          f"({100 * e.abs().mean() / j['entsoe'].mean():.1f}% of mean renewables)")


# ------------------------------------------------------------------------- helpers
def long_to_wide(s: pd.Series) -> pd.DataFrame:
    w = s.unstack("hour")
    w.index = pd.DatetimeIndex(w.index, name="date")
    return w.asfreq("D")


def pregate_inputs(zone: str, verbose: bool = True):
    """Returns (y, weather indices, pre-gate renewables series, first proxy date)."""
    path = f"data/processed/{zone}_dataset.parquet"
    y = load_long(path)["price"]
    wx = pd.read_parquet(f"data/processed/{zone}_weather.parquet")

    # ENTSO-E renewables (strict sum) on the (date, hour) grid
    ds = pd.read_parquet(path)
    ds.index = ds.index.set_levels(pd.to_datetime(ds.index.levels[0]), level=0)
    entsoe_res = ds[RES_COLS].sum(axis=1, min_count=len(RES_COLS))

    proxy = res_proxy(entsoe_res, wx)
    first_proxy = proxy.dropna().index.get_level_values("date").min()
    # splice: ENTSO-E only BEFORE the proxy exists; afterwards gaps stay NaN (no leakage)
    res_pregate = proxy.reindex(entsoe_res.index)
    early = res_pregate.index.get_level_values("date") < first_proxy
    res_pregate[early] = entsoe_res[early]
    if verbose:
        proxy_report(proxy, entsoe_res)
        n_gap = res_pregate[~early].isna().groupby(level="date").any().sum()
        print(f"proxy from {first_proxy.date()}; days after that without a pre-gate "
              f"forecast: {n_gap}")
    return y, wx, res_pregate, first_proxy


def main(a) -> None:
    path = f"data/processed/{a.zone}_dataset.parquet"
    y, wx, res_pregate, first_proxy = pregate_inputs(a.zone)
    eval_start = pd.Timestamp(a.eval_start)
    if eval_start < first_proxy + pd.Timedelta(days=7):
        raise SystemExit(f"eval start must be >= {first_proxy + pd.Timedelta(days=7)}")

    # forecast from eval_start - cal_days so the calibrated models are warm at eval_start
    dates = pd.date_range(eval_start - pd.Timedelta(days=a.cal_days), a.eval_end, freq="D")

    t0 = time.time()
    wide = load_wide(path)
    wide["res_fc"] = long_to_wide(res_pregate)
    X, Y = design_matrix(wide)
    lear_w = lear_forecast(X, Y, dates, recal_every=a.lear_recal, n_jobs=a.n_jobs)
    print(f"LEAR (pre-gate) done in {(time.time() - t0) / 60:.1f} min")
    lear_w.columns = lear_w.columns.astype(str)
    lear_w.to_parquet(FC_DIR / f"{a.zone}_lear_pregate.parquet")

    t0 = time.time()
    df_pg = load_long(path, res_override=res_pregate, extra=wx)
    lgbm_pg, imp = lgbm_forecast(df_pg, dates, a.window, 7, a.n_jobs)
    print(f"LightGBM (pre-gate) done in {(time.time() - t0) / 60:.1f} min")
    lgbm_pg.to_parquet(FC_DIR / f"{a.zone}_lgbm_q_pregate.parquet")

    # ---------------- three variants, each rebuilt the same way (EQ, CQR, ensemble)
    raw = {
        "full": (wide_to_long(pd.read_parquet(FC_DIR / f"{a.zone}_lear.parquet"), "lear"),
                 pd.read_parquet(FC_DIR / f"{a.zone}_lgbm_q.parquet")),
        "pre-gate": (wide_to_long(lear_w, "lear"), lgbm_pg),
        "no RES(D)": (wide_to_long(pd.read_parquet(FC_DIR / f"{a.zone}_lear_nores.parquet"), "lear"),
                      pd.read_parquet(FC_DIR / f"{a.zone}_lgbm_q_nores.parquet")),
    }
    naive = wide_to_long(pd.read_parquet(FC_DIR / f"{a.zone}_naive.parquet"), "naive")
    models = {}
    for k, (lear, lg) in raw.items():
        eq, cq, ens = build_prob_models(y, lear, lg, a.cal_days)
        models[k] = {"lear": lear, "lgbm": lg["q0.50"], "eq": eq, "cqr": cq, "ens": ens}
    models["pre-gate"]["ens"].to_parquet(FC_DIR / f"{a.zone}_ens_pregate.parquet")

    idx = y.dropna().index.intersection(naive.dropna().index)
    idx = idx[idx.get_level_values("date") >= eval_start]
    for m in models.values():
        for v in m.values():
            idx = idx.intersection(v.dropna().index)
    yv = y.loc[idx]
    dd = idx.get_level_values("date")
    print(f"\nEvaluation: {dd.nunique()} days ({dd.min().date()} -> {dd.max().date()}), "
          f"{len(idx):,} hours")

    naive_mae = np.mean(np.abs(naive.loc[idx] - yv))
    lo, hi = INTERVALS[90]
    rows = {}
    for k, m in models.items():
        ens = m["ens"].loc[idx]
        rows[k] = {
            "LEAR MAE": np.mean(np.abs(m["lear"].loc[idx] - yv)),
            "LGBM MAE": np.mean(np.abs(m["lgbm"].loc[idx] - yv)),
            "ENS MAE": np.mean(np.abs(ens["q0.50"] - yv)),
            "ENS rMAE": np.mean(np.abs(ens["q0.50"] - yv)) / naive_mae,
            "ENS pinball": pinball(yv.to_numpy(), ens),
            "ENS cov90 %": 100 * np.mean((yv >= ens[lo]) & (yv <= ens[hi])),
        }
    t = pd.DataFrame(rows)
    print("\n== Three information sets, same days (EUR/MWh) ==")
    print(t.round(3))

    f, p, n = t.loc["ENS pinball", "full"], t.loc["ENS pinball", "pre-gate"], t.loc["ENS pinball", "no RES(D)"]
    if n - f > 0.02 * f:   # only meaningful when the full model is clearly better
        print(f"\nPre-gate recovers {100 * (n - p) / (n - f):.0f}% of the pinball gap between "
              f"'no RES(D)' and 'full'")
    else:
        print("\nFull and no-RES(D) are too close for a meaningful 'gap recovered' figure")

    L = {k: daily_pinball(yv, m["ens"].loc[idx]) for k, m in models.items()}
    for a_, b_ in [("pre-gate", "no RES(D)"), ("full", "pre-gate"), ("pre-gate", "full")]:
        s, pv = dm_test(L[a_], L[b_])
        print(f"DM (ENS pinball) H1 '{a_} better than {b_}': stat {s:.2f}, p = {pv:.4f}")
    s, pv = dm_test(daily_abs_loss(yv, models["pre-gate"]["ens"].loc[idx, "q0.50"]),
                    daily_abs_loss(yv, models["full"]["lear"].loc[idx]))
    print(f"DM (MAE) H1 'pre-gate ENS better than full LEAR': stat {s:.2f}, p = {pv:.4f}")

    if imp is not None:
        print("\n== Top 10 features, pre-gate LightGBM (last refit, % gain) ==")
        print((100 * imp / imp.sum()).sort_values(ascending=False).head(10).round(1))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--zone", default="DK_1")
    ap.add_argument("--eval-start", default="2025-01-01")
    ap.add_argument("--eval-end", default="2026-08-31")
    ap.add_argument("--window", type=int, default=1092)
    ap.add_argument("--lear-recal", type=int, default=1)
    ap.add_argument("--cal-days", type=int, default=56)
    ap.add_argument("--n-jobs", type=int, default=-1)
    main(ap.parse_args())
