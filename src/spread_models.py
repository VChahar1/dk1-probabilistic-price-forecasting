"""
Intraday extension, Step 4: models for the imbalance - day-ahead spread (validation only).

Pre-registered (Step 3/4):
  * sign: logistic regression (one-hot hour/day-of-week, standardised numeric features,
    L2 penalty C chosen on VALIDATION log loss)
  * size: LightGBM quantile regression (7 quantiles, asinh target, fixed Week-3 parameters)
    + conformalized quantile regression (CQR, previous 56 days of out-of-sample errors)
  * walk-forward: refit monthly on all earlier data of the SAME regime; first month needs
    >= 120 training days
  * benchmarks: zero and trailing 7-day same-hour median (point); empirical quantiles of the
    previous 28 days, D-29..D-2 (probabilistic); trailing share of positive spreads (sign)
  * tests: daily multivariate Diebold-Mariano (Newey-West, 7-day lag)
Test-period predictions are produced and saved but NOT evaluated here (Step 5, once).

Usage:
    python -m src.spread_models --fit        # walk-forward fits (~5-10 min on 12 cores)
    python -m src.spread_models --evaluate   # validation tables + DM tests + diagnostics
    python -m src.spread_models --evaluate --split test   # Step 5: ONE-TIME test evaluation
"""
import argparse
import json
import time
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from src.dm import dm_test
from src.evaluate_prob import FC_DIR, INTERVALS, cqr, pinball
from src.infosets import INFOSETS, OUT as TABLE
from src.lear import AsinhScaler
from src.lgbm import PARAMS, QCOLS, QUANTILES
from src.revisions import ACTUALS

MIN_TRAIN_DAYS = 120
CAL_DAYS = 56
C_GRID = (0.01, 0.1, 1.0, 10.0)
CAT = ["hour", "dow"]
OUT_DIR = FC_DIR / "spread"
CONFIG = OUT_DIR / "config.json"


# --------------------------------------------------------------------- helpers
def load_table() -> pd.DataFrame:
    df = pd.read_parquet(TABLE)
    df.index = df.index.set_levels(pd.to_datetime(df.index.levels[0]), level=0)
    return df


def rows_of(df: pd.DataFrame, name: str) -> pd.DataFrame:
    hours = INFOSETS[name][2]
    return df[df["eligible_T2"]] if hours == "T2" else df


def _dates(df) -> pd.Series:
    return pd.Series(df.index.get_level_values("date"), index=df.index)


def fit_lgbm(Xtr, ytr, Xte) -> np.ndarray:
    sc = AsinhScaler().fit(ytr)
    z = sc.transform(ytr)
    out = []
    for q in QUANTILES:
        m = lgb.LGBMRegressor(objective="quantile", alpha=q, **PARAMS)
        m.fit(Xtr, z)
        out.append(sc.inverse(m.predict(Xte)))
    return np.sort(np.column_stack(out), axis=1)          # no quantile crossing


def logit_pipeline(feats, C) -> Pipeline:
    cat = [c for c in CAT if c in feats]
    num = [c for c in feats if c not in cat]
    pre = ColumnTransformer([
        ("cat", OneHotEncoder(handle_unknown="ignore"), cat),
        ("num", make_pipeline(SimpleImputer(strategy="median"), StandardScaler()), num),
    ])
    return Pipeline([("pre", pre), ("clf", LogisticRegression(C=C, max_iter=2000))])


def _block(df_reg, name, m0):
    """Fit on all regime data before month m0; predict month m0."""
    feats = INFOSETS[name][1]
    d = _dates(df_reg)
    tr = df_reg[(d < m0) & df_reg["spread"].notna()]
    te = df_reg[(d >= m0) & (d < m0 + pd.offsets.MonthBegin(1))]
    if tr.index.get_level_values("date").nunique() < MIN_TRAIN_DAYS or te.empty:
        return None
    q = pd.DataFrame(fit_lgbm(tr[feats], tr["spread"].to_numpy(), te[feats]),
                     index=te.index, columns=QCOLS)
    p = {}
    for C in C_GRID:
        pipe = logit_pipeline(feats, C).fit(tr[feats], tr["sign"].astype(int))
        p[C] = pd.Series(pipe.predict_proba(te[feats])[:, 1], index=te.index)
    return name, q, p


# ------------------------------------------------------------------------ fit
def fit(n_jobs=-1) -> None:
    df = load_table()
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    tasks = []
    for reg in ("pre", "post"):
        dr = df[df["regime"] == reg]
        if dr.empty:
            continue
        d = _dates(dr)
        months = pd.date_range(d.min().to_period("M").to_timestamp(), d.max(), freq="MS")
        for name in INFOSETS:
            sub = rows_of(dr, name)
            tasks += [(sub, name, m0) for m0 in months]
    t0 = time.time()
    res = [r for r in Parallel(n_jobs=n_jobs, verbose=5)(delayed(_block)(*t) for t in tasks) if r]
    print(f"{len(res)} monthly fits in {(time.time() - t0) / 60:.1f} min")

    y = df["spread"]
    for name in INFOSETS:
        parts = [r for r in res if r[0] == name]
        q = pd.concat([r[1] for r in parts]).sort_index()
        reg = df["regime"].reindex(q.index)
        # CQR within each regime, calibrated on that regime's own past out-of-sample errors
        q_cqr = pd.concat([cqr(q[reg == r], y.loc[q[reg == r].index], CAL_DAYS)
                           for r in reg.unique()]).sort_index()
        q.to_parquet(OUT_DIR / f"lgbm_{name}.parquet")
        q_cqr.to_parquet(OUT_DIR / f"cqr_{name}.parquet")
        probs = pd.DataFrame({str(C): pd.concat([r[2][C] for r in parts]) for C in C_GRID}).sort_index()
        probs.to_parquet(OUT_DIR / f"logit_{name}.parquet")
    benchmarks(df).to_parquet(OUT_DIR / "benchmarks.parquet")
    print(f"saved predictions to {OUT_DIR}/ (validation AND test; test is evaluated in Step 5)")


def benchmarks(df: pd.DataFrame) -> pd.DataFrame:
    """Trailing 28-day (D-29..D-2) empirical quantiles and share of positive spreads."""
    s = df["spread"]
    by_day = s.groupby(level="date").apply(lambda x: x.dropna().to_numpy())
    days = by_day.index
    rows = {}
    for D in days:
        win = by_day[(days >= D - pd.Timedelta(days=29)) & (days <= D - pd.Timedelta(days=2))]
        v = np.concatenate(win.to_numpy()) if len(win) else np.array([])
        if len(v) < 24 * 20:
            continue
        rows[D] = list(np.quantile(v, QUANTILES)) + [np.mean(v > 0)]
    b = pd.DataFrame.from_dict(rows, orient="index", columns=[f"clim_{c}" for c in QCOLS] + ["share_pos"])
    out = b.reindex(df.index.get_level_values("date"))
    out.index = df.index
    out["med7"] = df["spread_med7"]
    out["zero"] = 0.0
    return out


# -------------------------------------------------------------------- metrics
def _clip(p):
    return np.clip(p, 1e-4, 1 - 1e-4)


def row_logloss(y, p):
    p = _clip(p)
    return -(y * np.log(p) + (1 - y) * np.log(1 - p))


def row_pinball(y, fc) -> pd.Series:
    out = 0.0
    for q, c in zip(QUANTILES, QCOLS):
        d = y - fc[c]
        out = out + np.maximum(q * d, (q - 1) * d)
    return out / len(QUANTILES)


def daily(s: pd.Series) -> pd.Series:
    return s.groupby(level="date").mean()


def coverage(y, fc, level):
    lo, hi = INTERVALS[level]
    return 100 * np.mean((y >= fc[lo]) & (y <= fc[hi]))


def evaluate(split: str = "val") -> None:
    df = load_table()
    b = pd.read_parquet(OUT_DIR / "benchmarks.parquet")
    clim = b[[f"clim_{c}" for c in QCOLS]].set_axis(QCOLS, axis=1)
    val = df[df["split"] == split]
    pred = {n: {"lgbm": pd.read_parquet(OUT_DIR / f"lgbm_{n}.parquet"),
                "cqr": pd.read_parquet(OUT_DIR / f"cqr_{n}.parquet"),
                "logit": pd.read_parquet(OUT_DIR / f"logit_{n}.parquet")} for n in INFOSETS}

    if split == "val":
        # ---- choose C per (information set, regime) on validation log loss
        config = {}
        for n in INFOSETS:
            for reg in ("pre", "post"):
                v = rows_of(val[val["regime"] == reg], n)
                P = pred[n]["logit"].reindex(v.index)
                ok = v["sign"].notna() & P.notna().all(axis=1)
                if ok.sum() == 0:
                    continue
                ll = {C: row_logloss(v.loc[ok, "sign"], P.loc[ok, C]).mean() for C in P.columns}
                config[f"{n}/{reg}"] = {"C": min(ll, key=ll.get)}
        CONFIG.write_text(json.dumps(config, indent=2))
        print(f"chosen penalties saved to {CONFIG}: "
              + ", ".join(f"{k} C={v['C']}" for k, v in config.items()))
    else:
        if not CONFIG.exists():
            raise SystemExit("no config.json: run --evaluate on validation first (it fixes C)")
        config = json.loads(CONFIG.read_text())
        print("=" * 72 + f"\nONE-TIME {split.upper()} EVALUATION with settings fixed on validation "
              f"({CONFIG})\n" + "=" * 72)
    results = {}

    for reg in ("post", "pre"):
        vr = val[val["regime"] == reg]
        if vr.empty:
            continue
        print(f"\n################ {reg.upper()} regime, {split.upper()} ################")
        size, sign, losses = {}, {}, {}
        for n in INFOSETS:
            v = rows_of(vr, n)
            q, qc = pred[n]["lgbm"].reindex(v.index), pred[n]["cqr"].reindex(v.index)
            key = f"{n}/{reg}"
            if key not in config:
                continue
            p = pred[n]["logit"][config[key]["C"]].reindex(v.index)
            ok = (v["spread"].notna() & q.notna().all(axis=1) & qc.notna().all(axis=1)
                  & p.notna() & clim.reindex(v.index).notna().all(axis=1) & v["spread_med7"].notna())
            v, q, qc, p, cl = v[ok], q[ok], qc[ok], p[ok], clim.reindex(v.index)[ok]
            y = v["spread"]
            size[n] = {"rows": len(y), "pinball LGBM": row_pinball(y, q).mean(),
                       "pinball CQR": row_pinball(y, qc).mean(), "pinball clim": row_pinball(y, cl).mean(),
                       "MAE median": (y - qc["q0.50"]).abs().mean(),
                       "MAE med7": (y - v["spread_med7"]).abs().mean(), "MAE zero": y.abs().mean(),
                       "cov90 LGBM": coverage(y, q, 90), "cov90 CQR": coverage(y, qc, 90),
                       "cov90 clim": coverage(y, cl, 90)}
            share = b["share_pos"].reindex(v.index)
            sign[n] = {"log loss": row_logloss(v["sign"], p).mean(),
                       "log loss bench": row_logloss(v["sign"], share).mean(),
                       "Brier": ((p - v["sign"]) ** 2).mean(),
                       "accuracy": 100 * ((p > 0.5) == (v["sign"] == 1)).mean(),
                       "acc. majority": 100 * ((share > 0.5) == (v["sign"] == 1)).mean(),
                       "share>0 %": 100 * v["sign"].mean()}
            losses[n] = {"pin": row_pinball(y, qc), "pin_clim": row_pinball(y, cl),
                         "mae": (y - qc["q0.50"]).abs(), "mae_med7": (y - v["spread_med7"]).abs(),
                         "mae_zero": y.abs(),
                         "ll": row_logloss(v["sign"], p), "ll_bench": row_logloss(v["sign"], share)}
        print("\n== Size: quantile forecasts of the spread (EUR/MWh) ==")
        print(pd.DataFrame(size).T.round(2).to_string())
        print("\n== Sign: probability that imbalance > day-ahead ==")
        print(pd.DataFrame(sign).T.round(3).to_string())

        print("\n== DM tests (one-sided, daily losses; p < 0.05 = first model better) ==")
        dms = []
        def dm(a, b, la, lb, label):
            j = pd.concat([daily(a), daily(b)], axis=1).dropna()
            if len(j) < 10:
                return
            if (j.iloc[:, 0] - j.iloc[:, 1]).abs().max() < 1e-12:
                print(f"  {label:<40} {la} vs {lb}: identical losses, no test")
                return
            s, pv = dm_test(j.iloc[:, 0], j.iloc[:, 1])
            eff = 100 * (j.iloc[:, 0].mean() / j.iloc[:, 1].mean() - 1)
            print(f"  {label:<40} {la} vs {lb}: {eff:+6.2f}% loss, stat {s:+.2f}, p = {pv:.4f}  "
                  f"({len(j)} days)")
            dms.append({"test": label, "a": la, "b": lb, "change_%": eff, "stat": s, "p": pv,
                        "days": len(j)})
        for (a_, b_, lab) in [("A1", "A0", "H1 evening revision"), ("A2", "A1e", "H2 morning revision")]:
            if a_ in losses and b_ in losses:
                for k, nm in [("pin", "pinball"), ("mae", "MAE"), ("ll", "log loss")]:
                    dm(losses[a_][k], losses[b_][k], a_, b_, f"{lab}, {nm}")
        for n in ("B0", "A0", "A1", "A2"):
            if n in losses:
                dm(losses[n]["pin"], losses[n]["pin_clim"], n, "clim", f"{n} vs benchmark, pinball")
                dm(losses[n]["mae"], losses[n]["mae_med7"], n, "med7", f"{n} vs benchmark, MAE")
                dm(losses[n]["mae"], losses[n]["mae_zero"], n, "zero", f"{n} vs benchmark, MAE")
                dm(losses[n]["ll"], losses[n]["ll_bench"], n, "trail", f"{n} vs benchmark, log loss")
        results[reg] = {"size": pd.DataFrame(size).T.round(4).to_dict(orient="index"),
                        "sign": pd.DataFrame(sign).T.round(4).to_dict(orient="index"),
                        "dm": dms}
    Path("results").mkdir(exist_ok=True)
    out = Path(f"results/spread_{split}.json")
    out.write_text(json.dumps(results, indent=2, default=float))
    print(f"\nsaved {out}")
    if split == "val":
        print("Note: validation log losses use the C chosen on validation itself (mildly "
              "optimistic); Step 5 evaluates the untouched test period once.")
        diagnostics(df)


def diagnostics(df: pd.DataFrame) -> None:
    """Train + validation only. Ex-post quantities: never features."""
    a = pd.read_parquet(ACTUALS)
    a = a.groupby(a.index.floor("h")).mean()
    wind_act = a[[c for c in ("wind_on", "wind_off") if c in a]].sum(axis=1, min_count=1)
    tv = df[df["split"].isin(["train", "val"])].copy()
    tv["wind_act"] = wind_act.reindex(pd.DatetimeIndex(tv["start_utc"])).to_numpy()
    tv["resid_v0"] = tv["wind_act"] - tv["wind_mw_v0"]          # surprise vs gate-closure forecast
    tv["resid_v2"] = tv["wind_act"] - tv["wind_mw_v2"]          # surprise vs latest forecast
    cols = ["d1_wind_mw", "dtot_wind_mw", "d1_solar_mw", "resid_v0", "resid_v2"]
    rows = {f"{r}/{s}": {c: g[[c, "spread"]].corr(method="spearman").iloc[0, 1] for c in cols}
            for (r, s), g in tv.groupby(["regime", "split"])}
    print("\n== Diagnostics (TRAIN + VALIDATION): Spearman correlation with the spread ==")
    print(pd.DataFrame(rows).T.round(3).to_string())
    print("revisions (d1, dtot) are features; resid_* are EX-POST surprises (diagnostic only).\n"
          "If |resid| correlations >> |revision| correlations, imbalances are driven by the error\n"
          "remaining after the latest forecast, i.e. revisions are largely traded away intraday.")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--fit", action="store_true")
    ap.add_argument("--evaluate", action="store_true")
    ap.add_argument("--split", default="val", choices=["val", "test"])
    ap.add_argument("--n-jobs", type=int, default=-1)
    a = ap.parse_args()
    if a.fit:
        fit(a.n_jobs)
    if a.evaluate:
        evaluate(a.split)
    if not (a.fit or a.evaluate):
        ap.print_help()
