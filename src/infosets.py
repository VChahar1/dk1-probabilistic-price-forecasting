"Intraday extension, Step 3: decision times, information sets and an automated leakage audit."

import argparse
import time
from pathlib import Path

import holidays
import numpy as np
import pandas as pd

from src.evaluate_prob import FC_DIR, wide_to_long
from src.lear import design_matrix, lear_forecast, load_wide
from src.lgbm import lgbm_forecast, load_long
from src.pregate import long_to_wide, pregate_inputs

TZ = "Europe/Copenhagen"
IMB_LAG = pd.Timedelta(hours=2)
OUT = Path("data/processed/DK_1_infosets.parquet")
LEAR_EXT = FC_DIR / "DK_1_lear_pregate_ext.parquet"
LGBM_EXT = FC_DIR / "DK_1_lgbm_q_pregate_ext.parquet"

CAL = ["hour", "dow", "month", "is_holiday", "doy_sin", "doy_cos"]
LAGS = ["spread_d2", "spread_med7"]
WX0 = ["wind_mw_v0", "solar_mw_v0", "temp_mean_v0"]
DAFC = ["da_fc", "da_fc_width"]
DAACT = ["da", "da_err"]
D1 = ["d1_wind_mw", "d1_solar_mw", "d1_temp_mean"]
D2 = ["d2_wind_mw", "d2_solar_mw", "d2_temp_mean"]


INFOSETS = {
    "B0": ("T0", CAL + LAGS + WX0 + DAFC, "all", True),
    "A0": ("T1", CAL + LAGS + WX0 + DAFC + DAACT, "all", True),
    "A1": ("T1", CAL + LAGS + WX0 + DAFC + DAACT + D1, "all", True),
    "A1e": ("T2", CAL + LAGS + WX0 + DAFC + DAACT + D1, "T2", True),
    "A2": ("T2", CAL + LAGS + WX0 + DAFC + DAACT + D1 + D2, "T2", True),
    "C1": ("T0", CAL + LAGS + WX0 + DAFC + D1, "all", False),
    "C2": ("T0", CAL + LAGS + WX0 + DAFC + D1 + D2, "T2", False),
}


def da_forecasts(
    zone="DK_1", start="2024-06-01", end="2026-08-31", window=1092, recal=7, n_jobs=-1
) -> None:
    """Pre-gate LEAR + LightGBM forecasts from June 2024 (new file names; the saved
    results of the main project are not touched)."""
    path = f"data/processed/{zone}_dataset.parquet"
    y, wx, res_pregate, _ = pregate_inputs(zone, verbose=False)
    dates = pd.date_range(start, end, freq="D")

    t0 = time.time()
    wide = load_wide(path)
    wide["res_fc"] = long_to_wide(res_pregate)
    X, Y = design_matrix(wide)
    lear_w = lear_forecast(X, Y, dates, recal_every=recal, n_jobs=n_jobs)
    lear_w.columns = lear_w.columns.astype(str)
    lear_w.to_parquet(LEAR_EXT)
    print(f"LEAR done in {(time.time() - t0) / 60:.1f} min -> {LEAR_EXT}")

    t0 = time.time()
    lg, _ = lgbm_forecast(
        load_long(path, res_override=res_pregate, extra=wx),
        dates,
        window,
        recal,
        n_jobs,
    )
    lg.to_parquet(LGBM_EXT)
    print(f"LightGBM done in {(time.time() - t0) / 60:.1f} min -> {LGBM_EXT}")


def _lagged(s: pd.Series, days: int) -> pd.Series:
    w = s.unstack("hour")
    w.index = pd.DatetimeIndex(w.index, name="date")
    return w.asfreq("D").shift(days).stack().reindex(s.index)


def _median7(s: pd.Series) -> pd.Series:
    """Median of the same hour over D-8 .. D-2."""
    w = s.unstack("hour")
    w.index = pd.DatetimeIndex(w.index, name="date")
    m = w.asfreq("D").shift(2).rolling(7, min_periods=5).median()
    return m.stack().reindex(s.index)


def build_table(zone="DK_1") -> pd.DataFrame:
    sp = pd.read_parquet(f"data/processed/{zone}_spread_hourly.parquet")
    rv = pd.read_parquet(f"data/processed/{zone}_revisions.parquet")
    df = sp.join(rv, how="inner")
    dates = pd.Series(df.index.get_level_values("date"), index=df.index)

    df["sign"] = (df["spread"] > 0).astype(float).where(df["spread"].notna())

    dk = holidays.country_holidays(
        "DK", years=range(dates.dt.year.min(), dates.dt.year.max() + 1)
    )
    df["hour"] = df.index.get_level_values("hour")
    df["dow"] = dates.dt.dayofweek
    df["month"] = dates.dt.month
    df["is_holiday"] = dates.dt.date.isin(list(dk.keys())).astype(int)
    doy = dates.dt.dayofyear
    df["doy_sin"], df["doy_cos"] = (
        np.sin(2 * np.pi * doy / 365.25),
        np.cos(2 * np.pi * doy / 365.25),
    )

    df["spread_d2"] = _lagged(df["spread"], 2)
    df["spread_med7"] = _median7(df["spread"])

    lear = wide_to_long(pd.read_parquet(LEAR_EXT), "lear")
    lg = pd.read_parquet(LGBM_EXT)
    df["da_fc"] = 0.5 * (lear.reindex(df.index) + lg["q0.50"].reindex(df.index))
    df["da_fc_width"] = (lg["q0.90"] - lg["q0.10"]).reindex(df.index)
    df["da_err"] = df["da"] - df["da_fc"]

    d_prev = dates - pd.Timedelta(days=1)
    df["T0"] = (d_prev + pd.Timedelta(hours=12)).dt.tz_localize(TZ).dt.tz_convert("UTC")
    df["T1"] = d_prev.dt.tz_localize("UTC") + pd.Timedelta(hours=20)
    df["T2"] = dates.dt.tz_localize("UTC") + pd.Timedelta(hours=8)
    df["start_utc"] = pd.to_datetime(df["start_utc"], utc=True)
    df["eligible_T2"] = df["start_utc"] >= df["T2"] + pd.Timedelta(hours=1)
    return df


def availability(df: pd.DataFrame) -> pd.DataFrame:
    """When each feature becomes known (UTC), row by row."""
    dates = pd.Series(df.index.get_level_values("date"), index=df.index)
    never = pd.Series(pd.Timestamp("1900-01-01", tz="UTC"), index=df.index)
    d1_midnight = (dates - pd.Timedelta(days=1)).dt.tz_localize(TZ).dt.tz_convert("UTC")
    da_pub = (
        (dates - pd.Timedelta(days=1) + pd.Timedelta(hours=13))
        .dt.tz_localize(TZ)
        .dt.tz_convert("UTC")
    )
    av = {c: never for c in CAL}
    av.update({c: d1_midnight + IMB_LAG for c in LAGS})
    av.update({c: df["pub_v0"] for c in WX0 + DAFC})
    av.update({"da": da_pub, "da_err": da_pub})
    av.update({c: df["pub_v1"] for c in D1})
    av.update({c: df[["pub_v1", "pub_v2"]].max(axis=1) for c in D2})
    return pd.DataFrame(av)


def audit(df: pd.DataFrame) -> pd.DataFrame:
    av = availability(df)
    rows = {}
    for name, (tcol, feats, hours, realisable) in INFOSETS.items():
        m = df["eligible_T2"] if hours == "T2" else pd.Series(True, index=df.index)
        late = {}
        for f in feats:
            known = df[f].notna() & m
            late[f] = int((known & (av[f] > df[tcol])).sum())
        n_late = sum(late.values())
        rows[name] = {
            "decision": tcol,
            "hours": hours,
            "features": len(feats),
            "rows": int(m.sum()),
            "late values": n_late,
            "status": ("OK" if n_late == 0 else "LEAK")
            if realisable
            else "unrealisable by design",
        }
        if realisable and n_late:
            bad = {k: v for k, v in late.items() if v}
            raise SystemExit(f"LEAKAGE in {name}: features known after {tcol}: {bad}")
    return pd.DataFrame(rows).T


def build(zone="DK_1") -> None:
    df = build_table(zone)
    print("== Leakage audit (late values must be 0 for realisable sets) ==")
    print(audit(df).to_string())

    OUT.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(OUT)
    d = df.index.get_level_values("date")
    print(f"\nsaved {OUT}: {df.shape}, {d.min().date()} -> {d.max().date()}")

    print("\n== Complete cases (target + all features) per information set ==")
    rows = {}
    for name, (_, feats, hours, _) in INFOSETS.items():
        m = df["eligible_T2"] if hours == "T2" else pd.Series(True, index=df.index)
        ok = m & df[["spread"] + feats].notna().all(axis=1)
        rows[name] = ok.groupby([df["regime"], df["split"]]).sum()
    cc = pd.DataFrame(rows)
    print(cc[cc.index.get_level_values("split") != "none"].to_string())

    print(
        "\n== Physical prior check (TRAIN + VALIDATION only): corr with the spread =="
    )
    tv = df[df["split"].isin(["train", "val"])]
    rows = {}
    for (reg, spl), g in tv.groupby(["regime", "split"]):
        rows[f"{reg}/{spl}"] = {
            c: g[[c, "spread"]].corr().iloc[0, 1]
            for c in [
                "d1_wind_mw",
                "dtot_wind_mw",
                "d1_solar_mw",
                "wind_mw_v0",
                "da_err",
            ]
        }
    print(pd.DataFrame(rows).T.round(3).to_string())
    print(
        "expected: negative for wind revisions (more wind -> longer system -> lower imbalance price)"
    )


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--zone", default="DK_1")
    ap.add_argument("--da-forecasts", action="store_true")
    ap.add_argument("--build", action="store_true")
    ap.add_argument("--start", default="2024-06-01")
    ap.add_argument("--end", default="2026-08-31")
    ap.add_argument("--window", type=int, default=1092)
    ap.add_argument("--recal", type=int, default=7)
    ap.add_argument("--n-jobs", type=int, default=-1)
    a = ap.parse_args()
    if a.da_forecasts:
        da_forecasts(a.zone, a.start, a.end, a.window, a.recal, a.n_jobs)
    if a.build:
        build(a.zone)
    if not (a.da_forecasts or a.build):
        ap.print_help()
