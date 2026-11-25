"Intraday extension, Step 2: post-auction weather forecasts and their revisions."

import argparse
import os
from pathlib import Path

import numpy as np
import pandas as pd

from src.pregate import RES_COLS, WX, month_coefs, proxy_coefs
from src.weather import TZ, indices_by_run, load_runs

PUB_LAG = pd.Timedelta(hours=8)
VINTAGES = {"v0": (0, -1), "v1": (12, -1), "v2": (0, 0)}
OUT = Path("data/processed/DK_1_revisions.parquet")
ACTUALS = Path("data/raw/DK_1_actual_res.parquet")
IDX_COLS = ["wind_on_idx", "wind_off_idx", "solar_idx", "temp_mean"]


def vintage_table(idx: pd.DataFrame) -> pd.DataFrame:
    """(run, time_utc) indices -> one row per valid UTC hour, columns per vintage."""
    idx = idx.reset_index()
    local_day = pd.to_datetime(idx["time_utc"].dt.tz_convert(TZ).dt.date)
    run_day = pd.to_datetime(idx["run"].dt.tz_convert("UTC").dt.date)
    idx["offset"] = (run_day - local_day).dt.days
    idx["run_hour"] = idx["run"].dt.hour
    parts = []
    for v, (hour, off) in VINTAGES.items():
        sel = idx[(idx["run_hour"] == hour) & (idx["offset"] == off)]
        if v == "v0":
            fb = idx[(idx["run_hour"] == 0) & (idx["offset"] == -2)]
            fb = fb[~fb["time_utc"].isin(sel["time_utc"])]
            sel = pd.concat([sel.assign(fallback=False), fb.assign(fallback=True)])
        sel = sel.set_index("time_utc").sort_index()
        sel = sel[~sel.index.duplicated()]
        cols = {c: f"{c}_{v}" for c in IDX_COLS}
        part = sel[IDX_COLS].rename(columns=cols)
        part[f"pub_{v}"] = sel["run"] + PUB_LAG
        if v == "v0":
            part["v0_fallback"] = sel["fallback"].astype(float)
        parts.append(part)
    return pd.concat(parts, axis=1, sort=True)


def to_local_hours(t: pd.DataFrame) -> pd.DataFrame:
    """UTC hours -> (date, hour) in Danish local time; no interpolation."""
    local = t.index.tz_convert(TZ)
    num = t.select_dtypes("number")
    p = num.groupby([pd.to_datetime(local.date), local.hour]).agg(
        lambda x: x.mean(skipna=False)
    )
    pubs = t[[c for c in t.columns if c.startswith("pub_")]]
    pubs = pubs.groupby([pd.to_datetime(local.date), local.hour]).max()
    p = p.join(pubs)
    p["start_utc"] = (
        t.index.to_series().groupby([pd.to_datetime(local.date), local.hour]).min()
    )
    p.index = p.index.set_names(["date", "hour"])
    return p


def build(zone: str = "DK_1") -> pd.DataFrame:
    raw = load_runs((0, 12))
    print(
        f"runs loaded: 00 UTC {raw.loc[raw['run'].dt.hour == 0, 'run'].nunique()}, "
        f"12 UTC {raw.loc[raw['run'].dt.hour == 12, 'run'].nunique()}"
    )
    panel = to_local_hours(vintage_table(indices_by_run(raw)))

    ds = pd.read_parquet(f"data/processed/{zone}_dataset.parquet")
    ds.index = ds.index.set_levels(pd.to_datetime(ds.index.levels[0]), level=0)
    entsoe_res = ds[RES_COLS].sum(axis=1, min_count=len(RES_COLS))
    wx = pd.read_parquet(f"data/processed/{zone}_weather.parquet")
    coefs = proxy_coefs(entsoe_res, wx)
    c = month_coefs(panel.index, coefs)

    for v in VINTAGES:
        panel[f"wind_mw_{v}"] = (
            c["wind_on_idx"] * panel[f"wind_on_idx_{v}"]
            + c["wind_off_idx"] * panel[f"wind_off_idx_{v}"]
        )
        panel[f"solar_mw_{v}"] = c["solar_idx"] * panel[f"solar_idx_{v}"]
    for x in ("wind_mw", "solar_mw", "temp_mean"):
        panel[f"d1_{x}"] = panel[f"{x}_v1"] - panel[f"{x}_v0"]
        panel[f"d2_{x}"] = panel[f"{x}_v2"] - panel[f"{x}_v1"]
        panel[f"dtot_{x}"] = panel[f"{x}_v2"] - panel[f"{x}_v0"]

    proxy_v0 = (c["intercept"] + panel["wind_mw_v0"] + panel["solar_mw_v0"]).clip(
        lower=0
    )
    ref = (c["intercept"] + sum(c[k] * wx[k] for k in WX).reindex(panel.index)).clip(
        lower=0
    )
    j = pd.concat([proxy_v0, ref], axis=1).dropna()
    print(
        f"check: v0 wind+solar reproduces the pre-gate proxy on {len(j):,} hours, "
        f"max |diff| {(j.iloc[:, 0] - j.iloc[:, 1]).abs().max():.2e} MW"
    )

    panel = panel[panel.index.get_level_values("date") >= coefs.index.min()]
    OUT.parent.mkdir(parents=True, exist_ok=True)
    panel.to_parquet(OUT)
    d = panel.index.get_level_values("date")
    print(f"saved {OUT}: {panel.shape}, {d.min().date()} -> {d.max().date()}")

    print("\n== Availability (share of delivery hours with each vintage) ==")
    for v in VINTAGES:
        print(
            f"  {v}: {100 * panel[f'wind_mw_{v}'].notna().mean():.1f}%"
            + (
                f"  (fallback D-2 run: {100 * panel['v0_fallback'].eq(1).mean():.2f}%)"
                if v == "v0"
                else ""
            )
        )
    print(
        "  note: v2 starts at 00 UTC on D, so the first 1-2 local hours of D are never covered"
    )

    print(
        "\n== Size of revisions (MW or degC): mean absolute value, std, correlation of d1 and d2 =="
    )
    rows = {}
    for x in ("wind_mw", "solar_mw", "temp_mean"):
        rows[x] = {
            "|d1| mean": panel[f"d1_{x}"].abs().mean(),
            "|d2| mean": panel[f"d2_{x}"].abs().mean(),
            "|dtot| mean": panel[f"dtot_{x}"].abs().mean(),
            "d1 std": panel[f"d1_{x}"].std(),
            "d2 std": panel[f"d2_{x}"].std(),
            "corr(d1,d2)": panel[[f"d1_{x}", f"d2_{x}"]].corr().iloc[0, 1],
        }
    print(pd.DataFrame(rows).T.round(3).to_string())
    return panel


def download_actuals(zone: str = "DK_1", start="2024-06-01", end="2026-09-01") -> None:
    from entsoe import EntsoePandasClient

    client = EntsoePandasClient(api_key=os.environ["ENTSOE_API_KEY"])
    df = client.query_generation(
        zone, start=pd.Timestamp(start, tz=TZ), end=pd.Timestamp(end, tz=TZ)
    )
    if isinstance(df.columns, pd.MultiIndex):
        df = df.xs("Actual Aggregated", axis=1, level=1)
    keep = {"Wind Onshore": "wind_on", "Wind Offshore": "wind_off", "Solar": "solar"}
    df = df[[c for c in keep if c in df.columns]].rename(columns=keep)
    df.index = df.index.tz_convert("UTC")
    df.index.name = "timestamp_utc"
    ACTUALS.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(ACTUALS)
    print(
        f"saved {ACTUALS}: {df.shape}, {df.index.min()} -> {df.index.max()}, "
        f"resolution {pd.Series(df.index).diff().mode().iloc[0]}"
    )
    print(df.describe().loc[["mean", "max"]].round(0).to_string())


def skill(zone: str = "DK_1") -> None:
    """If later runs carry new information, their errors vs actual output must shrink,
    and the revision d1 must point in the direction of the v0 error. The grid operator's
    day-ahead forecast (ENTSO-E) is shown as a reference for the quality of the proxy."""
    p = pd.read_parquet(OUT)
    ds = pd.read_parquet(f"data/processed/{zone}_dataset.parquet")
    ds.index = ds.index.set_levels(pd.to_datetime(ds.index.levels[0]), level=0)
    p["wind_tso"] = (ds["wind_on_fc"] + ds["wind_off_fc"]).reindex(p.index)
    p["solar_tso"] = ds["solar_fc"].reindex(p.index)
    a = pd.read_parquet(ACTUALS)
    a = a.groupby(a.index.floor("h")).mean()
    a["wind"] = a[[c for c in ("wind_on", "wind_off") if c in a]].sum(
        axis=1, min_count=1
    )
    start = pd.DatetimeIndex(p["start_utc"])
    p["wind_act"] = a["wind"].reindex(start).to_numpy()
    p["solar_act"] = a["solar"].reindex(start).to_numpy() if "solar" in a else np.nan

    for x, act in (("wind_mw", "wind_act"), ("solar_mw", "solar_act")):
        cols = [f"{x}_{v}" for v in VINTAGES] + [act]
        j = p[cols].dropna()
        if x == "solar_mw":
            j = j[(j[f"{x}_v0"] > 1) | (j[act] > 1)]

        e = {v: j[f"{x}_{v}"] - j[act] for v in VINTAGES}
        mae = {v: (e[v] - e[v].mean()).abs().mean() for v in VINTAGES}
        print(
            f"\n== {x.split('_')[0]}: {len(j):,} hours covered by all three vintages =="
        )
        print(
            "  bias-corrected MAE vs actual (MW): "
            + ", ".join(f"{v} {mae[v]:.0f}" for v in VINTAGES)
            + f"  |  v1 vs v0: {100 * (mae['v1'] / mae['v0'] - 1):+.1f}%, "
            f"v2 vs v0: {100 * (mae['v2'] / mae['v0'] - 1):+.1f}%"
        )
        tso = p[x.split("_")[0] + "_tso"].reindex(j.index)
        jt = pd.concat([tso, j[act]], axis=1).dropna()
        if len(jt):
            et = jt.iloc[:, 0] - jt.iloc[:, 1]
            print(
                f"  reference, grid operator's day-ahead forecast (ENTSO-E): bias-corrected MAE "
                f"{(et - et.mean()).abs().mean():.0f} MW on {len(jt):,} of these hours"
            )
        err0 = j[act] - j[f"{x}_v0"]
        d1, d2 = j[f"{x}_v1"] - j[f"{x}_v0"], j[f"{x}_v2"] - j[f"{x}_v1"]
        err1 = j[act] - j[f"{x}_v1"]
        print(
            f"  corr(d1, actual - v0) = {np.corrcoef(d1, err0)[0, 1]:+.3f}   "
            f"corr(d2, actual - v1) = {np.corrcoef(d2, err1)[0, 1]:+.3f}   "
            "[> 0: revisions move towards the outcome]"
        )


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--zone", default="DK_1")
    ap.add_argument("--build", action="store_true")
    ap.add_argument("--actuals", action="store_true")
    ap.add_argument("--skill", action="store_true")
    a = ap.parse_args()
    if a.build:
        build(a.zone)
    if a.actuals:
        download_actuals(a.zone)
    if a.skill:
        skill(a.zone)
    if not (a.build or a.actuals or a.skill):
        ap.print_help()
