"Next step 1: leakage-free weather inputs from archived ECMWF IFS HRES runs (Open-Meteo)."

import argparse
import time
from pathlib import Path

import numpy as np
import pandas as pd
import requests

from src.features import to_hourly_panel

URL = "https://single-runs-api.open-meteo.com/v1/forecast"
CACHE = Path("data/raw/ecmwf_dk1")
CACHES = {0: CACHE, 12: Path("data/raw/ecmwf_dk1_12z")}
OUT = Path("data/processed/DK_1_weather.parquet")
TZ = "Europe/Copenhagen"
VARS = ["wind_speed_100m", "shortwave_radiation", "temperature_2m"]


SITES = {
    "horns_rev": (55.52, 7.90, "offshore"),
    "anholt": (56.60, 11.21, "offshore"),
    "ringkobing": (56.09, 8.24, "onshore"),
    "thisted": (56.96, 8.69, "onshore"),
    "tonder": (54.94, 8.86, "onshore"),
    "aarhus": (56.16, 10.20, "onshore"),
    "odense": (55.40, 10.39, "onshore"),
}


class MissingRun(ValueError):
    pass


def fetch_run(run: pd.Timestamp, retries: int = 5) -> pd.DataFrame:
    params = {
        "latitude": ",".join(str(v[0]) for v in SITES.values()),
        "longitude": ",".join(str(v[1]) for v in SITES.values()),
        "hourly": ",".join(VARS),
        "models": "ecmwf_ifs",
        "run": run.strftime("%Y-%m-%dT%H:%M"),
        "forecast_days": 3,
        "wind_speed_unit": "ms",
        "timezone": "GMT",
    }
    for attempt in range(retries):
        try:
            r = requests.get(URL, params=params, timeout=120)
            if r.status_code == 429:
                time.sleep(30 * (attempt + 1))
                continue
            if r.status_code == 400:
                raise MissingRun(f"run {run.date()} not available (400)")
            r.raise_for_status()
            return parse(r.json(), run)
        except (requests.Timeout, requests.ConnectionError, ValueError) as e:
            if isinstance(e, MissingRun):
                raise
            time.sleep(10 * (attempt + 1))
    raise RuntimeError(f"run {run.date()}: failed after {retries} attempts")


def parse(js, run: pd.Timestamp) -> pd.DataFrame:
    """API JSON (one object per site) -> long table: run, time_utc, site, variables."""
    js = js if isinstance(js, list) else [js]
    frames = []
    for name, loc in zip(SITES, js):
        h = loc["hourly"]
        df = pd.DataFrame(
            {
                v: pd.to_numeric(pd.Series(h.get(v), dtype="object"), errors="coerce")
                for v in VARS
            }
        ).astype(float)
        df["time_utc"] = pd.to_datetime(h["time"]).tz_localize("UTC")
        df["site"] = name
        frames.append(df)
    out = pd.concat(frames, ignore_index=True)
    out["run"] = run.tz_localize("UTC") if run.tzinfo is None else run
    return out


def download(start: str, end: str, pause: float = 0.5, run_hour: int = 0) -> None:
    cache = CACHES[run_hour]
    cache.mkdir(parents=True, exist_ok=True)
    runs = pd.date_range(start, end, freq="D") + pd.Timedelta(hours=run_hour)
    todo = [r for r in runs if not (cache / f"{r.date()}.parquet").exists()]
    print(
        f"{len(runs)} runs requested, {len(runs) - len(todo)} cached, {len(todo)} to fetch"
    )
    for i, run in enumerate(todo, 1):
        try:
            fetch_run(run).to_parquet(cache / f"{run.date()}.parquet")
        except Exception as e:
            print(f"  {run.date()}: FAILED ({e})")
        if i % 50 == 0:
            print(f"  {i}/{len(todo)} done")
        time.sleep(pause)


def power_curve(v: np.ndarray, cut_in=3.0, rated=12.0, cut_out=25.0) -> np.ndarray:
    """Generic normalised turbine power curve (0..1): cubic between cut-in and rated."""
    p = (v**3 - cut_in**3) / (rated**3 - cut_in**3)
    p = np.clip(p, 0.0, 1.0)
    return np.where((v < cut_in) | (v >= cut_out), 0.0, p)


def load_runs(run_hours=(0,)) -> pd.DataFrame:
    frames = []
    for h in run_hours:
        frames += [pd.read_parquet(f) for f in sorted(CACHES[h].glob("*.parquet"))]
    if not frames:
        raise SystemExit("No cached runs; run the download first.")
    raw = pd.concat(frames, ignore_index=True)
    raw[VARS] = raw[VARS].apply(pd.to_numeric, errors="coerce").astype(float)
    raw["run"] = pd.to_datetime(raw["run"], utc=True)
    return raw


def indices_by_run(raw: pd.DataFrame) -> pd.DataFrame:
    """Site values -> DK1 weather indices for every (run, valid time)."""
    kind = raw["site"].map({k: v[2] for k, v in SITES.items()})
    raw = raw.assign(
        power=power_curve(raw["wind_speed_100m"].to_numpy(dtype=float)), kind=kind
    )
    g = lambda mask, col: raw[mask].groupby(["run", "time_utc"])[col].mean()
    return pd.DataFrame(
        {
            "wind_on_idx": g(raw["kind"] == "onshore", "power"),
            "wind_off_idx": g(raw["kind"] == "offshore", "power"),
            "ws100_mean": g(raw["kind"].notna(), "wind_speed_100m"),
            "solar_idx": g(raw["kind"] == "onshore", "shortwave_radiation"),
            "temp_mean": g(raw["kind"] == "onshore", "temperature_2m"),
        }
    )


def build() -> pd.DataFrame:
    """Keep, for each local delivery day D, only values from the 00 UTC run of D-1."""
    raw = load_runs((0,))
    local_day = pd.to_datetime(raw["time_utc"].dt.tz_convert(TZ).dt.date)
    run_day = pd.to_datetime(raw["run"].dt.tz_convert("UTC").dt.date)
    raw["lead_days"] = (local_day - run_day).dt.days

    raw = raw[raw["lead_days"].isin([1, 2])]
    raw = raw.sort_values("lead_days").drop_duplicates(
        subset=["time_utc", "site"], keep="first"
    )
    days = raw.assign(d=local_day.loc[raw.index]).groupby("d")["lead_days"].min()
    print(
        f"delivery days: {len(days)} | from D-1 run: {(days == 1).sum()} | "
        f"fallback D-2 run: {(days == 2).sum()}"
    )

    idx = indices_by_run(raw).droplevel("run")
    panel = to_hourly_panel(idx, TZ)
    panel.index = panel.index.set_levels(pd.to_datetime(panel.index.levels[0]), level=0)
    OUT.parent.mkdir(parents=True, exist_ok=True)
    panel.to_parquet(OUT)
    d = panel.index.get_level_values("date")
    print(f"saved {OUT}: {panel.shape}, {d.min().date()} -> {d.max().date()}")
    print(panel.isna().mean().round(3).to_string())
    return panel


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--test", action="store_true")
    ap.add_argument("--build", action="store_true")
    ap.add_argument("--start", default="2024-03-14")
    ap.add_argument("--end", default="2026-08-30")
    ap.add_argument("--run-hour", type=int, default=0, choices=[0, 12])
    a = ap.parse_args()
    if a.test:
        df = fetch_run(pd.Timestamp("2025-01-15") + pd.Timedelta(hours=a.run_hour))
        print(df.head(10).to_string())
        print(df.groupby("site")[VARS].mean().round(2))
        print(
            f"rows: {len(df)} (expected {len(SITES)} sites x 72 h = {len(SITES) * 72})"
        )
    elif a.build:
        build()
    else:
        download(a.start, a.end, run_hour=a.run_hour)
