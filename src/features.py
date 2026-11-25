"""
Build a leakage-safe modelling dataset: one row per (delivery date, local hour).

Information set = what a trader knows at SDAC gate closure (12:00 CET on D-1):
  * prices for D-1 and earlier (D-1 prices were published ~12:45 on D-2)  -> OK
  * day-ahead load / wind / solar forecasts for D                           -> OK*
  * fuel/carbon settlements up to D-2 (D-1 settles after gate closure)      -> OK
  * NOT: actual load/generation for D, or D's own price.

*ENTSO-E rules allow wind/solar day-ahead forecasts to be published up to 18:00 on D-1,
 i.e. possibly AFTER gate closure. Document this assumption in the README and, later,
 test robustness with forecasts that are guaranteed to be pre-12:00 (e.g. Open-Meteo
 historical weather forecasts).

Usage:
    python -m src.features --zone DK_1 [--fuels data/raw/fuels.csv]
fuels.csv (optional): columns date, ttf, eua  (daily closes; weekends may be missing)
"""
import argparse
from pathlib import Path

import holidays
import numpy as np
import pandas as pd
from entsoe.mappings import lookup_area

RAW_DIR, OUT_DIR = Path("data/raw"), Path("data/processed")


def to_hourly_panel(x: pd.Series | pd.DataFrame, tz: str) -> pd.DataFrame:
    """UTC series at any resolution (60/15 min) -> (date, hour) panel in local time, 24 h/day.

    - 15-min data (SDAC after Oct 2025) is averaged to hourly.
    - 25-hour DST day: the duplicated local hour is averaged.
    - 23-hour DST day: the missing local hour is linearly interpolated.
    """
    df = x.to_frame() if isinstance(x, pd.Series) else x.copy()
    df = df.tz_convert("UTC").resample("h").mean()
    local = df.index.tz_convert(tz)
    df = df.groupby([local.date, local.hour]).mean()
    df.index = df.index.set_names(["date", "hour"])
    # keep only complete local days (>= 23 hours; 23 is the spring DST day).
    # ENTSO-E's end timestamp is inclusive, so the last date is often a 1-point stub.
    n_obs = df.notna().any(axis=1).groupby(level="date").sum()
    dates = pd.Index(sorted(n_obs.index[n_obs >= 23]), name="date")
    full = pd.MultiIndex.from_product([dates, range(24)], names=["date", "hour"])
    return df.reindex(full).interpolate(limit=2, limit_direction="both")


def _lag_days(panel: pd.Series, k: int) -> pd.Series:
    """Same hour, k days earlier (calendar-aware, not row-count based)."""
    s = panel.copy()
    idx = s.index
    s.index = pd.MultiIndex.from_arrays(
        [pd.to_datetime(idx.get_level_values("date")) + pd.Timedelta(days=k),
         idx.get_level_values("hour")], names=["date", "hour"])
    s.index = s.index.set_levels(s.index.levels[0].date, level=0)
    return s.reindex(panel.index)


def build_dataset(zone: str, fuels_csv: str | None = None) -> pd.DataFrame:
    tz = lookup_area(zone).tz
    price = pd.read_parquet(RAW_DIR / f"{zone}_price.parquet")["price"]
    load = pd.read_parquet(RAW_DIR / f"{zone}_load_fc.parquet")
    res = pd.read_parquet(RAW_DIR / f"{zone}_res_fc.parquet")

    df = to_hourly_panel(price, tz)
    df["load_fc"] = to_hourly_panel(load.iloc[:, 0], tz).iloc[:, 0]
    res_p = to_hourly_panel(res, tz)
    for col, new in [("Solar", "solar_fc"), ("Wind Onshore", "wind_on_fc"),
                     ("Wind Offshore", "wind_off_fc")]:
        df[new] = res_p[col] if col in res_p else 0.0
    df["res_fc"] = df[["solar_fc", "wind_on_fc", "wind_off_fc"]].sum(axis=1, min_count=1)
    df["residual_load_fc"] = df["load_fc"] - df["res_fc"]
    df["res_share_fc"] = df["res_fc"] / df["load_fc"]

    # --- price lags (all known before gate closure on D-1) ---
    for k in (1, 2, 7):
        df[f"price_d{k}"] = _lag_days(df["price"], k)
    daily = df["price"].groupby(level="date").agg(["mean", "min", "max", "std"])
    daily.index = pd.to_datetime(daily.index) + pd.Timedelta(days=1)  # D-1 stats for day D
    daily.index = daily.index.date
    daily = daily.add_prefix("price_d1_")
    df = df.join(daily, on="date")
    df["residual_load_fc_d7"] = _lag_days(df["residual_load_fc"], 7)

    # --- fuels: value for delivery D = last close on or before D-2 ---
    if fuels_csv:
        f = pd.read_csv(fuels_csv, parse_dates=["date"]).set_index("date").sort_index()
        f = f.asfreq("D").ffill()
        f.index = (f.index + pd.Timedelta(days=2)).date
        df = df.join(f[["ttf", "eua"]].add_suffix("_d2"), on="date")

    # --- calendar ---
    d = pd.to_datetime(df.index.get_level_values("date"))
    hol = holidays.country_holidays(zone.split("_")[0], years=range(d.year.min(), d.year.max() + 1))
    df["dow"] = d.dayofweek
    df["month"] = d.month
    df["is_holiday"] = np.isin(d.date, list(hol.keys())).astype(int)
    df["is_weekend"] = (d.dayofweek >= 5).astype(int)
    doy = d.dayofyear.to_numpy()
    df["doy_sin"], df["doy_cos"] = np.sin(2 * np.pi * doy / 365.25), np.cos(2 * np.pi * doy / 365.25)

    return df.dropna(subset=["price", "price_d7", "load_fc"])


def to_wide(df: pd.DataFrame, col: str = "price") -> pd.DataFrame:
    """(date, hour) long -> date x 24 matrix (the format LEAR / epftoolbox expect)."""
    return df[col].unstack("hour")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--zone", default="DK_1")
    ap.add_argument("--fuels", default=None)
    a = ap.parse_args()
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    out = build_dataset(a.zone, a.fuels)
    path = OUT_DIR / f"{a.zone}_dataset.parquet"
    out.to_parquet(path)
    print(f"saved {path}: {out.shape}, {out.index.get_level_values('date').min()} -> "
          f"{out.index.get_level_values('date').max()}")
    print(out.isna().mean().round(3).to_string())