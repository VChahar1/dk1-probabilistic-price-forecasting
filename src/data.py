"Download and cache raw ENTSO-E data for day-ahead price forecasting."

import argparse
import os
from pathlib import Path

import pandas as pd
from entsoe import EntsoePandasClient
from entsoe.mappings import lookup_area

RAW_DIR = Path("data/raw")


def _save(obj, path: Path) -> None:
    df = obj.to_frame() if isinstance(obj, pd.Series) else obj
    df.index = df.index.tz_convert("UTC")
    df.index.name = "timestamp_utc"
    df.to_parquet(path)
    print(f"  saved {path} ({len(df):,} rows, {df.index.min()} -> {df.index.max()})")


def fetch_zone(zone: str, start: str, end: str, force: bool = False) -> None:
    """Cache the requested ENTSO-E price, load and renewables series."""
    key = os.environ.get("ENTSOE_API_KEY")
    if not key:
        raise SystemExit(
            "Set ENTSOE_API_KEY (free token: register on transparency.entsoe.eu, "
            "then email transparency@entsoe.eu asking for RESTful API access)."
        )
    client = EntsoePandasClient(api_key=key)
    tz = lookup_area(zone).tz
    t0, t1 = pd.Timestamp(start, tz=tz), pd.Timestamp(end, tz=tz)
    RAW_DIR.mkdir(parents=True, exist_ok=True)

    jobs = {
        "price": lambda: client.query_day_ahead_prices(zone, start=t0, end=t1).rename(
            "price"
        ),
        "load_fc": lambda: client.query_load_forecast(zone, start=t0, end=t1),
        "res_fc": lambda: client.query_wind_and_solar_forecast(zone, start=t0, end=t1),
    }
    for name, job in jobs.items():
        path = RAW_DIR / f"{zone}_{name}.parquet"
        if path.exists() and not force:
            print(f"  cached {path}")
            continue
        print(f"Fetching {zone} {name} ...")
        _save(job(), path)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--zone", default="DK_1")
    ap.add_argument("--start", default="2019-01-01")
    ap.add_argument("--end", default="2026-09-01")
    ap.add_argument("--force", action="store_true")
    a = ap.parse_args()
    fetch_zone(a.zone, a.start, a.end, a.force)
