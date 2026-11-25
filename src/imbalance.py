"""
Intraday extension, Step 1: verify Energinet's imbalance-price data before building the pipeline.

Energi Data Service (Energinet, CC BY 4.0, "Source: Energinet"), no API key needed.

Usage:
    python -m src.imbalance --probe       # columns, date range, resolution of each dataset
    python -m src.imbalance --align       # time-stamp convention + DST + cross-check vs ENTSO-E
    python -m src.imbalance --download    # full history, cached in chunks (rerun fills gaps)
    python -m src.imbalance --build       # aligned hourly + 15-min spread tables
    python -m src.imbalance --status      # cached chunks per dataset
    python -m src.imbalance --entsoe-check     # ENTSO-E imbalance prices vs Energinet
    python -m src.imbalance --entsoe-download  # pre-regime imbalance prices (May 2024 - Mar 2025)
    python -m src.imbalance --gaps             # missing pre-regime hours in ENTSO-E
    python -m src.imbalance --fill-gaps        # fetch only the Energinet weeks that cover them

Conventions (verified in Step 1): TimeUTC = start of period; API start/end filter in Danish
local time; spread = imbalance price - day-ahead price (ENTSO-E); single imbalance price, so a
long position earns +spread and a short position -spread.
"""
import argparse
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
import requests

URL = "https://api.energidataservice.dk/dataset/{}"
AREA = "DK1"
TZ = "Europe/Copenhagen"
DATASETS = ["ImbalancePrice", "RegulatingBalancePowerdata", "DayAheadPrices"]


# ------------------------------------------------------------------ API client
def eds_get(dataset: str, params: dict, retries: int = 3, timeout: int = 180) -> dict:
    for attempt in range(retries):
        try:
            r = requests.get(URL.format(dataset), params=params, timeout=timeout)
        except (requests.Timeout, requests.ConnectionError):
            time.sleep(10 * (attempt + 1))
            continue
        if r.status_code == 429:
            time.sleep(20 * (attempt + 1))
            continue
        if 400 <= r.status_code < 500:                 # not found / bad request: permanent
            raise RuntimeError(f"{dataset}: HTTP {r.status_code}")
        r.raise_for_status()
        return r.json()
    raise RuntimeError(f"{dataset}: no response after {retries} attempts (timeout {timeout}s)")


def eds_records(dataset: str, start=None, end=None, sort=None, limit=10000,
                area: str = AREA, columns=None, timeout: int = 180) -> pd.DataFrame:
    params = {"filter": json.dumps({"PriceArea": [area]}), "limit": limit}
    if columns:
        params["columns"] = ",".join(columns)
    if start:
        params["start"] = start
    if end:
        params["end"] = end
    if sort:
        params["sort"] = sort
    js = eds_get(dataset, params, timeout=timeout)
    df = pd.DataFrame(js.get("records", []))
    df.attrs["total"] = js.get("total")
    return df


def utc_col(df: pd.DataFrame) -> str:
    cols = [c for c in df.columns if c.endswith("UTC")]
    if not cols:
        raise ValueError(f"no UTC time column in {list(df.columns)}")
    return cols[0]


def to_utc_index(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        raise ValueError("empty response (no records in this window)")
    c = utc_col(df)
    out = df.copy()
    out.index = pd.to_datetime(out[c]).dt.tz_localize("UTC")
    return out.sort_index()


def resolution(idx: pd.DatetimeIndex) -> str:
    d = pd.Series(idx.sort_values()).diff().dropna()
    return str(d.mode().iloc[0]) if len(d) else "n/a"


# ---------------------------------------------------------------------- probe
COVERAGE_DATES = ["2023-06-01", "2024-03-15", "2024-06-01", "2024-10-01", "2025-01-15",
                  "2025-02-24", "2025-03-10", "2025-03-24"]


def price_cols(df: pd.DataFrame) -> list[str]:
    return [c for c in df.columns if "Imbalance" in c and c.endswith("EUR")] or \
           [c for c in df.columns if c.endswith("EUR")][:1]


def coverage(dataset: str, dates=COVERAGE_DATES) -> None:
    """Rows, resolution and missing prices in 7-day windows (no server-side sorting)."""
    print(f"  coverage by 7-day window (expect 672 rows at 15 min, 168 at 1 h):")
    for d in dates:
        t0 = pd.Timestamp(d)
        try:
            wk = eds_records(dataset, start=t0.strftime("%Y-%m-%dT%H:%M"),
                             end=(t0 + pd.Timedelta(days=7)).strftime("%Y-%m-%dT%H:%M"))
        except Exception as e:
            print(f"    {d}: FAILED ({e})")
            continue
        if wk.empty:
            print(f"    {d}: no data")
            continue
        wk = to_utc_index(wk)
        pc = price_cols(wk)
        nan = {c: f"{100 * pd.to_numeric(wk[c], errors='coerce').isna().mean():.0f}%" for c in pc}
        print(f"    {d}: {len(wk):4d} rows, resolution {resolution(wk.index)}, "
              f"missing {nan}")


def probe(dataset: str) -> None:
    print(f"\n==================== {dataset} ====================")
    try:
        head = eds_records(dataset, limit=5)
    except Exception as e:
        print(f"  NOT AVAILABLE: {e}")
        return
    if head.empty:
        print("  no records returned for DK1")
        return
    tcol = utc_col(head)
    print(f"  columns : {list(head.columns)}")
    print(f"  time col: {tcol}")
    try:  # full-history sort can time out on large/discontinued datasets
        first = to_utc_index(eds_records(dataset, sort=f"{tcol} asc", limit=3))
        print(f"  first record: {first.index.min()}")
    except Exception as e:
        print(f"  first record: unknown (sort query failed: {e})")
    print(f"  latest records returned: {to_utc_index(head).index.max()}")
    coverage(dataset)
    print("  sample rows:")
    print(head.drop(columns=[c for c in head.columns if c.endswith(("DK", "DKK"))]).head(3).to_string())


# ------------------------------------------------------------------ alignment
def _q15(s: pd.Series) -> pd.Series:
    """Any-resolution UTC series -> 15-min grid (hourly values repeated over the hour).
    The last value is only extended by its own step (no stretching past the data)."""
    s = s[~s.index.duplicated()].sort_index()
    step = s.index[-1] - s.index[-2] if len(s) > 1 else pd.Timedelta(hours=1)
    grid = pd.date_range(s.index.min(), s.index.max() + step - pd.Timedelta(minutes=15),
                         freq="15min")
    return s.reindex(grid).ffill(limit=3)


def match_by_lag(a: pd.Series, b: pd.Series, tol=0.01, lags=range(-4, 5)) -> pd.Series:
    """Share of 15-min periods where a(t) == b(t + lag), for each lag (in quarter-hours)."""
    out = {}
    for k in lags:
        j = pd.concat([a, b.shift(-k)], axis=1).dropna()
        out[k] = float(np.mean(np.abs(j.iloc[:, 0] - j.iloc[:, 1]) < tol)) if len(j) else np.nan
    return pd.Series(out, name="match share")


def eds_day_ahead(start: str, end: str):
    """Energinet day-ahead prices for a window: DayAheadPrices (from Oct 2025, 15-min),
    falling back to the older Elspotprices (hourly, discontinued after Sep 2025)."""
    for dataset in ("DayAheadPrices", "Elspotprices"):
        try:
            df = eds_records(dataset, start=start, end=end)
        except Exception as e:
            print(f"  {dataset}: {e}")
            continue
        if df.empty:
            print(f"  {dataset}: no records in this window")
            continue
        col = [c for c in df.columns if "Price" in c and c.endswith("EUR")][0]
        df = to_utc_index(df)
        return pd.to_numeric(df[col], errors="coerce"), dataset
    return None, None


def align(zone: str = "DK_1") -> None:
    entsoe = pd.read_parquet(f"data/raw/{zone}_price.parquet")["price"]
    entsoe.index = entsoe.index.tz_convert("UTC")

    windows = {"hourly day-ahead era": ("2025-06-02", "2025-06-09"),
               "15-min day-ahead era": ("2025-11-03", "2025-11-10")}
    for label, (s, e) in windows.items():
        print(f"\n==================== {label}: {s} -> {e} ====================")
        imb = to_utc_index(eds_records("ImbalancePrice", start=s, end=e))
        print(f"  ImbalancePrice rows: {len(imb)} (resolution {resolution(imb.index)})")
        ent = _q15(entsoe[s:e])

        da, src = eds_day_ahead(s, e)
        best_a = None
        if da is not None:
            print(f"  {src} rows: {len(da)} (resolution {resolution(da.index)})")
            m = match_by_lag(_q15(da), ent)
            best_a = m.idxmax()
            print(f"  (a) Energinet day-ahead ({src}) == ENTSO-E day-ahead, by lag (quarter-hours):")
            print("      " + m.round(3).to_string().replace("\n", "\n      "))
        else:
            print("  (a) skipped: no Energinet day-ahead dataset covers this window")

        if "SpotPriceEUR" in imb:
            spot = pd.to_numeric(imb["SpotPriceEUR"], errors="coerce")
            m3 = match_by_lag(spot[~spot.index.duplicated()], ent)
            print(f"  (c) SpotPriceEUR (inside ImbalancePrice) == ENTSO-E day-ahead: "
                  f"lag 0 share {m3.loc[0]:.3f}, best lag {m3.idxmax()}")
        imb_eur = pd.to_numeric(imb["ImbalancePriceEUR"], errors="coerce")
        print(f"  missing imbalance prices in window: {int(imb_eur.isna().sum())}")
        imb_eur = imb_eur.dropna()
        imb_eur = imb_eur[~imb_eur.index.duplicated()]
        if "DominatingDirection" in imb:
            dd = pd.to_numeric(imb["DominatingDirection"], errors="coerce").value_counts(normalize=True)
            print(f"  DominatingDirection shares: {dd.round(3).to_dict()}")
        m2 = match_by_lag(imb_eur, ent)
        print(f"  (b) imbalance price == day-ahead price (no dominant direction), by lag:")
        print("      " + m2.round(3).to_string().replace("\n", "\n      "))
        best_b = m2.idxmax()
        print(f"  -> best lag (a) = {best_a}, (b) = {best_b}  "
              f"[0 means TimeUTC = start of period, same as ENTSO-E]")

        spread = (imb_eur - ent.reindex(imb_eur.index)).dropna()
        print(f"  spread (imbalance - day-ahead), EUR/MWh: mean {spread.mean():.1f}, "
              f"median {spread.median():.1f}, std {spread.std():.1f}, "
              f"1%/99% {spread.quantile(0.01):.0f}/{spread.quantile(0.99):.0f}")

    print("\n==================== DST days (rows per local day; expect 92 / 100) ====================")
    for s, e, day in [("2025-03-29", "2025-04-01", "2025-03-30"), ("2025-10-25", "2025-10-28", "2025-10-26")]:
        imb = to_utc_index(eds_records("ImbalancePrice", start=s, end=e))
        local = imb.index.tz_convert(TZ)
        n = int((local.date == pd.Timestamp(day).date()).sum())
        print(f"  {day}: {n} quarter-hours")


# -------------------------------------------------------------------- download
RAW = Path("data/raw/imbalance")
OUT_H = Path("data/processed/DK_1_spread_hourly.parquet")
OUT_Q = Path("data/processed/DK_1_spread_15min.parquet")

SOURCES = {
    # new design (+ mFRR EAM): 15-min, monthly chunks
    "ImbalancePrice": dict(
        start="2025-03-01", end="2026-09-01", freq="MS", limit=10000,
        columns=["TimeUTC", "TimeDK", "PriceArea", "ImbalancePriceEUR", "SpotPriceEUR",
                 "DominatingDirection", "mFRRMarginalPriceUpEUR", "mFRRMarginalPriceDownEUR",
                 "aFRRVWAUpEUR", "aFRRVWADownEUR"]),
    # old design: hourly, slow server -> weekly chunks
    "RegulatingBalancePowerdata": dict(
        start="2024-05-20", end="2025-03-05", freq="7D", limit=1000,
        columns=["HourUTC", "HourDK", "PriceArea", "ImbalancePriceEUR",
                 "BalancingPowerPriceUpEUR", "BalancingPowerPriceDownEUR", "ImbalanceMWh"],
        timeout=300),
}
# data are expected in these windows: an empty response there is an error, not "no data"
EXPECTED = {"ImbalancePrice": ("2025-03-05", "2026-09-01"),
            "RegulatingBalancePowerdata": ("2024-05-20", "2025-03-04")}

# regimes by local delivery date
PRE_END, POST_START = pd.Timestamp("2025-03-03"), pd.Timestamp("2025-03-19")
SPLITS = {  # pre-registered, chronological; test periods are touched once
    "post": {"train": ("2025-03-19", "2025-09-30"), "val": ("2025-10-01", "2026-01-31"),
             "test": ("2026-02-01", "2026-08-31")},
    "pre": {"train": ("2024-06-01", "2024-10-31"), "val": ("2024-11-01", "2024-12-31"),
            "test": ("2025-01-01", "2025-03-03")},
}


def chunk_edges(cfg: dict) -> list:
    starts = [t for t in pd.date_range(cfg["start"], cfg["end"], freq=cfg["freq"])
              if t < pd.Timestamp(cfg["end"])]
    return starts + [pd.Timestamp(cfg["end"])]


def download(include_old: bool = False, only_weeks=None) -> None:
    """Energinet chunks. The old-design endpoint is slow and rate-limited, so by default the
    pre-regime prices come from ENTSO-E instead (see --entsoe-download)."""
    RAW.mkdir(parents=True, exist_ok=True)
    for dataset, cfg in SOURCES.items():
        if dataset == "RegulatingBalancePowerdata" and not include_old:
            continue
        if only_weeks is not None and dataset != "RegulatingBalancePowerdata":
            continue
        edges = chunk_edges(cfg)
        ok = cached = failed = 0
        e0, e1 = map(pd.Timestamp, EXPECTED[dataset])
        for c0, c1 in zip(edges[:-1], edges[1:]):
            if only_weeks is not None and str(c0.date()) not in only_weeks:
                continue
            path = RAW / f"{dataset}_{c0.date()}.parquet"
            expected = c1 > e0 and c0 < e1
            if path.exists():
                if expected and pd.read_parquet(path).empty:
                    path.unlink()                      # empty cache where data must exist: refetch
                else:
                    cached += 1
                    continue
            # 1-day overlap on both sides: API windows are local time (DST-safe after dedupe)
            s0, s1 = c0 - pd.Timedelta(days=1), c1 + pd.Timedelta(days=1)
            try:
                df = eds_records(dataset, start=s0.strftime("%Y-%m-%dT%H:%M"),
                                 end=s1.strftime("%Y-%m-%dT%H:%M"),
                                 limit=cfg["limit"], columns=cfg["columns"],
                                 timeout=cfg.get("timeout", 180))
                if len(df) >= cfg["limit"]:
                    raise RuntimeError(f"hit the row limit ({cfg['limit']}); data truncated")
                if expected and df.empty:
                    raise RuntimeError("empty response where data are expected (not cached)")
                df.to_parquet(path)
                ok += 1
            except Exception as e:
                print(f"  {dataset} {c0.date()}: FAILED ({e})")
                failed += 1
            time.sleep(0.3)
        print(f"{dataset}: {ok} downloaded, {cached} cached, {failed} failed"
              + ("  -> rerun --download to fill the gaps" if failed else ""))


def status() -> None:
    """Per-chunk row counts, so gaps are visible before building."""
    if ENTSOE_IMB.exists():
        e = pd.read_parquet(ENTSOE_IMB)
        print(f"ENTSO-E imbalance (pre regime): {len(e):,} rows, {e.index.min()} -> {e.index.max()}")
    else:
        print("ENTSO-E imbalance (pre regime): not downloaded (run --entsoe-download)")
    for dataset, cfg in SOURCES.items():
        files = sorted(RAW.glob(f"{dataset}_*.parquet"))
        if dataset == "RegulatingBalancePowerdata" and ENTSOE_IMB.exists():
            print(f"{dataset}: {len(files)} cached weeks (used only as a cross-check)")
            continue
        rows = {f.stem.split("_")[-1]: len(pd.read_parquet(f)) for f in files}
        starts = [str(d.date()) for d in chunk_edges(cfg)[:-1]]
        missing = [d for d in starts if d not in rows]
        empty = [d for d, n in rows.items() if n == 0]
        print(f"{dataset}: {len(files)}/{len(starts)} chunks cached, {sum(rows.values()):,} rows")
        if missing:
            print(f"  not downloaded: {missing}")
        if empty:
            print(f"  empty chunks: {empty}")


# ------------------------------------------------ ENTSO-E as alternative source
def entsoe_imbalance(start: str, end: str, zone: str = "DK_1") -> pd.DataFrame:
    import os
    from entsoe import EntsoePandasClient
    client = EntsoePandasClient(api_key=os.environ["ENTSOE_API_KEY"])
    df = client.query_imbalance_prices(zone, start=pd.Timestamp(start, tz=TZ),
                                       end=pd.Timestamp(end, tz=TZ))
    df.index = df.index.tz_convert("UTC")
    return df[~df.index.duplicated()].sort_index()


ENTSOE_IMB = RAW / "DK_1_imbalance_entsoe.parquet"


def entsoe_download(zone: str = "DK_1", start: str = "2024-05-20", end: str = "2025-03-05") -> None:
    """Pre-regime DK1 imbalance prices from ENTSO-E (verified identical to Energinet),
    fetched month by month so a partial response is visible per month."""
    RAW.mkdir(parents=True, exist_ok=True)
    edges = [t for t in pd.date_range(start, end, freq="MS") if t > pd.Timestamp(start)]
    edges = [pd.Timestamp(start)] + edges + [pd.Timestamp(end)]
    parts = []
    print("month        rows  expected hours  missing hours")
    for a, b in zip(edges[:-1], edges[1:]):
        try:
            df = entsoe_imbalance(str(a.date()), str(b.date()), zone)
        except Exception as e:
            print(f"{a.date()}  FAILED ({e})")
            continue
        parts.append(df)
        num = df.select_dtypes("number").iloc[:, 0]
        hourly = num.groupby(num.index.floor("h")).mean()
        grid = pd.date_range(pd.Timestamp(a, tz=TZ).tz_convert("UTC"),
                             pd.Timestamp(b, tz=TZ).tz_convert("UTC"), freq="h", inclusive="left")
        miss = int(hourly.reindex(grid).isna().sum())
        print(f"{a.date()}  {len(df):6d}  {len(grid):14d}  {miss:13d}")
    df = pd.concat(parts)
    df = df[~df.index.duplicated()].sort_index()
    df.index.name = "timestamp_utc"
    df.to_parquet(ENTSOE_IMB)
    num = df.select_dtypes("number")
    print(f"saved {ENTSOE_IMB}: {len(df):,} rows, {df.index.min()} -> {df.index.max()}, "
          f"missing values {int(num.isna().sum().sum())}")
    if num.shape[1] >= 2:
        c0, c1 = num.columns[:2]
        print(f"'{c0}' == '{c1}' (single price): {100 * ((num[c0] - num[c1]).abs() < 0.01).mean():.2f}%")
    gaps()


def _entsoe_hourly() -> pd.Series:
    e = pd.read_parquet(ENTSOE_IMB)
    e.index = e.index.tz_convert("UTC")
    e = _num(e.select_dtypes("number").iloc[:, 0])
    return e.groupby(e.index.floor("h")).mean()


def _missing_pre_hours() -> pd.DatetimeIndex:
    """UTC hours of the old design (expected window) with no ENTSO-E imbalance price."""
    e0, e1 = EXPECTED["RegulatingBalancePowerdata"]
    grid = pd.date_range(pd.Timestamp(e0, tz=TZ).tz_convert("UTC"),
                         pd.Timestamp("2025-03-04 11:00", tz="UTC"), freq="h", inclusive="left")
    h = _entsoe_hourly().reindex(grid)
    return h.index[h.isna()]


def gaps() -> None:
    """Contiguous stretches of missing pre-regime hours in the ENTSO-E data."""
    miss = _missing_pre_hours()
    if len(miss) == 0:
        print("no missing pre-regime hours in ENTSO-E")
        return
    runs = (pd.Series(miss).diff() != pd.Timedelta(hours=1)).cumsum()
    print(f"missing pre-regime hours in ENTSO-E: {len(miss)} in {runs.nunique()} stretch(es):")
    for _, r in pd.Series(miss).groupby(runs.values):
        print(f"  {r.iloc[0]} -> {r.iloc[-1]}  ({len(r)} h)")


def fill_gaps() -> None:
    """Fetch the Energinet old-design weeks that overlap ENTSO-E gaps (and only those)."""
    miss = _missing_pre_hours()
    if len(miss) == 0:
        print("nothing to fill")
        return
    cfg = SOURCES["RegulatingBalancePowerdata"]
    edges = chunk_edges(cfg)
    local = miss.tz_convert(TZ).tz_localize(None)
    weeks = [str(a.date()) for a, b in zip(edges[:-1], edges[1:])
             if ((local >= a) & (local < b)).any()]
    print(f"{len(miss)} missing hours -> {len(weeks)} Energinet week(s) to fetch: {weeks}")
    download(include_old=True, only_weeks=set(weeks))
    gaps_after = len(_missing_pre_hours_combined())
    print(f"missing pre-regime hours after combining ENTSO-E + Energinet: {gaps_after}")


def _missing_pre_hours_combined() -> pd.DatetimeIndex:
    miss = _missing_pre_hours()
    try:
        old = _num(_load("RegulatingBalancePowerdata")["ImbalancePriceEUR"])
    except SystemExit:
        return miss
    have = old.dropna().index
    return miss[~miss.isin(have)]


def _compare(ent: pd.Series, ref: pd.Series, label: str) -> None:
    j = pd.concat([ent.rename("entsoe"), ref.rename("ref")], axis=1).dropna()
    if j.empty:
        print(f"    {label}: no overlapping timestamps")
        return
    d = (j["entsoe"] - j["ref"]).abs()
    ratio = (j["entsoe"] / j["ref"]).replace([np.inf, -np.inf], np.nan).dropna()
    print(f"    {label}: {len(j)} overlapping periods | equal (<0.01): {100 * (d < 0.01).mean():.1f}% "
          f"| corr {j.corr().iloc[0, 1]:.4f} | median |diff| {d.median():.2f} | max |diff| "
          f"{d.max():.2f} | median ratio {ratio.median():.3f}")


def entsoe_check(zone: str = "DK_1") -> None:
    """Is ENTSO-E's DK1 imbalance price the same series as Energinet's?"""
    windows = [("post, vs Energinet ImbalancePrice (15-min)", "2025-11-03", "2025-11-10"),
               ("pre, vs cached Energinet old dataset (hourly)", "2024-06-10", "2024-06-17"),
               ("pre, no Energinet data (coverage only)", "2024-11-04", "2024-11-11"),
               ("pre, just before the reform", "2025-02-24", "2025-03-03")]
    new = _load("ImbalancePrice")
    try:
        old = _load("RegulatingBalancePowerdata")
    except SystemExit:
        old = None
    for label, s0, s1 in windows:
        print(f"\n==================== {label}: {s0} -> {s1} ====================")
        try:
            ent = entsoe_imbalance(s0, s1, zone)
        except Exception as e:
            print(f"  ENTSO-E query FAILED: {e}")
            continue
        num = ent.select_dtypes("number")
        print(f"  columns: {list(ent.columns)} | rows: {len(ent)} | "
              f"resolution: {resolution(ent.index)} | missing: {int(num.isna().sum().sum())}")
        if num.shape[1] >= 2:
            a, b = num.columns[:2]
            print(f"  '{a}' == '{b}' (single price?): "
                  f"{100 * ((num[a] - num[b]).abs() < 0.01).mean():.1f}% of periods")
        print("  " + num.describe().loc[["mean", "50%", "min", "max"]].round(1)
              .to_string().replace("\n", "\n  "))
        for c in num.columns:
            if s0 >= "2025-03-19":
                ref = _num(new["ImbalancePriceEUR"])[s0:s1]
                _compare(_q15(num[c]), ref, f"'{c}' vs Energinet (15-min)")
            elif old is not None and len(old[s0:s1]):
                ref = _num(old["ImbalancePriceEUR"])[s0:s1]
                _compare(num[c].groupby(num.index.floor("h")).mean(), ref,
                         f"'{c}' (hourly mean) vs Energinet old dataset")


# ----------------------------------------------------------------------- build
def _load(dataset: str) -> pd.DataFrame:
    frames = [pd.read_parquet(f) for f in sorted(RAW.glob(f"{dataset}_*.parquet"))]
    frames = [f for f in frames if not f.empty]
    if not frames:
        raise SystemExit(f"no cached {dataset} chunks; run --download first")
    df = to_utc_index(pd.concat(frames, ignore_index=True))
    return df[~df.index.duplicated(keep="last")]


def _num(s: pd.Series) -> pd.Series:
    return pd.to_numeric(s, errors="coerce")


def regime_of(dates: pd.Series) -> pd.Series:
    return pd.Series(np.select([dates <= PRE_END, dates >= POST_START], ["pre", "post"],
                               default="transition"), index=dates.index)


def split_of(dates: pd.Series, regime: pd.Series) -> pd.Series:
    out = pd.Series("none", index=dates.index)
    for reg, parts in SPLITS.items():
        for name, (a, b) in parts.items():
            out[(regime == reg) & (dates >= a) & (dates <= b)] = name
    return out


def to_local_panel(h: pd.DataFrame) -> pd.DataFrame:
    """UTC-hourly frame -> (date, hour) in Danish local time. No interpolation of the target:
    the repeated hour of a 25-hour day is averaged (a missing value there stays missing), and a
    23-hour day simply has 23 rows."""
    local = h.index.tz_convert(TZ)
    p = h.groupby([pd.to_datetime(local.date), local.hour]).agg(lambda x: x.mean(skipna=False))
    p.index = p.index.set_names(["date", "hour"])
    return p


def _stats(s: pd.Series) -> dict:
    s = s.dropna()
    return {"hours": len(s), "mean": s.mean(), "median": s.median(), "std": s.std(),
            "p01": s.quantile(0.01), "p99": s.quantile(0.99),
            "share>0 %": 100 * (s > 0).mean(), "share=0 %": 100 * (s.abs() < 0.01).mean()}


def build(zone: str = "DK_1") -> None:
    entsoe = pd.read_parquet(f"data/raw/{zone}_price.parquet")["price"]
    entsoe.index = entsoe.index.tz_convert("UTC")
    entsoe = entsoe[~entsoe.index.duplicated()].sort_index()

    # ---- new design, 15-min
    new = _load("ImbalancePrice")
    q = pd.DataFrame({"imb": _num(new["ImbalancePriceEUR"]),
                      "spot_eds": _num(new["SpotPriceEUR"]),
                      "dom_dir": _num(new["DominatingDirection"])})
    q["da"] = _q15(entsoe).reindex(q.index)
    q["spread"] = q["imb"] - q["da"]
    chk = (q["spot_eds"] - q["da"]).abs().dropna()
    print(f"15-min rows: {len(q):,} ({q.index.min()} -> {q.index.max()}); "
          f"missing imbalance prices: {int(q['imb'].isna().sum())}")
    print(f"cross-check SpotPriceEUR vs ENTSO-E: {100 * (chk < 0.01).mean():.2f}% equal, "
          f"max |diff| {chk.max():.2f} EUR/MWh")
    bad = q.loc[chk[chk >= 0.01].index, ["spot_eds", "da"]]
    if len(bad):
        print(f"  {len(bad)} differing quarter-hours (first 10, UTC):")
        print("  " + bad.head(10).round(2).to_string().replace("\n", "\n  "))

    # hourly from 15-min: an hour counts only if all 4 quarter-hours are present
    g = q["spread"].groupby(q.index.floor("h"))
    h_new = pd.DataFrame({"spread": g.mean(), "n_q": g.count(),
                          "imb": q["imb"].groupby(q.index.floor("h")).mean(),
                          "da": q["da"].groupby(q.index.floor("h")).mean(),
                          "dom_dir": q["dom_dir"].groupby(q.index.floor("h")).mean()})
    h_new.loc[h_new["n_q"] < 4, ["spread", "imb"]] = np.nan

    # ---- old design, hourly: ENTSO-E (verified identical to Energinet), Energinet as fallback
    new_start_hour = q.index.min().floor("h")
    try:
        old_eds = _num(_load("RegulatingBalancePowerdata")["ImbalancePriceEUR"])
    except SystemExit:
        old_eds = None
    if ENTSOE_IMB.exists():
        e = _entsoe_hourly()
        imb_old = e[e.index < new_start_hour + pd.Timedelta(hours=1)]
        src = "ENTSO-E"
        if old_eds is not None:
            j = pd.concat([imb_old, old_eds], axis=1, sort=True).dropna()
            print(f"pre-regime source: ENTSO-E | cross-check vs {len(j)} cached Energinet hours: "
                  f"{100 * ((j.iloc[:, 0] - j.iloc[:, 1]).abs() < 0.01).mean():.2f}% equal")
            fill = old_eds[old_eds.index < new_start_hour + pd.Timedelta(hours=1)]
            fill = fill[~fill.index.isin(imb_old.dropna().index)].dropna()
            if len(fill):
                imb_old = pd.concat([imb_old.dropna(), fill]).sort_index()
                src = f"ENTSO-E + {len(fill)} h from Energinet (verified identical)"
    elif old_eds is not None:
        imb_old, src = old_eds, "Energinet RegulatingBalancePowerdata"
        print("pre-regime source: Energinet old dataset (run --entsoe-download for full coverage)")
    else:
        raise SystemExit("no pre-regime imbalance data: run --entsoe-download")
    h_old = pd.DataFrame({"imb": imb_old})
    da_h = entsoe.groupby(entsoe.index.floor("h")).mean()
    h_old["da"] = da_h.reindex(h_old.index)
    h_old["spread"] = h_old["imb"] - h_old["da"]
    h_old["n_q"] = 4
    h_old["dom_dir"] = np.nan
    print(f"old-design hourly rows ({src}): {len(h_old):,} "
          f"({h_old.index.min()} -> {h_old.index.max()})")

    # new data wins where it has a complete hour; the old data fills the hand-over hour
    h = h_new.dropna(subset=["spread"]).combine_first(h_old).sort_index()
    # full UTC-hour grid, so missing hours appear as NaN instead of silently disappearing
    grid = pd.date_range(h.index.min(), h.index.max(), freq="h", tz="UTC")
    h = h.reindex(grid)
    panel = to_local_panel(h[["spread", "imb", "da", "dom_dir"]])
    dates = pd.Series(panel.index.get_level_values("date"), index=panel.index)
    panel["regime"] = regime_of(dates)
    panel["split"] = split_of(dates, panel["regime"])

    # 15-min table (post regime only; robustness check)
    qd = pd.to_datetime(q.index.tz_convert(TZ).date)
    q15 = q[qd >= POST_START].copy()

    OUT_H.parent.mkdir(parents=True, exist_ok=True)
    panel.to_parquet(OUT_H)
    q15.to_parquet(OUT_Q)
    print(f"saved {OUT_H} {panel.shape} and {OUT_Q} {q15.shape}")

    # ---- coverage for every split; distribution ONLY for train + validation
    print("\n== Coverage by regime and split ==")
    rows = {}
    for (reg, spl), grp in panel.groupby(["regime", "split"]):
        d = grp.index.get_level_values("date")
        n_nan = grp["spread"].isna().groupby(d).sum()
        rows[f"{reg}/{spl}"] = {"days": d.nunique(), "complete days": int((n_nan == 0).sum()),
                                "missing hours": int(grp["spread"].isna().sum()),
                                "missing hours %": 100 * grp["spread"].isna().mean()}
    print(pd.DataFrame(rows).T.round(2).to_string())

    print("\n== Spread (imbalance - day-ahead, EUR/MWh), TRAIN and VALIDATION only ==")
    st = {f"{reg}/{spl}": _stats(grp["spread"])
          for (reg, spl), grp in panel.groupby(["regime", "split"]) if spl in ("train", "val")}
    print(pd.DataFrame(st).T.round(1).to_string())
    print("\nTest-period distributions are deliberately not shown (pre-registration).")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--probe", action="store_true")
    ap.add_argument("--align", action="store_true")
    ap.add_argument("--download", action="store_true")
    ap.add_argument("--build", action="store_true")
    ap.add_argument("--status", action="store_true")
    ap.add_argument("--entsoe-check", action="store_true")
    ap.add_argument("--entsoe-download", action="store_true")
    ap.add_argument("--gaps", action="store_true")
    ap.add_argument("--fill-gaps", action="store_true")
    ap.add_argument("--download-old", action="store_true",
                    help="also try the slow Energinet old-design endpoint")
    ap.add_argument("--zone", default="DK_1")
    a = ap.parse_args()
    if a.probe:
        for d in DATASETS:
            probe(d)
    if a.align:
        align(a.zone)
    if a.download or a.download_old:
        download(include_old=a.download_old)
    if a.entsoe_download:
        entsoe_download(a.zone)
    if a.gaps:
        gaps()
    if a.fill_gaps:
        fill_gaps()
    if a.status:
        status()
    if a.entsoe_check:
        entsoe_check(a.zone)
    if a.build:
        build(a.zone)
    if not (a.probe or a.align or a.download or a.download_old or a.build or a.status
            or a.entsoe_check or a.entsoe_download or a.gaps or a.fill_gaps):
        ap.print_help()
