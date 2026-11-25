"""Recompute documented metrics from locally cached forecasts without refitting models."""

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

from src.dm import daily_pinball, dm_test
from src.evaluate_prob import build_prob_models, interval_stats, pinball, wide_to_long
from src.lear import RES_COLS
from src.lgbm import load_long
from src.pregate import res_proxy


def main(output: str) -> None:
    """Recompute metrics from local forecasts and write an audit report."""
    root = Path("data")
    fc = root / "forecasts"
    y = load_long(str(root / "processed/DK_1_dataset.parquet"))["price"]
    naive = wide_to_long(pd.read_parquet(fc / "DK_1_naive.parquet"), "naive")
    models = {}
    for name, suffix in (("entsoe", ""), ("pregate", "_pregate"), ("no_res", "_nores")):
        lear = wide_to_long(pd.read_parquet(fc / f"DK_1_lear{suffix}.parquet"), "lear")
        raw = pd.read_parquet(fc / f"DK_1_lgbm_q{suffix}.parquet")
        eq, cq, ens = build_prob_models(y, lear, raw, 56)
        models[name] = dict(lear=lear, raw=raw, eq=eq, cqr=cq, ens=ens)

    idx = y.dropna().index.intersection(naive.dropna().index)
    dates = idx.get_level_values("date")
    idx = idx[(dates >= "2025-01-01") & (dates <= "2026-08-31")]
    for model in models.values():
        for v in model.values():
            idx = idx.intersection(v.dropna().index)
    yy = y.loc[idx]
    baseline = float((naive.loc[idx] - yy).abs().mean())
    compare = {}
    for name, model in models.items():
        ens = model["ens"].loc[idx]
        mae = float((ens["q0.50"] - yy).abs().mean())
        compare[name] = {
            "mae_eur_mwh": mae,
            "relative_mae": mae / baseline,
            "improvement_over_naive_percent": 100 * (1 - mae / baseline),
            "pinball_eur_mwh": pinball(yy.to_numpy(), ens),
            **interval_stats(yy.to_numpy(), ens),
        }
    stat, pvalue = dm_test(
        daily_pinball(yy, models["pregate"]["ens"].loc[idx]),
        daily_pinball(yy, models["entsoe"]["ens"].loc[idx]),
    )
    dates = idx.get_level_values("date")
    information_sets = {
        "start": str(dates.min().date()),
        "end": str(dates.max().date()),
        "days": int(dates.nunique()),
        "hours": len(idx),
        "naive_mae_eur_mwh": baseline,
        "models": compare,
        "dm_pregate_vs_entsoe_ensemble_pinball": {
            "statistic": stat,
            "one_sided_pvalue": pvalue,
            "newey_west_lag_days": 7,
        },
    }

    m = models["entsoe"]
    idx = y.dropna().index.intersection(naive.dropna().index)
    for v in m.values():
        idx = idx.intersection(v.dropna().index)
    yy = y.loc[idx]
    dates = idx.get_level_values("date")
    coverage = {
        "start": str(dates.min().date()),
        "end": str(dates.max().date()),
        "days": int(dates.nunique()),
        "hours": len(idx),
        "raw_lightgbm": interval_stats(yy.to_numpy(), m["raw"].loc[idx]),
        "conformal_lightgbm": interval_stats(yy.to_numpy(), m["cqr"].loc[idx]),
    }

    ds = pd.read_parquet(root / "processed/DK_1_dataset.parquet")
    ds.index = ds.index.set_levels(pd.to_datetime(ds.index.levels[0]), level=0)
    res = ds[RES_COLS].sum(axis=1, min_count=len(RES_COLS))
    wx = pd.read_parquet(root / "processed/DK_1_weather.parquet")
    proxy = res_proxy(res, wx)
    joined = pd.concat([proxy, res.rename("entsoe_forecast")], axis=1).dropna()
    error = joined["res_proxy"] - joined["entsoe_forecast"]
    target = joined["entsoe_forecast"]
    dates = joined.index.get_level_values("date")
    weather = {
        "target": "ENTSO-E renewables forecast, not observed power generation",
        "start": str(dates.min().date()),
        "end": str(dates.max().date()),
        "days": int(dates.nunique()),
        "hours": len(joined),
        "r_squared": float(
            1 - (error**2).sum() / ((target - target.mean()) ** 2).sum()
        ),
        "mae_mw": float(error.abs().mean()),
    }
    sources = {}
    for f in sorted(root.rglob("*.parquet")):
        sources[str(f)] = hashlib.sha256(f.read_bytes()).hexdigest()
    report = {
        "verification": "Metrics recomputed from saved local forecasts; models were not refitted.",
        "information_set_comparison": information_sets,
        "interval_calibration_comparison": coverage,
        "weather_to_power_proxy": weather,
        "local_input_sha256": sources,
    }
    result = json.dumps(report, indent=2, allow_nan=False) + "\n"
    Path(output).write_text(result)
    print(
        json.dumps(
            {k: v for k, v in report.items() if k != "local_input_sha256"}, indent=2
        )
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", default="results/verified_metrics.json")
    main(parser.parse_args().output)
