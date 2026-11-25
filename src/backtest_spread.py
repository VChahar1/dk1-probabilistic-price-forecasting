"Intraday extension, Step 6: hypothetical 1 MW spread backtest (pre-registered)."

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src.dm import dm_test
from src.spread_models import OUT_DIR, load_table

GRID = (0, 5, 10, 20, 40, 80)
COST = 2.0
COST_SENS = (0.0, 1.0, 2.0, 5.0)
STRATS = {
    "B0": ("B0", "all"),
    "C1": ("C1", "all"),
    "B0_T2": ("B0", "T2"),
    "C2": ("C2", "T2"),
}
THRESH = OUT_DIR / "thresholds.json"
RES = Path("results")


def signals(df: pd.DataFrame) -> dict:
    out = {}
    for name, (iset, hours) in STRATS.items():
        m = pd.read_parquet(OUT_DIR / f"lgbm_{iset}.parquet")["q0.50"].reindex(df.index)
        if hours == "T2":
            m = m.where(df["eligible_T2"])
        out[name] = m
    return out


def position(sig: pd.Series, theta: float) -> pd.Series:
    p = np.where(sig > theta, 1.0, np.where(sig < -theta, -1.0, 0.0))
    return pd.Series(p, index=sig.index).where(sig.notna())


def pnl(pos: pd.Series, spread: pd.Series, cost: float) -> pd.Series:
    return pos * spread - cost * pos.abs()


def benchmark_positions(df: pd.DataFrame) -> dict:
    share = pd.read_parquet(OUT_DIR / "benchmarks.parquet")["share_pos"].reindex(
        df.index
    )
    return {
        "always_short": pd.Series(-1.0, index=df.index),
        "majority": pd.Series(np.where(share > 0.5, 1.0, -1.0), index=df.index).where(
            share.notna()
        ),
    }


def p_mean_positive(hourly: pd.Series) -> float:
    """One-sided Newey-West test (7-day lag) of mean daily P&L > 0."""
    d = hourly.groupby(level="date").sum()
    if len(d) < 10 or d.std() == 0:
        return np.nan
    return dm_test(-d, pd.Series(0.0, index=d.index))[1]


def metrics(h: pd.Series, pos: pd.Series) -> dict:
    h, pos = h.dropna(), pos.reindex(h.dropna().index)
    traded = pos.abs() > 0
    daily = h.groupby(level="date").sum()
    cum = daily.cumsum()
    t = h[traded]
    k = max(1, int(np.ceil(0.01 * len(t))))
    top = t.abs().nlargest(k).index
    return {
        "hours": len(h),
        "traded %": 100 * traded.mean(),
        "total EUR/MW": h.sum(),
        "EUR/traded MWh": t.mean() if len(t) else np.nan,
        "hit %": 100 * (t > 0).mean() if len(t) else np.nan,
        "Sharpe (ann.)": daily.mean() / daily.std() * np.sqrt(365)
        if daily.std() > 0
        else np.nan,
        "max drawdown": (
            cum - np.maximum.accumulate(np.r_[0.0, cum.to_numpy()])[1:]
        ).min()
        if len(cum)
        else np.nan,
        "worst hour": h.min(),
        "worst day": daily.min(),
        "CVaR5% day": daily[daily <= daily.quantile(0.05)].mean(),
        "top 1% hours": t.loc[top].sum() if len(t) else 0.0,
        "ex top 1%": h.sum() - (t.loc[top].sum() if len(t) else 0.0),
        "p(mean>0)": p_mean_positive(h),
    }


def _rows(df, split, reg):
    return df[(df["split"] == split) & (df["regime"] == reg) & df["spread"].notna()]


def select() -> None:
    df = load_table()
    sig = signals(df)
    thresholds, report = {}, {}
    for reg in ("post", "pre"):
        v = _rows(df, "val", reg)
        if v.empty:
            continue
        tab = {}
        for name in STRATS:
            s = sig[name].reindex(v.index).dropna()
            tot = {
                th: pnl(position(s, th), v.loc[s.index, "spread"], COST).sum()
                for th in GRID
            }
            tab[name] = tot
            thresholds[f"{name}/{reg}"] = max(tot, key=tot.get)
        print(
            f"\n== {reg.upper()} / VALIDATION: total P&L per MW after {COST} EUR/MWh, by threshold =="
        )
        print(pd.DataFrame(tab).T.round(0).to_string())
        report[reg] = _evaluate(df, sig, "val", reg, thresholds)
    THRESH.write_text(json.dumps(thresholds, indent=2))
    print(f"\nthresholds fixed in {THRESH}: {thresholds}")


def test() -> None:
    if not THRESH.exists():
        raise SystemExit("no thresholds.json: run --select on validation first")
    thresholds = json.loads(THRESH.read_text())
    df = load_table()
    sig = signals(df)
    print(
        "=" * 72
        + f"\nONE-TIME TEST BACKTEST with thresholds fixed on validation ({THRESH})\n"
        + "=" * 72
    )
    results = {
        reg: _evaluate(df, sig, "test", reg, thresholds, plot=(reg == "post"))
        for reg in ("post", "pre")
        if not _rows(df, "test", reg).empty
    }
    RES.mkdir(exist_ok=True)
    (RES / "backtest_test.json").write_text(
        json.dumps(results, indent=2, default=float)
    )
    print(f"\nsaved {RES / 'backtest_test.json'}")


def _evaluate(df, sig, split, reg, thresholds, plot=False) -> dict:
    rows = _rows(df, split, reg)
    y = rows["spread"]
    print(
        f"\n################ {reg.upper()} regime, {split.upper()} "
        f"({rows.index.get_level_values('date').nunique()} days) ################"
    )
    if reg == "pre":
        print("(pre regime: descriptive only, too few days for inference)")
    pos, hourly = {}, {}
    for name in STRATS:
        th = thresholds.get(f"{name}/{reg}")
        if th is None:
            continue
        p = position(sig[name].reindex(rows.index), th)
        pos[name], hourly[name] = p, pnl(p, y, COST)
    base = pos["B0"].dropna().index
    for name, p in benchmark_positions(rows).items():
        p = p.reindex(base)
        pos[name], hourly[name] = p, pnl(p, y.reindex(base), COST)

    tab = {
        f"{n} (theta={thresholds.get(f'{n}/{reg}', '-')})": metrics(hourly[n], pos[n])
        for n in hourly
    }
    print(f"\n== P&L per 1 MW after {COST} EUR/MWh costs (EUR) ==")
    print(pd.DataFrame(tab).T.round(2).to_string())

    sens = {}
    for n in hourly:
        th = thresholds.get(f"{n}/{reg}")
        p = pos[n]
        sens[n] = {
            f"cost {c:g}": pnl(p, y.reindex(p.index), c).sum() for c in COST_SENS
        }
    print(
        "\n== Cost sensitivity: total P&L per MW (thresholds fixed at the 2 EUR/MWh choice) =="
    )
    print(pd.DataFrame(sens).T.round(0).to_string())

    print(
        "\n== Part C: value of post-auction information at the day-ahead price (UNREALISABLE bound) =="
    )
    info = {}
    for c, b in (("C1", "B0"), ("C2", "B0_T2")):
        if c not in hourly or b not in hourly:
            continue
        j = pd.concat([hourly[c], hourly[b]], axis=1).dropna()
        diff = (j.iloc[:, 0] - j.iloc[:, 1]).groupby(level="date").sum()
        p = (
            dm_test(-diff, pd.Series(0.0, index=diff.index))[1]
            if len(diff) >= 10 and diff.std() > 0
            else np.nan
        )
        per_h = (j.iloc[:, 0] - j.iloc[:, 1]).mean()
        info[f"{c} - {b}"] = {
            "hours": len(j),
            "EUR per MW-hour": per_h,
            "EUR per MW-year": per_h * 8760,
            "p(C better)": p,
        }
    print(pd.DataFrame(info).T.round(3).to_string())

    if plot:
        RES.mkdir(exist_ok=True)
        fig, ax = plt.subplots(figsize=(9, 4.5))
        styles = {"B0": "-", "C1": "-", "always_short": "--", "majority": ":"}
        for n in ("B0", "C1", "always_short", "majority"):
            if n in hourly:
                ax.plot(
                    hourly[n].dropna().groupby(level="date").sum().cumsum(),
                    styles[n],
                    label=n + (" (unrealisable)" if n == "C1" else ""),
                )
        ax.axhline(0, color="grey", lw=0.8)
        ax.set_ylabel("cumulative P&L per MW (EUR)")
        ax.set_title(
            f"DK1 imbalance - day-ahead spread, hypothetical 1 MW, {split} ({reg} regime), "
            f"costs {COST} EUR/MWh",
            fontsize=10,
        )
        ax.legend()
        fig.autofmt_xdate()
        fig.tight_layout()
        path = RES / f"backtest_{reg}_{split}.png"
        fig.savefig(path, dpi=150)
        plt.close(fig)
        print(f"saved {path}")
    return {"metrics": tab, "cost_sensitivity": sens, "information_value": info}


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--select", action="store_true")
    ap.add_argument("--test", action="store_true")
    a = ap.parse_args()
    if a.select:
        select()
    if a.test:
        test()
    if not (a.select or a.test):
        ap.print_help()
