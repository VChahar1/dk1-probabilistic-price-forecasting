"""
Week 4: are the accuracy differences statistically significant?

Multivariate Diebold-Mariano test (as in Lago et al., 2021): the 24 hourly losses of each
day are averaged into ONE daily loss, because all 24 prices of a day are forecast at the
same time. The test is one-sided:
    H0: model A is NOT more accurate than model B
    H1: model A is more accurate than model B     (small p-value => A significantly better)
The variance of the mean loss differential uses a Newey-West (Bartlett) estimator with a
7-day lag, which is conservative: it allows for weekly autocorrelation in the losses.

Usage (after src.evaluate and src.evaluate_prob):
    python -m src.dm --zone DK_1
Outputs: results/<zone>_dm_point.png, results/<zone>_dm_pinball.png + printed matrices.
"""
import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import norm

from src.evaluate_prob import FC_DIR, wide_to_long
from src.lgbm import QCOLS, QUANTILES, load_long

RES_DIR = Path("results")


# ---------------------------------------------------------------------- daily losses
def daily_abs_loss(y: pd.Series, point: pd.Series) -> pd.Series:
    return (point - y).abs().groupby(level="date").mean()


def daily_pinball(y: pd.Series, fc: pd.DataFrame) -> pd.Series:
    losses = []
    for q, c in zip(QUANTILES, QCOLS):
        d = y - fc[c]
        losses.append(np.maximum(q * d, (q - 1) * d))
    return pd.concat(losses, axis=1).mean(axis=1).groupby(level="date").mean()


# ------------------------------------------------------------------------- DM test
def dm_test(loss_a: pd.Series, loss_b: pd.Series, lag: int = 7) -> tuple[float, float]:
    """One-sided DM test. Returns (statistic, p-value); p small => A better than B."""
    d = (loss_a - loss_b).dropna().to_numpy()
    T = len(d)
    dc = d - d.mean()
    var = dc @ dc / T
    for k in range(1, lag + 1):
        var += 2 * (1 - k / (lag + 1)) * (dc[k:] @ dc[:-k]) / T
    stat = d.mean() / np.sqrt(var / T)
    return float(stat), float(norm.cdf(stat))


def pairwise(losses: dict[str, pd.Series], lag: int = 7) -> pd.DataFrame:
    """p[A, B] = p-value of H1 'row model A is more accurate than column model B'."""
    names = list(losses)
    idx = losses[names[0]].index
    for s in losses.values():
        idx = idx.intersection(s.dropna().index)
    p = pd.DataFrame(np.nan, index=names, columns=names)
    for a in names:
        for b in names:
            if a != b:
                p.loc[a, b] = dm_test(losses[a].loc[idx], losses[b].loc[idx], lag)[1]
    return p


def heatmap(p: pd.DataFrame, title: str, path: Path) -> None:
    fig, ax = plt.subplots(figsize=(1.3 * len(p) + 2, 1.1 * len(p) + 1.5))
    im = ax.imshow(p.clip(upper=0.1).to_numpy(dtype=float), cmap="RdYlGn_r", vmin=0, vmax=0.1)
    ax.set_xticks(range(len(p)), p.columns, rotation=30, ha="right")
    ax.set_yticks(range(len(p)), p.index)
    for i in range(len(p)):
        for j in range(len(p)):
            v = p.iat[i, j]
            ax.text(j, i, "" if np.isnan(v) else f"{v:.3f}", ha="center", va="center",
                    fontsize=9, color="black")
    ax.set_xlabel("... is more accurate than this model (column)")
    ax.set_ylabel("Row model ...")
    ax.set_title(title)
    fig.colorbar(im, ax=ax, label="p-value (clipped at 0.10)")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"saved {path}")


def run(zone: str, lag: int = 7) -> None:
    y = load_long(f"data/processed/{zone}_dataset.parquet")["price"]
    lear = wide_to_long(pd.read_parquet(FC_DIR / f"{zone}_lear.parquet"), "lear")
    naive = wide_to_long(pd.read_parquet(FC_DIR / f"{zone}_naive.parquet"), "naive")
    prob = {"LEAR-EQ": pd.read_parquet(FC_DIR / f"{zone}_lear_eq.parquet"),
            "LGBM": pd.read_parquet(FC_DIR / f"{zone}_lgbm_q.parquet"),
            "LGBM-CQR": pd.read_parquet(FC_DIR / f"{zone}_lgbm_cqr.parquet"),
            "ENS": pd.read_parquet(FC_DIR / f"{zone}_ens.parquet")}
    point = {"naive": naive, "LEAR": lear, "LGBM median": prob["LGBM"]["q0.50"],
             "ENS median": prob["ENS"]["q0.50"]}

    RES_DIR.mkdir(exist_ok=True)
    for label, losses in [
        ("point", {k: daily_abs_loss(y.reindex(v.index), v) for k, v in point.items()}),
        ("pinball", {k: daily_pinball(y.reindex(v.index), v) for k, v in prob.items()}),
    ]:
        p = pairwise(losses, lag)
        n = len(pd.concat(losses, axis=1).dropna())
        print(f"\n== DM p-values, {label} loss ({n} days; row better than column) ==")
        print(p.round(4))
        heatmap(p, f"{zone}: DM test, {label} loss ({n} days)",
                RES_DIR / f"{zone}_dm_{label}.png")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--zone", default="DK_1")
    ap.add_argument("--lag", type=int, default=7, help="Newey-West lag in days")
    a = ap.parse_args()
    run(a.zone, a.lag)
