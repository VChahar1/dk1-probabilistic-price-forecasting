"""Plot the committed saved-result audit without downloading market data."""

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


def main() -> None:
    """Write a chart of verified MAE and interval coverage."""
    root = Path(__file__).resolve().parents[1]
    metrics = json.loads((root / "results/verified_metrics.json").read_text())
    comparison = metrics["information_set_comparison"]
    coverage = metrics["interval_calibration_comparison"]
    fig, axes = plt.subplots(1, 2, figsize=(10, 4), constrained_layout=True)
    values = [
        comparison["naive_mae_eur_mwh"],
        comparison["models"]["entsoe"]["mae_eur_mwh"],
        comparison["models"]["pregate"]["mae_eur_mwh"],
    ]
    bars = axes[0].bar(
        ["Naive", "ENTSO-E", "Pre-gate"],
        values,
        color=["#9aa6b2", "#547da8", "#176b60"],
    )
    axes[0].bar_label(bars, fmt="%.3f", padding=3)
    axes[0].set(
        ylabel="MAE (€/MWh)",
        title=f"Saved forecasts: {comparison['days']} common days",
        ylim=(0, max(values) * 1.2),
    )
    values = [
        coverage["raw_lightgbm"]["cov90_%"],
        coverage["conformal_lightgbm"]["cov90_%"],
    ]
    bars = axes[1].bar(["Raw", "Conformal"], values, color=["#9aa6b2", "#176b60"])
    axes[1].bar_label(bars, fmt="%.3f%%", padding=3)
    axes[1].axhline(90, linestyle="--", color="#a7513b", label="Nominal 90%")
    axes[1].set(
        ylabel="Observed coverage (%)",
        title=f"Interval audit: {coverage['days']} days",
        ylim=(0, 108),
    )
    axes[1].legend(loc="lower right", frameon=False)
    for ax in axes:
        ax.spines[["top", "right"]].set_visible(False)
    (root / "assets").mkdir(exist_ok=True)
    fig.savefig(root / "assets/results.png", dpi=160)
    plt.close(fig)


if __name__ == "__main__":
    main()
