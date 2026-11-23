"""Figure 1 of the paper: in-drift SLO attainment vs load for the stale-predictor comparison.

    python scripts/fig_stale.py   # reads results/v4/bench_sim_drift*.jsonl, writes docs/figures/stale_predictors.png
"""
from __future__ import annotations

import json
from collections import defaultdict

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

SERIES = [  # (policy key, label, color, style)
    ("pacer-online", "Online conformal (ours)", "#2a78d6", "-"),
    ("pacer-online-point", "Online point (median)", "#1baf7a", "-"),
    ("pacer", "Offline calibration", "#eb6834", "-"),
    ("chunked-edf-128", "Fixed budget, no model", "#8a8984", "--"),
]


def load(path):
    acc = defaultdict(lambda: defaultdict(list))
    for line in open(path):
        r = json.loads(line)
        acc[r["policy"]][r["rate"]].append(r["slo_attainment_drift"] * 100)
    return acc


def main():
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 9, "axes.edgecolor": "#b5b4ae",
                         "axes.labelcolor": "#52514e", "xtick.color": "#52514e", "ytick.color": "#52514e"})
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 2.7), sharey=True)
    for ax, factor in zip(axes, ["1.5", "2.0"]):
        acc = load(f"results/v4/bench_sim_drift{factor}.jsonl")
        for key, label, color, ls in SERIES:
            rates = sorted(acc[key])
            y = [np.mean(acc[key][r]) for r in rates]
            ax.plot(rates, y, ls, color=color, lw=2, marker="o", ms=4.5, label=label, zorder=3,
                    markeredgecolor="white", markeredgewidth=1)
        ax.set_title(f"Hardware {factor}x slower mid-run", fontsize=9.5, color="#0b0b0b", loc="left")
        ax.set_xlabel("Offered load (requests/s)")
        ax.set_xticks([3, 4, 5, 6])
        ax.set_ylim(0, 102)
        ax.grid(axis="y", color="#e6e5e0", lw=0.8, zorder=0)
        for s in ("top", "right"):
            ax.spines[s].set_visible(False)
    axes[0].set_ylabel("In-flight requests in SLO (%)")
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=4, frameon=False, fontsize=8.5, bbox_to_anchor=(0.5, 1.02))
    fig.tight_layout(rect=(0, 0, 1, 0.9))
    fig.savefig("docs/figures/stale_predictors.png", dpi=220)
    print("wrote docs/figures/stale_predictors.png")


if __name__ == "__main__":
    main()
