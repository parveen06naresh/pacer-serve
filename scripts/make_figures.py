"""Render README figures from results/*.json(l)."""
from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

R = Path("results")
FIG = Path("docs/figures")
INK, INK2, GRID, SURF = "#0b0b0b", "#52514e", "#e6e5e0", "#fcfcfb"
BLUE, ORANGE, AQUA, YELLOW, MAGENTA, GREEN, VIOLET = "#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#4a3aa7"
GRAY = "#9a9893"

STYLE = {
    "pacer": dict(color=BLUE, lw=2.4, marker="o", label="Pacer (ours)"),
    "prefill-first": dict(color=ORANGE, lw=2, marker="s", label="Prefill-first (vLLM v0)"),
    "chunked-best": dict(color=AQUA, lw=2, marker="^", label="Chunked, best fixed budget (Sarathi)"),
    "chunked-edf-best": dict(color=VIOLET, lw=2, marker="D", ms=5, label="Chunked + EDF, best fixed budget"),
    "chunked-64": dict(color=GRAY, lw=1.2, ls=":", marker=None, label="Chunked-64"),
    "chunked-128": dict(color=GRAY, lw=1.2, ls="--", marker=None, label="Chunked-128"),
    "chunked-256": dict(color=GRAY, lw=1.2, ls="-.", marker=None, label="Chunked-256"),
    "chunked-512": dict(color=GRAY, lw=1.2, ls=(0, (1, 3)), marker=None, label="Chunked-512"),
    "pacer-no-slack": dict(color=YELLOW, lw=1.6, ls="--", marker="o", ms=4, label="Pacer w/o slack"),
    "pacer-no-edf": dict(color=MAGENTA, lw=1.6, ls="--", marker="o", ms=4, label="Pacer w/o EDF"),
}


def setup():
    plt.rcParams.update({
        "figure.facecolor": SURF, "axes.facecolor": SURF, "savefig.facecolor": SURF,
        "axes.edgecolor": GRID, "axes.labelcolor": INK2, "xtick.color": INK2, "ytick.color": INK2,
        "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.8, "axes.spines.top": False,
        "axes.spines.right": False, "font.size": 10.5, "axes.titlesize": 12, "axes.titleweight": "bold",
        "axes.titlecolor": INK, "legend.frameon": False, "lines.markersize": 6,
    })
    FIG.mkdir(parents=True, exist_ok=True)


def load(path):
    rows = [json.loads(l) for l in open(path)] if Path(path).exists() else []
    return rows


def aggregate(rows):
    """policy -> rate -> mean over seeds of each metric."""
    acc = defaultdict(lambda: defaultdict(list))
    for r in rows:
        acc[r["policy"]][r["rate"]].append(r)
    out = {}
    for p, by_rate in acc.items():
        out[p] = {rate: {k: float(np.mean([x[k] for x in xs])) for k in xs[0] if isinstance(xs[0][k], (int, float))}
                  for rate, xs in sorted(by_rate.items())}
    return out


def add_best_chunked(agg):
    best = None
    for prefix, key in [("chunked-edf-", "chunked-edf-best"), ("chunked-", "chunked-best")]:
        cands = [p for p in agg if p.startswith(prefix) and p[len(prefix)].isdigit()]
        if cands:
            b = max(cands, key=lambda p: np.mean([v["slo_attainment"] for v in agg[p].values()]))
            agg[key] = agg[b]
            if key == "chunked-best":
                best = b
    return agg, best


def slo_curve(agg, best, path, title, policies):
    fig, ax = plt.subplots(figsize=(7.2, 4.2))
    for p in policies:
        if p not in agg:
            continue
        rates = list(agg[p])
        st = dict(STYLE[p])
        if p == "chunked-best":
            st["label"] = f"Chunked, best fixed budget ({best.split('-')[-1]} tok)"
        ax.plot(rates, [agg[p][r]["slo_attainment"] * 100 for r in rates], **st)
    ax.axhline(90, color=INK2, lw=0.8, ls=":")
    ax.text(ax.get_xlim()[0], 91, " 90% SLO target", color=INK2, fontsize=9, va="bottom")
    ax.set_xlabel("Offered load (requests / second)")
    ax.set_ylabel("Requests meeting TTFT and TPOT SLOs (%)")
    ax.set_ylim(0, 102)
    ax.set_title(title, loc="left")
    ax.legend(loc="lower left", fontsize=9)
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


def costmodel_fig():
    p = R / "costmodel_preds.npz"
    if not p.exists():
        return
    d = np.load(p)
    y = d["y"] * 1e3
    fig, axes = plt.subplots(1, 2, figsize=(9.6, 4.2), sharex=True, sharey=True)
    for ax, (name, color, label) in zip(axes, [("roofline", ORANGE, "Calibrated roofline (2 fitted scalars)"),
                                                ("linear", BLUE, "Learned physics-informed model")]):
        pred = d[name] * 1e3
        err = np.mean(np.abs(pred - y) / y) * 100
        ax.scatter(y, pred, s=12, color=color, alpha=0.75, edgecolors="none")
        lim = [min(y.min(), pred.min()) * 0.8, max(y.max(), pred.max()) * 1.2]
        ax.plot(lim, lim, color=INK2, lw=1, ls="--")
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_xlim(lim)
        ax.set_ylim(lim)
        ax.set_title(f"{label}\nMAPE {err:.1f}% on held-out steps", loc="left", fontsize=11)
        ax.set_xlabel("Measured step latency (ms)")
    axes[0].set_ylabel("Predicted step latency (ms)")
    fig.tight_layout()
    fig.savefig(FIG / "costmodel.png", dpi=160)
    plt.close(fig)


def sim_vs_real_fig(real, sim):
    if not real or not sim:
        return
    fig, ax = plt.subplots(figsize=(5.2, 4.6))
    xs, ys = [], []
    for p in real:
        if p not in sim or p.endswith("-best"):
            continue
        for rate, m in real[p].items():
            if rate in sim[p]:
                xs.append(m["slo_attainment"] * 100)
                ys.append(sim[p][rate]["slo_attainment"] * 100)
    ax.scatter(xs, ys, s=22, color=BLUE, alpha=0.8, edgecolors="none")
    ax.plot([0, 100], [0, 100], color=INK2, lw=1, ls="--")
    err = np.mean(np.abs(np.array(xs) - np.array(ys)))
    ax.set_xlabel("SLO attainment, real engine (%)")
    ax.set_ylabel("SLO attainment, simulator (%)")
    ax.set_title(f"Simulator fidelity\nmean abs. error {err:.1f} points", loc="left")
    fig.tight_layout()
    fig.savefig(FIG / "sim_vs_real.png", dpi=160)
    plt.close(fig)


def roofline_fig():
    p = R / "roofline_projection.json"
    if not p.exists():
        return
    d = json.load(open(p))
    fig, ax = plt.subplots(figsize=(7.2, 4.2))
    for (name, v), c in zip(d.items(), [BLUE, ORANGE, AQUA, VIOLET]):
        xs = [pt["prefill_tokens"] for pt in v["curve"]]
        ys = [pt["step_ms"] / v["decode_only_step_ms"] for pt in v["curve"]]
        ax.plot(xs, ys, color=c, lw=2, marker="o", ms=4, label=f"{name}: {v['free_prefill_tokens']} free tokens")
    ax.set_xscale("symlog", linthresh=64)
    ax.set_xlabel("Prefill tokens added to a step with 32 decodes @ 2k context")
    ax.set_ylabel("Step time / decode-only step time")
    ax.set_title("Llama-3-8B BF16: how many prefill tokens ride free (roofline)", loc="left")
    ax.legend(fontsize=9)
    fig.tight_layout()
    fig.savefig(FIG / "roofline.png", dpi=160)
    plt.close(fig)


def robustness_fig():
    p = R / "summary.json"
    if not p.exists():
        return
    rob = json.load(open(p)).get("robustness")
    if not rob:
        return
    names = {"pacer": "Pacer (ours, no tuning)"}
    order = sorted(rob, key=lambda k: rob[k]["worst"])
    scen = list(next(iter(rob.values()))["relative"])
    labels = {"bench_sim": "Poisson, TPOT 100ms", "bench_sim_tpot0.06": "Poisson, TPOT 60ms",
              "bench_sim_tpot0.15": "Poisson, TPOT 150ms", "bench_sim_bursty": "Bursty, TPOT 100ms"}
    fig, ax = plt.subplots(figsize=(7.6, 4.6))
    for i, p in enumerate(order):
        vals = [rob[p]["relative"][k] * 100 for k in scen]
        ax.plot([min(vals), max(vals)], [i, i], color=GRID, lw=3, solid_capstyle="round", zorder=1)
    for k, c, m in zip(scen, [BLUE, ORANGE, AQUA, VIOLET], ["o", "s", "^", "D"]):
        ax.scatter([rob[p]["relative"][k] * 100 for p in order], range(len(order)), color=c, marker=m, s=46,
                   edgecolors=SURF, linewidths=1.5, zorder=3, label=labels.get(k, k))
    ax.set_yticks(range(len(order)))
    ax.set_yticklabels([names.get(p, p.replace("chunked-edf-", "chunked+EDF-")) for p in order])
    ax.set_xlabel("Capacity at 90% SLO attainment, % of best policy in that scenario")
    ax.set_title("No fixed budget wins everywhere; Pacer stays within 2%", loc="left")
    ax.legend(fontsize=8.5, loc="upper center", bbox_to_anchor=(0.4, -0.14), ncol=4, handletextpad=0.2, columnspacing=1)
    ax.set_xlim(40, 103)
    fig.set_size_inches(8.4, 5.0)
    fig.tight_layout()
    fig.savefig(FIG / "robustness.png", dpi=160)
    plt.close(fig)


def main():
    setup()
    robustness_fig()
    costmodel_fig()
    roofline_fig()
    real = aggregate(load(R / "bench_real.jsonl"))
    sim = aggregate(load(R / "bench_sim.jsonl"))
    main_p = ["pacer", "chunked-edf-best", "chunked-best", "prefill-first", "chunked-64", "chunked-128", "chunked-256", "chunked-512"]
    if real:
        real, best = add_best_chunked(real)
        slo_curve(real, best, FIG / "slo_real.png", "Real engine (measured on CPU): SLO attainment vs load", main_p)
    if sim:
        sim, best = add_best_chunked(sim)
        slo_curve(sim, best, FIG / "slo_sim.png", "Simulator: SLO attainment vs load", main_p)
        slo_curve(sim, best, FIG / "ablation.png", "Ablation (simulator): what each idea buys",
                  ["pacer", "pacer-no-slack", "pacer-no-edf", "chunked-edf-best", "chunked-best"])
    for tag in ["_bursty"]:
        b = aggregate(load(R / f"bench_sim{tag}.jsonl"))
        if b:
            b, best = add_best_chunked(b)
            slo_curve(b, best, FIG / f"slo_sim{tag}.png", "Bursty traffic (CV=3, simulator)", main_p)
    sim_vs_real_fig(real, sim)
    print("figures:", sorted(p.name for p in FIG.glob("*.png")))


if __name__ == "__main__":
    main()
