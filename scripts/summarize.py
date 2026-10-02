"""Print result tables (mean over seeds) and the max load each policy sustains at 90% SLO attainment."""
from __future__ import annotations

import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np


def capacity(rates, att, target=0.9):
    """Highest offered load with attainment >= target, linearly interpolated."""
    best = 0.0
    for i, (r, a) in enumerate(zip(rates, att)):
        if a >= target:
            best = r
            if i + 1 < len(rates) and att[i + 1] < target:
                r2, a2 = rates[i + 1], att[i + 1]
                best = r + (r2 - r) * (a - target) / (a - a2)
        else:
            break
    return best


def table(path: Path) -> dict:
    rows = [json.loads(l) for l in open(path)]
    by = defaultdict(lambda: defaultdict(list))
    for r in rows:
        by[r["policy"]][r["rate"]].append(r)
    rates = sorted({r["rate"] for r in rows})
    seeds = len({r["seed"] for r in rows})
    print(f"\n## {path.name}  ({seeds} seed{'s' if seeds > 1 else ''}, SLO TTFT {rows[0]['slo_ttft']}s / TPOT {rows[0]['slo_tpot']*1e3:.0f}ms)")
    print("SLO attainment %, mean" + (" ± std" if seeds > 1 else "") + " over seeds")
    print(f"{'policy':16s}" + "".join(f"{r:>12g}" for r in rates) + f"{'cap@90%':>10s}")
    drift = "slo_attainment_drift" in rows[0]
    out = {}
    for p, d in by.items():
        att = [np.mean([x["slo_attainment"] for x in d[r]]) for r in rates]
        sd = [np.std([x["slo_attainment"] for x in d[r]]) for r in rates]
        cap = capacity(rates, att)
        cells = "".join(f"{a*100:7.1f}±{s*100:4.1f}" if seeds > 1 else f"{a*100:12.1f}" for a, s in zip(att, sd))
        print(f"{p:16s}{cells}{cap:10.2f}")
        if drift:
            dv = [np.mean([x["slo_attainment_drift"] for x in d[r]]) for r in rates]
            print(f"{'  in drift':16s}" + "".join(f"{a*100:12.1f}" for a in dv))
        out[p] = {"rates": rates, "attainment": att, "std": sd, "capacity_90": cap,
                  "goodput": [np.mean([x["goodput"] for x in d[r]]) for r in rates]}
    return out


def main():
    files = [Path(f) for f in sys.argv[1:]] or sorted(Path("results").glob("bench_*.jsonl"))
    summary = {f.stem: table(f) for f in files}
    robustness(summary)
    with open("results/summary.json", "w") as f:
        json.dump(summary, f, indent=1)


SCENARIOS = {"bench_sim": "Poisson, TPOT 100ms", "bench_sim_tpot0.06": "Poisson, TPOT 60ms",
             "bench_sim_tpot0.15": "Poisson, TPOT 150ms", "bench_sim_bursty": "Bursty CV=3, TPOT 100ms",
             "bench_sim_azure_conv": "Azure chat trace", "bench_sim_azure_code": "Azure code trace"}


def robustness(summary):
    """Capacity of each policy relative to the best policy in each scenario (1.0 = best).
    The worst case across scenarios is the policy's regret if you cannot re-tune it."""
    scen = [k for k in SCENARIOS if k in summary]
    if not scen:
        return
    pols = sorted(set.intersection(*(set(summary[k]) for k in scen)))
    print("\n## Robustness: capacity@90% as a fraction of the best policy per scenario")
    print(f"{'policy':16s}" + "".join(f"{SCENARIOS[k]:>25s}" for k in scen) + f"{'worst case':>12s}")
    rob = {}
    for p in pols:
        rel = [summary[k][p]["capacity_90"] / max(summary[k][q]["capacity_90"] for q in pols) for k in scen]
        rob[p] = {"relative": dict(zip(scen, rel)), "worst": min(rel)}
    for p, v in sorted(rob.items(), key=lambda kv: -kv[1]["worst"]):
        print(f"{p:16s}" + "".join(f"{x:25.2f}" for x in v["relative"].values()) + f"{v['worst']:12.2f}")
    summary["robustness"] = rob


if __name__ == "__main__":
    main()
