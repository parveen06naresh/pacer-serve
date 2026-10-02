"""One simulated run through a hardware slowdown, step by step: what each scheduler's
steps actually cost against the TPOT SLO. Writes results/drift_timeline.json."""
from __future__ import annotations

import copy
import json

from pacer.costmodel import LinearModel, OnlineConformal
from pacer.engine import BlockManager, SimExecutor
from pacer.request import SLO
from pacer.runtime import serve
from pacer.scheduler import ChunkedFixed, Pacer
from pacer.workload import WorkloadSpec, generate


def main(rate=4.0, factor=1.5, seed=0):
    lat = LinearModel.load("results/latency_model.json")
    slo = SLO(2.0, 0.10)
    trace = generate(WorkloadSpec(rate=rate, num_requests=300, seed=seed))
    span = trace[-1].arrival
    a, b = 0.3 * span, 0.7 * span
    out = {"window": [a, b], "factor": factor, "slo_tpot": slo.tpot, "runs": {}}
    for key, sched in [("chunked-edf-128", ChunkedFixed(128, edf=Pacer(lat, slo))),
                       ("pacer", Pacer(lat, slo)),
                       ("pacer-online", Pacer(lat, slo, online=OnlineConformal(0.1)))]:
        reqs = copy.deepcopy(trace)
        bm = BlockManager(6144, 16)
        log = []
        ex = SimExecutor(lat, bm, noise=0.08, seed=seed, slowdown=lambda now: factor if a <= now < b else 1.0)
        serve(reqs, sched, ex, bm, step_log=log)
        # Per-request TPOT against completion time.
        out["runs"][key] = {
            "steps": [(s["t"], s["dt"]) for s in log if s["decodes"]],
            "requests": [(r.finish_time, r.tpot(), r.ttft()) for r in reqs],
        }
    with open("results/drift_timeline.json", "w") as f:
        json.dump(out, f)


if __name__ == "__main__":
    main()
