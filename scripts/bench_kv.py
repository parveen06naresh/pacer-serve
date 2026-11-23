"""Predictor 2 under drift: KV-memory reservation from a predicted output length.

The step scheduler is Pacer throughout (offline latency model, no hardware drift);
only the KV reservation policy changes. Midway through the trace the output-length
distribution shifts (requests arriving in the drift window generate FACTOR x more
tokens, e.g. a product switching on longer answers). Memory is made the binding
constraint (a small KV pool, as with long contexts or a large model on one GPU).

    python scripts/bench_kv.py --rates 3 4 5 6 7 --seeds 0 1 2
"""
from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path

import numpy as np

from pacer.costmodel import LinearModel
from pacer.engine import GrowingBlockManager, SimExecutor, StepShape
from pacer.kvpredict import build
from pacer.metrics import summarize
from pacer.request import SLO, Request
from pacer.runtime import serve
from pacer.scheduler import Pacer
from pacer.workload import WorkloadSpec, from_azure, generate

POLICIES = ["max", "ondemand", "offline", "online-point", "online", "oracle"]


def trace(kind: str, rate: float, n: int, seed: int, cap: int) -> list[Request]:
    if kind == "synthetic":
        return generate(WorkloadSpec(rate=rate, num_requests=n, seed=seed, output_median=48, output_sigma=0.6,
                                     output_max=cap))
    return from_azure(kind.split("-")[1], rate, n, window=seed, seed=seed, output_max=cap)


def apply_drift(reqs: list[Request], start: float, end: float, factor: float, cap: int):
    span = reqs[-1].arrival - reqs[0].arrival
    a, b = reqs[0].arrival + start * span, reqs[0].arrival + end * span
    for r in reqs:
        if a <= r.arrival < b:
            r.max_new_tokens = int(min(cap, max(4, round(r.max_new_tokens * factor))))
    return a, b


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--rates", type=float, nargs="+", default=[3, 4, 5, 6, 7])
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    ap.add_argument("--policies", nargs="+", default=POLICIES)
    ap.add_argument("--trace", default="synthetic", choices=["synthetic", "azure-conv", "azure-code"])
    ap.add_argument("--num-requests", type=int, default=400)
    ap.add_argument("--num-blocks", type=int, default=384)
    ap.add_argument("--block-size", type=int, default=16)
    ap.add_argument("--cap", type=int, default=384, help="max_tokens limit; what 'reserve max' reserves")
    ap.add_argument("--drift", type=float, nargs=3, default=[0.3, 0.7, 2.5], metavar=("START", "END", "FACTOR"))
    ap.add_argument("--alpha", type=float, default=0.1)
    ap.add_argument("--ttft", type=float, default=2.0)
    ap.add_argument("--tpot", type=float, default=0.10)
    ap.add_argument("--latency-model", default="results/latency_model.json")
    ap.add_argument("--out", default="results/bench_kv.jsonl")
    args = ap.parse_args()

    slo = SLO(args.ttft, args.tpot)
    lat = LinearModel.load(args.latency_model)
    cal = trace(args.trace, 5.0, 400, 100, args.cap)  # calibration traffic, before any drift

    def recompute(r: Request) -> float:
        ctx = r.context_len
        return lat.predict(StepShape(((ctx, ctx),), ()))

    Path(args.out).parent.mkdir(exist_ok=True)
    with open(args.out, "a") as f:
        for seed in args.seeds:
            for rate in args.rates:
                base = trace(args.trace, rate, args.num_requests, seed, args.cap)
                factor = args.drift[2]
                win = apply_drift(base, *args.drift[:2], factor, args.cap) if factor != 1.0 else None
                for kind in args.policies:
                    reqs = copy.deepcopy(base)
                    policy = build(kind, cal, args.cap, args.alpha)
                    bm = GrowingBlockManager(args.num_blocks, args.block_size, policy, recompute, policy.observe)
                    ex = SimExecutor(lat, bm, seed=seed)
                    steps = serve(reqs, Pacer(lat, slo), ex, bm)
                    m = summarize(kind, rate, reqs, slo, steps).as_dict()
                    m.update(trace=args.trace, seed=seed, drift=args.drift, num_blocks=args.num_blocks, cap=args.cap,
                             preemptions_per_req=bm.preemptions / len(reqs),
                             exceed_rate=len(bm.exceeded) / len(reqs), policy_name=policy.name)
                    if win:
                        hit = [r for r in reqs if r.arrival < win[1] and r.finish_time >= win[0]]
                        m["slo_attainment_drift"] = sum(slo.met(r) for r in hit) / max(len(hit), 1)
                        in_win = [r for r in reqs if win[0] <= r.arrival < win[1]]
                        m["exceed_rate_drift"] = float(np.mean([r.rid in bm.exceeded for r in in_win]))
                    calib = getattr(policy, "calib", None)
                    if calib is not None and calib.n:
                        m["online_miss_rate"] = calib.misses / calib.n
                    f.write(json.dumps(m) + "\n")
                    f.flush()
                    print(f"[{args.trace} seed={seed} rate={rate:3.1f}] {kind:13s} SLO={m['slo_attainment'] * 100:5.1f}% "
                          f"drift={m.get('slo_attainment_drift', 0) * 100:5.1f}% preempt/req={m['preemptions_per_req']:.2f} "
                          f"exceed={m['exceed_rate'] * 100:4.1f}% (drift {m.get('exceed_rate_drift', 0) * 100:4.1f}%)",
                          flush=True)


if __name__ == "__main__":
    main()
