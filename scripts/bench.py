"""Serve the same traces under every scheduling policy and record SLO metrics.

    python scripts/bench.py --mode real --rates 2 3 4 5 6       # real engine, measured
    python scripts/bench.py --mode sim  --rates 1 2 ... 10      # simulator (latency model)
"""
from __future__ import annotations

import argparse
import copy
import json
import pickle
from pathlib import Path

import torch

from pacer.costmodel import LinearModel
from pacer.engine import BlockManager, RealExecutor, SimExecutor
from pacer.metrics import summarize
from pacer.model import build_model
from pacer.request import SLO
from pacer.runtime import serve
from pacer.scheduler import ChunkedFixed, Pacer, PrefillFirst
from pacer.workload import WorkloadSpec, generate

NUM_BLOCKS, BLOCK_SIZE = 6144, 16


def policies(lat, slo, which: list[str]):
    table = {
        "prefill-first": lambda: PrefillFirst(),
        "chunked-64": lambda: ChunkedFixed(64),
        "chunked-128": lambda: ChunkedFixed(128),
        "chunked-256": lambda: ChunkedFixed(256),
        "chunked-512": lambda: ChunkedFixed(512),
        "chunked-edf-64": lambda: ChunkedFixed(64, edf=Pacer(lat, slo)),
        "chunked-edf-128": lambda: ChunkedFixed(128, edf=Pacer(lat, slo)),
        "chunked-edf-256": lambda: ChunkedFixed(256, edf=Pacer(lat, slo)),
        "chunked-edf-512": lambda: ChunkedFixed(512, edf=Pacer(lat, slo)),
        "pacer": lambda: Pacer(lat, slo),
        "pacer-no-slack": lambda: Pacer(lat, slo, use_slack=False, name="pacer w/o slack"),
        "pacer-no-edf": lambda: Pacer(lat, slo, use_edf=False, name="pacer w/o EDF"),
        "pacer-risk50": lambda: Pacer(lat, slo, coverage=0.5, name="pacer @50% coverage"),
        "pacer-risk80": lambda: Pacer(lat, slo, coverage=0.8, name="pacer @80% coverage"),
        "pacer-risk95": lambda: Pacer(lat, slo, coverage=0.95, name="pacer @95% coverage"),
        "pacer-risk99": lambda: Pacer(lat, slo, coverage=0.99, name="pacer @99% coverage"),
    }
    return [(k, table[k]) for k in which]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["real", "sim"], default="sim")
    ap.add_argument("--rates", type=float, nargs="+", default=[2, 3, 4, 5, 6])
    ap.add_argument("--policies", nargs="+", default=["prefill-first", "chunked-64", "chunked-128", "chunked-256",
                                                       "chunked-512", "pacer", "pacer-no-slack", "pacer-no-edf"])
    ap.add_argument("--num-requests", type=int, default=200)
    ap.add_argument("--burstiness", type=float, default=1.0)
    ap.add_argument("--ttft", type=float, default=2.0)
    ap.add_argument("--tpot", type=float, default=0.10)
    ap.add_argument("--seeds", type=int, nargs="+", default=[0])
    ap.add_argument("--sim-model", default="results/latency_model.json",
                    help="latency model the simulator replays (json=linear, pkl=pickled model)")
    ap.add_argument("--noise", type=float, default=0.0, help="simulator lognormal jitter")
    ap.add_argument("--preset", default="tiny")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--out", default=None)
    ap.add_argument("--tag", default="")
    args = ap.parse_args()

    slo = SLO(ttft=args.ttft, tpot=args.tpot)
    lat = LinearModel.load("results/latency_model.json")
    if args.sim_model.endswith(".pkl"):
        with open(args.sim_model, "rb") as f:
            sim_lat = pickle.load(f)
    else:
        sim_lat = LinearModel.load(args.sim_model)

    model = build_model(args.preset, device=args.device) if args.mode == "real" else None
    executor = None
    out_path = Path(args.out or f"results/bench_{args.mode}{args.tag}.jsonl")
    out_path.parent.mkdir(exist_ok=True)
    with open(out_path, "a") as f:
        for seed in args.seeds:
            for rate in args.rates:
                spec = WorkloadSpec(rate=rate, num_requests=args.num_requests, burstiness=args.burstiness, seed=seed)
                trace = generate(spec, vocab_size=model.cfg.vocab_size if model else None)
                for key, make in policies(lat, slo, args.policies):
                    reqs = copy.deepcopy(trace)
                    bm = BlockManager(NUM_BLOCKS, BLOCK_SIZE)
                    if args.mode == "real":
                        if executor is None:
                            executor = RealExecutor(model, bm, device=args.device)
                        executor.blocks = bm
                        ex = executor
                    else:
                        ex = SimExecutor(sim_lat, bm, noise=args.noise, seed=seed)
                    sched = make()
                    steps = serve(reqs, sched, ex, bm)
                    m = summarize(key, rate, reqs, slo, steps).as_dict()
                    m.update(mode=args.mode, seed=seed, burstiness=args.burstiness, slo_ttft=slo.ttft, slo_tpot=slo.tpot)
                    f.write(json.dumps(m) + "\n")
                    f.flush()
                    print(f"[{args.mode} seed={seed} rate={rate:4.1f}] {key:15s} SLO={m['slo_attainment']*100:5.1f}%  "
                          f"goodput={m['goodput']:.2f}/s  TTFT p99={m['ttft_p99']:.2f}s  TPOT p99={m['tpot_p99']*1e3:.0f}ms  "
                          f"steps={m['num_steps']} avg_tok={m['mean_step_tokens']:.0f}", flush=True)


if __name__ == "__main__":
    main()
