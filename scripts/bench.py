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

from pacer.costmodel import LinearModel, OnlineConformal
from pacer.engine import BlockManager, RealExecutor, SimExecutor
from pacer.metrics import summarize
from pacer.model import build_model
from pacer.request import SLO
from pacer.runtime import serve
from pacer.scheduler import ChunkedFixed, Pacer, PrefillFirst
from pacer.workload import WorkloadSpec, from_azure, generate

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
        "pacer-online": lambda: Pacer(lat, slo, online=OnlineConformal(alpha=0.1), name="pacer + online conformal"),
        "pacer-slack": lambda: Pacer(lat, slo, use_slack=True, name="pacer + slack banking"),
        "pacer-edf": lambda: Pacer(lat, slo, order="edf", name="pacer, EDF instead of Moore-Hodgson"),
        "pacer-fcfs": lambda: Pacer(lat, slo, use_edf=False, name="pacer, FCFS order"),
        "pacer-risk50": lambda: Pacer(lat, slo, coverage=0.5, name="pacer @50% coverage"),
        "pacer-risk80": lambda: Pacer(lat, slo, coverage=0.8, name="pacer @80% coverage"),
        "pacer-risk95": lambda: Pacer(lat, slo, coverage=0.95, name="pacer @95% coverage"),
        "pacer-risk99": lambda: Pacer(lat, slo, coverage=0.99, name="pacer @99% coverage"),
    }
    return [(k, table[k]) for k in which]


class Hog:
    """A noisy neighbour: a separate process that spins one core for `duty` of every 4 ms,
    switched on and off as the serving clock crosses the window (models a co-located
    job, thermal throttling, or a shared-GPU tenant). It stalls the engine's parallel
    regions, so even a fraction of one core costs far more than its share."""

    def __init__(self, window, duty: float, device: str = "cpu"):
        self.window, self.duty, self.device, self.proc = window, duty, device, None

    def __call__(self, now: float):
        inside = self.window[0] <= now < self.window[1]
        if inside and self.proc is None:
            import subprocess
            import sys
            d = self.duty
            if self.device.startswith("cuda"):
                # A co-tenant on the same GPU: matmul bursts for `duty` of every 20 ms.
                code = ("import time, torch\na = torch.randn(4096, 4096, device='cuda', dtype=torch.float16)\n"
                        "while True:\n    t = time.perf_counter()\n"
                        f"    while time.perf_counter() - t < {d} * 0.02:\n        a @ a; torch.cuda.synchronize()\n"
                        f"    time.sleep({1 - d} * 0.02)")
            else:
                code = ("import time\nwhile True:\n    t = time.perf_counter()\n"
                        f"    while time.perf_counter() - t < {d} * 0.004: pass\n    time.sleep({1 - d} * 0.004)")
            self.proc = subprocess.Popen([sys.executable, "-c", code])
        elif not inside and self.proc is not None:
            self.stop()

    def stop(self):
        if self.proc is not None:
            self.proc.kill()
            self.proc.wait()
            self.proc = None


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
    ap.add_argument("--dtype", default="float32", choices=["float32", "float16", "bfloat16"])
    ap.add_argument("--len-scale", type=float, default=1.0, help="multiply synthetic prompt/output lengths (GPU runs)")
    ap.add_argument("--latency-model", default="results/latency_model.json")
    ap.add_argument("--trace", default="synthetic", choices=["synthetic", "azure-conv", "azure-code"])
    ap.add_argument("--drift", type=float, nargs=3, metavar=("START", "END", "FACTOR"), default=None,
                    help="sim: steps between START and END (fractions of the trace span) run FACTOR x slower")
    ap.add_argument("--hog", type=float, nargs=3, metavar=("START", "END", "DUTY"), default=None,
                    help="real: run a noisy-neighbour process busy DUTY of the time between START and END "
                         "(fractions of span); DUTY 0.35 slows this engine ~1.5x, 0.5 ~1.75x")
    ap.add_argument("--out", default=None)
    ap.add_argument("--tag", default="")
    args = ap.parse_args()

    slo = SLO(ttft=args.ttft, tpot=args.tpot)
    lat = LinearModel.load(args.latency_model)
    if args.sim_model.endswith(".pkl"):
        with open(args.sim_model, "rb") as f:
            sim_lat = pickle.load(f)
    else:
        sim_lat = LinearModel.load(args.sim_model)

    model = build_model(args.preset, device=args.device, dtype=getattr(torch, args.dtype)) if args.mode == "real" else None
    executor = None
    out_path = Path(args.out or f"results/bench_{args.mode}{args.tag}.jsonl")
    out_path.parent.mkdir(exist_ok=True)
    with open(out_path, "a") as f:
        for seed in args.seeds:
            for rate in args.rates:
                vocab = model.cfg.vocab_size if model else None
                if args.trace == "synthetic":
                    k = args.len_scale
                    spec = WorkloadSpec(rate=rate, num_requests=args.num_requests, burstiness=args.burstiness, seed=seed,
                                        prompt_median=int(192 * k), prompt_max=int(1536 * k),
                                        output_median=int(48 * k), output_max=int(192 * k))
                    trace = generate(spec, vocab_size=vocab)
                else:
                    trace = from_azure(args.trace.split("-")[1], rate, args.num_requests, window=seed,
                                       vocab_size=vocab, seed=seed)
                span = trace[-1].arrival - trace[0].arrival
                window = None
                if args.drift:
                    window = (args.drift[0] * span, args.drift[1] * span)
                elif args.hog:
                    window = (args.hog[0] * span, args.hog[1] * span)
                for key, make in policies(lat, slo, args.policies):
                    reqs = copy.deepcopy(trace)
                    bm = BlockManager(NUM_BLOCKS, BLOCK_SIZE)
                    if args.mode == "real":
                        if executor is None:
                            executor = RealExecutor(model, bm, device=args.device)
                        executor.blocks = bm
                        ex = executor
                    else:
                        slow = None
                        if args.drift:
                            a, b, k = window[0], window[1], args.drift[2]
                            slow = lambda now, a=a, b=b, k=k: k if a <= now < b else 1.0  # noqa: E731
                        ex = SimExecutor(sim_lat, bm, noise=args.noise, seed=seed, slowdown=slow)
                    sched = make()
                    hog = Hog(window, args.hog[2], args.device) if (args.hog and args.mode == "real") else None
                    try:
                        steps = serve(reqs, sched, ex, bm, on_time=hog)
                    finally:
                        if hog:
                            hog.stop()
                    m = summarize(key, rate, reqs, slo, steps).as_dict()
                    m.update(mode=args.mode, seed=seed, burstiness=args.burstiness, slo_ttft=slo.ttft, slo_tpot=slo.tpot,
                             trace=args.trace, drift=args.drift, hog=args.hog)
                    if window:
                        # Requests that were in flight while the hardware was degraded.
                        hit = [r for r in reqs if r.arrival < window[1] and r.finish_time >= window[0]]
                        m["slo_attainment_drift"] = sum(slo.met(r) for r in hit) / max(len(hit), 1)
                        m["tpot_attainment_drift"] = sum(r.tpot() <= slo.tpot for r in hit) / max(len(hit), 1)
                    if getattr(sched, "online", None) is not None:
                        m["online_miss_rate"] = sched.online.misses / max(sched.online.n, 1)
                    f.write(json.dumps(m) + "\n")
                    f.flush()
                    print(f"[{args.mode} seed={seed} rate={rate:4.1f}] {key:15s} SLO={m['slo_attainment']*100:5.1f}%  "
                          f"goodput={m['goodput']:.2f}/s  TTFT p99={m['ttft_p99']:.2f}s  TPOT p99={m['tpot_p99']*1e3:.0f}ms  "
                          f"steps={m['num_steps']} avg_tok={m['mean_step_tokens']:.0f}", flush=True)


if __name__ == "__main__":
    main()
