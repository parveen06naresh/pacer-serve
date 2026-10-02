"""One command to reproduce Pacer's key results on an NVIDIA GPU (e.g. a free Colab T4).

    python scripts/run_gpu.py                  # ~30-40 min on a T4, writes results_gpu/

1. Profiles the ~0.5B `small` model in FP16 on the GPU and fits + calibrates the latency model.
2. Derives SLOs from the measured hardware (TPOT = 3x a 32-request decode step).
3. Uses the simulator to find the load where the best fixed budget starts failing.
4. Runs the real engine around that load: steady state, then with a GPU co-tenant
   (a second process issuing matmul bursts on the same GPU) for the drift result.
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "results_gpu"


def run(args: list[str]) -> None:
    print("+", " ".join(args), flush=True)
    subprocess.run([sys.executable, *args], cwd=ROOT, check=True, env={**__import__("os").environ, "PYTHONPATH": str(ROOT)})


def capacity(rows, policy, target=0.9):
    pts = sorted((r["rate"], r["slo_attainment"]) for r in rows if r["policy"] == policy)
    best = pts[0][0]
    for rate, att in pts:
        if att < target:
            break
        best = rate
    return best


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--preset", default="small")
    ap.add_argument("--dtype", default="float16")
    ap.add_argument("--shapes", type=int, default=400)
    ap.add_argument("--len-scale", type=float, default=2.0)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--quick", action="store_true", help="tiny smoke run (for CI / CPU checks)")
    ap.add_argument("--out", default=str(OUT))
    args = ap.parse_args()
    out = Path(args.out)
    out.mkdir(exist_ok=True)
    n = ["--num-requests", "40"] if args.quick else []
    lat_path = out / "latency_model.json"

    if not lat_path.exists():
        run(["scripts/profile_and_fit.py", "--preset", args.preset, "--dtype", args.dtype, "--device", args.device,
             "--shapes", str(min(args.shapes, 60) if args.quick else args.shapes), "--out", str(out), "--max-ctx", str(int(800 * args.len_scale))])

    from pacer.costmodel import LinearModel
    from pacer.engine import StepShape
    lat = LinearModel.load(lat_path)
    decode_step = lat.predict(StepShape((), tuple([int(300 * args.len_scale)] * 32)))
    tpot = round(3 * decode_step, 4)
    ttft = round(max(20 * tpot, 0.5), 3)
    common = ["--preset", args.preset, "--dtype", args.dtype, "--device", args.device, "--len-scale", str(args.len_scale),
              *n, "--latency-model", str(lat_path), "--sim-model", str(lat_path), "--tpot", str(tpot), "--ttft", str(ttft)]
    json.dump({"tpot": tpot, "ttft": ttft, "decode_step_32": decode_step}, open(out / "slo.json", "w"), indent=2)
    print(f"SLOs derived from hardware: TPOT {tpot * 1e3:.1f} ms, TTFT {ttft:.2f} s", flush=True)

    # Find the interesting load range in the simulator (seconds of compute).
    sim = out / "sim_scan.jsonl"
    if not sim.exists():
        base = 1.0 / (decode_step * 10)
        rates = [round(base * 1.25 ** i, 2) for i in range(6 if args.quick else 16)]
        run(["scripts/bench.py", "--mode", "sim", "--rates", *map(str, rates), "--policies", "chunked-128", "pacer",
             "--out", str(sim), *common])
    rows = [json.loads(line) for line in open(sim)]
    knee = capacity(rows, "chunked-128")
    rates = [round(knee * f, 2) for f in ((1.1,) if args.quick else (0.9, 1.1, 1.3, 1.5))]
    print(f"best fixed budget starts failing near {knee} req/s; real runs at {rates}", flush=True)

    pols = ["chunked-128", "chunked-edf-128", "pacer", "pacer-online"]
    run(["scripts/bench.py", "--mode", "real", "--rates", *map(str, rates), "--policies", *pols,
         "--out", str(out / "bench_real.jsonl"), *common])
    run(["scripts/bench.py", "--mode", "real", "--rates", str(rates[min(1, len(rates) - 1)]), "--hog", "0.3", "0.7", "0.5",
         "--policies", *pols, "--out", str(out / "bench_real_hog.jsonl"), *common])
    run(["scripts/summarize.py", str(out / "bench_real.jsonl"), str(out / "bench_real_hog.jsonl")])
    print(f"done: results in {out}", flush=True)


if __name__ == "__main__":
    main()
