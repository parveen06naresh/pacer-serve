"""Profile the engine, fit the four latency models, and report their accuracy.

    python scripts/profile_and_fit.py --shapes 600
"""
from __future__ import annotations

import argparse
import json
import pickle
from pathlib import Path

import numpy as np
import torch

from pacer.costmodel import (GBDTModel, HybridModel, LinearModel, RooflineModel,
                             conformal_ratio, featurize_many)
from pacer.engine import BlockManager, RealExecutor
from pacer.model import PRESETS, build_model
from pacer.profiler import measure_roofline, profile_engine


def apes(y, p):
    return np.abs(p - y) / y


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--preset", default="tiny")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--shapes", type=int, default=600)
    ap.add_argument("--out", default="results")
    ap.add_argument("--refit", action="store_true", help="reuse results/profile.pkl instead of profiling")
    args = ap.parse_args()
    out = Path(args.out)
    out.mkdir(exist_ok=True)

    if args.refit:
        with open(out / "profile.pkl", "rb") as f:
            d = pickle.load(f)
        shapes, y, hw = d["shapes"], d["latency"], d["hw"]
    else:
        hw = measure_roofline(args.device)
        print(f"roofline: {hw['peak_flops']/1e9:.0f} GFLOP/s, {hw['bandwidth']/1e9:.1f} GB/s")
        model = build_model(args.preset, device=args.device)
        ex = RealExecutor(model, BlockManager(num_blocks=8192, block_size=16), device=args.device)
        shapes, y = profile_engine(ex, args.shapes, seed=0)
        with open(out / "profile.pkl", "wb") as f:
            pickle.dump({"shapes": shapes, "latency": y, "hw": hw, "preset": args.preset, "device": args.device}, f)
    X = featurize_many(shapes)

    rng = np.random.default_rng(1)
    perm = rng.permutation(len(y))
    tr, te = perm[: int(0.8 * len(y))], perm[int(0.8 * len(y)):]
    # Extrapolation split: train only on small steps, test on large ones.
    small = X[:, 1] <= 512
    splits = {"random 80/20": (tr, te), "extrapolate (train <=512 tok, test >512)": (np.where(small)[0], np.where(~small)[0])}

    report = {"hardware": hw, "num_shapes": len(y), "device": args.device, "preset": args.preset, "models": {}}
    preds_random = {}
    for split, (a, b) in splits.items():
        print(f"\n[{split}] train={len(a)} test={len(b)}")
        models = [RooflineModel(PRESETS[args.preset], hw["peak_flops"], hw["bandwidth"]).fit(X[a], y[a]),
                  LinearModel().fit(X[a], y[a]), GBDTModel().fit(X[a], y[a]), HybridModel().fit(X[a], y[a])]
        for m in models:
            p = m.predict_X(X[b])
            e = apes(y[b], p)
            row = {"mape": float(e.mean()), "p50_ape": float(np.median(e)), "p90_ape": float(np.percentile(e, 90))}
            report["models"].setdefault(m.name, {})[split] = row
            print(f"  {m.name:9s} MAPE={row['mape']*100:5.1f}%  p50={row['p50_ape']*100:5.1f}%  p90={row['p90_ape']*100:5.1f}%")
            if split.startswith("random"):
                preds_random[m.name] = p

    # Conformal calibration: fit on one half of the training split, calibrate on the
    # other half, and check the promised coverage on the untouched test split.
    fit_idx, cal_idx = tr[: len(tr) // 2], tr[len(tr) // 2:]
    lin = LinearModel().fit(X[fit_idx], y[fit_idx])
    conformal, coverage = {}, {}
    print("\nconformal latency bounds (linear model):")
    for cov in [0.5, 0.8, 0.9, 0.95, 0.99]:
        q = conformal_ratio(lin, X[cal_idx], y[cal_idx], cov)
        emp = float(np.mean(y[te] <= q * lin.predict_X(X[te])))
        conformal[cov], coverage[cov] = q, emp
        print(f"  target {cov*100:4.0f}%  ratio bound {q:.3f}  empirical test coverage {emp*100:5.1f}%")
    report["conformal"] = {str(k): {"ratio": conformal[k], "test_coverage": coverage[k]} for k in conformal}

    final = LinearModel().fit(X, y)
    final.save(out / "latency_model.json", conformal={str(k): v for k, v in conformal.items()})
    report["linear_coefficients"] = final.describe()
    print("\nlinear model (all data):")
    for k, v in final.describe().items():
        print(f"  {k:13s} {v:.3e}")
    with open(out / "costmodel_report.json", "w") as f:
        json.dump(report, f, indent=2)
    np.savez(out / "costmodel_preds.npz", y=y[te], tokens=X[te, 1], **preds_random)
    with open(out / "hybrid_model.pkl", "wb") as f:
        pickle.dump(HybridModel().fit(X, y), f)


if __name__ == "__main__":
    main()
