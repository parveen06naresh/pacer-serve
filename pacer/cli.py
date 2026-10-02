"""`pacer` command line: serve a real model through the engine, or run the experiments.

    pacer demo                      # 8 concurrent Shakespeare requests through Pacer, streamed stats
    pacer test                      # correctness suite
    pacer bench ...                 # scripts/bench.py (same flags)
    pacer profile ...               # scripts/profile_and_fit.py
    pacer figures                   # summarize results and redraw every figure
"""
from __future__ import annotations

import argparse
import runpy
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


def _script(name: str, argv: list[str]) -> None:
    sys.argv = [name] + argv
    runpy.run_path(str(ROOT / "scripts" / name), run_name="__main__")


def demo(argv: list[str]) -> None:
    import torch

    from .costmodel import LinearModel, OnlineConformal
    from .engine import BlockManager, RealExecutor
    from .metrics import summarize
    from .model import PRESETS, Transformer
    from .request import SLO, Request
    from .runtime import serve
    from .scheduler import Pacer

    ap = argparse.ArgumentParser(prog="pacer demo")
    ap.add_argument("--checkpoint", default=str(ROOT / "results" / "shakespeare.pt"))
    ap.add_argument("--tokens", type=int, default=160)
    ap.add_argument("--temperature", type=float, default=0.8)
    args = ap.parse_args(argv)

    ck = torch.load(args.checkpoint, map_location="cpu")
    vocab = ck["vocab"]
    stoi = {c: i for i, c in enumerate(vocab)}
    model = Transformer(PRESETS["shakespeare"])
    model.load_state_dict(ck["state_dict"])
    model.eval()

    prompts = ["ROMEO:\n", "JULIET:\nO ", "KING HENRY:\n", "First Citizen:\n", "HAMLET:\nTo ",
               "LADY MACBETH:\n", "PROSPERO:\n", "FALSTAFF:\n"]
    reqs = [Request(i, 0.05 * i, len(p), args.tokens, prompt_ids=[stoi[c] for c in p]) for i, p in enumerate(prompts)]
    bm = BlockManager(1024, 16)
    ex = RealExecutor(model, bm, temperature=args.temperature)
    slo = SLO(ttft=1.0, tpot=0.05)
    lat = LinearModel.load(ROOT / "results" / "latency_model.json")
    steps = serve(reqs, Pacer(lat, slo, online=OnlineConformal(0.1)), ex, bm)
    for p, r in zip(prompts, reqs):
        print("-" * 60)
        print(p + "".join(vocab[t] for t in r.output_ids))
    m = summarize("pacer", 0.0, reqs, slo, steps)
    print("-" * 60)
    print(f"{len(reqs)} concurrent requests, {sum(len(r.output_ids) for r in reqs)} tokens in {len(steps)} engine steps; "
          f"TTFT p50 {m.ttft_p50 * 1e3:.0f} ms, TPOT p50 {m.tpot_p50 * 1e3:.1f} ms (val loss {ck['val_loss']:.3f})")


def main() -> None:
    cmds = {
        "demo": demo,
        "bench": lambda a: _script("bench.py", a),
        "profile": lambda a: _script("profile_and_fit.py", a),
        "figures": lambda a: (_script("summarize.py", a), _script("roofline_projection.py", []),
                              _script("make_figures.py", [])),
        "test": lambda a: sys.exit(__import__("pytest").main([str(ROOT / "tests"), "-q", *a])),
    }
    if len(sys.argv) < 2 or sys.argv[1] not in cmds:
        print(__doc__)
        sys.exit(1)
    cmds[sys.argv[1]](sys.argv[2:])


if __name__ == "__main__":
    main()
