"""Analytic roofline for one serving step of Llama-3-8B on NVIDIA GPUs.

Why chunked prefill works, in one number: a decode step must stream every weight from
HBM no matter how many tokens it carries, so extra tokens are nearly free until the
step's arithmetic intensity reaches the GPU's ridge point (peak FLOP/s / bandwidth).
This script computes that "free token" headroom per GPU, including the KV-cache reads
that eat into it as context grows. These are first-principles projections from
datasheet peaks, not measurements; real kernels reach roughly 60-80% of either roof.
"""
from __future__ import annotations

import json
from pathlib import Path

from pacer.model import PRESETS, model_flops_per_token

# Dense BF16 tensor-core peak (no sparsity) and HBM bandwidth, from NVIDIA datasheets.
GPUS = {
    "H100 SXM": (989e12, 3.35e12),
    "A100 80GB SXM": (312e12, 2.039e12),
    "L40S": (362e12, 0.864e12),
    "L4": (121e12, 0.300e12),
}


def step_time(cfg, peak, bw, prefill_tokens, n_decode, ctx, bytes_per_el=2):
    weights = (cfg.n_params() - cfg.vocab_size * cfg.d_model) * bytes_per_el
    kv_tok = 2 * cfg.n_layers * cfg.n_kv_heads * cfg.head_dim * bytes_per_el
    tokens = prefill_tokens + n_decode
    flops = tokens * model_flops_per_token(cfg) + 4 * cfg.n_layers * cfg.d_model * n_decode * ctx
    bytes_ = weights + n_decode * ctx * kv_tok
    return max(flops / peak, bytes_ / bw), flops / peak, bytes_ / bw


def main():
    cfg = PRESETS["llama3-8b"]
    out = {}
    print(f"Llama-3-8B: {cfg.n_params()/1e9:.2f}B params, BF16")
    print(f"{'GPU':14s} {'ridge':>7s} {'decode-only step':>17s} {'free prefill tokens (32 decodes @ ctx 2k)':>42s}")
    for name, (peak, bw) in GPUS.items():
        ridge = peak / bw
        base, _, _ = step_time(cfg, peak, bw, 0, 32, 2048)
        # Largest prefill chunk that keeps the step memory-bound (i.e. costs ~nothing extra).
        free = 0
        for c in range(0, 8192, 8):
            t, tc, tm = step_time(cfg, peak, bw, c, 32, 2048)
            if tc <= tm:
                free = c
        curve = []
        for c in [0, 64, 128, 256, 512, 1024, 2048, 4096]:
            t, _, _ = step_time(cfg, peak, bw, c, 32, 2048)
            curve.append({"prefill_tokens": c, "step_ms": t * 1e3})
        out[name] = {"peak_tflops": peak / 1e12, "bandwidth_tbs": bw / 1e12, "ridge_flop_per_byte": ridge,
                     "decode_only_step_ms": base * 1e3, "free_prefill_tokens": free, "curve": curve}
        print(f"{name:14s} {ridge:7.0f} {base*1e3:14.2f} ms {free:42d}")
    Path("results").mkdir(exist_ok=True)
    with open("results/roofline_projection.json", "w") as f:
        json.dump(out, f, indent=2)


if __name__ == "__main__":
    main()
