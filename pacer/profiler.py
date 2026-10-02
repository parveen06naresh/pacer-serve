"""Profile the real engine over randomly sampled step shapes and measure the
hardware roofline (peak GEMM FLOP/s, memory bandwidth)."""
from __future__ import annotations

import time

import numpy as np
import torch

from .engine import BlockManager, RealExecutor, StepShape
from .request import Request


def measure_roofline(device: str = "cpu", reps: int = 5) -> dict:
    def best(fn):
        fn()
        ts = []
        for _ in range(reps):
            t0 = time.perf_counter()
            fn()
            if device.startswith("cuda"):
                torch.cuda.synchronize()
            ts.append(time.perf_counter() - t0)
        return min(ts)

    n = 2048
    a, b = torch.randn(n, n, device=device), torch.randn(n, n, device=device)
    flops = 2 * n ** 3 / best(lambda: a @ b)
    x = torch.empty(64 * 2 ** 20, device=device)  # 256 MB
    y = torch.empty_like(x)
    bw = 2 * x.numel() * 4 / best(lambda: y.copy_(x))
    return {"peak_flops": flops, "bandwidth": bw}


def sample_shape(rng: np.random.Generator, max_tokens: int = 2048, max_ctx: int = 1600) -> StepShape:
    kind = rng.choice(["decode", "prefill", "mixed"], p=[0.3, 0.2, 0.5])
    dec: list[int] = []
    pre: list[tuple[int, int]] = []
    if kind in ("decode", "mixed"):
        k = int(rng.integers(1, 65))
        dec = rng.integers(16, max_ctx, size=k).tolist()
    if kind in ("prefill", "mixed"):
        budget = int(np.exp(rng.uniform(np.log(1), np.log(max_tokens - len(dec)))))
        while budget > 0 and len(pre) < 4:
            n = int(rng.integers(1, budget + 1)) if len(pre) < 3 else budget
            done = int(rng.integers(0, max(1, max_ctx - n)))
            pre.append((n, done + n))
            budget -= n
    if not dec and not pre:
        dec = [int(rng.integers(16, max_ctx))]
    return StepShape(tuple(pre), tuple(dec))


def _requests_for(shape: StepShape, bm: BlockManager, vocab: int, rng) -> list[tuple[Request, int]]:
    batch = []
    for i, (n, ctx) in enumerate(shape.prefill):
        r = Request(i, 0.0, prompt_len=ctx, max_new_tokens=1, prompt_ids=rng.integers(0, vocab, ctx).tolist())
        r.num_prefilled = ctx - n
        bm.admit(r)
        batch.append((r, n))
    for j, ctx in enumerate(shape.decode):
        # A decode step at ctx_after=ctx: prompt fully cached, one generated token pending.
        r = Request(1000 + j, 0.0, prompt_len=ctx - 1, max_new_tokens=2, prompt_ids=None)
        r.num_prefilled = ctx - 1
        r.output_ids = [int(rng.integers(0, vocab))]
        r.token_times = [0.0]
        bm.admit(r)
        batch.append((r, 1))
    return batch


def profile_engine(executor: RealExecutor, num_shapes: int, seed: int = 0, repeats: int = 3,
                   shape_sampler=sample_shape, log_every: int = 50):
    rng = np.random.default_rng(seed)
    bm = executor.blocks
    shapes, lat = [], []
    for _ in range(3):  # warm up allocator / threads
        s = shape_sampler(rng)
        b = _requests_for(s, bm, executor.model.cfg.vocab_size, rng)
        executor.time_shape(b)
        for r, _ in b:
            bm.release(r)
    for k in range(num_shapes):
        s = shape_sampler(rng)
        b = _requests_for(s, bm, executor.model.cfg.vocab_size, rng)
        assert StepShape.of(b) == s, (StepShape.of(b), s)
        t = float(np.median([executor.time_shape(b) for _ in range(repeats)]))
        for r, _ in b:
            bm.release(r)
        shapes.append(s)
        lat.append(t)
        if log_every and (k + 1) % log_every == 0:
            print(f"  profiled {k + 1}/{num_shapes} shapes", flush=True)
    return shapes, np.array(lat)
