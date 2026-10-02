"""Executors: run one scheduled step and report how long it took.

`RealExecutor` runs the transformer over a paged KV cache and times it with a wall
clock. `SimExecutor` asks a latency model instead, so the exact same serving loop can
replay hours of traffic in seconds. Both share the `BlockManager`, so memory behaviour
is identical between the two.
"""
from __future__ import annotations

import time
from dataclasses import dataclass

import torch

from .model import StepInput, Transformer
from .request import Request


@dataclass(frozen=True)
class StepShape:
    """What a latency model sees about a step: (new_tokens, context_after) per sequence."""
    prefill: tuple[tuple[int, int], ...]   # (chunk_len, ctx_len_after_chunk)
    decode: tuple[int, ...]                # ctx_len_after for each decode token

    @property
    def num_tokens(self) -> int:
        return sum(n for n, _ in self.prefill) + len(self.decode)

    @staticmethod
    def of(batch: list[tuple[Request, int]]) -> StepShape:
        pre, dec = [], []
        for r, n in batch:
            if r.in_prefill:
                pre.append((n, r.num_prefilled + n))
            else:
                dec.append(r.context_len + 1)
        return StepShape(tuple(pre), tuple(dec))


class BlockManager:
    """Paged KV-cache allocator. Reserves a request's full footprint on admission so
    no step can run out of memory mid-flight (no preemption needed)."""

    def __init__(self, num_blocks: int, block_size: int):
        self.block_size = block_size
        self.num_blocks = num_blocks
        self.free = list(range(num_blocks - 1, -1, -1))

    def blocks_needed(self, r: Request) -> int:
        return -(-(r.prompt_len + r.max_new_tokens) // self.block_size)

    def can_admit(self, r: Request) -> bool:
        return len(self.free) >= self.blocks_needed(r)

    def admit(self, r: Request) -> None:
        n = self.blocks_needed(r)
        r.blocks = [self.free.pop() for _ in range(n)]

    def release(self, r: Request) -> None:
        self.free.extend(reversed(r.blocks))
        r.blocks = []

    def slots(self, r: Request, start: int, end: int) -> torch.Tensor:
        pos = torch.arange(start, end)
        blocks = torch.tensor(r.blocks, dtype=torch.long)
        return blocks[pos // self.block_size] * self.block_size + pos % self.block_size

    @property
    def utilization(self) -> float:
        return 1.0 - len(self.free) / self.num_blocks


def _apply_progress(batch: list[tuple[Request, int]], new_tokens: list[int | None], t_end: float) -> None:
    for (r, n), tok in zip(batch, new_tokens):
        if r.in_prefill:
            r.num_prefilled += n
            if r.in_prefill:      # chunk did not finish the prompt: no token sampled
                continue
        r.output_ids.append(tok if tok is not None else 0)
        r.token_times.append(t_end)


class RealExecutor:
    def __init__(self, model: Transformer, blocks: BlockManager, device: str = "cpu",
                 temperature: float = 0.0, seed: int = 0):
        self.model = model
        self.blocks = blocks
        self.device = device
        self.temperature = temperature  # 0 = greedy (what the exactness tests use)
        self.gen = torch.Generator(device=device).manual_seed(seed)
        cfg = model.cfg
        dtype = next(model.parameters()).dtype
        shape = (blocks.num_blocks, blocks.block_size, cfg.n_kv_heads, cfg.head_dim)
        self.k = [torch.zeros(shape, dtype=dtype, device=device) for _ in range(cfg.n_layers)]
        self.v = [torch.zeros(shape, dtype=dtype, device=device) for _ in range(cfg.n_layers)]

    def build_input(self, batch: list[tuple[Request, int]]) -> StepInput:
        ids, pos, slots, spans, tables, lens = [], [], [], [], [], []
        off = 0
        for r, n in batch:
            if r.in_prefill:
                start = r.num_prefilled
                ids.extend(r.prompt_ids[start:start + n])
            else:
                start = r.context_len
                ids.append(r.output_ids[-1])
            end = start + n
            pos.extend(range(start, end))
            slots.append(self.blocks.slots(r, start, end))
            tables.append(r.blocks)
            lens.append(end)
            spans.append((off, off + n))
            off += n
        dev = self.device
        return StepInput(
            token_ids=torch.tensor(ids, device=dev),
            positions=torch.tensor(pos, device=dev),
            slot_mapping=torch.cat(slots).to(dev),
            seq_spans=spans,
            block_tables=tables,
            ctx_lens=lens,
            logits_idx=torch.tensor([b - 1 for _, b in spans], device=dev),
        )

    def _sync(self):
        if self.device.startswith("cuda"):
            torch.cuda.synchronize()

    def execute(self, batch: list[tuple[Request, int]], now: float) -> float:
        t0 = time.perf_counter()
        inp = self.build_input(batch)
        logits = self.model.forward_step(inp, self.k, self.v)
        if self.temperature > 0:
            probs = torch.softmax(logits / self.temperature, dim=-1)
            toks = torch.multinomial(probs, 1, generator=self.gen).squeeze(-1).tolist()
        else:
            toks = logits.argmax(-1).tolist()
        self._sync()
        dt = time.perf_counter() - t0
        _apply_progress(batch, toks, now + dt)
        return dt

    def time_shape(self, batch: list[tuple[Request, int]]) -> float:
        """Time a step without mutating request state (used by the profiler)."""
        inp = self.build_input(batch)
        self._sync()
        t0 = time.perf_counter()
        self.model.forward_step(inp, self.k, self.v).argmax(-1).tolist()
        self._sync()
        return time.perf_counter() - t0


class SimExecutor:
    """Replays a step using a latency model. Optional multiplicative lognormal noise
    reproduces the jitter of real hardware."""

    def __init__(self, latency_model, blocks: BlockManager, noise: float = 0.0, seed: int = 0, slowdown=None):
        import numpy as np
        self.lat = latency_model
        self.blocks = blocks
        self.noise = noise
        self.slowdown = slowdown  # optional f(now) -> multiplier, models hardware drift
        self.rng = np.random.default_rng(seed)

    def execute(self, batch: list[tuple[Request, int]], now: float) -> float:
        dt = float(self.lat.predict(StepShape.of(batch)))
        if self.noise:
            dt *= float(self.rng.lognormal(0.0, self.noise))
        if self.slowdown is not None:
            dt *= self.slowdown(now)
        _apply_progress(batch, [None] * len(batch), now + dt)
        return dt
