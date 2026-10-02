"""The serving loop shared by real execution and simulation.

Time is a virtual clock advanced by the *measured* cost of each step (scheduling
overhead plus model execution), and jumped forward when the engine is idle. On a
dedicated machine this is equivalent to wall-clock serving, but it makes runs
reproducible and lets the simulator reuse the exact same code path.
"""
from __future__ import annotations

import time

from .engine import BlockManager, StepShape
from .request import Request
from .scheduler import Scheduler


def serve(reqs: list[Request], scheduler: Scheduler, executor, bm: BlockManager,
          step_log: list | None = None, on_time=None) -> list[tuple[float, int]]:
    pending = sorted(reqs, key=lambda r: r.arrival)
    i, n = 0, len(pending)
    waiting: list[Request] = []
    running: list[Request] = []
    now = pending[0].arrival
    steps: list[tuple[float, int]] = []
    observe = getattr(scheduler, "observe", None)
    while i < n or waiting or running:
        if on_time is not None:
            on_time(now)
        while i < n and pending[i].arrival <= now:
            waiting.append(pending[i])
            i += 1
        t0 = time.perf_counter()
        batch = scheduler.schedule(now, waiting, running, bm)
        overhead = time.perf_counter() - t0
        now += overhead
        if not batch:
            if i < n:
                now = max(now, pending[i].arrival)
                continue
            raise RuntimeError("scheduler made no progress with work outstanding")
        shape = StepShape.of(batch) if observe else None
        dt = executor.execute(batch, now)
        if observe:
            observe(shape, dt + overhead)  # users feel scheduling time too
        now += dt
        ntok = sum(k for _, k in batch)
        steps.append((dt, ntok))
        if step_log is not None:
            step_log.append({"t": now, "dt": dt, "tokens": ntok, "seqs": len(batch),
                             "target": getattr(scheduler, "last_target", None),
                             "decodes": sum(1 for r, k in batch if k == 1 and r.num_prefilled == r.prompt_len)})
        for r, _ in batch:
            if r.done and r.finish_time is None:
                r.finish_time = now
                bm.release(r)
                running.remove(r)
    return steps
