"""Batch schedulers. Each step a scheduler decides which requests run and how many
tokens each contributes, under a paged-KV memory budget.

* `PrefillFirst`  vLLM's original policy: whole-prompt prefills preempt decoding.
                  Great throughput, but every arrival stalls all running decodes.
* `ChunkedFixed`  Sarathi-Serve style: decodes always run; prefills are chunked to
                  fill a fixed per-step token budget. The budget is a magic number
                  that must be re-tuned per model, GPU and workload.
* `Pacer`         (this project) picks the token budget *every step* by searching a
                  learned latency model for the largest step that still meets each
                  running request's TPOT deadline (including the slack it has banked),
                  and orders prefills earliest-deadline-first while deprioritising
                  requests that can no longer meet their TTFT SLO.
"""
from __future__ import annotations

from .engine import BlockManager, StepShape
from .request import SLO, Request

Batch = list[tuple[Request, int]]


def _admit_next(waiting: list[Request], running: list[Request], bm: BlockManager, r: Request) -> bool:
    if r in running:
        return True
    if not bm.can_admit(r):
        return False
    bm.admit(r)
    waiting.remove(r)
    running.append(r)
    return True


class Scheduler:
    name = "base"

    def schedule(self, now: float, waiting: list[Request], running: list[Request], bm: BlockManager) -> Batch:
        raise NotImplementedError


class PrefillFirst(Scheduler):
    def __init__(self, max_batched_tokens: int = 2048):
        self.max_tokens = max_batched_tokens
        self.name = "prefill-first (vLLM v0)"

    def schedule(self, now, waiting, running, bm):
        batch, used = [], 0
        for r in list(waiting):
            if batch and used + r.prompt_len > self.max_tokens:
                break
            if not _admit_next(waiting, running, bm, r):
                break
            batch.append((r, r.prompt_len))
            used += r.prompt_len
        if batch:
            return batch
        return [(r, 1) for r in running if not r.in_prefill]


class ChunkedFixed(Scheduler):
    def __init__(self, token_budget: int = 256, edf: "Pacer | None" = None):
        self.budget = token_budget
        self.edf = edf  # borrow Pacer's deadline ordering (a stronger baseline)
        self.name = f"chunked-{token_budget}" + ("+EDF" if edf else " (Sarathi)")

    def schedule(self, now, waiting, running, bm):
        batch: Batch = [(r, 1) for r in running if not r.in_prefill]
        left = max(self.budget - len(batch), 0)
        cands = [r for r in running if r.in_prefill] + list(waiting)
        if self.edf is not None:
            cands = self.edf._prefill_order(now, cands)
        for r in cands:
            if left <= 0:
                break
            if not _admit_next(waiting, running, bm, r):
                break
            n = min(r.remaining_prefill, left)
            batch.append((r, n))
            left -= n
        return batch


class Pacer(Scheduler):
    def __init__(self, latency_model, slo: SLO, margin: float | None = None, coverage: float = 0.9,
                 max_batched_tokens: int = 2048, use_slack: bool = False, use_edf: bool = True,
                 online=None, order: str = "mh", name: str | None = None):
        self.lat = latency_model
        self.slo = slo
        # Risk dial: by default the margin comes from the model's conformal bound, so a
        # step predicted to fit actually fits with probability >= `coverage`.
        if margin is None:
            bounds = getattr(latency_model, "conformal", {}) or {}
            margin = 1.0 / bounds[coverage] if coverage in bounds else 0.85
        self.margin = margin
        self.online = online  # OnlineConformal: re-calibrates the margin from live step timings
        self.max_tokens = max_batched_tokens
        self.use_slack = use_slack
        self.use_edf = use_edf
        self.order = order  # "edf" or "mh" (Moore-Hodgson: fewest TTFT misses)
        self.name = name or "pacer (ours)"
        self.last_target = 0.0

    def _margin(self) -> float:
        if self.online is not None and self.online.ready:
            return 1.0 / self.online.bound()
        return self.margin

    def _scale(self) -> float:
        """Typical actual/predicted ratio right now (1.0 without online calibration)."""
        if self.online is not None and self.online.ready:
            return self.online.center()
        return 1.0

    def observe(self, shape: StepShape, actual: float) -> None:
        if self.online is not None:
            self.online.update(actual / max(self.lat.predict(shape), 1e-6))

    # How long may this step take without breaking any running request's TPOT SLO?
    def _latency_target(self, now: float, decodes: list[Request]) -> float:
        if not decodes:
            return float("inf")
        if not self.use_slack:
            return self._margin() * self.slo.tpot
        # After k tokens, the next one keeps mean TPOT <= SLO iff it lands by first + SLO * k.
        slack = min(r.first_token_time + self.slo.tpot * len(r.token_times) - now for r in decodes)
        return self._margin() * max(slack, 0.0)

    def _prefill_order(self, now: float, cands: list[Request]) -> list[Request]:
        if not self.use_edf:
            return cands
        scale = self._scale()
        if self.order == "mh":
            return self._moore_hodgson(now, cands, scale)

        def key(r: Request):
            deadline = r.arrival + self.slo.ttft
            # Lower bound on its TTFT: its remaining prompt as one chunk on an idle engine.
            best_case = scale * self.lat.predict(StepShape(((r.remaining_prefill, r.prompt_len),), ()))
            late = now + best_case > deadline
            return (late, deadline if not late else r.arrival)
        return sorted(cands, key=key)

    def _moore_hodgson(self, now: float, cands: list[Request], scale: float) -> list[Request]:
        """Order prefills to minimise the number of TTFT-SLO misses.

        Treating the prefill stream as one machine, "maximise requests that meet their
        deadline" is the scheduling problem 1||sum U_j, which Moore-Hodgson (1968) solves
        exactly in O(n log n): walk jobs in deadline order and, whenever the running
        completion time overshoots the current deadline, evict the longest job so far.
        Processing times come from the latency model's marginal cost of the prompt.
        """
        import heapq
        c = self.lat.coef if hasattr(self.lat, "coef") else None

        def cost(r: Request) -> float:
            n, L = r.remaining_prefill, r.prompt_len
            if c is not None:  # marginal seconds of this chunk inside a shared step
                return scale * (c[1] * n + c[2] * n * L + c[5])
            return scale * self.lat.predict(StepShape(((n, L),), ()))

        on_time, late, heap, t = [], [], [], 0.0
        for r in sorted(cands, key=lambda r: r.arrival):
            p = cost(r)
            heapq.heappush(heap, (-p, r.rid, r))
            t += p
            if now + t > r.arrival + self.slo.ttft:
                neg_p, _, evicted = heapq.heappop(heap)
                t += neg_p
                late.append(evicted)
        kept = {r.rid for _, _, r in heap}
        on_time = [r for r in sorted(cands, key=lambda r: r.arrival) if r.rid in kept]
        return on_time + sorted(late, key=lambda r: r.arrival)

    @staticmethod
    def _pack(order: list[Request], budget: int) -> list[tuple[Request, int]]:
        out = []
        for r in order:
            if budget <= 0:
                break
            n = min(r.remaining_prefill, budget)
            out.append((r, n))
            budget -= n
        return out

    def schedule(self, now, waiting, running, bm):
        decodes = [r for r in running if not r.in_prefill]
        dec_shape = tuple(r.context_len + 1 for r in decodes)
        order = self._prefill_order(now, [r for r in running if r.in_prefill] + list(waiting))

        # Only requests we can actually hold in KV memory are candidates this step.
        free = len(bm.free)
        fits = []
        for r in order:
            if r in running:
                fits.append(r)
            elif bm.blocks_needed(r) <= free:
                free -= bm.blocks_needed(r)
                fits.append(r)

        def shape(c: int) -> StepShape:
            pre = tuple((n, r.num_prefilled + n) for r, n in self._pack(fits, c))
            return StepShape(pre, dec_shape)

        target = self._latency_target(now, decodes)
        hi = min(self.max_tokens - len(decodes), sum(r.remaining_prefill for r in fits))
        if target == float("inf"):
            c = hi
        else:
            # Largest prefill budget whose predicted step latency fits the target.
            lo, c = 0, 0
            while lo <= hi:
                mid = (lo + hi) // 2
                if self.lat.predict(shape(mid)) <= target:
                    c, lo = mid, mid + 1
                else:
                    hi = mid - 1
        self.last_target = target

        batch: Batch = [(r, 1) for r in decodes]
        for r, n in self._pack(fits, c):
            if not _admit_next(waiting, running, bm, r):
                break
            batch.append((r, n))
        return batch
