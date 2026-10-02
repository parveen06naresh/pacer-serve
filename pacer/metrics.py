from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np

from .request import SLO, Request


@dataclass
class RunMetrics:
    policy: str
    rate: float
    num_requests: int
    duration: float
    slo_attainment: float        # fraction of requests meeting both TTFT and TPOT SLOs
    goodput: float               # SLO-meeting requests per second
    throughput_tok: float        # output tokens per second
    ttft_p50: float
    ttft_p99: float
    tpot_p50: float
    tpot_p99: float
    ttft_attainment: float
    tpot_attainment: float
    mean_step_tokens: float
    num_steps: int

    def as_dict(self) -> dict:
        return asdict(self)


def summarize(policy: str, rate: float, reqs: list[Request], slo: SLO, steps: list[tuple[float, int]]) -> RunMetrics:
    done = [r for r in reqs if r.finish_time is not None]
    start = min(r.arrival for r in reqs)
    end = max(r.finish_time for r in done)
    duration = end - start
    ttft = np.array([r.ttft() for r in done])
    tpot = np.array([r.tpot() for r in done])
    ok = sum(slo.met(r) for r in reqs)
    toks = sum(len(r.output_ids) for r in done)
    return RunMetrics(
        policy=policy, rate=rate, num_requests=len(reqs), duration=duration,
        slo_attainment=ok / len(reqs), goodput=ok / duration, throughput_tok=toks / duration,
        ttft_p50=float(np.percentile(ttft, 50)), ttft_p99=float(np.percentile(ttft, 99)),
        tpot_p50=float(np.percentile(tpot, 50)), tpot_p99=float(np.percentile(tpot, 99)),
        ttft_attainment=float(np.mean(ttft <= slo.ttft)), tpot_attainment=float(np.mean(tpot <= slo.tpot)),
        mean_step_tokens=float(np.mean([n for _, n in steps])) if steps else 0.0, num_steps=len(steps),
    )
