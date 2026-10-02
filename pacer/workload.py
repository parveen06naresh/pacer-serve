"""Synthetic traffic: Poisson or bursty (Gamma) arrivals with heavy-tailed lengths.

Length distributions are lognormal, the shape reported for production chat traces
(e.g. the ShareGPT / Azure LLM inference traces), scaled down so a CPU can serve them.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .request import Request


@dataclass(frozen=True)
class WorkloadSpec:
    rate: float                   # mean requests / second
    num_requests: int = 200
    prompt_median: int = 192
    prompt_sigma: float = 0.8
    prompt_max: int = 1536
    output_median: int = 48
    output_sigma: float = 0.6
    output_max: int = 192
    burstiness: float = 1.0       # coefficient of variation of inter-arrivals; 1.0 = Poisson
    seed: int = 0


def generate(spec: WorkloadSpec, vocab_size: int | None = None) -> list[Request]:
    rng = np.random.default_rng(spec.seed)
    cv = spec.burstiness
    shape = 1.0 / (cv * cv)
    gaps = rng.gamma(shape, 1.0 / (spec.rate * shape), size=spec.num_requests)
    arrivals = np.cumsum(gaps) - gaps[0]
    p = np.clip(rng.lognormal(np.log(spec.prompt_median), spec.prompt_sigma, spec.num_requests), 8, spec.prompt_max)
    o = np.clip(rng.lognormal(np.log(spec.output_median), spec.output_sigma, spec.num_requests), 4, spec.output_max)
    reqs = []
    for i in range(spec.num_requests):
        pl, ol = int(p[i]), int(o[i])
        ids = rng.integers(0, vocab_size, pl).tolist() if vocab_size else None
        reqs.append(Request(i, float(arrivals[i]), pl, ol, prompt_ids=ids))
    return reqs
