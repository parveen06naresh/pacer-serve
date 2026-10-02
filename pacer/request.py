"""Request state shared by the real engine, the simulator, and the schedulers."""
from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class Request:
    rid: int
    arrival: float
    prompt_len: int
    max_new_tokens: int
    prompt_ids: list[int] | None = None

    # Mutable serving state.
    num_prefilled: int = 0                    # prompt tokens whose KV is in the cache
    output_ids: list[int] = field(default_factory=list)
    token_times: list[float] = field(default_factory=list)
    blocks: list[int] = field(default_factory=list)
    finish_time: float | None = None

    @property
    def in_prefill(self) -> bool:
        return self.num_prefilled < self.prompt_len

    @property
    def context_len(self) -> int:
        """Tokens currently in the KV cache for this request."""
        # Every generated token except the newest has been fed back through the model.
        return self.num_prefilled + max(0, len(self.output_ids) - 1)

    @property
    def remaining_prefill(self) -> int:
        return self.prompt_len - self.num_prefilled

    @property
    def done(self) -> bool:
        return len(self.output_ids) >= self.max_new_tokens

    @property
    def first_token_time(self) -> float | None:
        return self.token_times[0] if self.token_times else None

    # Metrics.
    def ttft(self) -> float:
        return self.token_times[0] - self.arrival

    def tpot(self) -> float:
        """Mean time per output token after the first (the DistServe / vLLM definition)."""
        if len(self.token_times) < 2:
            return 0.0
        return (self.token_times[-1] - self.token_times[0]) / (len(self.token_times) - 1)


@dataclass(frozen=True)
class SLO:
    ttft: float   # seconds
    tpot: float   # seconds per output token

    def met(self, r: Request) -> bool:
        return r.finish_time is not None and r.ttft() <= self.ttft and r.tpot() <= self.tpot
