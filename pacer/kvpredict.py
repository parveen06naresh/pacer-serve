"""Predictor 2: how much KV memory to reserve for a request's output.

A server must decide, at admission, how many output tokens of KV cache to set aside.
Reserving the maximum wastes memory (fewer requests fit); allocating on demand
(vLLM) packs tightly but preempts and recomputes requests when memory runs out;
reserving a *predicted* length sits in between, and is only as good as the
predictor's calibration. When the output-length distribution shifts (a product
turns on longer answers, a reasoning mode, a new client), an offline-calibrated
predictor under-reserves and preemptions spike.

Every policy here returns `reserve(r) -> tokens`. The learned ones predict
log(output) from log(prompt) and scale that point prediction by a calibration
factor on the ratio actual / predicted:

* offline conformal: the factor is the (1 - alpha) quantile on a calibration trace,
* online point: the median ratio of recently finished requests (tracks drift,
  no guarantee),
* online conformal: adaptive conformal inference on the same ratio, so in the long
  run about alpha of requests outgrow their reservation, whatever the drift.

The ratio is observed only when a request finishes, so feedback is delayed by
one request lifetime; the ACI guarantee still holds (it needs no timing assumption
beyond eventually seeing every outcome).
"""
from __future__ import annotations

import numpy as np

from .costmodel import OnlineConformal, OnlinePoint
from .request import Request


class LengthModel:
    """log(output_len) = a + b * log(prompt_len), fit by least squares."""

    def __init__(self, a: float = 0.0, b: float = 0.0):
        self.a, self.b = a, b

    def fit(self, prompts, outputs) -> LengthModel:
        x = np.log(np.asarray(prompts, float))
        y = np.log(np.asarray(outputs, float))
        A = np.stack([np.ones_like(x), x], 1)
        self.a, self.b = np.linalg.lstsq(A, y, rcond=None)[0]
        return self

    def predict(self, prompt_len: int) -> float:
        return float(np.exp(self.a + self.b * np.log(max(prompt_len, 1))))


class Reserve:
    """Base reservation policy; `observe` sees each finished request."""

    name = "base"

    def __call__(self, r: Request) -> int:
        raise NotImplementedError

    def observe(self, r: Request) -> None:
        pass


class ReserveMax(Reserve):
    def __init__(self, cap: int):
        self.cap, self.name = cap, f"reserve max ({cap})"

    def __call__(self, r):
        return self.cap


class OnDemand(Reserve):
    name = "on demand (vLLM)"

    def __call__(self, r):
        return 1


class Oracle(Reserve):
    name = "oracle (true length)"

    def __call__(self, r):
        return r.max_new_tokens


class Calibrated(Reserve):
    """Point prediction times a calibration factor from `calib` (None = offline)."""

    def __init__(self, model: LengthModel, cap: int, offline_factor: float, calib=None, name: str = ""):
        self.model, self.cap, self.offline = model, cap, offline_factor
        self.calib = calib
        self.name = name

    def factor(self) -> float:
        if self.calib is not None and self.calib.ready:
            return self.calib.bound()
        return self.offline

    def __call__(self, r):
        return int(min(self.cap, np.ceil(self.model.predict(r.prompt_len) * self.factor())))

    def observe(self, r):
        if self.calib is not None:
            self.calib.update(r.max_new_tokens / max(self.model.predict(r.prompt_len), 1e-6))


def build(kind: str, cal: list[Request], cap: int, alpha: float = 0.1) -> Reserve:
    """Reservation policies, calibrated on `cal` (a trace from before any drift)."""
    if kind == "max":
        return ReserveMax(cap)
    if kind == "ondemand":
        return OnDemand()
    if kind == "oracle":
        return Oracle()
    m = LengthModel().fit([r.prompt_len for r in cal], [r.max_new_tokens for r in cal])
    ratios = np.sort([r.max_new_tokens / m.predict(r.prompt_len) for r in cal])
    n = len(ratios)
    q = float(ratios[min(n - 1, int(np.ceil((n + 1) * (1 - alpha))) - 1)])
    med = float(np.median(ratios))
    if kind == "offline":
        return Calibrated(m, cap, q, name="predicted, offline conformal")
    if kind == "online-point":
        return Calibrated(m, cap, med, OnlinePoint(window=64, warmup=16), name="predicted, online point")
    if kind == "online":
        return Calibrated(m, cap, q, OnlineConformal(alpha=alpha, gamma=0.05, window=64, warmup=16),
                          name="predicted, online conformal")
    raise ValueError(kind)
