"""Step-latency models: predict how long one engine step takes from its shape.

Four models, from zero-shot physics to fully learned:

* `RooflineModel`  analytic: max(FLOPs / peak, bytes / bandwidth), with peak and
                   bandwidth measured by micro-benchmarks; only an efficiency factor
                   and a fixed overhead are calibrated.
* `LinearModel`    physics-informed features (dense tokens, attention FLOPs, KV bytes,
                   padding, per-sequence overhead) with non-negative coefficients fit to
                   minimise *relative* error. Every coefficient has a unit (s/token, s/FLOP).
* `GBDTModel`      gradient-boosted trees on the same features, fit in log space.
* `HybridModel`    linear model times a GBDT correction of its log residual: keeps the
                   linear model's extrapolation and the trees' accuracy in-distribution.

All are deliberately cheap to evaluate, because the scheduler calls `predict` inside a
search on every step.
"""
from __future__ import annotations

import json
from dataclasses import dataclass

import numpy as np

from .engine import StepShape
from .model import ModelConfig, attn_flops, model_flops_per_token

FEATURES = ["bias", "tokens", "prefill_attn", "decode_kv", "decode_pad", "prefill_seqs", "decode_seqs"]


def featurize(s: StepShape) -> np.ndarray:
    dec = s.decode
    n_dec = len(dec)
    return np.array([
        1.0,
        float(s.num_tokens),
        float(sum(n * L for n, L in s.prefill)),
        float(sum(dec)),
        float(n_dec * max(dec)) if dec else 0.0,
        float(len(s.prefill)),
        float(n_dec),
    ])


def featurize_many(shapes) -> np.ndarray:
    return np.stack([featurize(s) for s in shapes])


def mape(y_true, y_pred) -> float:
    y_true, y_pred = np.asarray(y_true), np.asarray(y_pred)
    return float(np.mean(np.abs(y_pred - y_true) / y_true))


def conformal_ratio(model, X_cal, y_cal, coverage: float) -> float:
    """Split-conformal upper bound on the latency ratio y / y_hat.

    With exchangeable calibration data, P(y <= q * y_hat) >= coverage for a new step,
    with no distributional assumptions. The scheduler divides its latency target by
    q, turning a point forecast into a risk-controlled one (the serving analogue of
    sizing a position from a calibrated VaR rather than a mean forecast).
    """
    r = np.sort(np.asarray(y_cal) / model.predict_X(X_cal))
    n = len(r)
    k = min(n - 1, int(np.ceil((n + 1) * coverage)) - 1)
    return float(r[k])


class OnlineConformal:
    """Adaptive conformal inference (Gibbs & Candes, 2021) on the latency ratio y / y_hat.

    Offline conformal bounds assume tomorrow's hardware behaves like profiling day. A
    noisy neighbour, thermal throttling or a driver update breaks that, and a fixed
    bound silently stops covering. ACI keeps a sliding window of observed ratios and
    nudges its miscoverage level after every step:

        alpha_{t+1} = alpha_t + gamma * (alpha - 1[ratio_t > bound_t])

    which guarantees the long-run fraction of steps exceeding the bound converges to
    alpha for *any* sequence of ratios, adversarial drift included.
    """

    def __init__(self, alpha: float = 0.1, gamma: float = 0.02, window: int = 256, warmup: int = 32):
        from collections import deque
        self.alpha = alpha
        self.alpha_t = alpha
        self.gamma = gamma
        self.buf = deque(maxlen=window)
        self.warmup = warmup
        self.n = 0
        self.misses = 0

    @property
    def ready(self) -> bool:
        return len(self.buf) >= self.warmup

    def bound(self) -> float:
        r = np.fromiter(self.buf, float)
        if self.alpha_t <= 0:
            return float(r.max()) * 1.25
        if self.alpha_t >= 1:
            return float(r.min())
        return float(np.quantile(r, 1.0 - self.alpha_t, method="higher"))

    def center(self) -> float:
        return float(np.median(np.fromiter(self.buf, float)))

    def update(self, ratio: float) -> None:
        if self.ready:
            miss = ratio > self.bound()
            self.alpha_t += self.gamma * (self.alpha - miss)
            self.n += 1
            self.misses += miss
        self.buf.append(ratio)


class LatencyModel:
    name = "base"

    def predict(self, s: StepShape) -> float:
        return float(self.predict_X(featurize(s)[None])[0])

    def predict_X(self, X: np.ndarray) -> np.ndarray:
        raise NotImplementedError


@dataclass
class RooflineModel(LatencyModel):
    cfg: ModelConfig
    peak_flops: float          # achieved FLOP/s of a large GEMM
    bandwidth: float           # achieved bytes/s of a large copy
    overhead: float = 0.0      # fixed per-step cost
    efficiency: float = 1.0    # how far below the roofs real kernels run (>= 1)
    bytes_per_el: int = 4
    name: str = "roofline"

    def predict_X(self, X):
        cfg, b = self.cfg, self.bytes_per_el
        tokens, pre_attn, dec_kv = X[:, 1], X[:, 2], X[:, 3]
        flops = tokens * model_flops_per_token(cfg) + attn_flops(cfg, 1, 1) * (pre_attn + dec_kv)
        kv_bytes_per_tok = 2 * cfg.n_layers * cfg.n_kv_heads * cfg.head_dim * b
        weight_bytes = (cfg.n_params() - cfg.vocab_size * cfg.d_model) * b
        bytes_ = weight_bytes + dec_kv * kv_bytes_per_tok + pre_attn / np.maximum(tokens, 1) * kv_bytes_per_tok
        return self.efficiency * np.maximum(flops / self.peak_flops, bytes_ / self.bandwidth) + self.overhead

    def fit(self, X, y):
        """Calibrate only two scalars (kernel efficiency, fixed overhead), the strongest
        fair version of a roofline: the *shape* of the model stays first-principles."""
        from scipy.optimize import nnls
        self.efficiency, self.overhead = 1.0, 0.0
        A = np.stack([self.predict_X(X), np.ones(len(y))], 1)
        w = 1.0 / y
        (self.efficiency, self.overhead), _ = nnls(A * w[:, None], y * w)
        return self


class LinearModel(LatencyModel):
    name = "linear"

    def __init__(self, coef: np.ndarray | None = None):
        self.coef = coef

    def fit(self, X, y):
        from scipy.optimize import nnls
        w = 1.0 / y                         # minimise relative error
        self.coef, _ = nnls(X * w[:, None], y * w)
        return self

    def predict_X(self, X):
        return X @ self.coef

    def predict(self, s: StepShape) -> float:  # hot path: skip numpy stacking
        return float(featurize(s) @ self.coef)

    def describe(self) -> dict:
        return dict(zip(FEATURES, self.coef.tolist()))

    def save(self, path, conformal: dict | None = None):
        with open(path, "w") as f:
            json.dump({"features": FEATURES, "coef": self.coef.tolist(), "conformal": conformal or {}}, f, indent=2)

    @classmethod
    def load(cls, path):
        with open(path) as f:
            d = json.load(f)
        m = cls(np.array(d["coef"]))
        m.conformal = {float(k): v for k, v in d.get("conformal", {}).items()}
        return m


class GBDTModel(LatencyModel):
    name = "gbdt"

    def fit(self, X, y):
        from sklearn.ensemble import HistGradientBoostingRegressor
        self.m = HistGradientBoostingRegressor(max_iter=400, learning_rate=0.05, max_leaf_nodes=31,
                                               min_samples_leaf=8, random_state=0)
        self.m.fit(X[:, 1:], np.log(y))
        return self

    def predict_X(self, X):
        return np.exp(self.m.predict(X[:, 1:]))


class HybridModel(LatencyModel):
    name = "hybrid"

    def fit(self, X, y):
        from sklearn.ensemble import HistGradientBoostingRegressor
        self.lin = LinearModel().fit(X, y)
        resid = np.log(y) - np.log(np.maximum(self.lin.predict_X(X), 1e-6))
        self.m = HistGradientBoostingRegressor(max_iter=200, learning_rate=0.05, max_leaf_nodes=15,
                                               min_samples_leaf=16, l2_regularization=1.0, random_state=0)
        self.m.fit(X[:, 1:], resid)
        return self

    def predict_X(self, X):
        return self.lin.predict_X(X) * np.exp(self.m.predict(X[:, 1:]))
