"""Predictor 2 (KV reservation) and the online point baseline."""
import copy
from pathlib import Path

import numpy as np

from pacer.costmodel import LinearModel, OnlineConformal, OnlinePoint
from pacer.engine import GrowingBlockManager, SimExecutor, StepShape
from pacer.kvpredict import LengthModel, build
from pacer.request import SLO
from pacer.runtime import serve
from pacer.scheduler import ChunkedFixed, Pacer
from pacer.workload import WorkloadSpec, generate

LAT = LinearModel.load(Path(__file__).resolve().parent.parent / "results" / "latency_model.json")


def _run(policy_kind, blocks=96, drift=1.0, sched=None):
    cal = generate(WorkloadSpec(rate=4, num_requests=200, seed=100, output_max=384))
    reqs = generate(WorkloadSpec(rate=4, num_requests=150, seed=1, output_max=384))
    for r in reqs[50:100]:
        r.max_new_tokens = int(min(384, r.max_new_tokens * drift))
    pol = build(policy_kind, cal, 384)
    bm = GrowingBlockManager(blocks, 16, pol, lambda r: LAT.predict(StepShape(((r.context_len, r.context_len),), ())),
                             pol.observe)
    slo = SLO(2.0, 0.1)
    serve(reqs, sched or Pacer(LAT, slo), SimExecutor(LAT, bm), bm)
    return reqs, bm, pol


def test_every_request_finishes_with_full_output_under_preemption():
    reqs, bm, _ = _run("ondemand", blocks=128)
    assert bm.preemptions > 0
    assert all(r.finish_time is not None and len(r.output_ids) == r.max_new_tokens for r in reqs)
    assert len(bm.free) == bm.num_blocks  # nothing leaked


def test_chunked_baseline_runs_on_growing_allocator():
    reqs, bm, _ = _run("ondemand", blocks=128, sched=ChunkedFixed(128))
    assert all(r.finish_time is not None for r in reqs)
    assert len(bm.free) == bm.num_blocks


def test_reserve_max_never_preempts():
    _, bm, _ = _run("max", blocks=256)
    assert bm.preemptions == 0


def test_online_conformal_keeps_exceed_rate_near_alpha_under_drift():
    _, bm_off, _ = _run("offline", blocks=512, drift=3.0)
    _, bm_on, pol = _run("online", blocks=512, drift=3.0)
    assert len(bm_on.exceeded) < len(bm_off.exceeded)
    assert pol.calib.misses / pol.calib.n < 0.2


def test_length_model_recovers_power_law():
    rng = np.random.default_rng(0)
    p = rng.integers(10, 1000, 500)
    o = 3.0 * p ** 0.5
    m = LengthModel().fit(p, o)
    assert abs(m.b - 0.5) < 1e-6 and abs(m.predict(400) - 60.0) < 1e-6


def test_online_point_tracks_median_without_margin():
    pt, cf = OnlinePoint(window=200, warmup=10), OnlineConformal(alpha=0.1, window=200, warmup=10)
    for x in np.random.default_rng(0).lognormal(0, 0.2, 2000):
        pt.update(x)
        cf.update(x)
    assert abs(pt.bound() - 1.0) < 0.1
    assert 0.4 < pt.misses / pt.n < 0.6          # a point forecast is overshot about half the time
    assert cf.misses / cf.n < 0.15                # the conformal bound is not
    assert copy.deepcopy(pt).bound() == pt.bound()
