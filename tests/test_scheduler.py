import copy

import numpy as np
import pytest

from pacer.costmodel import LinearModel
from pacer.engine import BlockManager, SimExecutor, StepShape
from pacer.request import SLO, Request
from pacer.runtime import serve
from pacer.scheduler import ChunkedFixed, Pacer, PrefillFirst
from pacer.workload import WorkloadSpec, generate

# bias, tokens, prefill_attn, decode_kv, decode_pad, prefill_seqs, decode_seqs
TOY = LinearModel(np.array([0.010, 2e-4, 1e-8, 2e-7, 0.0, 1e-3, 5e-4]))
SLO_ = SLO(ttft=2.0, tpot=0.08)


@pytest.mark.parametrize("make", [lambda: PrefillFirst(), lambda: ChunkedFixed(128), lambda: Pacer(TOY, SLO_),
                                  lambda: Pacer(TOY, SLO_, use_slack=False), lambda: Pacer(TOY, SLO_, use_edf=False)])
def test_every_policy_finishes_every_request(make):
    trace = generate(WorkloadSpec(rate=6.0, num_requests=120, seed=3))
    bm = BlockManager(2048, 16)
    reqs = copy.deepcopy(trace)
    serve(reqs, make(), SimExecutor(TOY, bm), bm)
    assert all(r.finish_time is not None for r in reqs)
    assert all(len(r.output_ids) == r.max_new_tokens and r.num_prefilled == r.prompt_len for r in reqs)
    assert all(r.ttft() >= 0 and r.finish_time >= r.first_token_time for r in reqs)
    assert len(bm.free) == bm.num_blocks  # every KV block returned


def test_pacer_step_fits_tightest_deadline():
    bm = BlockManager(4096, 16)
    pacer = Pacer(TOY, SLO_, margin=1.0)
    now = 10.0
    running = []
    for i in range(8):  # decoding requests with assorted banked slack
        r = Request(i, 0.0, prompt_len=100, max_new_tokens=50)
        bm.admit(r)
        r.num_prefilled = 100
        r.output_ids = [0] * (i + 1)
        r.token_times = [now - 0.05 * (i + 1) + 0.05 * k for k in range(i + 1)]
        running.append(r)
    waiting = [Request(100 + j, now, prompt_len=900, max_new_tokens=10) for j in range(3)]
    batch = pacer.schedule(now, waiting, running, bm)
    shape = StepShape.of(batch)
    assert TOY.predict(shape) <= pacer.last_target + 1e-9
    assert shape.num_tokens > 8, "should pack prefill tokens into the slack"
    # One more prefill token would have broken the target (budget is maximal).
    bigger = StepShape(((shape.prefill[0][0] + 1, shape.prefill[0][1] + 1),) + shape.prefill[1:], shape.decode)
    assert TOY.predict(bigger) > pacer.last_target


def test_deadline_ordering_puts_hopeless_requests_last():
    pacer = Pacer(TOY, SLO_)
    old = Request(0, arrival=0.0, prompt_len=200, max_new_tokens=5)       # deadline long gone
    fresh = Request(1, arrival=9.9, prompt_len=200, max_new_tokens=5)
    order = pacer._prefill_order(10.0, [old, fresh])
    assert order == [fresh, old]


def test_online_conformal_tracks_coverage_through_drift():
    from pacer.costmodel import OnlineConformal
    rng = np.random.default_rng(0)
    oc = OnlineConformal(alpha=0.1, gamma=0.02)
    # Ratios jump 1.8x a third of the way in (a noisy neighbour arrives) and recover later.
    scale = np.r_[np.ones(3000), np.full(3000, 1.8), np.ones(3000)]
    for s in scale:
        oc.update(float(s * rng.lognormal(0, 0.1)))
    assert abs(oc.misses / oc.n - 0.1) < 0.02


def test_azure_trace_replay_scales_rate_and_lengths():
    from pacer.workload import from_azure
    reqs = from_azure("conv", rate=4.0, num_requests=200)
    span = reqs[-1].arrival - reqs[0].arrival
    assert abs(200 / span - 4.0) < 1e-6
    assert all(8 <= r.prompt_len <= 1536 and 4 <= r.max_new_tokens <= 192 for r in reqs)


def test_moore_hodgson_saves_more_deadlines_than_edf():
    # A long prompt due first, then three short ones. EDF serves the long one first and
    # makes two short ones late; Moore-Hodgson gives up the long one and saves all three.
    long = Request(0, arrival=0.00, prompt_len=1500, max_new_tokens=4)
    shorts = [Request(i, arrival=0.01 * i, prompt_len=150, max_new_tokens=4) for i in (1, 2, 3)]
    slo = SLO(ttft=0.4, tpot=0.08)
    assert Pacer(TOY, slo, order="mh")._prefill_order(0.05, [long, *shorts]) == [*shorts, long]
    assert Pacer(TOY, slo, order="edf")._prefill_order(0.05, [long, *shorts])[0] is long
