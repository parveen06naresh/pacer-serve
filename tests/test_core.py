"""The C++ core must make exactly the same decisions as the Python reference."""
import random

import numpy as np
import pytest

from pacer.core import HAVE_CORE, _core
from pacer.costmodel import LinearModel, OnlineConformal
from pacer.engine import BlockManager
from pacer.request import SLO, Request
from pacer.scheduler import Pacer

pytestmark = pytest.mark.skipif(not HAVE_CORE, reason="C++ core not built")
COEF = np.array([2e-3, 4e-5, 3e-9, 2e-8, 1e-8, 1e-4, 5e-5])


def _state(rng, n_wait, n_run):
    waiting, running = [], []
    for i in range(n_wait):
        waiting.append(Request(i, rng.uniform(0, 5), rng.randint(16, 2048), rng.randint(8, 256)))
    for j in range(n_run):
        r = Request(1000 + j, rng.uniform(0, 5), rng.randint(16, 2048), 512)
        r.num_prefilled = r.prompt_len if rng.random() < 0.7 else rng.randint(0, r.prompt_len - 1)
        if not r.in_prefill:
            k = rng.randint(1, 200)
            r.output_ids = [0] * k
            r.token_times = sorted(rng.uniform(1, 6) for _ in range(k))
        running.append(r)
    return waiting, running


@pytest.mark.parametrize("seed", range(40))
def test_schedule_parity(seed):
    tpot = random.Random(seed + 99).choice([0.02, 0.05, 0.1])
    out = []
    for use_core in (False, True):
        r2 = random.Random(seed)
        waiting, running = _state(r2, r2.randint(0, 60), r2.randint(0, 120))
        bm = BlockManager(100_000, 16)
        for r in running:
            bm.admit(r)
        sched = Pacer(LinearModel(COEF), SLO(ttft=2.0, tpot=tpot), margin=0.9, use_core=use_core)
        assert (sched._planner is not None) == use_core
        batch = sched.schedule(6.0, waiting, running, bm)
        out.append([(r.rid, n) for r, n in batch])
    assert out[0] == out[1]


def test_moore_hodgson_parity():
    rng = random.Random(1)
    sched = Pacer(LinearModel(COEF), SLO(ttft=1.0, tpot=0.05), margin=0.9)
    for _ in range(200):
        cands = [Request(i, rng.uniform(0, 3), rng.randint(1, 4000), 8) for i in range(rng.randint(0, 80))]
        now = rng.uniform(0, 4)
        py = [r.rid for r in sched._moore_hodgson(now, cands, 1.0)]
        c = sched.lat.coef
        cost = [c[1] * r.remaining_prefill + c[2] * r.remaining_prefill * r.prompt_len + c[5] for r in cands]
        idx = _core.moore_hodgson([r.arrival for r in cands], cost, [r.rid for r in cands], now, 1.0)
        assert [cands[i].rid for i in idx] == py


def test_online_conformal_parity():
    rng = np.random.default_rng(0)
    py, cc = OnlineConformal(window=64, warmup=16), _core.OnlineConformal(0.1, 0.02, 64, 16)
    for t in range(3000):
        x = float(rng.lognormal(0, 0.1) * (1.6 if t > 1500 else 1.0))
        py.update(x)
        cc.update(x)
        if py.ready:
            assert cc.bound() == pytest.approx(py.bound(), rel=1e-12)
            assert cc.center() == pytest.approx(py.center(), rel=1e-12)
    assert (cc.n, cc.misses) == (py.n, py.misses)
    assert cc.alpha_t == pytest.approx(py.alpha_t)


def test_block_allocator_parity():
    rng = random.Random(3)
    bm, cc = BlockManager(512, 16), _core.BlockAllocator(512, 16)
    live = []
    for i in range(2000):
        if live and rng.random() < 0.45:
            r, blocks = live.pop(rng.randrange(len(live)))
            bm.release(r)
            cc.release(blocks)
        else:
            r = Request(i, 0.0, rng.randint(1, 300), rng.randint(1, 100))
            assert cc.can_admit(r.prompt_len + r.max_new_tokens) == bm.can_admit(r)
            if bm.can_admit(r):
                bm.admit(r)
                blocks = cc.admit(r.prompt_len + r.max_new_tokens)
                assert blocks == r.blocks
                assert cc.slots(blocks, 0, r.prompt_len) == bm.slots(r, 0, r.prompt_len).tolist()
                live.append((r, blocks))
        assert cc.num_free == len(bm.free)
