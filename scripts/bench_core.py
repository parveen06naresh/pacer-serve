"""Per-step scheduler overhead: pure Python vs the C++17 core (same decisions).

    python scripts/bench_core.py  ->  results/core_bench.json
"""
import copy
import json
import platform
import random
import statistics
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from pacer import core  # noqa: E402
from pacer.costmodel import LinearModel, OnlineConformal  # noqa: E402
from pacer.engine import BlockManager  # noqa: E402
from pacer.request import SLO, Request  # noqa: E402
from pacer.scheduler import Pacer  # noqa: E402

# Same benchmark run on v0.2.0 (commit 3c3efc4, pure Python, before the O(n^2) fix), same machine.
BASELINE_V020_US = {16: 178.2, 64: 424.8, 256: 3289.6, 1024: 25616.3}
COEF = np.array([2e-3, 4e-5, 3e-9, 2e-8, 1e-8, 1e-4, 5e-5])


def make_state(n_run, n_wait, seed=0):
    rng = random.Random(seed)
    running, waiting = [], []
    for j in range(n_run):
        r = Request(j, rng.uniform(0, 5), rng.randint(64, 2048), 512)
        r.num_prefilled = r.prompt_len if j % 8 else r.prompt_len // 2
        if not r.in_prefill:
            r.output_ids = [0] * 50
            r.token_times = [1 + 0.02 * i for i in range(50)]
        running.append(r)
    for i in range(n_wait):
        waiting.append(Request(10_000 + i, rng.uniform(0, 5), rng.randint(64, 2048), 128))
    return running, waiting


def time_schedule(use_core, n_run, n_wait, reps):
    run0, wait0 = make_state(n_run, n_wait)
    ts, batch = [], None
    for _ in range(reps):
        running, waiting = copy.deepcopy(run0), copy.deepcopy(wait0)
        bm = BlockManager(1_000_000, 16)
        for r in running:
            bm.admit(r)
        s = Pacer(LinearModel(COEF), SLO(ttft=2.0, tpot=0.05), margin=0.9, use_core=use_core)
        t = time.perf_counter()
        batch = s.schedule(6.0, waiting, running, bm)
        ts.append(time.perf_counter() - t)
    return statistics.median(ts) * 1e6, [(r.rid, n) for r, n in batch]


def time_conformal(impl, steps=20_000):
    rng = np.random.default_rng(0)
    xs = rng.lognormal(0, 0.1, steps).tolist()
    t = time.perf_counter()
    for x in xs:
        impl.update(x)
        if impl.ready:
            impl.bound()
    return (time.perf_counter() - t) / steps * 1e6


def main():
    assert core.HAVE_CORE, "build first: python setup.py build_ext --inplace"
    rows = []
    for n_run, n_wait in [(16, 8), (64, 32), (256, 128), (1024, 256)]:
        reps = 200 if n_run < 1024 else 40
        py_us, py_b = time_schedule(False, n_run, n_wait, reps)
        cc_us, cc_b = time_schedule(True, n_run, n_wait, reps)
        assert py_b == cc_b, "decisions differ"
        rows.append(dict(running=n_run, waiting=n_wait, python_us=round(py_us, 1), cpp_us=round(cc_us, 1),
                         speedup=round(py_us / cc_us, 1), v020_us=BASELINE_V020_US[n_run],
                         speedup_vs_v020=round(BASELINE_V020_US[n_run] / cc_us, 1)))
        print(rows[-1])
    aci = dict(python_us=round(time_conformal(OnlineConformal()), 2),
               cpp_us=round(time_conformal(core._core.OnlineConformal(0.1, 0.02, 256, 32)), 2))
    aci["speedup"] = round(aci["python_us"] / aci["cpp_us"], 1)
    print("online conformal update+bound", aci)
    out = dict(machine=platform.processor() or platform.machine(), python=platform.python_version(),
               schedule_step=rows, online_conformal=aci)
    Path("results").mkdir(exist_ok=True)
    Path("results/core_bench.json").write_text(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
