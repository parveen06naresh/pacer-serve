"""The paged, chunked, continuously-batched engine must produce exactly the tokens a
naive full-recompute forward pass produces."""
import random

import torch

from pacer.engine import BlockManager, RealExecutor
from pacer.model import build_model
from pacer.request import Request


def greedy_reference(model, prompt, n_new):
    ids = list(prompt)
    for _ in range(n_new):
        logits = model.forward_dense(torch.tensor(ids))
        ids.append(int(logits[-1].argmax()))
    return ids[len(prompt):]


def test_paged_chunked_mixed_batch_matches_dense():
    torch.manual_seed(0)
    model = build_model("tiny", seed=1)
    bm = BlockManager(num_blocks=256, block_size=16)
    ex = RealExecutor(model, bm)
    rng = random.Random(0)
    reqs = []
    for i, (p, o) in enumerate([(37, 6), (5, 9), (70, 4), (1, 5)]):
        r = Request(i, 0.0, p, o, prompt_ids=[rng.randrange(model.cfg.vocab_size) for _ in range(p)])
        bm.admit(r)
        reqs.append(r)

    # Irregular chunk sizes so prefill chunks and decodes get mixed in the same steps.
    chunks = [7, 13, 3, 64]
    step = 0
    while not all(r.done for r in reqs):
        batch = []
        for r in reqs:
            if r.done:
                continue
            n = min(r.remaining_prefill, chunks[(step + r.rid) % len(chunks)]) if r.in_prefill else 1
            batch.append((r, n))
        ex.execute(batch, now=float(step))
        step += 1

    for r in reqs:
        assert r.output_ids == greedy_reference(model, r.prompt_ids, r.max_new_tokens), r.rid


def test_block_manager_reserves_and_releases():
    bm = BlockManager(num_blocks=8, block_size=16)
    r = Request(0, 0.0, prompt_len=40, max_new_tokens=10)
    assert bm.blocks_needed(r) == 4
    bm.admit(r)
    assert len(bm.free) == 4
    slots = bm.slots(r, 14, 18).tolist()
    assert slots[:2] == [r.blocks[0] * 16 + 14, r.blocks[0] * 16 + 15]
    assert slots[2:] == [r.blocks[1] * 16, r.blocks[1] * 16 + 1]
    bm.release(r)
    assert len(bm.free) == 8
