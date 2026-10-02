# Pacer design notes

## 1. The problem

An LLM server answers two kinds of work with very different hardware profiles:

| Phase   | Work per step                    | Bottleneck                          | User-facing metric |
|---------|----------------------------------|-------------------------------------|--------------------|
| Prefill | the whole prompt (100s-1000s tok)| compute (GEMMs, attention FLOPs)    | TTFT (time to first token) |
| Decode  | 1 token per running request      | memory bandwidth (weights + KV cache) | TPOT (time per output token) |

Continuous batching runs both in the same step. That creates a trade-off on every
single step: each prefill token added makes the step slower for every request that
is decoding (hurting TPOT), and each prefill token withheld delays someone's first
token (hurting TTFT). Production systems care about **goodput**: requests per second
that meet *both* SLOs, not raw tokens per second.

* **Prefill-first** (vLLM's original scheduler) runs whole prompts as soon as they
  arrive. A 1,500-token prompt stalls every running decode for a full prefill.
* **Chunked prefill** (Sarathi-Serve, now default in vLLM/TensorRT-LLM) caps each
  step at a fixed token budget and fills leftover room with prompt chunks. The budget
  is a static knob: too small wastes the GPU and inflates TTFT, too large blows TPOT.
  The right value depends on the model, the GPU, context lengths and the live mix
  of requests, all of which change.

## 2. Pacer's idea: model-predictive token budgets

Pacer replaces the static knob with a per-step optimisation:

```
maximise   prefill tokens c in this step
subject to predicted_latency(decodes + chunks(c)) <= margin * min_i slack_i
```

1. **Learned step-latency model.** `pacer/costmodel.py` fits latency as a function of
   the step's *shape*: dense tokens (GEMM work), prefill attention FLOPs (sum of
   chunk x context), decode KV bytes (sum of contexts), padding, and per-sequence
   overhead. Coefficients are non-negative and fit to relative error, so each has a
   physical unit (seconds per token, per FLOP, per KV byte) and the model extrapolates
   instead of memorising. It is profiled once per (model, hardware) in minutes.

2. **Slack-aware latency target.** TPOT is an *average*. A request that has been
   decoding faster than its SLO has banked slack: its next token may land as late as
   `first_token_time + SLO_tpot * tokens_so_far`. Pacer takes the tightest such
   deadline across running requests, so it spends banked slack on prefill when it
   exists and tightens automatically when a request is falling behind.

3. **Deadline-ordered, feasibility-aware prefill.** Waiting prompts are ordered
   earliest-TTFT-deadline-first. A request whose deadline cannot be met even on an
   idle engine is moved to the back (it is still served, it just stops stealing
   capacity from requests that can still make it). This is what turns raw
   throughput into goodput under overload.

4. **Binary search on the model.** The budget is found by binary search over `c`
   using the linear model (a dot product), so a scheduling decision costs
   microseconds, not a GPU step.

## 3. Engine (`pacer/model.py`, `pacer/engine.py`)

* Llama-architecture decoder: RMSNorm, RoPE, grouped-query attention, SwiGLU.
* **Paged KV cache**: `[num_blocks, block_size, kv_heads, head_dim]` per layer, block
  tables per request, block-granular gathers (one index per 16 tokens), persistent
  gather workspace.
* **Packed variable-length batches**: every token of every scheduled request in one
  `[T, d]` tensor; decode tokens are attended in one batched padded call, prefill
  chunks with a causal mask offset by their cached context.
* **Exactness test**: tokens produced with paging, irregular chunking and mixed
  batches are bit-for-bit the greedy tokens of a naive full-recompute forward
  (`tests/test_engine.py`).
* Device agnostic: `--device cuda` runs the same code on an NVIDIA GPU.

## 4. Simulator (`SimExecutor`)

The serving loop (`pacer/runtime.py`) takes an executor. The real executor runs the
transformer; the simulator advances the clock by the latency model's prediction.
Because the loop, the schedulers and the block manager are shared, the simulator
replays exactly the decisions the real system would make. We validate it against
real runs (figure `sim_vs_real.png`) before trusting its load sweeps.

## 5. Why this matters on GPUs (`scripts/roofline_projection.py`)

A decode step streams every weight from HBM regardless of batch size, so added
tokens are almost free until the step's arithmetic intensity reaches the GPU's
ridge point. For Llama-3-8B in BF16 with 32 decodes at 2k context, the roofline
says the free prefill headroom is ~200 tokens on A100, ~420 on H100, ~600 on
L40S/L4. A single hard-coded budget is wrong by 2-3x on at least one of those
GPUs; a profiled latency model finds the knee automatically.

## 6. Limitations (honest list)

* Benchmarks in this repo were measured on a 4-core CPU with a 63M-parameter model.
  The absolute numbers are CPU numbers; the scheduling dynamics (compute-bound
  prefill vs. bandwidth-bound decode) are the same shape as on GPUs, and the GPU
  section is an analytic projection, not a measurement.
* Weights are random (latency does not depend on weight values; the exactness test
  covers correctness).
* No preemption/swapping: requests reserve their full KV footprint at admission.
* Single-node, single-model. Disaggregated prefill/decode (DistServe, Dynamo) is a
  complementary direction.
