# Showcasing Pacer

Every number below is in `results/`. Quote them exactly; don't round up.

## Resume bullets (pick 3; lead with the one that fits the role)

**Pacer: LLM inference server with online risk-controlled scheduling** | Python, PyTorch, SciPy, scikit-learn | github.com/<you>/pacer-serve

- Built an LLM inference engine from scratch (Llama-style GQA model, PagedAttention-style
  paged KV cache, continuous batching, chunked prefill), verified token-exact against a
  full-recompute reference.
- Designed a scheduler that keeps latency SLOs under hardware drift using online
  conformal prediction; under a real noisy-neighbour slowdown it kept **85%** of users
  within SLO vs **49%** for the best standard scheduler, holding its 10% risk target
  (8.5-10.4% observed) on real hardware.
- Learned a physics-informed step-latency model (7.3% error, 4.4% when extrapolating vs
  50.8% for gradient-boosted trees) and used it for model-predictive batching, reaching
  **+49% capacity** at 90% SLO attainment over vLLM/Sarathi-style fixed budgets.
- Applied Moore-Hodgson (exact minimisation of missed deadlines) to prompt ordering:
  **+21%** capacity on the bursty Azure production code trace; across six workloads
  including two Azure production traces, never below 94% of the best policy, where every
  fixed configuration dropped to 70% or worse.

Lead with:
- **NVIDIA / AI infra:** the engine, the KV-cache/roofline analysis, and the drift result
  (shared GPUs and noisy neighbours are a real datacenter problem).
- **FAANG SWE/MLE:** system design, measured impact on production traces, test discipline,
  honest ablations.
- **Quant:** "online risk control with a distribution-free guarantee, verified empirically;
  model selection driven by out-of-sample extrapolation; an exact combinatorial
  optimisation in the hot loop". Quant interviewers will push on ACI's guarantee; know it.

## LinkedIn post (draft)

> Most LLM servers have a fixed safety margin baked in. What happens when the hardware
> gets slower mid-flight?
>
> I built Pacer, an LLM inference server from scratch (paged KV cache, continuous
> batching, chunked prefill), to find out. Its scheduler predicts how long every step will
> take, and wraps that prediction in an online conformal bound: a statistical guarantee
> that re-calibrates after every step, so only ~10% of steps run over budget no matter
> how the hardware drifts.
>
> When I started a noisy-neighbour process mid-run, Pacer kept 85% of users within their
> latency targets. The best standard scheduler kept 49%. And a version of my own scheduler
> that trusted its offline calibration did worse than both, at 38%. Being clever without
> being calibrated is a liability.
>
> Also inside: a learned latency model that still holds within 4.4% on step sizes it never
> saw (tree models fall apart at 51%), Moore-Hodgson scheduling to save the most requests
> during traffic bursts, and replay of Microsoft's public Azure LLM production traces.
>
> Honest notes are in the README: one of my ideas didn't work and is reported as a negative
> result, and the benchmarks are on CPU with GPU projections, because that's the hardware
> I had.
>
> Code, tests, every number, and how to reproduce them: <link>
>
> #LLM #MLSystems #InferenceOptimization #GPU #ConformalPrediction

## 60-second interview pitch

"LLM serving mixes compute-bound prefill and bandwidth-bound decode in every step, so the
scheduler solves a constrained optimisation every few milliseconds. Production systems use
a fixed token budget with a hand-set safety margin. I built an engine from scratch,
profiled it, and found a physics-shaped linear model predicts step latency within 7% and,
unlike trees, extrapolates. The scheduler binary-searches that model for the biggest step
that fits the SLO. The interesting part is the margin: I replaced it with adaptive
conformal inference, which re-calibrates after every step and guarantees the long-run
miss rate. Under a real noisy neighbour that's 85% of users on target versus 49%, and the
10% miss target held at 8.5 to 10.4% on real hardware. The biggest surprise was that my
own scheduler with offline calibration did worst of all under drift."

## Questions to be ready for

1. **What exactly does ACI guarantee?** The long-run average of 1[ratio > bound] converges
   to alpha for any sequence of ratios (no exchangeability needed), at rate ~1/(gamma*T).
   It does not guarantee coverage at every moment, and it is a step-level guarantee, not
   a per-request one; the per-request effect is measured, not proven.
2. **Why did offline-calibrated Pacer do worst under drift?** It packs steps right up to
   the bound it believes; when the hardware is 1.5x slower, every step overshoots. Static
   schedulers leave accidental slack. Online calibration removes the overconfidence.
3. **Why Moore-Hodgson and not EDF?** EDF is optimal when everything can be on time;
   under overload it lets one long late job push many short jobs late. Moore-Hodgson is
   exactly optimal for the number of on-time jobs on one machine. It matters only under
   bursts, which the ablation shows.
4. **Why a linear model, not a neural net?** Extrapolation and microsecond inference
   inside a control loop; the GBDT comparison is the evidence.
5. **What changes on a GPU?** The coefficients and absolute numbers. The structure
   (weights streamed every step, KV bandwidth for decode, noisy neighbours on shared GPUs)
   is the same. Re-profile with `--device cuda` and everything else runs unchanged.
6. **Why does Pacer's p99 TTFT look worse at overload?** It lets hopeless requests wait so
   savable ones make their deadline. That is the goodput objective; say it before they do.

## Next upgrades

- One run on a rented GPU (an L4 or A10G for an hour costs a few dollars) for a measured
  GPU column.
- Replace the decode gather with a Triton kernel and benchmark it against the PyTorch path.
- A short arXiv-style write-up (4 pages) of the drift result.
