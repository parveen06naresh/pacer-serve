# Showcasing Pacer

Everything below quotes numbers that are in `results/`. Do not round them up.

## Resume bullets (pick 2-3; tailor the first word to the role)

**Pacer: SLO-aware LLM inference server** | Python, PyTorch, NumPy, SciPy, scikit-learn | github.com/<you>/pacer-serve

- Built an LLM inference engine from scratch (Llama-style GQA model, PagedAttention-style
  paged KV cache, continuous batching, chunked prefill), verified token-exact against a
  full-recompute reference.
- Designed a model-predictive scheduler driven by a learned step-latency model (7.3% MAPE;
  4.4% when extrapolating, vs 50.8% for gradient-boosted trees), serving **24% more load
  at 90% SLO attainment** than the best-tuned Sarathi-style chunked-prefill baseline on
  the real engine.
- Added split-conformal risk bounds (90% bound: 93.3% empirical coverage on held-out
  steps) so the scheduler targets a chosen tail probability instead of a mean forecast.
- Showed with a validated discrete-event simulator (4.4-point mean error vs. real runs)
  that every fixed-budget policy loses 30-54% capacity in some scenario while Pacer stays
  within 2% of the best in all four; projected the effect to H100/A100/L40S/L4 with a
  roofline model.

Track-specific first lines:
- **NVIDIA / AI infra:** lead with the engine + roofline (KV-cache bandwidth, ridge point,
  "free" prefill tokens per GPU).
- **FAANG SWE/MLE:** lead with system design + measured goodput gain + test discipline.
- **Quant:** lead with "risk-calibrated real-time optimization": conformal bounds, honest
  ablations, tail latency, extrapolation failure of tree models.

## LinkedIn post (draft)

> I built an LLM inference server from scratch to answer one question: how much work
> should go into each GPU step?
>
> Every chat model mixes two jobs on the same chip: reading prompts (compute-bound) and
> generating tokens (memory-bound). Put too much prompt work in a step and everyone
> mid-answer stutters; too little and new users wait. Today's servers use a fixed
> per-step token budget that has to be re-tuned for every model, GPU and SLO.
>
> Pacer learns the step latency instead (7.3% error, and it still holds at 4.4% when
> extrapolating to step sizes it never saw, where gradient-boosted trees fall apart at
> 51%), wraps it in conformal risk bounds, and picks the largest step that keeps every
> user inside their latency target.
>
> Results on my engine: 24% more traffic served at a 90% SLO than the best-tuned
> chunked-prefill baseline, and within 2% of the best policy across four traffic and SLO
> scenarios with zero tuning, where every fixed budget loses 30%+ somewhere.
>
> The honest parts are in the README too: one of my three ideas (slack banking) didn't
> help, and the benchmarks are on CPU with GPU projections, because that's the hardware I
> had.
>
> Code, tests, figures and every number: <link>
>
> #LLM #MLSystems #GPU #InferenceOptimization #MachineLearning

## 60-second interview pitch

"LLM serving mixes compute-bound prefill and bandwidth-bound decode in the same step, so
the scheduler is solving a constrained optimisation every few milliseconds. Production
systems use a fixed token budget. I built an engine from scratch, profiled 600 step
shapes, and found a physics-shaped linear model predicts step latency within 7%, and,
crucially, extrapolates, where trees don't. The scheduler binary-searches that model for
the biggest step that meets every running request's TPOT deadline, scales by a conformal
bound so it's a probabilistic guarantee, and orders prompts by deadline. It serves 24%
more load at the same SLO on the real engine. The ablation surprised me: deadline ordering
is the biggest single win, adaptivity is what makes it robust, and slack banking didn't
help, which I reported."

## Questions you should be ready for

1. *Why not just tune the static budget?* Section 3 of the README: the best budget moves
   from 64 to 128 tokens as the SLO changes, and from ~200 to ~600 across NVIDIA GPUs.
2. *Why linear and not a neural net?* Extrapolation and microsecond inference inside a
   control loop; the GBDT comparison is the evidence.
3. *What does conformal prediction guarantee?* Marginal coverage under exchangeability.
   The 99% bound is huge (13x) because rare VM stalls dominate it; that's a real finding
   about the hardware, not a bug.
4. *What would change on a GPU?* Absolute numbers and coefficients; the structure
   (weights streamed per step, KV bandwidth for decode) is the same. Re-profile with
   `--device cuda` (about 10 minutes) and everything else runs unchanged.
5. *Why does Pacer's p99 TTFT look worse at overload?* It lets hopeless requests wait so
   savable ones make their deadline; that is the goodput objective. Say it before they do.

## Next steps that would make it stronger

- Run the same benchmarks on one rented GPU (A10G/L4 on a cloud spot instance costs a few
  dollars) and add a "measured on GPU" column.
- Replace the decode gather with a Triton kernel and compare against the PyTorch path.
- Use the public Azure LLM inference trace instead of synthetic arrivals.
