# Putting Pacer on LinkedIn

Do the steps in this order. Steps 1 to 4 take about 15 minutes.

## 1. The repo

Your GitHub link is what recruiters click: https://github.com/parveen06naresh/pacer-serve.
Pin it on your profile and add the topics before you post.

## 2. Add it to the Projects section

Profile > **Add profile section** > Recommended > **Add projects**.

- **Project name:** Pacer: LLM Inference Server with Risk-Controlled Scheduling
- **Associated with:** your university
- **Start date:** Sep 2026 (check "I am currently working on this project")
- **Skills:** PyTorch, Python, Large Language Models (LLM), Machine Learning, Distributed
  Systems, Statistics, Performance Optimization (you can add up to 5)
- **Media:** add a link to the GitHub repo, a link to the project page, and upload the
  paper PDF (`docs/paper/pacer.pdf`).
- **Description** (paste this):

```
I built an LLM inference server from scratch (paged KV cache, continuous batching, chunked prefill) and designed a new scheduler for it.

Most LLM servers use a fixed token budget per step and a hand-tuned safety margin. Pacer predicts every step's latency with a physics-informed model and wraps that prediction in an online conformal bound. The bound recalibrates after every step, so only about 10% of steps run over budget, whatever the hardware does.

Results:
• Under a real noisy-neighbour slowdown, 85% of requests stayed within latency targets, vs 49% for vLLM-style scheduling.
• The 10% risk target held at 8.5–10.4% on real hardware.
• +49% capacity over the best fixed budget, and never below 94% of the best policy across six workloads, including two Azure production traces.
• The latency model is within 4.4% on step sizes it never saw; gradient-boosted trees miss by 51%.

Written up in a 5-page paper with a formal guarantee. Measured on CPU; a GPU benchmark notebook is included.

Python · PyTorch · SciPy · scikit-learn
```

## 3. Add it to the Featured section (top of your profile)

Profile > **Add profile section** > Recommended > **Add featured**. Add three items:

1. **Link:** the GitHub repo. Title: "Pacer: LLM inference server built from scratch".
2. **Media:** upload `pacer.pdf`. Title: "Pacer paper: risk-controlled batch scheduling for
   LLM serving".
3. **Link:** the project page. Open it, use its **Share** menu to make it viewable by
   anyone with the link, then paste that link. Title: "Pacer results, interactive".

After you publish the post in step 5, add it to Featured too (the "..." menu on the post >
Feature on top of profile).

## 4. Update your headline and skills

Headline (paste, then edit the school part):

```
CS @ [Your University] | ML Systems & LLM Inference | Built Pacer, an LLM server with risk-controlled scheduling (85% vs 49% SLO under drift) | Seeking 2027 SWE / ML / Quant internships
```

Skills section: add PyTorch, Large Language Models (LLM), CUDA (only once you've run the
GPU notebook), Statistics, Python, C++ (only if you have it). Pin the top three: PyTorch,
LLM, Python.

About section, add one line at the top:

```
I build ML systems from the engine up. Most recently: Pacer, an LLM inference server whose scheduler uses online conformal prediction to keep latency promises when the hardware slows down.
```

## 5. Post about it

Use the post draft in the Pacer thread. Tips that work:

- Post Tuesday to Thursday, 8 to 10 am in your time zone.
- Attach the drift chart image (`docs/figures/drift.png`) or a 20-second screen recording
  of `pacer demo`. Posts with a visual get far more reach than text only.
- Put the GitHub link in the **first comment**, not the post body. LinkedIn shows posts
  with external links to fewer people.
- Reply to every comment in the first two hours.
- Don't tag companies or recruiters in the post itself.

## 6. After posting

- Send the repo link in connection requests to engineers on NVIDIA's inference teams
  (TensorRT-LLM, Triton, Dynamo), quant firm infra teams, and FAANG ML infra. One line:
  "I built an LLM serving scheduler with online conformal risk control. I'd value your
  take on it: <link>".
- When you run the GPU notebook, post a short follow-up with the measured GPU numbers.
