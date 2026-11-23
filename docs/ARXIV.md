# Posting the paper on arXiv

The paper is `docs/paper/pacer.pdf`, built from `docs/paper/pacer.html` (not from TeX, so
arXiv accepts the PDF on its own). Only the author can submit it: arXiv needs your account,
your endorsement and your agreement to its license.

## 1. Get an endorser (required for a first paper)

Since January 21, 2026, a university email alone no longer qualifies a new author. You need a
personal endorsement from an established arXiv author in computer science
([arXiv blog](https://blog.arxiv.org/2026/01/21/attention-authors-updated-endorsement-policy)).
arXiv staff will not endorse anyone.

1. Ask one USF professor who works in the area (see the plan doc). Send the PDF and the repo.
2. Do not email many endorsers at once; arXiv calls that inappropriate.

## 2. Submit

1. Create an account at https://arxiv.org/user/register with your usf.edu email.
2. Start a new submission. Primary category: **cs.DC** (Distributed, Parallel, and Cluster
   Computing). Cross-list: **cs.LG**.
3. arXiv shows an endorsement request with a 6-character code. Send that code to your endorser.
   They enter it at https://arxiv.org/auth/endorse.
4. License: CC BY 4.0 (lets anyone reuse with credit; the usual choice).
5. Upload `pacer.pdf`.
6. Metadata:
   - **Title:** Stale Predictors in LLM Serving: Online Conformal Control under Drift
   - **Authors:** Parveen Kumar Naresh Kumar (add your advisor only if they agree to be a co-author)
   - **Abstract:** copy it from the PDF as plain text (no line breaks inside sentences).
   - **Comments:** `7 pages, 1 figure, 7 tables. Code: https://github.com/parveen06naresh/pacer-serve`
7. Preview the PDF arXiv generates, then submit. Papers announced the next weekday evening
   (US Eastern) if submitted before the 14:00 ET cutoff.

## Rules to remember

- Rate limit: 2 submissions per calendar month, 3 active at once
  ([arXiv blog](https://blog.arxiv.org/2026/10/01/updated-rate-limit-policy/)).
- The paper's AI usage disclosure must stay: arXiv sanctions unchecked AI-generated text.
  Read every sentence and be ready to defend every number.
- If you later submit to MLSys (double-blind), its call asks that an arXiv version use a
  different system name and title during review.

## Before you submit

- [ ] Run the Colab notebook (`notebooks/gpu_benchmark.ipynb`) and decide whether to add the
      GPU table to the paper first. Strongly recommended: it answers the first question every
      reviewer will ask.
- [ ] Have your advisor or endorser read the PDF.
- [ ] Re-read Section 6 (Limitations). It is what makes the paper credible.
