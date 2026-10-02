#!/usr/bin/env bash
# Every simulator sweep in the README (a few minutes, runs in parallel).
set -euo pipefail
export PYTHONPATH=.
BASE="prefill-first chunked-64 chunked-128 chunked-256 chunked-512 chunked-edf-64 chunked-edf-128 chunked-edf-256 chunked-edf-512 pacer pacer-online"
S="--seeds 0 1 2"
python scripts/bench.py --mode sim --rates 1 2 3 4 5 6 7 8 9 --seeds 0 1 2 3 4 --policies $BASE pacer-slack pacer-edf pacer-fcfs &
python scripts/bench.py --mode sim --rates 3 4 5 6 7 8 $S --tpot 0.06 --policies $BASE --tag _tpot0.06 &
python scripts/bench.py --mode sim --rates 3 4 5 6 7 8 $S --tpot 0.15 --policies $BASE --tag _tpot0.15 &
python scripts/bench.py --mode sim --rates 1 2 3 4 5 6 7 $S --burstiness 3 --policies $BASE --tag _bursty &
wait
python scripts/bench.py --mode sim --trace azure-conv --rates 3 4 5 6 7 8 9 10 --seeds 0 1 2 3 4 --policies $BASE --tag _azure_conv &
python scripts/bench.py --mode sim --trace azure-code --rates 0.25 0.5 0.75 1 1.25 1.5 2 --seeds 0 1 2 3 4 --policies $BASE --tag _azure_code &
python scripts/bench.py --mode sim --rates 4 5 6 7 8 $S --noise 0.08 --policies pacer pacer-risk50 pacer-risk80 pacer-risk95 pacer-risk99 --tag _risk &
for f in 1.5 2.0; do
  python scripts/bench.py --mode sim --rates 3 4 5 6 $S --noise 0.08 --drift 0.3 0.7 $f \
    --policies chunked-128 chunked-edf-128 pacer pacer-online --tag _drift$f &
done
wait
