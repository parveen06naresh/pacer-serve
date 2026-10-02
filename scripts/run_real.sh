#!/usr/bin/env bash
# Every real-engine measurement in the README. Needs a quiet machine (~50 min on 4 CPU cores).
set -euo pipefail
export PYTHONPATH=.
ALL="prefill-first chunked-64 chunked-128 chunked-256 chunked-512 chunked-edf-128 pacer pacer-online"
python scripts/bench.py --mode real --rates 3 4 5 6 7 --policies $ALL
python scripts/bench.py --mode real --trace azure-conv --rates 5 7 9 --policies chunked-128 chunked-edf-128 pacer pacer-online --tag _azure_conv
python scripts/bench.py --mode real --rates 4 5 --hog 0.3 0.7 0.35 --policies chunked-128 chunked-edf-128 pacer pacer-online --tag _hog
