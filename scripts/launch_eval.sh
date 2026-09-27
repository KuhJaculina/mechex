#!/usr/bin/env bash
# Evaluate a trained checkpoint on the external site and the reported analyses.
#   scripts/launch_eval.sh runs/seed_20260927/mechex.pt
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

export PYTHONPATH="$ROOT/src${PYTHONPATH:+:$PYTHONPATH}"

CHECKPOINT="${1:-}"
if [[ -z "$CHECKPOINT" ]]; then
  echo "usage: scripts/launch_eval.sh <checkpoint>" >&2
  exit 2
fi
shift

python -m mechex.console.evaluate --checkpoint "$CHECKPOINT" "$@"
