#!/usr/bin/env bash
# Train one reported configuration. Pass the study file as the first argument.
#   scripts/launch_train.sh main
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

export PYTHONPATH="$ROOT/src${PYTHONPATH:+:$PYTHONPATH}"

STUDY="${1:-main}"
shift || true

python -m mechex.console.train --overlay "study/${STUDY}.yaml" "$@"
