#!/usr/bin/env bash
# Verify the release and rewrite the three root artefacts plus the text summary.
# Run it twice: the manifest-freshness check can only pass on the second run, because the
# first run's tree differs from the manifest it reads.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

export PYTHONPATH="$ROOT/src${PYTHONPATH:+:$PYTHONPATH}"

python -m mechex.console.audit --print-summary "$@"
