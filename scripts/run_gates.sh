#!/usr/bin/env bash
# The three tool gates on their own, for a quick check before a full audit.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

export PYTHONPATH="$ROOT/src${PYTHONPATH:+:$PYTHONPATH}"

python -m pytest -q
python -m ruff check .
python -m ruff format --check .
python -m mypy --strict src/mechex
