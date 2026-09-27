#!/usr/bin/env bash
# Build the schema-compatible cohort fixture and write it under data/.
# The primary clinical cohort is not distributed; see dataset_urls.txt.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

export PYTHONPATH="$ROOT/src${PYTHONPATH:+:$PYTHONPATH}"

python - "$ROOT" <<'PY'
import sys
from pathlib import Path

from mechex.cohort.construction import build_cohort, stage
from mechex.provenance.artefacts import default_root
from mechex.settings import load_config

root = Path(sys.argv[1]) if len(sys.argv) > 1 else default_root()
config = load_config(root / "configs")
staged = build_cohort(config)
stage(staged, root / "data")
print(f"fixture written: {len(staged.cohort)} records, split {staged.split.as_dict()}")
PY
