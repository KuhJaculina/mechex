#!/usr/bin/env bash
# Single-process launch by default. Under torchrun the same entry point picks up the
# process group and wraps the model, so no code change is needed.
#   torchrun --nproc_per_node=2 scripts/launch_distributed.sh main
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

export PYTHONPATH="$ROOT/src${PYTHONPATH:+:$PYTHONPATH}"

STUDY="${1:-main}"
shift || true

if [[ -n "${WORLD_SIZE:-}" ]]; then
  python -m mechex.console.train --overlay "study/${STUDY}.yaml" "$@"
else
  echo "WORLD_SIZE is unset; launching a single process on CPU."
  python -m mechex.console.train --overlay "study/${STUDY}.yaml" --device cpu "$@"
fi
