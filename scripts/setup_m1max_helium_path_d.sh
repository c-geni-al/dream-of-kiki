#!/usr/bin/env bash
# G6-M1Max Path D — reproducible bootstrap script.
#
# Sets up everything needed to run the G6 Path D pilot on a fresh
# M1 Max boot : (a) downloads Kyutai Helium-1 2B from HF if missing,
# (b) rsyncs the KIKI-Mac_tunner mlx_lm fork from Studio to /tmp/
# (which Path D's driver imports via PYTHONPATH=/tmp), (c) verifies
# the imports resolve correctly. Idempotent — safe to re-run.
#
# Usage ::
#
#     bash scripts/setup_m1max_helium_path_d.sh
#
# Prerequisites :
#   - SSH access to Studio user `clems` at 100.116.92.12 (for the
#     fork rsync). If unavailable, set FORK_SKIP=1 to skip the rsync
#     and rely on a pre-existing /tmp/mlx_lm directory.
#   - `uv` installed (https://docs.astral.sh/uv/).
#   - Python 3.12+ in the dream-of-kiki venv.
#
# Reference :
# - Pre-reg : ``docs/osf-prereg-g6-m1max-path-d.md`` §5
# - Tracking : STATUS.md "As of 2026-05-04 G6 family in flight"
# - Op-finding : ``docs/osf-prereg-g6-studio-path-a-star.md`` §9.2

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
HELIUM_REPO="kyutai/helium-1-2b"
HELIUM_CACHE_PATTERN="$HOME/.cache/huggingface/hub/models--kyutai--helium-1-2b/snapshots"
FORK_SRC="clems@100.116.92.12:/Users/clems/KIKI-Mac_tunner/lib/mlx_lm_fork/"
FORK_DST="/tmp/mlx_lm"

cd "$REPO_ROOT"

echo "==> [1/4] Checking Helium-1 2B HF cache..."
if compgen -G "$HELIUM_CACHE_PATTERN/*" > /dev/null; then
    echo "    Helium-1 2B already cached :"
    ls -d "$HELIUM_CACHE_PATTERN"/*
else
    echo "    Cache miss — downloading from HF (~3.86 GB)..."
    uv run python -c "
from huggingface_hub import snapshot_download
p = snapshot_download('$HELIUM_REPO', allow_patterns=['*.json','*.model','*.safetensors','tokenizer*'])
print(f'    Downloaded to {p}')
"
fi

echo "==> [2/4] Syncing mlx_lm fork from Studio..."
if [[ "${FORK_SKIP:-0}" == "1" ]]; then
    echo "    FORK_SKIP=1 — skipping rsync, relying on existing $FORK_DST"
    if [[ ! -d "$FORK_DST" ]]; then
        echo "    ERROR: $FORK_DST does not exist. Set FORK_SKIP=0 or create it." >&2
        exit 1
    fi
else
    rsync -av -e "ssh -i $HOME/.ssh/id_ed25519" \
        "$FORK_SRC" "$FORK_DST/" \
        | tail -5
fi

echo "==> [3/4] Verifying fork import resolution..."
PYTHONPATH=/tmp uv run python -c "
import mlx_lm
from mlx_lm.tuner.trainer import train
print(f'    mlx_lm location : {mlx_lm.__file__}')
print(f'    train  location : {train.__module__}')
assert mlx_lm.__file__.startswith('/tmp/mlx_lm/'), 'mlx_lm did NOT resolve to fork'
assert train.__module__ == 'mlx_lm.tuner.trainer', 'train not from fork tuner.trainer'
print('    OK : fork is active.')
"

echo "==> [4/4] Verifying Helium-1 2B loads via mlx_lm..."
HELIUM_PATH=$(ls -d "$HELIUM_CACHE_PATTERN"/* | head -1)
PYTHONPATH=/tmp uv run python -c "
from mlx_lm import load
m, _ = load('$HELIUM_PATH')
print(f'    Helium-1 2B loaded : {type(m).__name__}')
"

echo ""
echo "Setup complete. To launch G6 Path D Step 1 (real MMLU) ::"
echo ""
echo "    PYTHONPATH=/tmp DREAM_MICRO_KIKI_REAL=1 \\"
echo "        uv run python experiments/g6_studio_path_a/run_g6_studio_path_a.py \\"
echo "        --n-seeds 5 \\"
echo "        --fixture-path tests/fixtures/mmlu_g6_real.jsonl \\"
echo "        --n-train 50 --n-eval 50 \\"
echo "        --base-path '$HELIUM_PATH' \\"
echo "        --adapter-path /tmp/helium-fresh-adapters \\"
echo "        --out-json docs/milestones/g6-m1max-path-d-mmlu-2026-05-04.json \\"
echo "        --out-md   docs/milestones/g6-m1max-path-d-mmlu-2026-05-04.md"
