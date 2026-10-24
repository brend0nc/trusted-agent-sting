#!/usr/bin/env bash
# One-time setup on a fresh RunPod pod (also safe to re-run after a restart). Run from the repo root:
#   bash scripts/runpod_setup.sh
# Assumes the repo is cloned under the persistent volume (e.g. /workspace/delegated-influence)
# so logs/ survives pod restarts, and OPENROUTER_API_KEY is set as a pod environment variable.
# The venv itself lives on LOCAL disk ($VENV_REAL) — the persistent volume can't host a uv venv
# (cross-filesystem hardlinks fail) — so a pod restart wipes it; re-running this (or launch.sh)
# rebuilds it. Idempotent and loud.
set -euo pipefail

# 1. uv (installs to ~/.local/bin)
if ! command -v uv >/dev/null 2>&1; then
    curl -LsSf https://astral.sh/uv/install.sh | sh
    export PATH="$HOME/.local/bin:$PATH"
fi

# 2. standard (GIL) CPython 3.13 venv + deps (inspect-ai has no free-threaded wheels), on local disk
#    with a repo-root .venv symlink. The health check imports inspect_ai (present only after a real
#    install), so a missing/dangling/half-built venv rebuilds; a working one is left untouched.
VENV_REAL="${DI_VENV:-/root/di-venv}"
ensure_venv() {
    if .venv/bin/python -c 'import inspect_ai' >/dev/null 2>&1; then
        echo "venv OK: .venv -> $(readlink .venv 2>/dev/null || echo '(real dir)')  [inspect-ai importable]"
        return 0
    fi
    echo "venv MISSING/BROKEN — rebuilding at $VENV_REAL (a pod restart wipes it, leaving .venv dangling) ..."
    rm -rf .venv "$VENV_REAL"          # unlink a dangling symlink / clear a half-built venv
    uv venv --python 3.13 "$VENV_REAL"
    ln -sfn "$VENV_REAL" .venv
    UV_LINK_MODE=copy uv pip install --python .venv/bin/python -e '.[dev]'
    echo "venv REBUILT: .venv -> $VENV_REAL  [UV_LINK_MODE=copy]"
}
ensure_venv

# 3. sanity: key present, credits above the floor, tests green, every config builds and prices
[ -n "${OPENROUTER_API_KEY:-}" ] || { echo "ERROR: OPENROUTER_API_KEY is not set"; exit 1; }
bash "$(dirname "$0")/check_credits.sh"
PYTHONPATH=src .venv/bin/python -m pytest -q
for f in configs/*.yaml; do
    PYTHONPATH=src .venv/bin/python -m delegated_influence.run "$f" --dry-run | tail -1 | sed "s|^|$f |"
done

echo "Setup OK. Launch experiments with: bash scripts/launch.sh <config> [<config> ...]"
