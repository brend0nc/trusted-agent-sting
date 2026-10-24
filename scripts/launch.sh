#!/usr/bin/env bash
# Launch one or more experiments concurrently, each surviving SSH disconnects:
#   bash scripts/launch.sh configs/02_background_check.yaml configs/03_five_principals.yaml
# Each experiment gets its own nohup process + logfile under logs/launcher/.
# Re-running the same native config reuses saved calls and complete games in its log_dir.
# Auto-rebuilds a missing/broken .venv first (a pod restart wipes the local-disk venv).
#
# Watch progress:   tail -f logs/launcher/<name>.out
# List running:     pgrep -fa delegated_influence.run
# Stop one:         pkill -f <config-name>
set -euo pipefail

# The pod's persistent volume can't host a uv venv (cross-filesystem hardlinks fail), so the venv
# lives on local disk at $VENV_REAL and repo-root .venv is a symlink to it. A pod restart wipes
# local disk, leaving .venv a dangling symlink — detect that and rebuild before launching. The
# health check imports inspect_ai (present only after a real install), so a half-built venv also
# rebuilds. Idempotent (skips a working venv) and loud. Override the location with DI_VENV=.
VENV_REAL="${DI_VENV:-/root/di-venv}"
ensure_venv() {
    if .venv/bin/python -c 'import inspect_ai' >/dev/null 2>&1; then
        echo "venv OK: .venv -> $(readlink .venv 2>/dev/null || echo '(real dir)')  [inspect-ai importable]"
        return 0
    fi
    echo "venv MISSING/BROKEN — rebuilding at $VENV_REAL (a pod restart wipes it, leaving .venv dangling) ..."
    command -v uv >/dev/null 2>&1 || { echo "ERROR: uv not on PATH; run scripts/runpod_setup.sh first"; exit 1; }
    rm -rf .venv "$VENV_REAL"          # unlink a dangling symlink / clear a half-built venv
    uv venv --python 3.13 "$VENV_REAL"
    ln -sfn "$VENV_REAL" .venv
    UV_LINK_MODE=copy uv pip install --python .venv/bin/python -e '.[dev]'
    echo "venv REBUILT: .venv -> $VENV_REAL  [UV_LINK_MODE=copy]"
}

[ -n "${OPENROUTER_API_KEY:-}" ] || { echo "ERROR: OPENROUTER_API_KEY is not set"; exit 1; }
bash "$(dirname "$0")/check_credits.sh"   # refuse to launch on an empty account (402 protection)
ensure_venv
mkdir -p logs/launcher

for cfg in "$@"; do
    name=$(basename "$cfg" .yaml)
    if pgrep -f "delegated_influence.run $cfg" >/dev/null 2>&1; then
        echo "SKIP $name — already running"
        continue
    fi
    nohup .venv/bin/python -m delegated_influence.run "$cfg" \
        > "logs/launcher/${name}.out" 2>&1 &
    echo "LAUNCHED $name (pid $!) -> logs/launcher/${name}.out"
done

echo
echo "Note: all experiments share one OpenRouter account, so its rate limits are shared."
echo "Each config has its own budget; add those budgets before launching concurrent runs."
echo "Native progress and billing coverage are in each log_dir/status.json. Provider errors pause runs."
