#!/usr/bin/env bash
# Sprint work: start one config in the background with a log file, and return at once.
#   bash scripts/run_bg.sh configs/sting_pilot.yaml
# API keys come from compute.env (gitignored). Re-running resumes saved calls and complete games.
# Watch:   tail -f <log printed below>
# Status:  uv run python -m delegated_influence.run configs/NAME.yaml --status   (no model calls)
# Stop:    pkill -f "delegated_influence.run configs/NAME.yaml"   (paused games resume on the next start)
set -euo pipefail
cd "$(dirname "$0")/.."
cfg=${1:?usage: bash scripts/run_bg.sh configs/NAME.yaml}
[ -f "$cfg" ] || { echo "no such config: $cfg"; exit 1; }
grep -q '^OPENROUTER_API_KEY=.' compute.env 2>/dev/null || { echo "compute.env has no OPENROUTER_API_KEY"; exit 1; }
name=$(basename "$cfg" .yaml)
# Match the config by file name, so ./configs/x.yaml and configs/x.yaml count as the same run.
if pgrep -f "delegated_influence\.run .*${name}\.yaml" >/dev/null; then
    echo "already running: $cfg"
    exit 1
fi
# The run log goes next to the run's own data (the config's log_dir).
log_dir=$(sed -n 's/^log_dir:[[:space:]]*//p' "$cfg" | head -1)
log_dir=${log_dir:-logs/$name}
mkdir -p "$log_dir"
log="$log_dir/run-$(date -u +%Y%m%dT%H%M%SZ).log"
nohup uv run --env-file compute.env python -m delegated_influence.run "$cfg" > "$log" 2>&1 &
echo "started $cfg (pid $!)"
echo "log:   $log"
echo "watch: tail -f $log"
