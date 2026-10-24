#!/usr/bin/env bash
# Pre-launch OpenRouter credit check: prints the remaining balance and FAILS if it is below the
# floor, so a long run can't silently forfeit through an empty account (the 2026-07-12 exp-7 rerun
# burned 405 games on 402s). Floor defaults to $5; override with CREDIT_MIN. Fails OPEN on network
# problems (a curl blip must not block a launch) and CLOSED on a low balance.
set -euo pipefail

[ -n "${OPENROUTER_API_KEY:-}" ] || { echo "ERROR: OPENROUTER_API_KEY is not set"; exit 1; }

resp=$(curl -s --max-time 15 -H "Authorization: Bearer $OPENROUTER_API_KEY" \
    https://openrouter.ai/api/v1/credits) || { echo "WARN: credit check unreachable; continuing"; exit 0; }

remaining=$(printf '%s' "$resp" | python3 -c \
    'import json,sys; d=json.load(sys.stdin)["data"]; print(round(d["total_credits"]-d["total_usage"], 2))' \
    2>/dev/null) || { echo "WARN: credit response unparseable ($resp); continuing"; exit 0; }

min="${CREDIT_MIN:-5}"
echo "OpenRouter credits remaining: \$$remaining (launch floor: \$$min)"
if python3 -c "import sys; sys.exit(0 if float('$remaining') < float('$min') else 1)"; then
    echo "ERROR: credits below the floor — top up at https://openrouter.ai/settings/credits"
    echo "       (or lower the floor for a small run: CREDIT_MIN=1 bash scripts/launch.sh ...)"
    exit 1
fi
