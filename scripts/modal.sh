#!/usr/bin/env bash
# Run one command with the Modal token in env, pinned to OUR environment.
#   scripts/modal.sh modal app list
#   scripts/modal.sh python some_launcher.py
# The token is decrypted in memory from the encrypted key file. Nothing is written to disk.
# Blocks modal subcommands that can reach resources outside our environment.
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
KEY_FILE="${MODAL_KEY_FILE:-$ROOT/UtqedEcK.txt}"
# Passphrase order: environment, then the macOS Keychain item, then an interactive prompt.
# Store it once with: security add-generic-password -s di-modal-key-passphrase -a "$USER" -w
if [ -z "${MODAL_KEY_PASSPHRASE:-}" ]; then
  MODAL_KEY_PASSPHRASE=$(security find-generic-password -s di-modal-key-passphrase -w 2>/dev/null || true)
fi
if [ -z "${MODAL_KEY_PASSPHRASE:-}" ]; then
  read -r -s -p "Modal key passphrase: " MODAL_KEY_PASSPHRASE; echo >&2
fi
export MODAL_KEY_PASSPHRASE
PT=$(sed -n '3p' "$KEY_FILE" | openssl enc -d -aes-256-cbc -pbkdf2 -iter 100000 -base64 -A -pass env:MODAL_KEY_PASSPHRASE)
unset MODAL_KEY_PASSPHRASE
MODAL_TOKEN_ID=$(printf '%s\n' "$PT" | grep -o 'ak-[A-Za-z0-9_-]*' | head -1)
MODAL_TOKEN_SECRET=$(printf '%s\n' "$PT" | grep -o 'as-[A-Za-z0-9_-]*' | head -1)
unset PT
[ -n "$MODAL_TOKEN_ID" ] && [ -n "$MODAL_TOKEN_SECRET" ] || { echo "modal.sh: could not parse token" >&2; exit 1; }
export MODAL_TOKEN_ID MODAL_TOKEN_SECRET
export MODAL_ENVIRONMENT=delegated-influence      # every call lands in our environment
export MODAL_CONFIG_PATH=/nonexistent/modal.toml  # ~/.modal.toml is never read or written

if [ "${1:-}" = "modal" ]; then
  case "${2:-}" in
    shell|container|token|config|profile|workspace|nfs|launch|endpoint|curl)
      echo "modal.sh: blocked 'modal $2' (acts by id or on the workspace, outside our environment)" >&2; exit 2;;
    environment)
      case "${3:-}" in list|create) ;; *) echo "modal.sh: blocked 'modal environment ${3:-}'" >&2; exit 2;; esac;;
  esac
  for a in "$@"; do
    case "$a" in --env|--env=*|-e) echo "modal.sh: blocked '$a'; the wrapper pins MODAL_ENVIRONMENT" >&2; exit 2;; esac
  done
fi
exec "$@"
