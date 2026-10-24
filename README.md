# Trusted-agent sting

Apart Research AI Collusion Research Sprint, 23 to 25 October 2026. Track 3, question 3.2: can a trusted agent expose or disrupt a coalition?

A trusted agent offers a collusion pact to the other players in a multi-agent ledger game. We record who accepts, who refuses and who becomes suspicious, and we check this against a known coalition (the positive control).

Status: setup. The design is not frozen yet. See `docs/TASKS.md`.

## What is preexisting and what is new

The game engine is our own earlier work. It was written before the sprint for the delegated-influence project, a multi-agent benchmark about persuasion, favours and promises (`docs/ENGINE-README.md`).

| Path | Source |
| --- | --- |
| `src/`, `tests/`, `configs/`, `site/`, `scripts/`, `pyproject.toml`, `uv.lock`, `LICENSE`, `.gitignore` | Copied unchanged from the delegated-influence repo at commit `7e44c638` (2 October 2026), before the sprint started. MIT license. |
| `docs/ENGINE-README.md` | The delegated-influence README at the same commit. |

The first commit holds only that preexisting code. All sprint work is in later commits, and each commit message says what it adds. The report lists the same split.

## Setup

1. Install: `uv sync --extra dev --extra site`.
2. Tests: `PYTHONPATH=src .venv/bin/python -m pytest -q tests`.
3. Model access: see `docs/ENGINE-README.md` for the API keys and the run commands.
