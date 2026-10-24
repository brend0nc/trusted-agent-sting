"""Build the results site: derive JSON from .eval logs, draw figures, render HTML.

Run with the project venv (needs inspect_ai):
    PYTHONPATH=src .venv/bin/python site/build.py [--skip-derive]
"""

import argparse
import subprocess
import sys
from pathlib import Path

SITE = Path(__file__).parent


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--skip-derive", action="store_true", help="reuse existing site/data")
    args = parser.parse_args()

    steps = [SITE / "derive.py", SITE / "figures.py", SITE / "render.py"]
    if args.skip_derive:
        steps = steps[1:]
    for step in steps:
        print(f"== {step.name}")
        result = subprocess.run([sys.executable, str(step)])
        if result.returncode != 0:
            print(f"{step.name} failed ({result.returncode})", file=sys.stderr)
            return result.returncode
    print(f"== done: {SITE / 'dist'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
