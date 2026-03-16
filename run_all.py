"""
run_all.py — Run the full CTG validation pipeline for both pH thresholds.

Executes each step sequentially for RF, CNN, and LSTM classifiers:
  1. load_data.py           — download/cache features and raw signals
  2. rf_classifier.py       — train Random Forest
  3. cnn_classifier.py      — train 1D CNN
  4. lstm_classifier.py     — train LSTM
  5. falsification.py rf    — falsify RF
  6. falsification.py cnn   — falsify CNN
  7. falsification.py lstm  — falsify LSTM
  8. compare_results.py     — generate comparison tables and plots

Usage:
    python run_all.py
"""

import subprocess
import sys
from pathlib import Path

BASE_DIR = Path(__file__).parent
PYTHON = sys.executable

THRESHOLDS = [7.15, 7.05]

# (script, extra positional args)
STEPS = [
    ("load_data.py",       []),
    ("rf_classifier.py",   []),
    ("cnn_classifier.py",  []),
    ("lstm_classifier.py", []),
    ("falsification.py",   ["rf"]),
    ("falsification.py",   ["cnn"]),
    ("falsification.py",   ["lstm"]),
]


def run(script: str, extra_args: list[str], threshold: float):
    cmd = [
        PYTHON,
        str(BASE_DIR / script),
        *extra_args,
        "--ph-threshold",
        str(threshold),
    ]
    header = "=" * 70
    print(f"\n{header}")
    print(f"  {' '.join(cmd)}")
    print(f"{header}\n")

    result = subprocess.run(cmd, cwd=str(BASE_DIR))
    if result.returncode != 0:
        print(f"\nERROR: {script} exited with code {result.returncode}")
        sys.exit(result.returncode)


def main():
    for threshold in THRESHOLDS:
        print(f"\n{'#' * 70}")
        print(f"#  PIPELINE — pH threshold = {threshold}")
        print(f"{'#' * 70}")

        for script, extra in STEPS:
            run(script, extra, threshold)

    # Final comparison (threshold-independent)
    print(f"\n{'#' * 70}")
    print("#  COMPARISON")
    print(f"{'#' * 70}\n")

    compare_cmd = [PYTHON, str(BASE_DIR / "compare_results.py")]
    result = subprocess.run(compare_cmd, cwd=str(BASE_DIR))
    if result.returncode != 0:
        print(f"\nERROR: compare_results.py exited with code {result.returncode}")
        sys.exit(result.returncode)

    print("\n" + "=" * 70)
    print("ALL DONE.  Results in results/ and plots/.")
    print("=" * 70)


if __name__ == "__main__":
    main()