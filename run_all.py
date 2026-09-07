"""Run the public COTS ENHM workflow."""

import argparse
import subprocess
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent
SCRIPTS_DIR = PROJECT_ROOT / "scripts"
PYTHON = sys.executable

ANALYSIS_SCRIPTS = [
    ("Core ENHM analysis", "01_enhm_analysis.py"),
]

ROBUSTNESS_SCRIPTS = [
    ("Hierarchical null-model comparison", "13_robustness_null_comparison.py"),
]

FIGURE_SCRIPTS = [
    ("Figure 1: study area and outbreak context", "04_cots_overview.py"),
    ("Figure 2 and Figures S2/S5: residual periodicity", "02_paper_figures.py"),
    ("Figure S3: spatial heterogeneity", "04_spatial_heterogeneity.py"),
    ("Figure 3: phase organization", "05_propagation.py"),
    ("Figure 4: hydroclimatic coupling", "06_mechanism_fig6.py"),
    ("Figure S4: model comparison", "08_model_comparison.py"),
]


def run_script(script_name, description):
    script_path = SCRIPTS_DIR / script_name
    if not script_path.exists():
        print(f"  [MISSING] {script_name}")
        return False

    print(f"\n{'=' * 60}")
    print(f"  {description}")
    print(f"  Script: {script_name}")
    print(f"{'=' * 60}")

    try:
        result = subprocess.run(
            [PYTHON, str(script_path)],
            cwd=str(PROJECT_ROOT),
            capture_output=True,
            text=True,
            timeout=600,
        )
    except Exception as exc:
        print(f"  [ERROR] {description}: {exc}")
        return False

    if result.returncode == 0:
        for line in result.stdout.strip().split("\n")[-10:]:
            print(f"  {line}")
        print(f"  [SUCCESS] {description}")
        return True

    print(f"  [FAILED] {description}")
    stderr = result.stderr[-1000:] if len(result.stderr) > 1000 else result.stderr
    print(f"  Error: {stderr}")
    return False


def main():
    parser = argparse.ArgumentParser(
        description="Run the compact-data analysis, with optional robustness and figure stages."
    )
    parser.add_argument(
        "--robustness",
        action="store_true",
        help="run the hierarchical white-noise, calendar-block, and AR(1) checks",
    )
    parser.add_argument(
        "--figures",
        action="store_true",
        help="also draw manuscript figures; requires the external data listed in data/README.md",
    )
    args = parser.parse_args()

    results = []
    fatal_failure = False

    for desc, script in ANALYSIS_SCRIPTS:
        success = run_script(script, desc)
        results.append((desc, success))
        if not success:
            fatal_failure = True
            break

    if not fatal_failure and args.robustness:
        for desc, script in ROBUSTNESS_SCRIPTS:
            success = run_script(script, desc)
            results.append((desc, success))
            if not success:
                fatal_failure = True

    if not fatal_failure and args.figures:
        for desc, script in FIGURE_SCRIPTS:
            success = run_script(script, desc)
            results.append((desc, success))
            if not success:
                fatal_failure = True

    print("\n" + "=" * 60)
    print("  EXECUTION SUMMARY")
    print("=" * 60)
    for desc, success in results:
        print(f"    {'[OK]' if success else '[FAIL]'}  {desc}")
    print("=" * 60)

    if fatal_failure:
        sys.exit(1)


if __name__ == "__main__":
    main()
