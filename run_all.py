# -*- coding: utf-8 -*-
"""
Master Runner Script for COTS enhm Analysis
涓€閿繍琛屽叏娴佺▼锛氭牳蹇冨垎鏋?-> 鍥捐〃鐢熸垚
"""

import argparse
import subprocess
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent
SCRIPTS_DIR = PROJECT_ROOT / "scripts"
PYTHON = sys.executable

# Default public workflow for the formal manuscript-supporting pipeline.
ANALYSIS_SCRIPTS = [
    ("01 - Core enhm Analysis", "01_enhm_analysis.py"),
]

FIGURE_SCRIPTS = [
    ("03 - Figure 1: Study Area Map", "03_study_area_map.py"),
    ("02 - Figure 2-3: Mechanism & Periodicity", "02_paper_figures.py"),
    ("04 - Figure 4: Spatial Heterogeneity", "04_spatial_heterogeneity.py"),
    ("05 - Figure 5: Propagation Dynamics", "05_propagation.py"),
    ("06 - Figure 6: IPO Mechanism", "06_mechanism_fig6.py"),
    ("Supplementary: Combined Overview", "04_cots_overview.py"),
    ("08 - Supplementary: Model Comparison", "08_model_comparison.py"),
]

CONSISTENCY_SCRIPTS = [
    ("12 - Manuscript Consistency Audit", "12_consistency_audit.py"),
]

AUDIT_SCRIPTS = [
    ("11 - Audit: 242 km/yr Propagation", "11_audit_242.py"),
]


def run_script(script_name, description):
    script_path = SCRIPTS_DIR / script_name
    if not script_path.exists():
        print(f"  [MISSING] {script_name}")
        return False
        
    print(f"\n{'='*60}")
    print(f"  {description}")
    print(f"  Script: {script_name}")
    print(f"{'='*60}")

    try:
        result = subprocess.run(
            [PYTHON, str(script_path)],
            cwd=str(PROJECT_ROOT),
            capture_output=True,
            text=True,
            timeout=600
        )

        if result.returncode == 0:
            lines = result.stdout.strip().split('\n')
            for line in lines[-10:]:
                print(f"  {line}")
            print(f"  [SUCCESS] {description}")
            return True
        else:
            print(f"  [FAILED] {description}")
            stderr = result.stderr[-500:] if len(result.stderr) > 500 else result.stderr
            print(f"  Error: {stderr}")
            return False

    except Exception as e:
        print(f"  [ERROR] {description}: {e}")
        return False


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--figs', action='store_true')
    parser.add_argument('--audit242', action='store_true')
    args = parser.parse_args()

    results = []

    fatal_failure = False

    if not args.figs:
        for desc, script in ANALYSIS_SCRIPTS:
            success = run_script(script, desc)
            results.append((desc, success))
            if not success:
                fatal_failure = True
                break
    
    if not fatal_failure:
        for desc, script in FIGURE_SCRIPTS:
            success = run_script(script, desc)
            results.append((desc, success))
            if not success:
                fatal_failure = True

    if not fatal_failure:
        for desc, script in CONSISTENCY_SCRIPTS:
            success = run_script(script, desc)
            results.append((desc, success))
            if not success:
                fatal_failure = True

    if not fatal_failure and args.audit242:
        for desc, script in AUDIT_SCRIPTS:
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
