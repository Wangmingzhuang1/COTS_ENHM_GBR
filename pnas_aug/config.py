from pathlib import Path
PROJECT_ROOT=Path(__file__).resolve().parents[1]
TABLES_DIR=PROJECT_ROOT/'results/supplementary/tables'
FIGURES_DIR=PROJECT_ROOT/'results/supplementary/figures'
SUMMARY_DIR=PROJECT_ROOT/'results/supplementary/summary'
ALL_RESIDUALS=PROJECT_ROOT/'results/tables/all_residuals.csv'
MODEL_RESULTS=PROJECT_ROOT/'results/tables/model_results.csv'
for p in [TABLES_DIR,FIGURES_DIR,SUMMARY_DIR]:
    p.mkdir(parents=True,exist_ok=True)
