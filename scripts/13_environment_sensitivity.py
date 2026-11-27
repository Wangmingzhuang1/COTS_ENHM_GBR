"""Recompute environmental sensitivities after provider grids have been staged."""
import importlib.util
from pathlib import Path
import shutil
import pandas as pd
from config import PROJECT_ROOT, RESULTS_DIR, TABLES_DIR
spec=importlib.util.spec_from_file_location("robustness",Path(__file__).with_name("13_robustness_null_comparison.py"))
module=importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)
dest=RESULTS_DIR/"source_data"
dest.mkdir(parents=True,exist_ok=True)
for p in (PROJECT_ROOT/"source_data/environment").iterdir():
    if p.is_file():shutil.copy2(p,dest/p.name)
module.run_environmental_sensitivity(pd.read_csv(TABLES_DIR/"period_results.csv"),pd.read_csv(TABLES_DIR/"all_residuals.csv")).to_csv(TABLES_DIR/"ndjf_and_tpi_sensitivity.csv",index=False)
