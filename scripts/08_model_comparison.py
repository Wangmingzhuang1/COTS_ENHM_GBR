# -*- coding: utf-8 -*-
"""
鐢熸垚琛ュ厖鍥? 妯″瀷姣旇緝楠岃瘉
Supplementary Figure: Model Comparison Validation
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

# Configure plot style
plt.rcParams['font.family'] = 'Times New Roman'
plt.rcParams['font.size'] = 10
plt.rcParams['axes.labelsize'] = 11
plt.rcParams['axes.titlesize'] = 12
plt.rcParams['xtick.labelsize'] = 10
plt.rcParams['ytick.labelsize'] = 10
plt.rcParams['svg.fonttype'] = 'none'

# Paths
SCRIPTS_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPTS_DIR.parent
RESULTS_DIR = PROJECT_ROOT / 'results' / 'tables'
SUMMARY_PATH = RESULTS_DIR / 'model_validation_summary.csv'


def ensure_model_validation_summary():
    if SUMMARY_PATH.exists():
        return SUMMARY_PATH

    raise FileNotFoundError(
        "model_validation_summary.csv is missing from the formal results directory: "
        f"{SUMMARY_PATH}. Run the formal enhm/model-validation workflow first."
    )


df = pd.read_csv(ensure_model_validation_summary())

print(f"Loaded {len(df)} reefs for model comparison")
print(f"Columns: {df.columns.tolist()}")

# Create figure with 2x3 layout
fig, axes = plt.subplots(2, 3, figsize=(15, 10))
fig.suptitle('Model Performance Comparison', fontsize=14, fontweight='bold', y=0.98)

# ===== Panel (a): R^2 Distribution =====
ax_r2 = axes[0, 0]
data_r2 = [df['rto_r2'].values, df['gam_r2'].values, df['xgb_r2'].values]
labels_r2 = ['RTO', 'GAM', 'XGBoost']
colors_r2 = ['#6BAED6', '#4DAF4A', '#984EA3']

bp_r2 = ax_r2.boxplot(data_r2, labels=labels_r2, patch_artist=True, widths=0.6,
                       medianprops=dict(color='orange', linewidth=2),
                       flierprops=dict(marker='o', markerfacecolor='black', markersize=5, alpha=0.5))
for patch, color in zip(bp_r2['boxes'], colors_r2):
    patch.set_facecolor(color)
    patch.set_alpha(0.7)

ax_r2.set_ylabel('R^2', fontsize=11, fontweight='bold')
ax_r2.set_title('(a) Model Performance (R^2)', loc='left', fontsize=12, fontweight='bold')
ax_r2.axhline(0, color='gray', linestyle='--', linewidth=0.8)
ax_r2.spines[['top', 'right']].set_visible(False)

# ===== Panel (b): RMSE Distribution =====
ax_rmse = axes[0, 1]
data_rmse = [df['rto_rmse'].values, df['gam_rmse'].values, df['xgb_rmse'].values]
colors_rmse = ['#FB8072', '#B3DE69', '#FDB462']

bp_rmse = ax_rmse.boxplot(data_rmse, labels=labels_r2, patch_artist=True, widths=0.6,
                           medianprops=dict(color='orange', linewidth=2),
                           flierprops=dict(marker='o', markerfacecolor='black', markersize=5, alpha=0.5))
for patch, color in zip(bp_rmse['boxes'], colors_rmse):
    patch.set_facecolor(color)
    patch.set_alpha(0.7)

ax_rmse.set_ylabel('RMSE', fontsize=11, fontweight='bold')
ax_rmse.set_title('(b) Prediction Error (RMSE)', loc='left', fontsize=12, fontweight='bold')
ax_rmse.spines[['top', 'right']].set_visible(False)

# ===== Panel (c): Residual Similarity =====
ax_corr = axes[0, 2]

# Remove null values for plotting
corr_rto_gam = df['corr_rto_gam'].dropna().values
corr_rto_xgb = df['corr_rto_xgb'].dropna().values
corr_gam_xgb = df['corr_gam_xgb'].dropna().values

data_corr = [corr_rto_gam, corr_rto_xgb, corr_gam_xgb]
labels_corr = ['ENHM-GAM', 'ENHM-XGB', 'GAM-XGB']
colors_corr = ['#BEBADA', '#8DD3C7', '#FFFFB3']

bp_corr = ax_corr.boxplot(data_corr, labels=labels_corr, patch_artist=True, widths=0.6,
                           medianprops=dict(color='darkblue', linewidth=2),
                           flierprops=dict(marker='o', markerfacecolor='black', markersize=5, alpha=0.5))
for patch, color in zip(bp_corr['boxes'], colors_corr):
    patch.set_facecolor(color)
    patch.set_alpha(0.7)

ax_corr.axhline(0.8, color='red', linestyle='--', linewidth=1.5, label='High correlation threshold')
ax_corr.set_ylabel('Residual Correlation', fontsize=11, fontweight='bold')
ax_corr.set_title('(c) Residual Similarity', loc='left', fontsize=12, fontweight='bold')
ax_corr.set_ylim(0, 1.05)
ax_corr.spines[['top', 'right']].set_visible(False)

# ===== Panel (d): IPO Detection Rate =====
ax_ipo = axes[1, 0]
n_rto_ipo = df['rto_ipo'].sum()
n_gam_ipo = df['gam_ipo'].sum()
n_xgb_ipo = df['xgb_ipo'].sum()
n_total = len(df)

detection_rates = [n_rto_ipo/n_total*100, n_gam_ipo/n_total*100, n_xgb_ipo/n_total*100]
bars = ax_ipo.bar(labels_r2, detection_rates, color=['#66C2A5', '#FC8D62', '#8DA0CB'], alpha=0.8, width=0.6)

for i, (bar, count, rate) in enumerate(zip(bars, [n_rto_ipo, n_gam_ipo, n_xgb_ipo], detection_rates)):
    height = bar.get_height()
    ax_ipo.text(bar.get_x() + bar.get_width()/2., height + 2,
                f'n={count}', ha='center', va='bottom', fontsize=9, fontweight='bold')

# Add warning text for GAM
ax_ipo.text(1, 20, 'Over-fitting\nSignal Absorption', ha='center', va='center',
            fontsize=9, color='red', bbox=dict(boxstyle='round', facecolor='white', edgecolor='red', linewidth=1.5))

ax_ipo.set_ylabel('Detection Rate (%)', fontsize=11, fontweight='bold')
ax_ipo.set_title('(d) IPO Band (15-30y) Detection', loc='left', fontsize=12, fontweight='bold')
ax_ipo.set_ylim(0, 65)
ax_ipo.spines[['top', 'right']].set_visible(False)

# ===== Panel (e): ENHM vs GAM Period Consistency =====
ax_enhm_gam = axes[1, 1]
valid_rto = df[df['rto_ipo']].copy()
x_e = valid_rto['rto_period'].values
y_g = valid_rto['gam_period'].values

# Remove NaN values
mask = ~(np.isnan(x_e) | np.isnan(y_g))
x_e = x_e[mask]
y_g = y_g[mask]

ax_enhm_gam.scatter(x_e, y_g, s=60, alpha=0.6, color='#4292C6', edgecolors='black', linewidths=0.5)
ax_enhm_gam.plot([0, 35], [0, 35], 'r--', lw=1.5, label='1:1 line')
ax_enhm_gam.set_xlabel('ENHM Primary Period (years)', fontweight='bold')
ax_enhm_gam.set_ylabel('GAM Primary Period (years)', fontweight='bold')
ax_enhm_gam.set_title(f'(e) Period Consistency (ENHM vs GAM)', loc='left', fontsize=12, fontweight='bold')
ax_enhm_gam.set_xlim(0, 36)
ax_enhm_gam.set_ylim(0, 36)
ax_enhm_gam.set_aspect('equal')
ax_enhm_gam.spines[['top', 'right']].set_visible(False)

# ===== Panel (f): ENHM vs XGBoost Period Consistency =====
ax_enhm_xgb = axes[1, 2]
y_x = valid_rto['xgb_period'].values[mask]

ax_enhm_xgb.scatter(x_e, y_x, s=60, alpha=0.6, color='#74C476', edgecolors='black', linewidths=0.5)
ax_enhm_xgb.plot([0, 35], [0, 35], 'r--', lw=1.5, label='1:1 line')
ax_enhm_xgb.set_xlabel('ENHM Primary Period (years)', fontweight='bold')
ax_enhm_xgb.set_ylabel('XGBoost Primary Period (years)', fontweight='bold')
ax_enhm_xgb.set_title(f'(f) Period Consistency (ENHM vs XGBoost)', loc='left', fontsize=12, fontweight='bold')
ax_enhm_xgb.set_xlim(0, 36)
ax_enhm_xgb.set_ylim(0, 36)
ax_enhm_xgb.set_aspect('equal')
ax_enhm_xgb.spines[['top', 'right']].set_visible(False)

plt.tight_layout()

# Save figure
FIGURES_DIR = PROJECT_ROOT / 'results' / 'figures'
output_path_png = FIGURES_DIR / 'SuppFig_Model_Comparison.png'
output_path_pdf = FIGURES_DIR / 'SuppFig_Model_Comparison.pdf'
output_path_svg = FIGURES_DIR / 'SuppFig_Model_Comparison.svg'

plt.savefig(output_path_png, dpi=300, bbox_inches='tight')
plt.savefig(output_path_pdf, dpi=300, bbox_inches='tight')
plt.savefig(output_path_svg, bbox_inches='tight')

print(f"\n[SUCCESS] Supplementary Figure saved:")
print(f"  - {output_path_png}")
print(f"  - {output_path_pdf}")
print(f"  - {output_path_svg}")

# Print summary statistics
print("\n" + "="*60)
print("SUMMARY STATISTICS")
print("="*60)
print(f"\nModel Performance (n={len(df)} reefs):")
print(f"  RTO:      R^2 = {df['rto_r2'].median():.3f} (median), RMSE = {df['rto_rmse'].median():.3f}")
print(f"  GAM:      R^2 = {df['gam_r2'].median():.3f} (median), RMSE = {df['gam_rmse'].median():.3f}")
print(f"  XGBoost:  R^2 = {df['xgb_r2'].median():.3f} (median), RMSE = {df['xgb_rmse'].median():.3f}")

print(f"\nResidual Correlation:")
print(f"  ENHM-GAM:    r = {df['corr_rto_gam'].median():.3f} (median)")
print(f"  ENHM-XGBoost: r = {df['corr_rto_xgb'].median():.3f} (median)")
print(f"  GAM-XGBoost:  r = {df['corr_gam_xgb'].median():.3f} (median)")

print(f"\nIPO Detection (15-30 years):")
print(f"  ENHM:     {n_rto_ipo}/{n_total} ({n_rto_ipo/n_total*100:.1f}%)")
print(f"  GAM:      {n_gam_ipo}/{n_total} ({n_gam_ipo/n_total*100:.1f}%)")
print(f"  XGBoost:  {n_xgb_ipo}/{n_total} ({n_xgb_ipo/n_total*100:.1f}%)")
print("="*60)

plt.show()
