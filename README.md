# COTS ENHM GBR: dated-survey analysis and source data

Version **2026.11.25** accompanies *The Interdecadal Pacific Oscillation Organizes Starfish Outbreak Windows on the Great Barrier Reef*. This version replaces the earlier calendar-year workflow with the exact dated-survey pipeline used by the current manuscript.

## Reproduce the core analysis

Use Python 3.12. Install `requirements-tested.txt` for the package versions recorded in the analysis environment, or `requirements.txt` for compatible minimum versions. Cartopy and other spatial packages are used for optional figure stages.

```bash
python -m venv .venv
# Windows
.venv/Scripts/python -m pip install -r requirements-tested.txt
.venv/Scripts/python run_all.py
# macOS/Linux: use .venv/bin/python instead
```

The first run safely expands `data/aims_local_stock/ecology_snapshot_20261120.zip` and verifies every file against the SHA-256 manifest. The archive preserves the exact 6,104-row observation table, reef coordinates, and 335 referenced unmodified AIMS dashboard exports. The table includes both manta and photo-transect cover; the analysis **explicitly selects LTMP manta hard-coral cover**, not photo-transect cover. No accounts or gridded downloads are required for the core analysis.

The default command fits the baseline, runs exact floating-mean Lomb–Scargle screening and the spectral null hierarchy, then audits original source rows through to results. Expected checkpoints are:

| Quantity | Reference value |
| --- | ---: |
| Coordinate-bearing COTS reefs | 113 |
| Reefs with both COTS and manta hard-coral data | 112 |
| Fitted reefs / dated lag-3 pairs | 92 / 1,051 |
| Spectrally screened reefs | 58 |
| White-noise significant / IPO-band significant reefs | 15 / 12 |
| 3-calendar-year block significant / IPO-band reefs | 9 / 6 |
| Continuous-time AR(1) significant / IPO-band reefs | 1 / 0 |
| Mean centered / non-centered R² (80 defined reefs) | 0.076 / 0.310 |
| Primary 12-reef circular phase R / apparent speed | 0.876 / 31.2 km/year |

`reference_results/tables/` contains the manuscript output snapshot. Newly computed outputs go to ignored `results/`, so reference files are not overwritten.

## Methods encoded by this version

- Every COTS survey is paired to the nearest strictly prior coral survey within ±183 days of the nominal 3-year anniversary. Alternative lags use the same dated matching rule, rather than shifting observation rows or averaging first into calendar years.
- The 1–4-year lag profile holds the 30-reef membership and response survey dates constant. The empirical peak is at 2 years; 3 years is the biologically motivated default, with 2-year sensitivity.
- Lomb–Scargle uses actual survey dates, `method="slow"`, a 1,000-point uniform frequency grid, at least five residual observations, and periods from 3 years to `min(30 years, record span)`.
- The primary white-noise screen uses 5,000 permutations and `(k+1)/(N+1)`. Short-memory and continuous-time AR(1) nulls are also retained; single-reef IPO-band periodicity is not robust to the latter.
- Circular–linear phase scans avoid ordinary phase unwrapping. All-58-reef and 2-year-lag diagnostics explicitly show competing branches. Reported approximately 30–32 km/year sensitivity values refer to the **north-to-south branch**. Its plotted bootstrap interval is conditional on that branch, not uncertainty across all competing solutions.
- Annual unfiltered TPI retains interannual variance. Environmental composites are climate-state associations, and Eulerian meridional velocity does not directly estimate residence times or larval trajectories.

## Optional stages

```bash
python run_all.py --robustness
python run_all.py --audit-only
python run_all.py --environment
python run_all.py --figures
python run_all.py --supplementary
```

`--robustness` adds baseline, lag, phase-null and all-reef phase sensitivity analyses, Table S1 and the phase-sensitivity figure. It does not require the external environmental grids.

`--environment` exports the exact environmental composite fields, annual chlorophyll/TPI and discharge series, nearest-grid-cell reef currents and regional summaries. `--figures` generates the current main and supplementary figures and requires the provider grids and spatial resources listed in [data/README.md](data/README.md). Cartopy may need its usual Natural Earth assets on first use. Times New Roman is the original figure font; an unavailable font may change appearance but not numeric analyses.

`--supplementary` runs additional reef-level climate comparisons and growth/threshold-transition diagnostics. These have different estimands from the primary spectral/phase analysis and are not an independently validated outbreak forecasting model. They are included for transparency.

## Source data and figure mapping

| Paper display | Implementation | Public supporting data |
| --- | --- | --- |
| Figure 1 | `04_cots_overview.py` | AIMS snapshot; reef coordinates; 2026 is partial through February |
| Figure 2; Figure S2 and S4 | `02_paper_figures.py` | matched observations, model and spectral tables |
| Figure 3 | `05_propagation.py` | residual/period/phase tables; `source_data/environment/` current composites |
| Figure 4 | `06_mechanism_fig6.py` | `source_data/environment/` arrays and annual series; reference environmental tests |
| Figure S1 | `16_conceptual_susceptibility_schematic.py` | conceptual interpretation, not a reconstructed circulation or particle field |
| Figure S3 | `04_spatial_heterogeneity.py` | model and spectral tables; external spatial backgrounds |
| Figure S5; Table S1 | `13_robustness_null_comparison.py` | phase sensitivity and complete screened-reef tables |

File stems retain historical numbering where necessary; this table maps them to the current paper. The obsolete calendar-year implementation is not an alternative entry point for reproducing this release.

## Data, attribution and citation

The software is MIT-licensed under [LICENSE](LICENSE). Source and derived data retain their providers' attribution and applicable terms; the software license does not relicense AIMS, NOAA, BoM, Queensland, Copernicus or GBRMPA data. [data/README.md](data/README.md) records source URLs, staging paths and exclusions. [CITATION.cff](CITATION.cff) identifies this software version; no archival DOI has yet been assigned.
