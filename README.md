# COTS ENHM GBR

Dated-survey crown-of-thorns starfish analysis and source data for the Great Barrier Reef. Version **2026.11.27** synchronizes the November 27 implementation and reference tables. It supersedes v2026.11.25 (15 significant / 12 IPO-band reefs).

## Reproduce

Use Python 3.12. Install `requirements-tested.txt` for recorded versions or `requirements.txt` for compatible minimum versions. Run from the repository root:

```bash
python run_all.py
python run_all.py --robustness
```

The first command verifies and expands the included ecological snapshot, fits the dated baseline, runs 10,000-permutation screening, and audits source rows through to results. The second also runs baseline, spectral-field, phase, lag, fixed-period, unfolded-time and direct-IPO diagnostics without large environmental grids. Each invocation reruns the core analysis. Computed outputs go to `results/`; frozen comparison tables are in `reference_results/tables/`.

| Checkpoint | Reference |
| --- | ---: |
| Coordinate-bearing COTS reefs | 113 |
| Reefs with COTS and manta hard-coral records | 112 |
| Fitted reefs / dated lag-3 pairs | 92 / 1,051 |
| Analyzable reefs | 58 |
| White-noise significant / IPO-band reefs | 16 / 13 |
| IPO fraction among white-noise detections | 81.25% |
| Three-calendar-year block significant / IPO-band reefs | 9 / 6 |
| Continuous-time AR(1) significant / IPO-band reefs | 1 / 0 |
| Mean centered / non-centered R² (80 defined reefs) | 0.076 / 0.310 |
| Common period of the 13 IPO-band reefs | 21.3 years |
| All-58 weighted southward phase speed / R | 31.2 km/year / 0.601 |
| 13-reef subset southward phase speed / R | 31.0 km/year / 0.874 |

## Definitions and limits

COTS abundance is count divided by tow effort. The baseline selects **LTMP manta hard-coral cover**, not photo-transect cover. Each response survey is matched to the nearest strictly prior coral survey within ±183 days of its nominal three-year anniversary. The 30-reef lag-comparison cohort is a separate subset, not an intermediate filter yielding the 58 analyzable reefs.

Lomb–Scargle uses actual dates, exact floating-mean power, and a 1,000-point uniform frequency grid covering periods from 3 years to min(30 years, record span), with at least five observations. White-noise screening uses 10,000 permutations, `(k+1)/(N+1)`, and deterministic per-reef random streams. Four baseline specifications include raw log abundance, regression through the origin, nonnegative OLS with intercept, and log-transformed coral cover.

Circular–linear fits avoid ordinary phase unwrapping. Reported southward fits and bootstrap intervals are conditional on the selected branch; competing solutions remain in the diagnostics. Two-year-lag unconstrained fits favor the opposite branch. The 21.3-year common period comes from the screened IPO subset; fixed-period sensitivity is separate.

White-noise screening is permissive. Pooled IPO-band power rejects white-noise and three-year block nulls but **does not reject AR(1)** (p approximately 0.140). Spatial nulls use independent reef surrogates. Spatial ordering and an IPO-band period alone do not establish IPO causation. `direct_ipo_phase_locking_check.csv` retains direct correspondence tests, including nonsignificant results. This release does not certify causal attribution or validated forecasting lead time.

## Optional stages

```bash
python run_all.py --audit-only
python run_all.py --environment
python run_all.py --figures
python run_all.py --supplementary
```

`--environment` regenerates environmental source data and annual/NDJF sensitivities using monthly grids and discharge records listed in `data/README.md`. `--figures` additionally requires GIS layers and Cartopy assets. `--robustness` explicitly skips environmental recomputation rather than substituting archived outputs for new computations. `--supplementary` runs additional reef-level climate and growth/threshold diagnostics with different estimands.

Included `source_data/environment/` arrays support inspection of annual composites; they do not replace all monthly fields required for NDJF regeneration. Annual unfiltered TPI contains interannual variability. The 13-year Butterworth sensitivity is a separate transformation, and removing linear Niño3.4 does not isolate every ENSO contribution. Eulerian meridional velocity does not measure residence time, flushing rate, or larval settlement.

## Files

- `data/aims_local_stock/ecology_snapshot_20261120.zip`: unchanged 6,104-row observation table, coordinates, and 335 AIMS dashboard exports; all 337 members verified against the snapshot manifest.
- `scripts/` and `run_all.py`: versioned analysis and public entry point.
- `reference_results/tables/`: November 27 reference outputs, including bidirectional lags, field significance, environmental, fixed-period, unfolded-time and direct-IPO diagnostics.
- `source_data/environment/`: derived annual environmental fields and series.
- `REPRODUCTION.json`: checks actually performed, distinguished from archived reference results.
- `FILES_SHA256.json`: bytes and SHA-256 of release files, excluding itself and generated outputs.

Historical stems map as follows: `05_propagation.py` produces paper Figure 3; `06_mechanism_fig6.py` Figure 4; robustness phase-sensitivity plotting produces SI Figure S5. Manuscripts, private reviews, credentials, unrelated backups, and large provider grids are excluded.

## Attribution and license

Software is MIT licensed. Source and derived data retain provider attribution requirements and terms; this license does not relicense AIMS, NOAA, BoM, Queensland, Copernicus or GBRMPA data. AIMS dashboard exports include provider-modelled coral-cover estimates and must not be relabelled as raw tow-level observations. Provider metadata: [10.25845/q40d-7a72](https://doi.org/10.25845/q40d-7a72). Access routes are in `data/README.md`; version citation is in `CITATION.cff`. No archival DOI has been assigned.

## Changes in this version

Updated deterministic spectral screening and reference tables from 15/12 at N=5,000 to 16/13 at N=10,000; added four-baseline, bidirectional-lag, field-significance, seasonal, fixed-period, unfolded-time and direct-IPO diagnostics. Earlier v2026.11.25 remains at commit e95af1f89459fb15637c166a4fdf23c8907f2c6f. This synchronization does not imply that scientific review concerns have all been resolved.
