# Ecological Null Hypothesis Model (ENHM) for Crown-of-Thorns Starfish Outbreaks

This repository contains the core analytical pipeline for separating coral-limited trophic baselines from environmentally forced ecological anomalies on the Great Barrier Reef (GBR). The framework uses the Ecological Null Hypothesis Model (ENHM), biological stripping, and residual spectral analysis as presented in the associated manuscript.

## Overview

A central challenge in ecological forecasting is distinguishing slowly varying climatic background conditions from immediate outbreak triggers. Because crown-of-thorns starfish (COTS) are obligate corallivores, raw abundance series often mix climatic forcing with prey depletion and recovery cycles.

This pipeline implements **ENHM** to address that confound explicitly:

1. **Biological Stripping:** Evaluates the empirical delay between coral resources and COTS abundance, establishing a 2-year lag as the optimal trophic baseline.
2. **Residual Separation:** Removes density-dependent constraints by fitting a regression-through-the-origin baseline model.
3. **Frequency and Propagation Extraction:** Evaluates residual periodicity using iteratively applied Double Lomb-Scargle (DLS) analysis to identify low-frequency interdecadal structure, then tracks phase anomalies spatially.
4. **Climate Coupling:** Tests Wavelet Coherence Analysis (WCA) against teleconnection indices such as the Interdecadal Pacific Oscillation (IPO) and El Nino-Southern Oscillation (ENSO).

## Project Structure

```text
COTS_ENHM_GBR/
|-- data/                 # Empty staging tree plus download instructions
|-- scripts/              # Core analytic Python scripts
|-- run_all.py            # Pipeline entry point
|-- requirements.txt      # Python dependencies
`-- README.md
```

## Data Availability

This public repository does not include raw project datasets. Populate the `data/` directory locally before executing the pipeline. The repository is configured to ignore staged data files so that only code and documentation are tracked in git.

### Required data sources

- **AIMS LTMP Data (Manta Tow):** Open-access data available from the [AIMS eAtlas or AIMS monitoring portal](https://apps.aims.gov.au/reef-monitoring/).
- **Climate Indices (IPO and ENSO):** The Tripole Index (TPI) for the IPO should be downloaded from the [NOAA Physical Sciences Laboratory](https://psl.noaa.gov/data/timeseries/IPOTPI/).
- **Hydrological Data (Rainfall and Runoff):** Rainfall inputs come from the Australian Bureau of Meteorology archives. River discharge gauge data comes from the Queensland Water Monitoring Information Portal.
- **Oceanographic Grids (Chlorophyll and Currents):** Obtain from the CSIRO [eReefs model products](https://ereefs.org.au/ereefs).
- **Spatial Boundaries:** NRM and reef shapefiles should be downloaded from Great Barrier Reef Marine Park Authority spatial data sources.

Detailed local-path expectations are listed in [data/README.md](data/README.md). If your local filenames differ from the paths hard-coded in `scripts/config.py`, update the config before running the workflow.

## Running the Pipeline

1. Install Python 3.9 or later.
2. Install dependencies:

   ```bash
   pip install -r requirements.txt
   ```

3. Populate `data/` with the required local inputs described in [data/README.md](data/README.md).
4. Execute the full pipeline:

   ```bash
   python run_all.py
   ```

This orchestrates the ENHM analysis modules and writes statistical outputs and SVG figures into the `results/` directory.

## License and Citation

If you use this code or analytical framework, cite the associated manuscript. For methodological details, equations, and scope limitations, refer to the manuscript and supplementary information.
