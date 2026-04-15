# Ecological Null Hypothesis Model (ENHM) for Crown-of-Thorns Starfish Outbreaks

This repository contains the core analytical pipeline for separating coral-limited trophic baselines from environmentally forced ecological anomalies on the Great Barrier Reef (GBR). The framework utilizes the Ecological Null Hypothesis Model (ENHM), biological stripping, and residual spectral analysis as presented in our manuscript.

## Overview

A central challenge in ecological forecasting is distinguishing slowly varying climatic background conditions from immediate outbreak triggers. Because Crown-of-Thorns Starfish (COTS) are obligate corallivores, raw abundance series often mix climatic forcing with prey depletion and recovery cycles. 

Our pipeline implements **ENHM** to address this confound explicitly:
1. **Biological Stripping:** Evaluates the empirical delay between coral resources and COTS abundance, establishing a 2-year lag as the optimal trophic baseline.
2. **Residual Separation:** Removes the density-dependent constraints by fitting a regression-through-the-origin model.
3. **Frequency & Propagation Extraction:** Evaluates residual periodicity using iteratively applied Double Lomb-Scargle (DLS) analysis to identify low-frequency interdecadal structure. Phase anomalies are then tracked spatially.
4. **Climate Coupling:** Tests Wavelet Coherence Analysis (WCA) against teleconnection indices like the Interdecadal Pacific Oscillation (IPO) and El Niño-Southern Oscillation (ENSO).

## Project Structure

```text
COTS_ENHM_GBR/
│
├── data/                       # Contains data metadata placeholders (large files are excluded)
│   ├── climate/                # IPO (NOAA), ENSO indices, Rainfall (BoM), Chlorophyll-a (eReefs)
│   ├── hydrology/              # River discharge datasets
│   ├── juvenile/               # Supplementary recruitment data
│   ├── reef_surveys/           # AIMS Long-Term Monitoring Project (LTMP) data
│   └── spatial/                # GBRMPA geographic features
│
├── scripts/                    # Core analytic Python scripts
│   ├── 01_enhm_analysis.py     # Main ENHM baseline extraction and Double Lomb-Scargle periodicity
│   ├── 02_paper_figures.py     # Figure 1-4 generator matching manuscript style (SVG)
│   ├── ...                     # Supplementary sensitivities, recruitment-context modules 
│   └── config.py               # Shared parameters and unified styling
```

## Data Availability & Download Instructions

Due to GitHub repository size limits, heavy multidimensional NetCDF datasets (chlorophyll grids, currents, raw rainfall) and high-resolution shapefiles are **not included** in this repository. Ensure you populate the `data/` directory with the required inputs before executing the pipeline.

### Required Datasets:
- **AIMS LTMP Data (Manta Tow):** Open-access data available from the [AIMS eAtlas or the AIMS Data Platform](https://apps.aims.gov.au/reef-monitoring/).
- **Climate Indices (IPO & ENSO):** The Tripole Index (TPI) for the IPO is derived from Henley et al. (2015) and should be downloaded from the [NOAA Physical Sciences Laboratory](https://psl.noaa.gov/data/timeseries/IPOTPI/). ENSO specific indices (e.g., NINO3.4) are similarly available.
- **Hydrological Data (Rainfall & Runoff):** Rainfall data is sourced from the Australian Bureau of Meteorology (BoM) gridded archives. River discharge gauge data is provided by the Queensland Government Water Monitoring Information Portal (WMIP).
- **Oceanographic Grids (Chlorophyll & Currents):** Data available through the CSIRO [eReefs modeled outputs](https://ereefs.org.au/ereefs).
- **Spatial Boundaries:** NRM and Reef Shapefiles are hosted broadly by the Great Barrier Reef Marine Park Authority (GBRMPA) Master Spatial Data Hub.

Place the unzipped raw datasets into their respective `climate/`, `hydrology/`, and `reef_surveys/` folders according to the path requirements configured in `scripts/config.py`.

## Running the Pipeline

1. **Environment Setup:** Ensure Python 3.9+ is installed. Install requirements:
   ```bash
   pip install -r requirements.txt
   ```
2. **Execute ENHM Pipeline:** To run the full analysis and generate all figures, execute the main entry file:
   ```bash
   python run_all.py
   ```
   *Note: This will orchestrate data extraction, run the ENHM models reef by reef, and produce all statistical outputs and `SVG` figures into the `results/` directory.*

## License & Citation
If you use this code or analytical framework, please cite the associated manuscript as outlined in the publication. For details on manuscript formatting, equations, and explicit methodological limits, refer to the "Materials and Methods" and SI files.
