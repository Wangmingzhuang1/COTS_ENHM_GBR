# Data Staging Guide

This repository retains compact ecological CSV inputs needed for the core ENHM workflow, but excludes larger external resources. Populate the missing assets locally before running the pipeline.

## Required sources

| Local path | Contents | Download source |
| --- | --- | --- |
| `data/reef_surveys/` | AIMS LTMP manta tow reef survey CSV files | Retained in repository |
| `data/juvenile/` | Juvenile recruitment CSV files used for supporting checks | Retained in repository |
| `data/climate/ipo_index.csv` | IPO Tripole Index time series | Retained in repository |
| `data/climate/rainfall/` | Rainfall inputs used for climate-hydrology contrasts | [BOM Climate Data Archives](http://www.bom.gov.au/climate/data/) or [AGCD Grids](http://www.bom.gov.au/climate/maps/rainfall/) |
| `data/climate/GBR_chlorophyll_monthly_1998_2025.nc` | Chlorophyll grid | [AIMS eReefs THREDDS / eReefs portal](https://ereefs.aims.gov.au/) |
| `data/climate/GBR_currents_monthly_1993_2025.nc` | Current field grid | [AIMS eReefs THREDDS / eReefs portal](https://ereefs.aims.gov.au/) |
| `data/hydrology/runoff/` | River discharge gauge CSV files | [Queensland Water Monitoring Information Portal](https://water-monitoring.information.qld.gov.au/) |
| `data/spatial/site_lon_lat.csv` | Reef coordinate table | Retained in repository |
| `data/spatial/GBR_Features/` | GBR feature shapefile components | [GBRMPA Geoportal](https://geoportal.gbrmpa.gov.au/) |
| `data/spatial/NRM_Regions/` | NRM region shapefile components | [GBRMPA Geoportal](https://geoportal.gbrmpa.gov.au/) |

## Expected layout

```text
data/
|-- climate/
|   |-- .gitkeep
|   |-- ipo_index.csv
|   `-- rainfall/
|-- hydrology/
|   |-- .gitkeep
|   `-- runoff/
|-- juvenile/
|   |-- .gitkeep
|   `-- *.csv
|-- reef_surveys/
|   |-- .gitkeep
|   `-- *.csv
`-- spatial/
    |-- .gitkeep
    |-- GBR_Features/
    `-- NRM_Regions/
```

## Notes

- The repository intentionally tracks only compact ecological CSV inputs and ignores heavier runoff downloads, shapefiles, and large grid products.
- If your local filenames differ from the paths hard-coded in `scripts/config.py`, update the config before running `python run_all.py`.
