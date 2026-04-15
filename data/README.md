# Data Staging Guide

This repository does not ship raw input datasets. Populate `data/` locally before running the pipeline.

## Required sources

| Local path | Contents | Download source |
| --- | --- | --- |
| `data/reef_surveys/` | AIMS LTMP manta tow reef survey CSV files | https://apps.aims.gov.au/reef-monitoring/ |
| `data/juvenile/` | Juvenile recruitment CSV files used for supporting checks | Project-specific exports from the same monitoring workflow |
| `data/climate/ipo_index.csv` | IPO Tripole Index time series | https://psl.noaa.gov/data/timeseries/IPOTPI/ |
| `data/climate/rainfall/` | Rainfall inputs used for climate-hydrology contrasts | Australian Bureau of Meteorology archives |
| `data/climate/GBR_chlorophyll_monthly_1998_2025.nc` | Chlorophyll grid | https://ereefs.org.au/ereefs |
| `data/climate/GBR_currents_monthly_1993_2025.nc` | Current field grid | https://ereefs.org.au/ereefs |
| `data/hydrology/runoff/` | River discharge gauge CSV files | Queensland Water Monitoring Information Portal |
| `data/spatial/site_lon_lat.csv` | Reef coordinate table | Derived from project spatial preprocessing |
| `data/spatial/GBR_Features/` | GBR feature shapefile components | GBRMPA spatial data hub |
| `data/spatial/NRM_Regions/` | NRM region shapefile components | GBRMPA spatial data hub |

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
|   `-- .gitkeep
|-- reef_surveys/
|   `-- .gitkeep
`-- spatial/
    |-- .gitkeep
    |-- GBR_Features/
    `-- NRM_Regions/
```

## Notes

- Keep raw downloads out of git. The repository is intentionally configured to ignore files under `data/`.
- If your local filenames differ from the paths hard-coded in `scripts/config.py`, update the config before running `python run_all.py`.
