# Data staging guide

The compact inputs already tracked in this repository are sufficient for the core ENHM analysis. The optional figure and environmental-context stages require larger external files that must be downloaded from their official providers.

## Source-to-file mapping

| Local path | Role in the workflow | Availability and official source |
| --- | --- | --- |
| `data/reef_surveys/*.csv` | Reef-level AIMS manta-tow COTS and hard-coral-cover series | Retained here; source: [AIMS Reef Monitoring Dashboard](https://apps.aims.gov.au/reef-monitoring/) |
| `data/juvenile/*.csv` | Juvenile-coral supporting series | Retained here; source: [AIMS Reef Monitoring Dashboard](https://apps.aims.gov.au/reef-monitoring/) |
| `data/climate/ipo_index.csv` | Processed annual IPO index used in the workflow | Retained here; source: [NOAA PSL TPI (IPO)](https://psl.noaa.gov/data/timeseries/IPOTPI/), unfiltered NOAA ERSST v5 monthly series |
| `data/climate/rainfall/` | Monthly gridded rainfall used for IPO-phase contrasts | Download from [Bureau of Meteorology AGCD](https://www.bom.gov.au/climate/austmaps/about-agcd-maps.shtml) and [rainfall grid downloads](https://www.bom.gov.au/climate/maps/rainfall/) |
| `data/hydrology/runoff/` | River discharge for the Pascoe, Burdekin, and Fitzroy gauges | Download from the Queensland Government [Water Monitoring Information Portal](https://water-monitoring.information.qld.gov.au/) |
| `data/climate/GBR_chlorophyll_monthly_1998_2025.nc` | Monthly surface chlorophyll-a grid | Obtain through the [eReefs Data Explorer](https://portal.ereefs.org.au/) or [Data Extraction Tool](https://extraction.ereefs.aims.gov.au/) |
| `data/climate/GBR_currents_monthly_1993_2025.nc` | Monthly surface-current grid | Obtain through the [eReefs Data Explorer](https://portal.ereefs.org.au/) or [Data Extraction Tool](https://extraction.ereefs.aims.gov.au/) |
| `data/spatial/site_lon_lat.csv` | Reef coordinates | Retained here; derived from the monitoring-site records used in the analysis |
| `data/spatial/GBR_Features/` | Reef-feature boundaries | Download through GBRMPA [Spatial Data Information Services](https://www.gbrmpa.gov.au/about-us/spatial-data-information-services) |
| `data/spatial/NRM_Regions/` | Natural Resource Management region boundaries | Download through GBRMPA [Spatial Data Information Services](https://www.gbrmpa.gov.au/about-us/spatial-data-information-services) |

## Expected layout

```text
data/
|-- climate/
|   |-- ipo_index.csv
|   |-- rainfall/
|   |-- GBR_chlorophyll_monthly_1998_2025.nc
|   `-- GBR_currents_monthly_1993_2025.nc
|-- hydrology/
|   `-- runoff/
|-- juvenile/
|   `-- *.csv
|-- reef_surveys/
|   `-- *.csv
`-- spatial/
    |-- site_lon_lat.csv
    |-- GBR_Features/
    `-- NRM_Regions/
```

Do not rename the retained compact files. If an official provider changes an external filename, update the corresponding path in `scripts/config.py` rather than creating a second copy with a suffix such as `(1)`.

The MIT licence at the repository root applies to software only. Data users must retain provider attribution and comply with the licence attached to each source dataset or downloaded layer.
