# -*- coding: utf-8 -*-
"""
Unified Path Configuration for COTS enhm Project
All scripts import paths from this module to ensure consistency.
"""
from pathlib import Path

# Project root (parent of scripts/)
PROJECT_ROOT = Path(__file__).resolve().parent.parent

# Data directories
DATA_DIR = PROJECT_ROOT / "data"
REEF_SURVEY_DIR = DATA_DIR / "reef_surveys"
CLIMATE_DIR = DATA_DIR / "climate"
RAINFALL_DIR = CLIMATE_DIR / "rainfall"
HYDROLOGY_DIR = DATA_DIR / "hydrology"
RUNOFF_DIR = HYDROLOGY_DIR / "runoff"
SPATIAL_DIR = DATA_DIR / "spatial"

# Key data files
IPO_FILE = CLIMATE_DIR / "ipo_index.csv"
CHL_FILE = CLIMATE_DIR / "GBR_chlorophyll_monthly_1998_2025.nc"
CURRENTS_FILE = CLIMATE_DIR / "GBR_currents_monthly_1993_2025.nc"
COORD_FILE = SPATIAL_DIR / "site_lon_lat.csv"
NRM_SHP = SPATIAL_DIR / "NRM_Regions"
GBR_SHP = SPATIAL_DIR / "GBR_Features"

# Output directories
RESULTS_DIR = PROJECT_ROOT / "results"
FIGURES_DIR = RESULTS_DIR / "figures"
TABLES_DIR = RESULTS_DIR / "tables"
DIAGNOSTICS_DIR = RESULTS_DIR / "diagnostics"

# Ensure output dirs exist
for d in [RESULTS_DIR, FIGURES_DIR, TABLES_DIR, DIAGNOSTICS_DIR]:
    d.mkdir(parents=True, exist_ok=True)
