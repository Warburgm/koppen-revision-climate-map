# Köppen Climate Map Revision

*A physically motivated global climate classification derived from ERA5 reanalysis (1991–2020).*

## Overview

This repository contains a revised global climate classification system inspired by the Köppen–Geiger framework, but rebuilt from first principles using modern reanalysis data and explicit thermodynamic and precipitation thresholds.

Unlike traditional Köppen maps, which rely on station climatologies and discrete empirical rules, this system:

- operates on gridded ERA5 reanalysis at 0.1° resolution,
- uses internally consistent thermal and hydrologic metrics,
- explicitly resolves continentality, seasonality, and aridity regimes,
- introduces refined distinctions within mid-latitude and tropical climates.

The result is a classification that is both physically interpretable and globally reproducible.

## Key Features

- Global coverage (land points only)
- Fully vectorized, reproducible Python implementation
- Temperature metrics derived from monthly and annual means
- Precipitation seasonality and aridity thresholds based on energy balance logic and ecosystem distribution
- Explicit treatment of continentality, oceanicity, and subtropical transitions
- Designed for analysis, not just visualization

## Data Sources

All calculations are based on ERA5 reanalysis products:

| Variable | Description | Temporal Resolution |
|----------|------------|---------------------|
| `t`      | Near-surface air temperature | Monthly mean |
| `pr`     | Total precipitation | Monthly mean |

- Period: **1991–2020**
- Spatial resolution: **0.1° × 0.1°**
- Units are converted internally where required

ERA5 is produced by ECMWF and provides globally consistent, physically constrained atmospheric fields.

## Climate Classification Philosophy

This system preserves the conceptual spirit of Köppen while revising its implementation.

## Thermal Regimes

Thermal regimes are defined using:

- annual mean temperature,
- warmest and coldest month means,
- seasonal thermal range,
- number of thermally active months.

This allows explicit separation of:

- cold/polar, mid-latitude, and tropical regimes,
- continental vs. oceanic behavior,
- hypercontinental extremes (including corrections for certain ERA5 precipitation artifacts).

## Moisture Regimes

Moisture regimes are defined using:

- climatological annual precipitation,
- seasonal precipitation partitioning,
- temperature-scaled aridity thresholds.

Aridity thresholds are vectorized functions of mean temperature, with regime-dependent offsets to account for:

- tropical convection dominance,
- winter-dominant precipitation,
- continental amplification.

## Climate Classes

The classification currently includes **22 distinct climate types**, grouped broadly as:

- Polar & Subpolar  
- Wet Summer Temperate  
- Dry Summer Temperate  
- Semi-Arid & Desert (temperate and tropical)  
- Tropical (monsoon savanna, rainforest)

Each grid cell is assigned exactly one class, with logical safeguards to prevent physically inconsistent combinations.

## Output

The primary outputs are:

- a global gridded classification field (`xarray.DataArray`)
- a high-resolution global map visualization

## Reproducibility

The workflow is fully deterministic:

1. Load ERA5 reanalysis fields  
2. Compute climatological metrics  
3. Apply logical classification rules  
4. Map classes to labeled indices  
5. Visualize using Cartopy  

No machine learning or tuning against external maps is performed.

## Repository Structure

.
├── src/
│   └── climateclass/
│       ├── __init__.py
│       ├── classify.py
│       ├── palette.py
│       └── plotting.py
├── scripts/
│   └── run_global.py
├── notebooks/
├── README.md
├── environment.yml
├── LICENSE
└── .gitignore

## Intended Use

This project is intended for:

- climate analysis and comparison studies  
- educational and teaching use  
- climate regime diagnostics  
- alternative baselines for impact studies  

## Data Requirements

This workflow expects two NetCDF files:

- `t.nc` — monthly mean near-surface air temperature  
- `pr.nc` — monthly mean precipitation rate  

## Expected variable names

The files must contain:

- temperature variable named `t`  
- precipitation variable named `pr`  

## Expected dimensions

Both datasets should share compatible dimensions and coordinates, typically:

- `time`  
- `lat`  
- `lon`  

## Expected units

- `t`: degrees Celsius  
- `pr`: mean precipitation rate in `mm/day` or equivalent `kg m^-2 day^-1`  

The script converts precipitation rates to monthly totals internally using the number of days in each month.

## Time period used in the current workflow

The current implementation slices the data to:

- **1991-01-01 to 2020-12-31**

## Spatial coverage

The workflow is designed for gridded global or near-global datasets on a regular latitude–longitude grid.

## Data Availability

Raw climate input files are not included in this repository.

Users must provide their own NetCDF input files matching the required structure and variable names.

This project was developed using high-resolution ERA5 reanalysis-based temperature and precipitation climatologies.

## Usage 

1. Create the environment
   
conda env create -f environment.yml
conda activate climateclass

2. Run the global classification workflow
   
From the repository root:

PYTHONPATH=src python scripts/run_global.py \
  --t /path/to/t.nc \
  --pr /path/to/pr.nc

## Reproducibility Note

Because raw NetCDF inputs are not bundled with the repository, exact reproduction of the published map requires access to equivalent source datasets, preprocessing choices, and variable definitions.

The codebase is designed to make the classification logic fully transparent and reusable.

## License

This project is released under the MIT License.

## Acknowledgements

Inspired by:

Wladimir Köppen

Köppen–Geiger climatologies

modern reanalysis-based climate diagnostics

ERA5 data © ECMWF.

## Citation

If you use this work, please cite the repository and acknowledge ERA5 as the underlying data source.

## Status

Active development
Initial public release forthcoming.