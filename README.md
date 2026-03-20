# Koppen Climate Map Revision Proposal

Köppen Revision Climate Map

A physically motivated global climate classification derived from ERA5 reanalysis (1991–2020).

Overview

This repository contains a revised global climate classification system inspired by the Köppen–Geiger framework, but rebuilt from first principles using modern reanalysis data and explicit thermodynamic and precipitation thresholds.

Unlike traditional Köppen maps, which rely on station climatologies and discrete empirical rules, this system:

operates on gridded ERA5 reanalysis at 0.1° resolution,

uses internally consistent thermal and hydrologic metrics,

explicitly resolves continentality, seasonality, and aridity regimes,

and introduces refined distinctions within mid-latitude and tropical climates.

The result is a classification that is both physically interpretable and globally reproducible.

Key Features:

Global coverage (land points only)

Fully vectorized, reproducible Python implementation

Temperature metrics derived from monthly and annual means

Precipitation seasonality and aridity thresholds based on energy balance logic and ecosystem distribution

Explicit treatment of continentality, oceanicity, and subtropical transitions

Designed for analysis, not just visualization

Data Sources

All calculations are based on ERA5 reanalysis products:

Variable Description Temporal Resolution t Near-surface air temperature Monthly mean Monthly pr Total precipitation Monthly

Period: 1991–2020

Spatial resolution: 0.1° × 0.1°

Units converted internally where required

ERA5 is produced by ECMWF and provides globally consistent, physically constrained atmospheric fields.

Climate Classification Philosophy

This system preserves the conceptual spirit of Köppen, while revising its implementation:

Thermal Regimes

Defined using:

annual mean temperature,

warmest and coldest month means,

seasonal thermal range,

number of thermally “active” months.

This allows explicit separation of:

cold/polar, mid-latitude, and tropical regimes,

continental vs oceanic behavior,

hypercontinental extremes (to isolate and correct certain ERA5 precipitation biases).

Moisture Regimes

Defined using:

climatological annual precipitation,

seasonal precipitation partitioning,

temperature-scaled aridity thresholds.

Aridity thresholds are vectorized functions of mean temperature, with regime-dependent offsets to account for:

tropical convection dominance,

winter-dominant precipitation,

continental amplification.

Climate Classes

The classification includes 22 distinct climate types, grouped broadly as:

Polar & Subpolar

Wet Summer Temperate

Dry Summer Temperate

Semi-Arid & Desert (temperate & tropical)

Tropical (monsoon savanna, rainforest)

Each grid cell is assigned exactly one class, with logical safeguards preventing physically inconsistent combinations.

Output

The primary output is:

a global gridded classification field (xarray DataArray)

an accompanying high-resolution global map visualization

Example output:

Improvement on Köppen Climate Classification (1991–2020 Averages) (ERA5 reanalysis, 0.1° resolution)

Reproducibility

The workflow is fully deterministic:

Load ERA5 reanalysis fields

Compute climatological metrics

Apply logical classification rules

Map classes to labeled indices

Visualize using Cartopy

No machine learning or tuning against external maps is performed.

Repository Structure (planned) . ├── src/ │ ├── classify.py # climate classification logic │ ├── metrics.py # climatological calculations │ └── plot.py # visualization utilities ├── notebooks/ │ └── exploration.ipynb ├── README.md ├── environment.yml └── .gitignore

Intended Use

This project is intended for:

climate analysis and comparison studies

educational and teaching use

climate regime diagnostics

alternative baselines for impact studies

License

This project is released under the MIT License.

Acknowledgements

Inspired by:

Wladimir Köppen

Köppen–Geiger climatologies

Modern reanalysis-based climate diagnostics

ERA5 data © ECMWF.

Citation

If you use this work, please cite the repository and acknowledge ERA5 as the underlying data source.

Status

Active development Initial public release forthcoming.