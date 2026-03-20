import argparse
from pathlib import Path
import xarray as xr
import regionmask

from climateclass.classify import build_classification, classification_to_index
from climateclass.palette import LABELS, LEGEND_ITEMS, CBAR_LABELS
from climateclass.plotting import quick_plot


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--t", required=True, help="Path to temperature NetCDF")
    parser.add_argument("--pr", required=True, help="Path to precipitation NetCDF")
    parser.add_argument(
        "--out",
        default="global_climate_types_1991_2020.png",
        help="Output figure filename",
    )
    args = parser.parse_args()

    t_path = Path(args.t)
    pr_path = Path(args.pr)

    for p in [t_path, pr_path]:
        assert p.exists(), f"Missing file: {p}"

    t = xr.open_dataset(
        t_path,
        engine="netcdf4",
        chunks={"time": 12},
    )["t"].astype("float32")

    pr = xr.open_dataset(
        pr_path,
        engine="netcdf4",
        chunks={"time": 12},
    )["pr"].astype("float32")

    t = t.sel(time=slice("1991-01-01", "2020-12-31"))
    pr = pr.sel(time=slice("1991-01-01", "2020-12-31"))

    result = build_classification(t, pr)
    classification = result["classification"]

    classification_idx = classification_to_index(classification, LABELS)

    lon = classification_idx.lon.values
    lat = classification_idx.lat.values
    landmask = regionmask.defined_regions.natural_earth_v5_0_0.land_110.mask(lon, lat)
    land = landmask.notnull()
    classification_idx = classification_idx.where(land)

    quick_plot(
        classification_idx=classification_idx,
        legend_items=LEGEND_ITEMS,
        cbar_labels=CBAR_LABELS,
        outfile=args.out,
    )


if __name__ == "__main__":
    main()