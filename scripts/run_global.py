import argparse
from pathlib import Path

import xarray as xr

from climateclass.classify import build_classification, classification_to_index
from climateclass.palette import LABELS
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

    classification = build_classification(
        t,
        pr,
        time_slice=("1991-01-01", "2020-12-31"),
    )

    classification_idx = classification_to_index(classification, LABELS)

    quick_plot(
        class_int=classification_idx,
        out=args.out,
        title="Köppen Revision Climate Types (1991–2020 Averages)",
        subtitle=(
            "Based on 0.1° × 0.1° resolution ERA5 reanalysis "
            "temperature and precipitation monthly means"
        ),
        show=True,
    )


if __name__ == "__main__":
    main()