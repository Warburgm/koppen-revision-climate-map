import numpy as np
import xarray as xr

from .palette import LABELS


def _as_dataarray(x):
    """Return x as a DataArray."""
    return x if isinstance(x, xr.DataArray) else x.to_array().squeeze()


def _monthly_precip_totals(pr: xr.DataArray) -> xr.DataArray:
    """
    Convert precipitation from mean daily values to monthly totals
    using days in each month.
    """
    days = pr.time.dt.days_in_month.astype("int32")
    return pr * days


def _lat_bands(lat2d: xr.DataArray):
    """
    Return boolean masks for poleward/equatorward of 32 degrees latitude.
    """
    pole_of_32 = (lat2d >= 32) | (lat2d <= -32)
    eq_of_32 = ~pole_of_32
    return pole_of_32, eq_of_32


def build_classification(
    t: xr.DataArray,
    pr: xr.DataArray,
    time_slice=("1991-01-01", "2020-12-31"),
) -> xr.DataArray:
    """
    Build the climate classification DataArray from temperature and precipitation.

    Parameters
    ----------
    t : xr.DataArray or xr.Dataset
        Temperature field with dimensions including time, lat, lon.
    pr : xr.DataArray or xr.Dataset
        Precipitation field with dimensions including time, lat, lon.
    time_slice : tuple[str, str] or None
        Optional (start, end) time slice. Set to None to use the full record.

    Returns
    -------
    xr.DataArray
        Object-valued DataArray of climate class names.
    """
    t = _as_dataarray(t).astype("float32")
    pr = _as_dataarray(pr).astype("float32")

    if time_slice is not None:
        start, end = time_slice
        t = t.sel(time=slice(start, end))
        pr = pr.sel(time=slice(start, end))

    pr_monthly = _monthly_precip_totals(pr)

    # 2D latitude grid
    lat2d, _ = xr.broadcast(t["lat"], t["lon"])
    pole_of_32, eq_of_32 = _lat_bands(lat2d)

    # Core climatological variables
    avg_mean = t.mean("time", skipna=True)

    annual_precip = pr_monthly.groupby("time.year").sum("time").mean("year")

    t_monthly_mean = t.groupby("time.month").mean("time")
    pr_monthly_mean = pr_monthly.groupby("time.month").mean("time")

    driest_month = pr_monthly_mean.min("month")
    wettest_month = pr_monthly_mean.max("month")

    coldest_month_avg_mean = t_monthly_mean.min("month")
    warmest_month_avg_mean = t_monthly_mean.max("month")

    ranks = t_monthly_mean.chunk({"month": -1}).rank("month")
    summer_mask = ranks >= 7
    winter_mask = ranks <= 6

    summer_temp = t_monthly_mean.where(summer_mask).mean("month", skipna=True)
    winter_temp = t_monthly_mean.where(winter_mask).mean("month", skipna=True)
    summer_precip = pr_monthly_mean.where(summer_mask).sum("month", skipna=True)
    winter_precip = pr_monthly_mean.where(winter_mask).sum("month", skipna=True)

    warm_months = (t_monthly_mean >= 18).sum("month")
    mild_months = (t_monthly_mean >= 10).sum("month")
    grow_months = (t_monthly_mean >= 6).sum("month")
    cool_months = (t_monthly_mean >= 0).sum("month")

    continental = (warmest_month_avg_mean - coldest_month_avg_mean) > 20
    oceanic = (warmest_month_avg_mean - coldest_month_avg_mean) <= 20
    acontinental = (warmest_month_avg_mean - coldest_month_avg_mean) > 20
    aoceanic = (warmest_month_avg_mean - coldest_month_avg_mean) <= 20

    hypercontinental = (
        (coldest_month_avg_mean < 0)
        & (avg_mean < 10)
        & ((warmest_month_avg_mean - coldest_month_avg_mean) >= 26)
    )

    polar = (grow_months < 3)
    subpolar = (grow_months >= 3) & (
        (mild_months < 4) | (avg_mean < 0)
    )
    midlat = (
        (mild_months >= 4)
        & (avg_mean >= 0)
        & ((coldest_month_avg_mean < 4) | (warm_months < 4))
    )
    mild = (warm_months >= 4) & (coldest_month_avg_mean >= 4) & (
        coldest_month_avg_mean < 18
    )
    tropical = coldest_month_avg_mean >= 18

    winter_dominant = mild & pole_of_32 & (winter_precip >= 0.6 * annual_precip)
    summer_dominant = winter_precip < 0.6 * annual_precip

    precip_threshold = xr.where(
        tropical,
        (avg_mean * 18) + 260,
        xr.where(
            hypercontinental,
            (avg_mean * 18) + 440,
            xr.where(
                winter_dominant,
                (avg_mean * 18) + 100,
                (avg_mean * 18) + 380,
            ),
        ),
    )

    # Aridity classes
    humid = (annual_precip >= precip_threshold)
    semiarid = ((precip_threshold > annual_precip) & (annual_precip >= 0.4 * precip_threshold))
    arid = (annual_precip < 0.4 * precip_threshold)
    winter_wet = (((driest_month * 4) > (wettest_month)) | (summer_precip >= winter_precip))
    winter_dry = (summer_precip >= winter_precip) & ((driest_month * 4) <= (wettest_month))
    summer_dry = (winter_precip >= summer_precip) & ((driest_month * 4) <= (wettest_month))

    cls = xr.full_like(avg_mean, "Unknown", dtype=object)

    cls = xr.where(polar & (cool_months == 0), "Cold, Ice Cap", cls)
    cls = xr.where(polar & (cool_months >= 1), "Cold, Tundra", cls)

    cls = xr.where(subpolar & continental, "Cold, Taiga, Severe Winter", cls)
    cls = xr.where(subpolar & oceanic, "Cold, Taiga, Moderated Winter", cls)

    cls = xr.where(
        midlat & humid & continental
        & (warm_months < 3)
        & ((driest_month >= 40) | (summer_precip >= winter_precip)),
        "Continental, Short Wet Summer",
        cls,
    )

    cls = xr.where(
        midlat & humid & continental
        & (warm_months >= 3)
        & ((driest_month >= 40) | (winter_wet)),
        "Continental, Long Summer, No Dry Season",
        cls,
    )

    cls = xr.where(
        midlat & humid
        & (warm_months >= 3)
        & (summer_precip >= winter_precip)
        & winter_dry,
        "Continental, Long Summer, Dry Winter",
        cls,
    )

    cls = xr.where(
        midlat & humid
        & continental
        & (winter_precip >= summer_precip)
        & (driest_month < 40),
        "Continental, Dry Summer",
        cls,
    )

    cls = xr.where(
        midlat & humid & oceanic & ((driest_month >= 40) | (winter_wet)),
        "Oceanic, No Dry Season",
        cls,
    )

    cls = xr.where(
        midlat & humid & oceanic & (summer_dry) & (driest_month < 40),
        "Oceanic, Dry Summer",
        cls,
    )

    cls = xr.where(
        midlat & humid & oceanic & (summer_precip >= winter_precip)
        & winter_dry,
        "Oceanic, Dry Winter",
        cls,
    )

    cls = xr.where(
        humid & mild
        & (winter_precip > summer_precip)
        & (driest_month < 40),
        "Subtropical, Dry Summer",
        cls,
    )

    cls = xr.where(
        mild & humid 
        & ((driest_month >= 40) | ((summer_precip >= winter_precip) & (winter_wet))),
        "Subtropical, No Dry Season",
        cls,
    )

    cls = xr.where(
        mild & humid
        & (summer_precip >= winter_precip)
        & (winter_dry)
        & (driest_month < 40),
        "Subtropical, Dry Winter",
        cls,
    )

    cls = xr.where(
        (mild_months >= 4) & (avg_mean >= 0)
        & semiarid
        & acontinental
        & (coldest_month_avg_mean < 6),
        "Dry, Variable Steppe",
        cls,
    )

    cls = xr.where(
        (mild_months >= 4) & (avg_mean >= 0) 
        & semiarid
        & aoceanic
        & (warmest_month_avg_mean < 26),
        "Dry, Moderated Steppe",
        cls,
    )

    cls = xr.where(
        semiarid
        & (coldest_month_avg_mean >= 6)
        & (warmest_month_avg_mean >= 26),
        "Dry, Hot Steppe",
        cls,
    )

    cls = xr.where(
        (mild_months >= 4) & (avg_mean >= 0) 
        & arid
        & acontinental
        & (coldest_month_avg_mean < 6),
        "Dry, Variable Desert",
        cls,
    )

    cls = xr.where(
        (mild_months >= 4) & (avg_mean >= 0)
        & arid
        & aoceanic
        & (warmest_month_avg_mean < 26),
        "Dry, Moderated Desert",
        cls,
    )

    cls = xr.where(
        arid
        & (coldest_month_avg_mean >= 6)
        & (warmest_month_avg_mean >= 26),
        "Dry, Hot Desert",
        cls,
    )

    cls = xr.where(
        tropical & humid
        & ((annual_precip < 1250) | (driest_month < 50)),
        "Tropical, Wet & Dry Season",
        cls,
    )

    cls = xr.where(
        tropical & humid
        & (annual_precip >= 1250)
        & (driest_month >= 50),
        "Tropical, No Dry Season",
        cls,
    )
    
    cls.name = "climate_class"
    return cls


def classification_to_index(
    classification: xr.DataArray,
    labels=None,
) -> xr.DataArray:
    """
    Convert string class labels to numeric indices for plotting.
    """
    if labels is None:
        labels = LABELS

    label_to_int = {lab: i for i, lab in enumerate(labels)}

    class_int = xr.DataArray(
        np.full(classification.shape, np.nan, dtype=np.float32),
        coords=classification.coords,
        dims=classification.dims,
        name="class_index",
    )

    for lab, i in label_to_int.items():
        class_int = xr.where(classification == lab, float(i), class_int)

    return class_int