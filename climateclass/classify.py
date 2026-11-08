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
    acontinental = (warmest_month_avg_mean - coldest_month_avg_mean) > 18
    aoceanic = (warmest_month_avg_mean - coldest_month_avg_mean) <= 18

    hypercontinental = (
        (coldest_month_avg_mean < 0)
        & (avg_mean < 10)
        & ((warmest_month_avg_mean - coldest_month_avg_mean) >= 26)
    )

    polar = (grow_months < 3) 
    subpolar = (grow_months >= 3) & ((mild_months < 4) | (avg_mean < 0))
    midlat = (mild_months >= 4) & (avg_mean >= 0) & ((coldest_month_avg_mean < 6) | (mild_months < 8))
    mild = (mild_months >= 8) & (coldest_month_avg_mean >= 6) & (coldest_month_avg_mean < 18)
    tropical = (coldest_month_avg_mean >= 18)

    winter_dominant = mild & pole_of_32 & (winter_precip >= 0.6 * annual_precip)
    summer_dominant = winter_precip < 0.6 * annual_precip

    precip_threshold = xr.where(
        tropical,
        (avg_mean * 18) + 340,
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

    humid = (annual_precip >= precip_threshold)
    semiarid = ((precip_threshold > annual_precip) & (annual_precip >= 0.4 * precip_threshold))
    arid = (annual_precip < 0.4 * precip_threshold)
    winter_wet = ((driest_month * 6) > (wettest_month))
    winter_dry = (summer_precip >= winter_precip) & ((driest_month * 6) <= (wettest_month))
    summer_dry = (winter_precip >= summer_precip)

    cls = xr.full_like(avg_mean, "Unknown", dtype=object)

    cls = xr.where(polar & (cool_months == 0), "Ice Cap", cls)
    cls = xr.where(polar & (cool_months >= 1), "Tundra", cls)

    cls = xr.where(subpolar & continental, "Short Summer Boreal", cls)
    cls = xr.where(subpolar & oceanic, "Short Summer Moderated", cls)

    cls = xr.where(
        midlat & humid & continental
        & (warmest_month_avg_mean < 20)
        & ((winter_wet & (summer_precip >= winter_precip)) | (driest_month >= 30)),
        "Humid Cool Temperate",
        cls,
    )

    cls = xr.where(
        midlat & humid
        & (coldest_month_avg_mean < 0) & (warmest_month_avg_mean >= 20)
        & ((winter_wet & (summer_precip >= winter_precip)) | (driest_month >= 30)),
        "Humid Mixed Temperate",
        cls,
    )

    cls = xr.where(
        midlat & humid
        & ((coldest_month_avg_mean >= 0) | oceanic) & (warmest_month_avg_mean < 22)
        & ((winter_wet & (summer_precip >= winter_precip)) | (driest_month >= 30)),
        "Humid Mild Temperate",
        cls,
    )

    cls = xr.where(
        midlat & humid
        & (coldest_month_avg_mean >= 0) & (warmest_month_avg_mean >= 22) & ((winter_wet & (summer_precip >= winter_precip)) | (driest_month >= 40)),
        "Humid Warm Temperate",
        cls,
    )

    cls = xr.where(
        midlat & humid & winter_dry
        & continental
        & (warmest_month_avg_mean < 20),
        "Dry Winter Cool Temperate",
        cls,
    )

    cls = xr.where(
        midlat & humid & winter_dry
        & (coldest_month_avg_mean < 0) & (warmest_month_avg_mean >= 20),
        "Dry Winter Mixed Temperate",
        cls,
    )

    cls = xr.where(
        midlat & humid & winter_dry
        & ((coldest_month_avg_mean >= 0) | oceanic) & (warmest_month_avg_mean < 22),
        "Dry Winter Mild Temperate",
        cls,
    )

    cls = xr.where(
        midlat & humid & winter_dry
        & (coldest_month_avg_mean >= 0) & (warmest_month_avg_mean >= 22),
        "Dry Winter Warm Temperate",
        cls,
    )

    cls = xr.where(
        midlat & humid & (winter_precip >= summer_precip)
        & continental
        & (warmest_month_avg_mean < 20)
        & (driest_month < 30),
        "Dry Summer Cool Temperate",
        cls,
    )

    cls = xr.where(
        midlat & humid & (winter_precip >= summer_precip)
        & (coldest_month_avg_mean < 0) & (warmest_month_avg_mean >= 20)
        & (driest_month < 30),
        "Dry Summer Mixed Temperate",
        cls,
    )

    cls = xr.where(
        midlat & humid & (winter_precip >= summer_precip)
        & ((coldest_month_avg_mean >= 0) | oceanic) & (warmest_month_avg_mean < 22)
        & (driest_month < 30),
        "Dry Summer Mild Temperate",
        cls,
    )

    cls = xr.where(
        midlat & humid & (winter_precip >= summer_precip)
        & (coldest_month_avg_mean >= 0) & (warmest_month_avg_mean >= 22) & (driest_month < 40),
        "Dry Summer Warm Temperate",
        cls,
    )

    cls = xr.where(
        mild & humid
        & (warmest_month_avg_mean < 22)
        & ((winter_wet & (summer_precip >= winter_precip)) | (driest_month >= 40)),
        "Humid Mild Subtropical",
        cls,
    )

    cls = xr.where(
        mild & humid
        & (warmest_month_avg_mean >= 22) & ((winter_wet & (summer_precip >= winter_precip)) | (driest_month >= 40)),
        "Humid Warm Subtropical",
        cls,
    )

    cls = xr.where(
        mild & humid
        & (summer_precip >= winter_precip)
        & (warmest_month_avg_mean < 22)
        & winter_dry,
        "Dry Winter Mild Subtropical",
        cls,
    )

    cls = xr.where(
        mild & humid
        & winter_dry
        & (warmest_month_avg_mean >= 22),
        "Dry Winter Warm Subtropical",
        cls,
    )

    cls = xr.where(
        mild & humid & (winter_precip >= summer_precip)
        & (warmest_month_avg_mean < 22)
        & (driest_month < 40),
        "Dry Summer Mild Subtropical",
        cls,
    )

    cls = xr.where(
        mild & humid & (winter_precip >= summer_precip)
        & (warmest_month_avg_mean >= 22) & (driest_month < 40),
        "Dry Summer Warm Subtropical",
        cls,
    )

    cls = xr.where(
        (mild_months >= 4) & (avg_mean >= 0) 
        & semiarid
        & (coldest_month_avg_mean < 6)
        & acontinental,
        "Variable Steppe",
        cls,
    )

    cls = xr.where(
        (mild_months >= 4) & (avg_mean >= 0) 
        & semiarid
        & aoceanic
        & (warmest_month_avg_mean < 24),
        "Moderated Steppe",
        cls,
    )

    cls = xr.where(
        semiarid
        & (coldest_month_avg_mean >= 6)
        & (warmest_month_avg_mean >= 24),
        "Hot Steppe",
        cls,
    )

    cls = xr.where(
        (mild_months >= 4) & (avg_mean >= 0) 
        & arid
        & (coldest_month_avg_mean < 6)
        & acontinental,
        "Variable Desert",
        cls,
    )

    cls = xr.where(
        (mild_months >= 4) & (avg_mean >= 0)
        & arid
        & aoceanic
        & (warmest_month_avg_mean < 24),
        "Moderated Desert",
        cls,
    )

    cls = xr.where(
        arid
        & (coldest_month_avg_mean >= 6)
        & (warmest_month_avg_mean >= 24),
        "Hot Desert",
        cls,
    )

    cls = xr.where(
        tropical & humid
        & (driest_month < 50),
        "Tropical Monsoon Savanna",
        cls,
    )

    cls = xr.where(
        tropical & humid
        & (driest_month >= 50),
        "Tropical Rainforest",
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