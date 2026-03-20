import numpy as np
import xarray as xr


def da(x):
    return x if isinstance(x, xr.DataArray) else x.to_array().squeeze()


def lat_bands(lat2d: xr.DataArray) -> dict:
    pole_of_32 = (lat2d >= 32) | (lat2d <= -32)
    eq_of_32 = ~pole_of_32
    return {
        "pole_of_32": pole_of_32,
        "eq_of_32": eq_of_32,
    }


def compute_metrics(t: xr.DataArray, pr: xr.DataArray) -> dict:
    """
    Compute climatological temperature and precipitation metrics.

    Assumes:
    - t is monthly mean temperature
    - pr is monthly mean precipitation rate in mm/day or kg m-2 day-1 equivalent
    """
    days = pr.time.dt.days_in_month.astype("int32")
    pr_monthly = pr * days

    avg_mean = t.mean("time", skipna=True)
    annual_precip = pr_monthly.groupby("time.year").sum("time").mean("year")

    t_monthly_mean = t.groupby("time.month").mean("time")
    pr_monthly_mean = pr_monthly.groupby("time.month").mean("time")

    driest_month_precip = pr_monthly_mean.min("month")
    coldest_month_avg_mean = t_monthly_mean.min("month")
    warmest_month_avg_mean = t_monthly_mean.max("month")

    ranks = t_monthly_mean.chunk({"month": -1}).rank("month")
    summer_mask = ranks >= 7
    winter_mask = ranks <= 6

    summer_temp = t_monthly_mean.where(summer_mask).mean("month", skipna=True)
    winter_temp = t_monthly_mean.where(winter_mask).mean("month", skipna=True)
    summer_precip = pr_monthly_mean.where(summer_mask).sum("month", skipna=True)
    winter_precip = pr_monthly_mean.where(winter_mask).sum("month", skipna=True)

    mild_months = (t_monthly_mean >= 10).sum("month")
    grow_months = (t_monthly_mean >= 6).sum("month")
    cool_months = (t_monthly_mean >= 0).sum("month")

    return {
        "pr_monthly": pr_monthly,
        "avg_mean": avg_mean,
        "annual_precip": annual_precip,
        "t_monthly_mean": t_monthly_mean,
        "pr_monthly_mean": pr_monthly_mean,
        "driest_month_precip": driest_month_precip,
        "coldest_month_avg_mean": coldest_month_avg_mean,
        "warmest_month_avg_mean": warmest_month_avg_mean,
        "summer_temp": summer_temp,
        "winter_temp": winter_temp,
        "summer_precip": summer_precip,
        "winter_precip": winter_precip,
        "mild_months": mild_months,
        "grow_months": grow_months,
        "cool_months": cool_months,
    }


def compute_regimes(metrics: dict, latbands: dict) -> dict:
    avg_mean = metrics["avg_mean"]
    annual_precip = metrics["annual_precip"]
    summer_precip = metrics["summer_precip"]
    winter_precip = metrics["winter_precip"]
    coldest_month_avg_mean = metrics["coldest_month_avg_mean"]
    warmest_month_avg_mean = metrics["warmest_month_avg_mean"]
    mild_months = metrics["mild_months"]
    grow_months = metrics["grow_months"]

    pole_of_32 = latbands["pole_of_32"]

    continental = (warmest_month_avg_mean - coldest_month_avg_mean >= 22)
    hypercontinental = (
        (coldest_month_avg_mean < 0)
        & (avg_mean < 10)
        & (warmest_month_avg_mean - coldest_month_avg_mean >= 26)
    )
    oceanic = (warmest_month_avg_mean - coldest_month_avg_mean < 22)

    polar = (grow_months < 3) | (mild_months == 0)
    subpolar = (mild_months >= 1) & (grow_months >= 3) & (mild_months < 4)
    midlat = (mild_months >= 4) & (coldest_month_avg_mean < 16)
    mild = midlat & (coldest_month_avg_mean >= 4)
    tropical = coldest_month_avg_mean >= 16

    polar_da = da(polar)
    subpolar_da = da(subpolar)
    midlat_da = da(midlat)
    tropical_da = da(tropical)
    continental_da = da(continental)
    hypercontinental_da = da(hypercontinental)
    oceanic_da = da(oceanic)
    mild_da = da(mild)

    winter_dominant = mild_da & pole_of_32 & (winter_precip >= 0.6 * annual_precip)

    precip_threshold = xr.where(
        tropical_da,
        (avg_mean * 18) + 260,
        xr.where(
            hypercontinental_da,
            (avg_mean * 18) + 440,
            xr.where(
                winter_dominant,
                (avg_mean * 18) + 100,
                (avg_mean * 18) + 380,
            ),
        ),
    )

    humid = annual_precip >= precip_threshold
    semiarid = (precip_threshold >= annual_precip) & (
        annual_precip >= 0.4 * precip_threshold
    )
    arid = annual_precip < 0.4 * precip_threshold

    return {
        "continental": continental,
        "hypercontinental": hypercontinental,
        "oceanic": oceanic,
        "polar": polar,
        "subpolar": subpolar,
        "midlat": midlat,
        "mild": mild,
        "tropical": tropical,
        "polar_da": polar_da,
        "subpolar_da": subpolar_da,
        "midlat_da": midlat_da,
        "tropical_da": tropical_da,
        "continental_da": continental_da,
        "hypercontinental_da": hypercontinental_da,
        "oceanic_da": oceanic_da,
        "mild_da": mild_da,
        "winter_dominant": winter_dominant,
        "precip_threshold": precip_threshold,
        "humid": humid,
        "semiarid": semiarid,
        "arid": arid,
        "humid_da": da(humid),
        "semiarid_da": da(semiarid),
        "arid_da": da(arid),
    }


def classify(metrics: dict, regimes: dict, latbands: dict) -> xr.DataArray:
    avg_mean = metrics["avg_mean"]
    annual_precip = metrics["annual_precip"]
    summer_precip = metrics["summer_precip"]
    winter_precip = metrics["winter_precip"]
    driest_month_precip = metrics["driest_month_precip"]
    coldest_month_avg_mean = metrics["coldest_month_avg_mean"]
    warmest_month_avg_mean = metrics["warmest_month_avg_mean"]
    cool_months = metrics["cool_months"]

    polar_da = regimes["polar_da"]
    subpolar_da = regimes["subpolar_da"]
    midlat_da = regimes["midlat_da"]
    tropical_da = regimes["tropical_da"]
    continental_da = regimes["continental_da"]
    oceanic_da = regimes["oceanic_da"]
    mild_da = regimes["mild_da"]
    humid_da = regimes["humid_da"]
    semiarid_da = regimes["semiarid_da"]
    arid_da = regimes["arid_da"]

    pole_of_32 = latbands["pole_of_32"]
    eq_of_32 = latbands["eq_of_32"]

    cls = xr.full_like(avg_mean, "Unknown", dtype=object)

    cls = xr.where(polar_da & (cool_months == 0), "Ice Cap", cls)
    cls = xr.where(polar_da & (cool_months >= 1), "Tundra", cls)

    cls = xr.where(subpolar_da & continental_da, "Cold Taiga", cls)
    cls = xr.where(subpolar_da & oceanic_da, "Cool Taiga", cls)

    cls = xr.where(
        midlat_da
        & humid_da
        & continental_da
        & ((avg_mean < 6) | (warmest_month_avg_mean < 20))
        & ((summer_precip >= winter_precip) | (driest_month_precip >= 40)),
        "Hemiboreal Continental",
        cls,
    )

    cls = xr.where(
        midlat_da
        & humid_da
        & continental_da
        & (coldest_month_avg_mean < 0)
        & (avg_mean >= 6)
        & (warmest_month_avg_mean >= 20)
        & ((summer_precip >= winter_precip) | (driest_month_precip >= 40)),
        "Humid Continental",
        cls,
    )

    cls = xr.where(
        midlat_da
        & humid_da
        & (coldest_month_avg_mean < 4)
        & (continental_da | (warmest_month_avg_mean >= 22))
        & (winter_precip > summer_precip)
        & (driest_month_precip < 40),
        "Dry Summer Continental",
        cls,
    )

    cls = xr.where(
        midlat_da
        & humid_da
        & (coldest_month_avg_mean >= 0)
        & (coldest_month_avg_mean < 6)
        & (warmest_month_avg_mean >= 22)
        & ((summer_precip >= winter_precip) | (driest_month_precip >= 40)),
        "Humid Warm Temperate",
        cls,
    )

    cls = xr.where(
        midlat_da
        & humid_da
        & oceanic_da
        & (coldest_month_avg_mean < 4)
        & (warmest_month_avg_mean < 22)
        & (winter_precip > summer_precip)
        & (driest_month_precip < 40)
        & pole_of_32,
        "Cool West Coast",
        cls,
    )

    cls = xr.where(
        midlat_da
        & humid_da
        & oceanic_da
        & (coldest_month_avg_mean < 4)
        & (warmest_month_avg_mean < 22)
        & (
            (summer_precip >= winter_precip)
            | (driest_month_precip >= 40)
            | eq_of_32
        ),
        "Cool Oceanic",
        cls,
    )

    cls = xr.where(
        humid_da
        & mild_da
        & (warmest_month_avg_mean < 22)
        & (winter_precip > summer_precip)
        & (driest_month_precip < 40)
        & pole_of_32,
        "Mild West Coast",
        cls,
    )

    cls = xr.where(
        humid_da
        & mild_da
        & (warmest_month_avg_mean < 22)
        & (
            (summer_precip >= winter_precip)
            | (driest_month_precip >= 40)
            | eq_of_32
        ),
        "Mild Oceanic",
        cls,
    )

    cls = xr.where(
        humid_da
        & mild_da
        & (warmest_month_avg_mean >= 22)
        & (winter_precip > summer_precip)
        & (driest_month_precip < 40),
        "Mediterranean",
        cls,
    )

    cls = xr.where(
        midlat_da
        & humid_da
        & (coldest_month_avg_mean >= 6)
        & (warmest_month_avg_mean >= 22)
        & ((summer_precip >= winter_precip) | (driest_month_precip >= 40)),
        "Humid Subtropical",
        cls,
    )

    cls = xr.where(
        midlat_da
        & semiarid_da
        & (warmest_month_avg_mean - coldest_month_avg_mean >= 18)
        & (coldest_month_avg_mean < 8),
        "Variable Semi-Arid",
        cls,
    )

    cls = xr.where(
        (midlat_da | tropical_da)
        & semiarid_da
        & (warmest_month_avg_mean - coldest_month_avg_mean < 18)
        & (warmest_month_avg_mean < 26),
        "Moderated Semi-Arid",
        cls,
    )

    cls = xr.where(
        semiarid_da
        & (coldest_month_avg_mean >= 8)
        & (warmest_month_avg_mean >= 26),
        "Hot Semi-Arid",
        cls,
    )

    cls = xr.where(
        midlat_da
        & arid_da
        & (warmest_month_avg_mean - coldest_month_avg_mean >= 18)
        & (coldest_month_avg_mean < 8),
        "Variable Desert",
        cls,
    )

    cls = xr.where(
        (midlat_da | tropical_da)
        & arid_da
        & (warmest_month_avg_mean - coldest_month_avg_mean < 18)
        & (warmest_month_avg_mean < 26),
        "Moderated Desert",
        cls,
    )

    cls = xr.where(
        arid_da
        & (coldest_month_avg_mean >= 8)
        & (warmest_month_avg_mean >= 26),
        "Hot Desert",
        cls,
    )

    cls = xr.where(
        tropical_da
        & humid_da
        & ((annual_precip < 1250) | (driest_month_precip < 50)),
        "Tropical Monsoon Savanna",
        cls,
    )

    cls = xr.where(
        tropical_da
        & humid_da
        & (annual_precip >= 1250)
        & (driest_month_precip >= 50),
        "Tropical Rainforest",
        cls,
    )

    return cls


def classification_to_index(
    classification: xr.DataArray, labels: list[str]
) -> xr.DataArray:
    label_to_index = {label: i for i, label in enumerate(labels)}
    vec_map = np.vectorize(lambda x: label_to_index.get(x, np.nan), otypes=[float])

    return xr.DataArray(
        vec_map(classification.values),
        dims=classification.dims,
        coords=classification.coords,
    )


def build_classification(
    t: xr.DataArray,
    pr: xr.DataArray,
) -> dict:
    """
    End-to-end classification builder.

    Returns a dict containing:
    - classification
    - metrics
    - regimes
    - latbands
    """
    lat2d, _ = xr.broadcast(t.lat, t.lon)
    latbands = lat_bands(lat2d)

    metrics = compute_metrics(t, pr)
    regimes = compute_regimes(metrics, latbands)
    classification = classify(metrics, regimes, latbands)

    return {
        "classification": classification,
        "metrics": metrics,
        "regimes": regimes,
        "latbands": latbands,
    }