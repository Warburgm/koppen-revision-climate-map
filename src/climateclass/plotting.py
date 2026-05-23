import numpy as np
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import regionmask

from .palette import LABELS, CBAR_LABELS, CMAP, NORM, LEGEND_ITEMS


def land_mask(class_int, land_resolution="110m"):
    """
    Create a land mask using Natural Earth via regionmask.

    For ERA5 data already on -180 to 180 longitude, do not use wrap_lon=True.
    """
    if land_resolution == "110m":
        land = regionmask.defined_regions.natural_earth_v5_0_0.land_110
    elif land_resolution == "50m":
        land = regionmask.defined_regions.natural_earth_v5_0_0.land_50
    elif land_resolution == "10m":
        land = regionmask.defined_regions.natural_earth_v5_0_0.land_10
    else:
        raise ValueError(
            "land_resolution must be one of: '110m', '50m', or '10m'"
        )

    mask = land.mask(
        class_int["lon"],
        class_int["lat"],
    )

    return mask.notnull()


def mask_to_land(class_int, land_resolution="110m"):
    """
    Mask ocean grid cells from a numeric climate-class DataArray.
    """
    mask = land_mask(class_int, land_resolution=land_resolution)
    return class_int.where(mask)


def quick_plot(
    class_int,
    out=None,
    title="Köppen Revision Climate Types (1991–2020 Averages)",
    subtitle=(
        "Based on 0.1° × 0.1° resolution ERA5 reanalysis "
        "temperature and precipitation monthly means"
    ),
    labels=None,
    cbar_labels=None,
    cmap=None,
    norm=None,
    figsize=(27, 15),
    dpi=900,
    land_resolution="110m",
    ocean_resolution="10m",
    boundary_resolution="10m",
    add_colorbar=True,
    add_legend=False,
    show=False,
):
    """
    Plot indexed climate classes on a global PlateCarree map.

    Parameters
    ----------
    class_int : xr.DataArray
        Numeric climate-class index DataArray, usually produced by
        classification_to_index(...).
    out : str or Path, optional
        Output filename. If None, the figure is not saved.
    title : str
        Main figure title.
    subtitle : str
        Text placed below the map.
    labels : list[str], optional
        Climate-class labels.
    cbar_labels : list[str], optional
        Labels used on the colorbar.
    cmap : matplotlib.colors.Colormap, optional
        Listed colormap.
    norm : matplotlib.colors.BoundaryNorm, optional
        Boundary norm for discrete classes.
    figsize : tuple
        Figure size.
    dpi : int
        Output resolution when saving.
    land_resolution : {"110m", "50m", "10m"}
        Resolution used for regionmask land masking.
    ocean_resolution : {"110m", "50m", "10m"}
        Resolution used for the white ocean overlay.
    boundary_resolution : {"110m", "50m", "10m"}
        Resolution used for coastlines and borders.
    add_colorbar : bool
        Whether to add a vertical colorbar.
    add_legend : bool
        Whether to add a patch legend instead of/in addition to colorbar.
    show : bool
        Whether to call plt.show().

    Returns
    -------
    fig, ax
        Matplotlib figure and axes.
    """
    if labels is None:
        labels = LABELS

    if cbar_labels is None:
        cbar_labels = CBAR_LABELS

    if cmap is None:
        cmap = CMAP

    if norm is None:
        norm = NORM

    masked = mask_to_land(class_int, land_resolution=land_resolution)

    fig = plt.figure(figsize=figsize, facecolor="white")
    ax = plt.axes(projection=ccrs.PlateCarree())
    ax.set_facecolor("white")

    im = ax.pcolormesh(
        masked["lon"],
        masked["lat"],
        masked,
        cmap=cmap,
        norm=norm,
        shading="auto",
        transform=ccrs.PlateCarree(),
        zorder=1,
    )

    ocean = cfeature.NaturalEarthFeature(
        "physical",
        "ocean",
        ocean_resolution,
        facecolor="white",
        edgecolor="none",
    )
    ax.add_feature(ocean, zorder=2)

    ax.add_feature(
        cfeature.COASTLINE.with_scale(boundary_resolution),
        linewidth=0.2,
        zorder=3,
    )
    ax.add_feature(
        cfeature.BORDERS.with_scale(boundary_resolution),
        linewidth=0.2,
        zorder=3,
    )

    ax.set_global()

    ax.text(
        0.5,
        -0.12,
        subtitle,
        transform=ax.transAxes,
        ha="center",
        va="top",
        fontsize=10,
        style="italic",
    )

    if add_colorbar:
        cbar = plt.colorbar(
            im,
            ax=ax,
            orientation="vertical",
            pad=0.02,
            ticks=np.arange(len(labels)),
        )
        cbar.set_ticklabels(cbar_labels)

    if add_legend:
        ax.legend(
            handles=LEGEND_ITEMS,
            loc="lower center",
            bbox_to_anchor=(0.5, -0.25),
            ncol=4,
            frameon=False,
            fontsize=9,
        )

    ax.set_title(title, fontsize=16, pad=12)

    if out is not None:
        fig.savefig(out, dpi=dpi, bbox_inches="tight")

    if show:
        plt.show()

    return fig, ax