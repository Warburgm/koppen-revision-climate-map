import numpy as np
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature as cfeature
from matplotlib.colors import ListedColormap, BoundaryNorm


def make_cmap(legend_items):
    """
    Create a categorical colormap from legend_items.

    Parameters
    ----------
    legend_items : list[tuple[str, str]]
        List of (label, color) pairs.

    Returns
    -------
    cmap : ListedColormap
    norm : BoundaryNorm
    colors : list[str]
    """
    colors = [color for _, color in legend_items]
    cmap = ListedColormap(colors)
    cmap.set_bad("white")
    norm = BoundaryNorm(
        boundaries=np.arange(len(colors) + 1),
        ncolors=len(colors),
    )
    return cmap, norm, colors


def prepare_lon_lat(da):
    """
    Extract lon/lat 1D coordinates and corresponding 2D meshgrid.
    """
    lon = da.lon.values
    lat = da.lat.values
    lon_grid, lat_grid = np.meshgrid(lon, lat)
    return lon, lat, lon_grid, lat_grid


def plot_global_map(
    classification_idx,
    legend_items,
    cbar_labels,
    title="Köppen Revision Climate Types",
    subtitle=None,
    figsize=(14, 7),
    dpi=900,
    outfile=None,
    coastline_lw=0.1,
    borders_lw=0.1,
    states_lw=0.1,
    cbar_pad=0.02,
):
    """
    Plot a global categorical climate classification map.

    Parameters
    ----------
    classification_idx : xr.DataArray
        2D indexed classification field with dims (lat, lon).
        Ocean / masked regions should be NaN.
    legend_items : list[tuple[str, str]]
        List of (label, color) pairs in the same order as class indices.
    cbar_labels : list[str]
        Colorbar labels in class-index order.
    title : str
        Plot title.
    subtitle : str or None
        Optional subtitle / caption placed below the map.
    figsize : tuple
        Figure size.
    dpi : int
        Output DPI if saved.
    outfile : str or None
        Path to save figure. If None, figure is not saved.
    """
    if classification_idx.ndim != 2:
        raise ValueError(
            f"classification_idx must be 2D (lat, lon); got dims={classification_idx.dims}"
        )

    _, _, lon_grid, lat_grid = prepare_lon_lat(classification_idx)
    cmap, norm, _ = make_cmap(legend_items)

    fig, ax = plt.subplots(
        figsize=figsize,
        subplot_kw={"projection": ccrs.PlateCarree()},
    )

    mesh = ax.pcolormesh(
        lon_grid,
        lat_grid,
        classification_idx,
        cmap=cmap,
        norm=norm,
        shading="auto",
        transform=ccrs.PlateCarree(),
    )

    ax.set_facecolor("white")
    ax.add_feature(cfeature.COASTLINE, linewidth=coastline_lw)
    ax.add_feature(cfeature.BORDERS, linewidth=borders_lw)
    ax.add_feature(cfeature.STATES, linewidth=states_lw)

    if subtitle is not None:
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

    cbar = plt.colorbar(mesh, ax=ax, orientation="vertical", pad=cbar_pad)
    cbar.set_ticks(np.arange(len(cbar_labels)) + 0.5)
    cbar.set_ticklabels(cbar_labels)

    plt.title(title)
    plt.tight_layout()

    if outfile is not None:
        plt.savefig(outfile, dpi=dpi, bbox_inches="tight")

    return fig, ax, mesh


def quick_plot(
    classification_idx,
    legend_items,
    cbar_labels,
    outfile="global_climate_types_1991_2020.png",
):
    """
    Convenience wrapper using your current preferred defaults.
    """
    return plot_global_map(
        classification_idx=classification_idx,
        legend_items=legend_items,
        cbar_labels=cbar_labels,
        title="Köppen Revision Climate Types (1991–2020 Averages)",
        subtitle="Based on 0.1°× 0.1° resolution ERA5 reanalysis temperature and precipitation monthly means",
        outfile=outfile,
        dpi=900,
    )