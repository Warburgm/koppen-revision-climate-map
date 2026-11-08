import textwrap

import numpy as np
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import regionmask

from matplotlib.patches import Patch, Rectangle

from .palette import (
    LABELS,
    CBAR_LABELS,
    CMAP,
    NORM,
    LEGEND_ITEMS,
    LEGEND_GROUPS,
    GROUP_ORDER,
)


def land_mask(class_int, land_resolution="110m"):
    """
    Create a land mask using Natural Earth via regionmask.

    Important:
    For ERA5 data already using -180 to 180 longitude, do not force
    wrap_lon=True. That can cause one hemisphere to disappear.
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


def _wrap_label(text, width):
    """
    Wrap legend text to a fixed character width.
    """
    return "\n".join(
        textwrap.wrap(
            text,
            width=width,
            break_long_words=False,
            break_on_hyphens=False,
        )
    )


def draw_grouped_legend(
    fig,
    legend_groups=None,
    group_order=None,
    left=0.045,
    bottom=0.025,
    width=0.91,
    height=0.18,
    group_gap=0.014,
    box_size=0.0105,
    title_fontsize=16,
    label_fontsize=8.4,
):
    """
    Draw a grouped climate legend beneath the map using figure coordinates.

    This avoids relying on matplotlib's default legend layout and gives
    tighter control over grouped 5/4/3/A-B/1-2 columns.
    """
    if legend_groups is None:
        legend_groups = LEGEND_GROUPS

    if group_order is None:
        group_order = GROUP_ORDER

    # Relative column widths. These can be adjusted if one group needs more room.
    relative_widths = {
        "5": 0.12,
        "4": 0.19,
        "3": 0.27,
        "A-B": 0.27,
        "1-2": 0.11,
    }

    total_relative_width = sum(relative_widths.get(group, 0.18) for group in group_order)
    available_width = width - group_gap * (len(group_order) - 1)

    x = left
    title_y = bottom + height
    item_start_y = title_y - 0.030

    for group in group_order:
        group_data = legend_groups[group]
        group_width = (
            available_width
            * relative_widths.get(group, 0.18)
            / total_relative_width
        )

        fig.text(
            x,
            title_y,
            group_data["title"],
            transform=fig.transFigure,
            ha="left",
            va="top",
            fontsize=title_fontsize,
            fontweight="bold",
        )

        y = item_start_y

        # Approximate wrap width based on visual column width.
        wrap_width = max(16, int(group_width * 115))

        for item in group_data["items"]:
            color = item["color"]
            name = item["name"]
            code = item.get("code", "")

            if code:
                label = f"{name} ({code})"
            else:
                label = name

            wrapped_label = _wrap_label(label, wrap_width)
            n_lines = wrapped_label.count("\n") + 1

            rect = Rectangle(
                (x, y - box_size + 0.002),
                box_size,
                box_size,
                transform=fig.transFigure,
                facecolor=color,
                edgecolor="black",
                linewidth=0.35,
                clip_on=False,
            )
            fig.add_artist(rect)

            fig.text(
                x + box_size + 0.005,
                y,
                wrapped_label,
                transform=fig.transFigure,
                ha="left",
                va="top",
                fontsize=label_fontsize,
                linespacing=1.04,
            )

            # Move down. Multi-line labels get extra space.
            y -= 0.018 + (n_lines - 1) * 0.012

        x += group_width + group_gap


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
    dpi=300,
    land_resolution="110m",
    ocean_resolution="10m",
    boundary_resolution="10m",
    add_colorbar=False,
    add_flat_legend=False,
    add_grouped_legend=True,
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
    add_flat_legend : bool
        Whether to add a standard matplotlib patch legend.
    add_grouped_legend : bool
        Whether to draw the custom grouped A/B/C/D/E legend.
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
    ax = fig.add_subplot(1, 1, 1, projection=ccrs.PlateCarree())
    ax.set_facecolor("white")

    if add_grouped_legend:
        fig.subplots_adjust(
            left=0.02,
            right=0.98,
            top=0.94,
            bottom=0.24,
        )
        subtitle_y = -0.055
    else:
        fig.subplots_adjust(
            left=0.02,
            right=0.98,
            top=0.94,
            bottom=0.08,
        )
        subtitle_y = -0.045

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
        subtitle_y,
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

    if add_flat_legend:
        ax.legend(
            handles=LEGEND_ITEMS,
            loc="lower center",
            bbox_to_anchor=(0.5, -0.25),
            ncol=4,
            frameon=False,
            fontsize=9,
        )

    if add_grouped_legend:
        draw_grouped_legend(fig)

    ax.set_title(title, fontsize=16, pad=12)

    if out is not None:
        fig.savefig(out, dpi=dpi, bbox_inches="tight")

    if show:
        plt.show()

    return fig, ax