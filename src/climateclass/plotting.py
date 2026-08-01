import textwrap
import numpy as np
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import regionmask

from matplotlib.patches import Rectangle

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


def _wrap_plain_text(text, width):
    """
    Wrap plain text to a fixed character width.
    """
    return "\n".join(
        textwrap.wrap(
            text,
            width=width,
            break_long_words=False,
            break_on_hyphens=False,
        )
    )


def _plain_from_math_italic(text):
    """
    Convert a simple mathtext italic label like:

        $\\it{Humid\\ Tropical}$

    into:

        Humid Tropical

    If the text is not in that simple format, return it unchanged.
    """
    text = text.strip()

    if text.startswith(r"$\it{") and text.endswith("}$"):
        text = text[len(r"$\it{"):-2]
        text = text.replace(r"\ ", " ")

    return text


def _math_italic_from_plain(text):
    """
    Convert plain text into a mathtext italic label.
    """
    return r"$\it{" + text.replace(" ", r"\ ") + "}$"


def _wrap_math_italic(text, width):
    """
    Wrap a simple mathtext italic label while preserving italic rendering.
    """
    plain = _plain_from_math_italic(text)

    wrapped_lines = textwrap.wrap(
        plain,
        width=width,
        break_long_words=False,
        break_on_hyphens=False,
    )

    return "\n".join(
        _math_italic_from_plain(line)
        for line in wrapped_lines
    )


def _format_legend_label(label, group_data):
    """
    Format a two-line legend label.

    The expected palette.py format is:

        "Main label (Code)\\n$\\it{Subtitle}$"

    Groups can request wrapping with:
        wrap_main=True
        main_wrap_width=...
        italic_wrap_width=...

    Groups can disable wrapping with:
        wrap=False
    """
    if not group_data.get("wrap", True):
        return label

    if "\n" not in label:
        if group_data.get("wrap_main", False):
            return _wrap_plain_text(
                label,
                group_data.get("main_wrap_width", 18),
            )
        return label

    main_label, italic_label = label.split("\n", 1)

    if group_data.get("wrap_main", False):
        main_label = _wrap_plain_text(
            main_label,
            group_data.get("main_wrap_width", 18),
        )

    if "italic_wrap_width" in group_data:
        italic_label = _wrap_math_italic(
            italic_label,
            group_data["italic_wrap_width"],
        )

    return main_label + "\n" + italic_label


def draw_grouped_legend(
    fig,
    legend_groups=None,
    group_order=None,
    legend_box=(0.03, 0.03, 0.94, 0.21),
    group_title_y=0.94,
    entries_top_y=0.78,
    entries_bottom_y=0.04,
    group_title_fs=13,
    item_fs=7.4,
    swatch_w=0.018,
    swatch_h=0.045,
    title_gap=0.020,
):
    """
    Draw grouped A/B/C/D/E legend using a dedicated legend axis.

    This preserves the original working legend behavior:
    - one separate legend box below the map
    - normalized 0–1 legend coordinates
    - group titles centered above each cluster
    - entries fill downward first, then move to next column
    - swatches centered above centered text
    """
    if legend_groups is None:
        legend_groups = LEGEND_GROUPS

    if group_order is None:
        group_order = GROUP_ORDER

    leg_ax = fig.add_axes(legend_box)

    leg_ax.set_xlim(0, 1)
    leg_ax.set_ylim(0, 1)
    leg_ax.axis("off")

    leg_ax.add_patch(
        Rectangle(
            (0, 0),
            1,
            1,
            facecolor="white",
            edgecolor="black",
            linewidth=1.4,
        )
    )

    def draw_group(ax, cfg):
        group_title = cfg["title"]
        items = cfg["items"]
        x0 = cfg["x0"]
        width = cfg["w"]
        ncol = cfg["ncol"]

        wrap_titles = cfg.get("wrap", True)

        ax.text(
            x0 + width / 2,
            group_title_y,
            group_title,
            ha="center",
            va="top",
            fontsize=group_title_fs,
            fontweight="bold",
        )

        n_items = len(items)
        nrows = int(np.ceil(n_items / ncol))

        col_gap = 0.012 if ncol > 1 else 0.0
        col_w = (width - col_gap * (ncol - 1)) / ncol
        row_h = (entries_top_y - entries_bottom_y) / nrows

        for i, item in enumerate(items):
            if isinstance(item, dict):
                class_name = item["name"]
                display_title = item.get("legend_label", item["name"])
                color = item["color"]
            else:
                class_name, display_title = item
                color = LABEL_TO_COLOR[class_name]

            # Fill downward first, then move to the next column.
            col = i // nrows
            row = i % nrows

            cell_x = x0 + col * (col_w + col_gap)
            cell_y_top = entries_top_y - row * row_h

            sw_x = cell_x + (col_w - swatch_w) / 2
            sw_y = cell_y_top - swatch_h - row_h * 0.10

            ax.add_patch(
                Rectangle(
                    (sw_x, sw_y),
                    swatch_w,
                    swatch_h,
                    facecolor=color,
                    edgecolor="black",
                    linewidth=1.0,
                )
            )

            wrap_width = cfg.get(
                "main_wrap_width",
                20 if col_w >= 0.09 else 17,
            )

            if cfg.get("wrap_main", False) and "\n" in display_title:
                main_line, italic_line = display_title.split("\n", 1)

                main_wrapped = textwrap.fill(
                    main_line,
                    width=wrap_width,
                    break_long_words=False,
                    break_on_hyphens=False,
                )

                italic_wrap_width = cfg.get(
                    "italic_wrap_width",
                    wrap_width,
                )

                italic_wrapped = _wrap_math_italic(
                    italic_line,
                    width=italic_wrap_width,
                )

                wrapped = main_wrapped + "\n" + italic_wrapped

            elif (
                wrap_titles
                and "\n" not in display_title
                and "$" not in display_title
            ):
                wrapped = textwrap.fill(
                    display_title,
                    width=wrap_width,
                    break_long_words=False,
                    break_on_hyphens=False,
                )

            else:
                wrapped = display_title

            ax.text(
                cell_x + col_w / 2,
                sw_y - title_gap,
                wrapped,
                ha="center",
                va="top",
                fontsize=item_fs,
                fontweight="bold",
                linespacing=0.95,
            )

    for group in group_order:
        draw_group(
            leg_ax,
            legend_groups[group],
        )

    return leg_ax


def quick_plot(
    class_int,
    out=None,
    title="Köppen Revision Climate Types (1991–2020 Averages)",
    subtitle=None,
    labels=None,
    cbar_labels=None,
    cmap=None,
    norm=None,
    figsize=(27, 15),
    dpi=900,
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

    if subtitle:
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
            fontsize=6,
        )

    if add_grouped_legend:
        draw_grouped_legend(fig)

    ax.set_title(title, fontsize=16, pad=12)

    if out is not None:
        fig.savefig(out, dpi=dpi, bbox_inches="tight")

    if show:
        plt.show()

    return fig, ax