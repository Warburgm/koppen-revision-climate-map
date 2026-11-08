import numpy as np
from matplotlib.colors import BoundaryNorm, ListedColormap
from matplotlib.patches import Patch


CLASS_INFO = [
    {
        "name": "Tropical Rainforest",
        "code": "5E",
        "group": "5",
        "group_title": "Tropical",
        "color": "#1C14FC",
    },
    {
        "name": "Tropical Monsoon Savanna",
        "code": "5CD",
        "group": "5",
        "group_title": "Tropical",
        "color": "#2F69BA",
    },
    {
        "name": "Dry Summer Mild Subtropical",
        "code": "4Ca",
        "group": "4",
        "group_title": "Subtropical",
        "color": "#f6c203",
    },
    {
        "name": "Dry Summer Warm Subtropical",
        "code": "4Cb",
        "group": "4",
        "group_title": "Subtropical",
        "color": "#d67814",
    },
    {
        "name": "Dry Winter Mild Subtropical",
        "code": "4Da",
        "group": "4",
        "group_title": "Subtropical",
        "color": "#1aeeb4",
    },
    {
        "name": "Dry Winter Warm Subtropical",
        "code": "4Db",
        "group": "4",
        "group_title": "Subtropical",
        "color": "#2d8b76",
    },
    {
        "name": "Humid Mild Subtropical",
        "code": "4Ea",
        "group": "4",
        "group_title": "Subtropical",
        "color": "#8db954",
    },
    {
        "name": "Humid Warm Subtropical",
        "code": "4Eb",
        "group": "4",
        "group_title": "Subtropical",
        "color": "#b5f80b",
    },
    {
        "name": "Dry Summer Cool Temperate",
        "code": "3Ca",
        "group": "3",
        "group_title": "Temperate",
        "color": "#cc2be4",
    },
    {
        "name": "Dry Summer Mixed Temperate",
        "code": "3Cb",
        "group": "3",
        "group_title": "Temperate",
        "color": "#7c26f7",
    },
    {
        "name": "Dry Summer Mild Temperate",
        "code": "3Cc",
        "group": "3",
        "group_title": "Temperate",
        "color": "#abaa2f",
    },
    {
        "name": "Dry Summer Warm Temperate",
        "code": "3Cd",
        "group": "3",
        "group_title": "Temperate",
        "color": "#FFF824",
    },
    {
        "name": "Dry Winter Cool Temperate",
        "code": "3Da",
        "group": "3",
        "group_title": "Temperate",
        "color": "#7549b1",
    },
    {
        "name": "Dry Winter Mixed Temperate",
        "code": "3Db",
        "group": "3",
        "group_title": "Temperate",
        "color": "#a56de5",
    },
    {
        "name": "Dry Winter Mild Temperate",
        "code": "3Dc",
        "group": "3",
        "group_title": "Temperate",
        "color": "#251a9b",
    },
    {
        "name": "Dry Winter Warm Temperate",
        "code": "3Dd",
        "group": "3",
        "group_title": "Temperate",
        "color": "#93da84",
    },
    {
        "name": "Humid Cool Temperate",
        "code": "3Ea",
        "group": "3",
        "group_title": "Temperate",
        "color": "#48b1cd",
    },
    {
        "name": "Humid Mixed Temperate",
        "code": "3Eb",
        "group": "3",
        "group_title": "Temperate",
        "color": "#04fafe",
    },
    {
        "name": "Humid Mild Temperate",
        "code": "3Ec",
        "group": "3",
        "group_title": "Temperate",
        "color": "#13f30a",
    },
    {
        "name": "Humid Warm Temperate",
        "code": "3Ed",
        "group": "3",
        "group_title": "Temperate",
        "color": "#27ad34",
    },
    {
        "name": "Variable Steppe",
        "code": "3B",
        "group": "A-B",
        "group_title": "Dry",
        "color": "#8a7a5b",
    },
    {
        "name": "Moderated Steppe",
        "code": "345B",
        "group": "A-B",
        "group_title": "Dry",
        "color": "#4d3d29",
    },
    {
        "name": "Hot Steppe",
        "code": "45B",
        "group": "A-B",
        "group_title": "Dry",
        "color": "#c7b079",
    },
    {
        "name": "Variable Desert",
        "code": "3A",
        "group": "A-B",
        "group_title": "Dry",
        "color": "#FCE1C7",
    },
    {
        "name": "Moderated Desert",
        "code": "345A",
        "group": "A-B",
        "group_title": "Dry",
        "color": "#E38DCD",
    },
    {
        "name": "Hot Desert",
        "code": "45A",
        "group": "A-B",
        "group_title": "Dry",
        "color": "#FF0000",
    },
    {
        "name": "Short Summer Moderated",
        "code": "2b",
        "group": "1-2",
        "group_title": "Polar/Subpolar/Alpine",
        "color": "#49914d",
    },
    {
        "name": "Short Summer Boreal",
        "code": "2a",
        "group": "1-2",
        "group_title": "Polar/Subpolar/Alpine",
        "color": "#0a530e",
    },
    {
        "name": "Tundra",
        "code": "1b",
        "group": "1-2",
        "group_title": "Polar/Subpolar/Alpine",
        "color": "#c1c6c8",
    },
    {
        "name": "Ice Cap",
        "code": "1a",
        "group": "1-2",
        "group_title": "Polar/Subpolar/Alpine",
        "color": "#727475",
    },
]


LABELS = [item["name"] for item in CLASS_INFO]
COLORS = [item["color"] for item in CLASS_INFO]
CODES = [item["code"] for item in CLASS_INFO]

LABEL_TO_INT = {
    label: i
    for i, label in enumerate(LABELS)
}

DISPLAY_LABELS = [
    f"{item['name']} ({item['code']})"
    for item in CLASS_INFO
]

CBAR_LABELS = DISPLAY_LABELS.copy()

CMAP = ListedColormap(COLORS)
CMAP.set_bad("white")

NORM = BoundaryNorm(
    np.arange(-0.5, len(LABELS) + 0.5, 1),
    CMAP.N,
)

LEGEND_ITEMS = [
    Patch(
        facecolor=item["color"],
        edgecolor="black",
        linewidth=0.4,
        label=f"{item['name']} ({item['code']})",
    )
    for item in CLASS_INFO
]


GROUP_ORDER = ["5", "4", "3", "A-B", "1-2"]

LEGEND_GROUPS = {
    group: {
        "title": next(
            item["group_title"]
            for item in CLASS_INFO
            if item["group"] == group
        ),
        "items": [
            item
            for item in CLASS_INFO
            if item["group"] == group
        ],
    }
    for group in GROUP_ORDER
}