import numpy as np
from matplotlib.colors import BoundaryNorm, ListedColormap
from matplotlib.patches import Patch


LEGEND_GROUPS = {
    "A": {
        "title": "A (Tropical)",
        "x0": 0.02,
        "w": 0.10,
        "ncol": 1,
        "items": [
            {
                "name": "Tropical, No Dry Season",
                "legend_label": (
                    r"Tropical, No Dry Season (Af)"
                    "\n"
                    r"$\it{Humid\ Tropical}$"
                ),
                "color": "#0021F3",
            },
            {
                "name": "Tropical, Wet & Dry Season",
                "legend_label": (
                    r"Tropical, Wet & Dry Season (Aw)"
                    "\n"
                    r"$\it{Monsoonal\ Tropical}$"
                ),
                "color": "#6784DB",
            },
        ],
    },
    "B": {
        "title": "B (Mild)",
        "x0": 0.15,
        "w": 0.28,
        "ncol": 2,
        "items": [
            {
                "name": "Subtropical, No Dry Season",
                "legend_label": (
                    r"Subtropical, No Dry Season (Bfa)"
                    "\n"
                    r"$\it{Humid\ Subtropical}$"
                ),
                "color": "#CDFC65",
            },
            {
                "name": "Subtropical, Dry Winter",
                "legend_label": (
                    r"Subtropical, Dry Winter (Bwa)"
                    "\n"
                    r"$\it{Monsoonal\ Subtropical}$"
                ),
                "color": "#90EE90",
            },
            {
                "name": "Subtropical, Dry Summer",
                "legend_label": (
                    r"Subtropical, Dry Summer (Bma)"
                    "\n"
                    r"$\it{Mediterranean}$"
                ),
                "color": "#FFF824",
            },
            {
                "name": "Oceanic, No Dry Season",
                "legend_label": (
                    r"Oceanic, No Dry Season (Bfb)"
                    "\n"
                    r"$\it{Oceanic}$"
                ),
                "color": "#40FC00",
            },
            {
                "name": "Oceanic, Dry Winter",
                "legend_label": (
                    r"Oceanic, Dry Winter (Bwb)"
                    "\n"
                    r"$\it{Subequatorial\ Highland}$"
                ),
                "color": "#15B01A",
            },
            {
                "name": "Oceanic, Dry Summer",
                "legend_label": (
                    r"Oceanic, Dry Summer (Bmb)"
                    "\n"
                    r"$\it{West\ Coast}$"
                ),
                "color": "#ABAB29",
            },
        ],
    },
    "C": {
        "title": "C (Continental)",
        "x0": 0.45,
        "w": 0.17,
        "ncol": 2,
        "wrap_main": True,
        "main_wrap_width": 17,
        "italic_wrap_width": 16,
        "items": [
            {
                "name": "Continental, Long Summer, No Dry Season",
                "legend_label": (
                    r"Long Summer, No Dry Season (Cfa)"
                    "\n"
                    r"$\it{Humid\ Continental}$"
                ),
                "color": "#00fff9",
            },
            {
                "name": "Continental, Long Summer, Dry Winter",
                "legend_label": (
                    r"Long Summer, Dry Winter (Cwa)"
                    "\n"
                    r"$\it{Monsoonal\ Continental}$"
                ),
                "color": "#B18EED",
            },
            {
                "name": "Continental, Short Wet Summer",
                "legend_label": (
                    r"Short Wet Summer (Cb)"
                    "\n"
                    r"$\it{Short\ Summer\ Continental}$"
                ),
                "color": "#00b8ff",
            },
            {
                "name": "Continental, Dry Summer",
                "legend_label": (
                    r"Dry Summer (Cm)"
                    "\n"
                    r"$\it{Dry\ Summer\ Continental}$"
                ),
                "color": "#7F1799",
            },
        ],
    },
    "D": {
        "title": "D (Dry)",
        "x0": 0.64,
        "w": 0.24,
        "ncol": 2,
        "items": [
            {
                "name": "Dry, Variable Steppe",
                "legend_label": (
                    r"Variable Steppe (Ds1)"
                    "\n"
                    r"$\it{Variable\ Steppe}$"
                ),
                "color": "#BF9982",
            },
            {
                "name": "Dry, Moderated Steppe",
                "legend_label": (
                    r"Moderated Steppe (Ds2)"
                    "\n"
                    r"$\it{Moderated\ Steppe}$"
                ),
                "color": "#784122",
            },
            {
                "name": "Dry, Hot Steppe",
                "legend_label": (
                    r"Hot Steppe (Ds3)"
                    "\n"
                    r"$\it{Hot\ Steppe}$"
                ),
                "color": "#FACC4B",
            },
            {
                "name": "Dry, Variable Desert",
                "legend_label": (
                    r"Variable Desert (Dd1)"
                    "\n"
                    r"$\it{Variable\ Desert}$"
                ),
                "color": "#FCE1C7",
            },
            {
                "name": "Dry, Moderated Desert",
                "legend_label": (
                    r"Moderated Desert (Dd2)"
                    "\n"
                    r"$\it{Moderated\ Desert}$"
                ),
                "color": "#FC74DA",
            },
            {
                "name": "Dry, Hot Desert",
                "legend_label": (
                    r"Hot Desert (Dd3)"
                    "\n"
                    r"$\it{Hot\ Desert}$"
                ),
                "color": "#FF0000",
            },
        ],
    },
    "E": {
        "title": "E (Cold)",
        "x0": 0.88,
        "w": 0.10,
        "ncol": 1,
        "wrap": False,
        "items": [
            {
                "name": "Cold, Ice Cap",
                "legend_label": (
                    r"Ice Cap (Eb1)"
                    "\n"
                    r"$\it{Ice\ Cap}$"
                ),
                "color": "#808080",
            },
            {
                "name": "Cold, Tundra",
                "legend_label": (
                    r"Tundra (Eb2)"
                    "\n"
                    r"$\it{Tundra}$"
                ),
                "color": "#C5C9C7",
            },
            {
                "name": "Cold, Taiga, Severe Winter",
                "legend_label": (
                    r"Taiga, Severe Winter (Ea1)"
                    "\n"
                    r"$\it{Cold\ Taiga}$"
                ),
                "color": "#22421C",
            },
            {
                "name": "Cold, Taiga, Moderated Winter",
                "legend_label": (
                    r"Taiga, Moderated Winter (Ea2)"
                    "\n"
                    r"$\it{Cool\ Taiga}$"
                ),
                "color": "#568F5A",
            },
        ],
    },
}


GROUP_ORDER = ["A", "B", "C", "D", "E"]

CLASS_INFO = [
    {
        **item,
        "group": group,
        "group_title": LEGEND_GROUPS[group]["title"],
    }
    for group in GROUP_ORDER
    for item in LEGEND_GROUPS[group]["items"]
]

LABELS = [item["name"] for item in CLASS_INFO]
COLORS = [item["color"] for item in CLASS_INFO]
DISPLAY_LABELS = [item["legend_label"] for item in CLASS_INFO]

LABEL_TO_INT = {
    label: i
    for i, label in enumerate(LABELS)
}

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
        label=item["legend_label"],
    )
    for item in CLASS_INFO
]