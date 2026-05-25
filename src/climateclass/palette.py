import numpy as np
from matplotlib.colors import BoundaryNorm, ListedColormap
from matplotlib.patches import Patch


LABELS = [
    "Ice Cap",
    "Tundra",
    "Cold Taiga",
    "Cool Taiga",
    "Hemiboreal Continental",
    "Humid Continental",
    "Dry Summer Continental",
    "Monsoonal Continental",
    "Dry Summer Montane",
    "Suboceanic",
    "Dry Summer West Coast",
    "Oceanic",
    "Dry Summer Subtropical",
    "Humid Subtropical",
    "Monsoonal Subtropical",
    "Variable Semi-Arid",
    "Moderated Semi-Arid",
    "Warm Semi-Arid",
    "Variable Arid",
    "Moderated Arid",
    "Warm Arid",
    "Monsoonal Tropical",
    "Humid Tropical",
]

COLORS = [
    "#808080",  # Ice Cap
    "#C5C9C7",  # Tundra
    "#22421C",  # Cold Taiga
    "#2C7AAB",  # Cool Taiga
    "#487A40",  # Hemiboreal Continental
    "#84B07D",  # Humid Continental
    "#B18EED",  # Dry Summer Continental
    "#90EE90",  # Monsoonal Continental
    "#5316B8",  # Dry Summer Montane
    "#030764",  # Suboceanic
    "#BABF24",  # Dry Summer West Coast
    "#48E8B0",  # Oceanic
    "#FFFF14",  # Dry Summer Subtropical
    "#15B01A",  # Humid Subtropical
    "#CDFC65",  # Monsoonal Subtropical
    "#BF9982",  # Variable Semi-Arid
    "#784122",  # Moderated Semi-Arid
    "#BD5615",  # Warm Semi-Arid
    "#FCE1C7",  # Variable Arid
    "#FC74DA",  # Moderated Arid
    "#FACC4B",  # Warm Arid
    "#08FF24",  # Monsoonal Tropical
    "#1D7555",  # Humid Tropical
]

LABEL_TO_INT = {lab: i for i, lab in enumerate(LABELS)}
CBAR_LABELS = LABELS.copy()

CMAP = ListedColormap(COLORS)
CMAP.set_bad("white")

NORM = BoundaryNorm(
    np.arange(-0.5, len(LABELS) + 0.5, 1),
    CMAP.N,
)

LEGEND_ITEMS = [
    Patch(facecolor=color, edgecolor="black", linewidth=0.4, label=label)
    for label, color in zip(LABELS, COLORS)
]