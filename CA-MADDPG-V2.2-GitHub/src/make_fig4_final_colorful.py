from pathlib import Path

import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt


# =========================================================
# Official fixed 100-episode results
# =========================================================
METHODS = [
    {
        "name": "Latency heuristic",
        "latency": 1613.159708,
        "energy": 37113.174659,
        "success": 50.83,
        "color": "#1F77B4",
        "marker": "o",
        "size": 62,
    },
    {
        "name": "CA-MADDPG-V2.2",
        "latency": 1684.140031,
        "energy": 38666.154115,
        "success": 49.36,
        "color": "#2CA02C",
        "marker": "D",
        "size": 88,
    },
    {
        "name": "MADDPG",
        "latency": 1785.912933,
        "energy": 43354.071118,
        "success": 48.43,
        "color": "#FF7F0E",
        "marker": "s",
        "size": 68,
    },
    {
        "name": "CA-MADDPG-V2",
        "latency": 1925.812946,
        "energy": 29940.520838,
        "success": 45.72,
        "color": "#9467BD",
        "marker": "^",
        "size": 72,
    },
    {
        "name": "Random allocation",
        "latency": 2601.074407,
        "energy": 86513.626403,
        "success": 38.49,
        "color": "#D62728",
        "marker": "X",
        "size": 78,
    },
]


# =========================================================
# Label positions chosen manually to prevent overlap
# =========================================================
LABEL_OFFSETS = {
    "Latency heuristic": (8, -32),
    "CA-MADDPG-V2.2": (20, 12),
    "MADDPG": (12, 12),
    "CA-MADDPG-V2": (10, 10),
    "Random allocation": (-8, -22),
}


OUTPUT_DIRECTORY = Path(
    "results/final_conference_package/"
    "figures_final_single_column"
)

OUTPUT_DIRECTORY.mkdir(
    parents=True,
    exist_ok=True,
)

JPG_PATH = (
    OUTPUT_DIRECTORY
    / "Fig_4_Final_Colorful_Latency_Energy_Tradeoff.jpg"
)

PDF_PATH = (
    OUTPUT_DIRECTORY
    / "Fig_4_Final_Colorful_Latency_Energy_Tradeoff.pdf"
)


# =========================================================
# IEEE single-column style
# =========================================================
plt.rcParams.update(
    {
        "font.family": "serif",
        "font.serif": ["DejaVu Serif"],
        "font.size": 8,
        "axes.labelsize": 8.5,
        "xtick.labelsize": 7.3,
        "ytick.labelsize": 7.3,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
        "axes.linewidth": 0.8,
    }
)


# Broken y-axis avoids large empty space and
# separates Random Allocation from clustered methods.
figure = plt.figure(
    figsize=(3.5, 4.1)
)

grid = figure.add_gridspec(
    2,
    1,
    height_ratios=[1, 3],
    hspace=0.05,
)

upper_axis = figure.add_subplot(
    grid[0]
)

lower_axis = figure.add_subplot(
    grid[1],
    sharex=upper_axis,
)


# =========================================================
# Plot points
# =========================================================
for item in METHODS:
    target_axis = (
        upper_axis
        if item["energy"] > 70000
        else lower_axis
    )

    proposed = (
        item["name"]
        == "CA-MADDPG-V2.2"
    )

    target_axis.scatter(
        item["latency"],
        item["energy"],
        s=item["size"],
        marker=item["marker"],
        facecolor=item["color"],
        edgecolor="black",
        linewidth=(
            1.3 if proposed else 0.8
        ),
        zorder=4,
    )

    horizontal_alignment = (
        "right"
        if item["name"] == "Random allocation"
        else "left"
    )

    label = (
        f"{item['name']}\n"
        f"{item['success']:.2f}%"
    )

    target_axis.annotate(
        label,
        xy=(
            item["latency"],
            item["energy"],
        ),
        xytext=LABEL_OFFSETS[
            item["name"]
        ],
        textcoords="offset points",
        ha=horizontal_alignment,
        va="center",
        fontsize=6.8,
        fontweight=(
            "bold" if proposed else "normal"
        ),
        bbox={
            "boxstyle": "round,pad=0.20",
            "facecolor": "white",
            "edgecolor": item["color"],
            "linewidth": 0.75,
            "alpha": 0.97,
        },
        arrowprops={
            "arrowstyle": "-",
            "color": item["color"],
            "linewidth": 0.75,
            "shrinkA": 1,
            "shrinkB": 3,
        },
        annotation_clip=False,
        zorder=5,
    )


# =========================================================
# Axis ranges
# =========================================================
lower_axis.set_xlim(
    1500,
    2725,
)

lower_axis.set_ylim(
    27000,
    50000,
)

upper_axis.set_ylim(
    82000,
    90000,
)


# =========================================================
# Broken-axis visual marks
# =========================================================
upper_axis.spines[
    "bottom"
].set_visible(False)

lower_axis.spines[
    "top"
].set_visible(False)

upper_axis.tick_params(
    bottom=False,
    labelbottom=False,
)

diagonal_size = 0.012

upper_kwargs = {
    "transform": upper_axis.transAxes,
    "color": "black",
    "clip_on": False,
    "linewidth": 0.8,
}

upper_axis.plot(
    (-diagonal_size, diagonal_size),
    (-diagonal_size, diagonal_size),
    **upper_kwargs,
)

upper_axis.plot(
    (
        1 - diagonal_size,
        1 + diagonal_size,
    ),
    (
        -diagonal_size,
        diagonal_size,
    ),
    **upper_kwargs,
)

lower_kwargs = {
    "transform": lower_axis.transAxes,
    "color": "black",
    "clip_on": False,
    "linewidth": 0.8,
}

lower_axis.plot(
    (-diagonal_size, diagonal_size),
    (
        1 - diagonal_size,
        1 + diagonal_size,
    ),
    **lower_kwargs,
)

lower_axis.plot(
    (
        1 - diagonal_size,
        1 + diagonal_size,
    ),
    (
        1 - diagonal_size,
        1 + diagonal_size,
    ),
    **lower_kwargs,
)


# =========================================================
# Labels and styling
# =========================================================
lower_axis.set_xlabel(
    "Mean latency (ms)"
)

figure.text(
    0.03,
    0.50,
    "Mean energy (mJ)",
    rotation="vertical",
    va="center",
    fontsize=8.5,
)

lower_axis.text(
    0.02,
    0.96,
    "Lower-left is preferable",
    transform=lower_axis.transAxes,
    ha="left",
    va="top",
    fontsize=6.7,
    style="italic",
)

for axis in [
    upper_axis,
    lower_axis,
]:
    axis.grid(
        linestyle=":",
        linewidth=0.55,
        alpha=0.50,
    )

    axis.set_axisbelow(
        True
    )


# No internal title.
# The manuscript caption will provide the figure title.

figure.subplots_adjust(
    left=0.19,
    right=0.98,
    bottom=0.12,
    top=0.98,
)


# =========================================================
# Save PDF and JPG
# =========================================================
figure.savefig(
    JPG_PATH,
    dpi=600,
    format="jpg",
    bbox_inches="tight",
    pad_inches=0.04,
)

figure.savefig(
    PDF_PATH,
    format="pdf",
    bbox_inches="tight",
    pad_inches=0.04,
)

plt.close(
    figure
)


print("=" * 88)
print("FINAL COLORFUL FIGURE 4 SCRIPT COMPLETED")
print("=" * 88)
print("Methods included:", len(METHODS))
print("CA-MADDPG-V2 included: True")
print("Colorful markers: True")
print("Broken y-axis: True")
print("Legend removed: True")
print("Text overlap avoided: True")
print("Single-column width: 3.5 inches")
print("JPG:", JPG_PATH)
print("PDF:", PDF_PATH)
print("=" * 88)
