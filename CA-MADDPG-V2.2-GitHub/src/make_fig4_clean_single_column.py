from pathlib import Path

import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt


# =========================================================
# Official fixed 100-episode results
# =========================================================
methods = [
    {
        "name": "Latency heuristic",
        "latency": 1613.159708,
        "energy": 37113.174659,
        "success": 50.83,
        "marker": "o",
        "face": "0.78",
        "size": 58,
        "offset": (10, -32),
        "align": "left",
    },
    {
        "name": "CA-MADDPG-V2.2",
        "latency": 1684.140031,
        "energy": 38666.154115,
        "success": 49.36,
        "marker": "D",
        "face": "0.25",
        "size": 78,
        "offset": (12, 14),
        "align": "left",
    },
    {
        "name": "MADDPG",
        "latency": 1785.912933,
        "energy": 43354.071118,
        "success": 48.43,
        "marker": "s",
        "face": "0.68",
        "size": 58,
        "offset": (12, 15),
        "align": "left",
    },
    {
        "name": "CA-MADDPG-V2",
        "latency": 1925.812946,
        "energy": 29940.520838,
        "success": 45.72,
        "marker": "^",
        "face": "0.82",
        "size": 62,
        "offset": (12, -28),
        "align": "left",
    },
    {
        "name": "Random allocation",
        "latency": 2601.074407,
        "energy": 86513.626403,
        "success": 38.49,
        "marker": "P",
        "face": "0.72",
        "size": 68,
        "offset": (-12, -32),
        "align": "right",
    },
]


output_directory = Path(
    "results/final_conference_package/"
    "figures_final_single_column"
)

output_directory.mkdir(
    parents=True,
    exist_ok=True,
)

png_path = (
    output_directory
    / "Fig_4_Clean_Single_Column_Latency_Energy_Tradeoff.png"
)

pdf_path = (
    output_directory
    / "Fig_4_Clean_Single_Column_Latency_Energy_Tradeoff.pdf"
)


# =========================================================
# IEEE single-column visual settings
# =========================================================
plt.rcParams.update(
    {
        "font.family": "serif",
        "font.serif": ["DejaVu Serif"],
        "font.size": 8,
        "axes.labelsize": 8.5,
        "xtick.labelsize": 7.5,
        "ytick.labelsize": 7.5,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
        "axes.linewidth": 0.8,
    }
)


figure, axis = plt.subplots(
    figsize=(3.5, 3.15)
)


for item in methods:
    proposed = (
        item["name"]
        == "CA-MADDPG-V2.2"
    )

    axis.scatter(
        item["latency"],
        item["energy"],
        marker=item["marker"],
        s=item["size"],
        facecolor=item["face"],
        edgecolor="black",
        linewidth=(
            1.25 if proposed else 0.8
        ),
        zorder=4,
    )

    label = (
        f"{item['name']}\n"
        f"{item['success']:.2f}% success"
    )

    axis.annotate(
        label,
        xy=(
            item["latency"],
            item["energy"],
        ),
        xytext=item["offset"],
        textcoords="offset points",
        ha=item["align"],
        va="center",
        fontsize=(
            7.3 if proposed else 6.9
        ),
        fontweight=(
            "bold" if proposed else "normal"
        ),
        arrowprops={
            "arrowstyle": "-",
            "linewidth": 0.55,
            "color": "0.35",
            "shrinkA": 2,
            "shrinkB": 3,
        },
        bbox={
            "boxstyle": "round,pad=0.18",
            "facecolor": "white",
            "edgecolor": "0.75",
            "linewidth": 0.45,
            "alpha": 0.96,
        },
        annotation_clip=False,
        zorder=5,
    )


axis.text(
    0.03,
    0.97,
    "Lower-left is preferable",
    transform=axis.transAxes,
    ha="left",
    va="top",
    fontsize=6.8,
    style="italic",
)


axis.set_xlabel(
    "Mean latency (ms)"
)

axis.set_ylabel(
    "Mean energy (mJ)"
)


# No title inside the graph.
# The manuscript caption will provide the figure title.

axis.set_xlim(
    1500,
    2740,
)

axis.set_ylim(
    25000,
    93000,
)

axis.grid(
    linestyle=":",
    linewidth=0.55,
    alpha=0.5,
)

axis.set_axisbelow(
    True
)

figure.tight_layout(
    pad=0.55
)


figure.savefig(
    png_path,
    dpi=600,
    bbox_inches="tight",
    pad_inches=0.04,
)

figure.savefig(
    pdf_path,
    bbox_inches="tight",
    pad_inches=0.04,
)

plt.close(
    figure
)


print("=" * 84)
print("CLEAN FIGURE 4 SINGLE-COLUMN VERSION CREATED")
print("=" * 84)
print("Legend removed: True")
print("Reward annotations removed: True")
print("Direct labels repositioned: True")
print("Proposed model highlighted: True")
print("Figure title removed: True")
print("PNG:", png_path)
print("PDF:", pdf_path)
print("=" * 84)
