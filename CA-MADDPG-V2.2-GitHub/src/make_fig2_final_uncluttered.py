from pathlib import Path

import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np


# =========================================================
# Official held-out validation results
# =========================================================
resource_floors = np.array(
    [0.50, 0.75, 0.85, 0.90, 0.95, 0.98],
    dtype=float,
)

success_rates = np.array(
    [45.95, 47.57, 48.15, 48.50, 48.97, 49.41],
    dtype=float,
)


# =========================================================
# Output paths
# =========================================================
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
    / "Fig_2_Final_Uncluttered_Resource_Floor_Sensitivity.png"
)

pdf_path = (
    output_directory
    / "Fig_2_Final_Uncluttered_Resource_Floor_Sensitivity.pdf"
)


# =========================================================
# IEEE single-column visual settings
# =========================================================
plt.rcParams.update(
    {
        "font.family": "serif",
        "font.serif": ["DejaVu Serif"],
        "font.size": 8,
        "axes.labelsize": 9,
        "xtick.labelsize": 7.5,
        "ytick.labelsize": 7.5,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
        "axes.linewidth": 0.8,
    }
)


figure, axis = plt.subplots(
    figsize=(3.5, 2.9)
)


# =========================================================
# Main colorful sensitivity curve
# =========================================================
axis.plot(
    resource_floors,
    success_rates,
    color="#1565C0",
    linewidth=1.8,
    zorder=2,
)

point_colors = [
    "#42A5F5",
    "#26A69A",
    "#66BB6A",
    "#FFA726",
    "#AB47BC",
]

for index in range(
    len(resource_floors) - 1
):
    axis.scatter(
        resource_floors[index],
        success_rates[index],
        s=38,
        marker="o",
        facecolor=point_colors[index],
        edgecolor="black",
        linewidth=0.65,
        zorder=4,
    )


# Selected floor
axis.scatter(
    resource_floors[-1],
    success_rates[-1],
    s=115,
    marker="*",
    facecolor="#E53935",
    edgecolor="black",
    linewidth=0.85,
    zorder=5,
)


# =========================================================
# Compact success-value labels only
# The x-axis already provides each resource-floor value.
# =========================================================
label_offsets = [
    (7, 7),      # 0.50
    (-7, 9),     # 0.75
    (7, 8),      # 0.85
    (-7, -14),   # 0.90
    (-9, 9),     # 0.95
]

horizontal_alignments = [
    "left",
    "right",
    "left",
    "right",
    "right",
]

for index in range(
    len(resource_floors) - 1
):
    axis.annotate(
        f"{success_rates[index]:.2f}%",
        xy=(
            resource_floors[index],
            success_rates[index],
        ),
        xytext=label_offsets[index],
        textcoords="offset points",
        ha=horizontal_alignments[index],
        va="center",
        fontsize=7,
        color="#202020",
        bbox={
            "boxstyle": "round,pad=0.13",
            "facecolor": "white",
            "edgecolor": "0.78",
            "linewidth": 0.45,
            "alpha": 0.94,
        },
        zorder=6,
    )


# Compact selected-floor annotation
axis.annotate(
    "Selected: 0.98\n49.41%",
    xy=(
        resource_floors[-1],
        success_rates[-1],
    ),
    xytext=(-60, -26),
    textcoords="offset points",
    ha="left",
    va="center",
    fontsize=7.2,
    fontweight="bold",
    color="#C62828",
    bbox={
        "boxstyle": "round,pad=0.18",
        "facecolor": "#FFF7F7",
        "edgecolor": "#E53935",
        "linewidth": 0.65,
        "alpha": 0.97,
    },
    arrowprops={
        "arrowstyle": "->",
        "color": "#E53935",
        "linewidth": 0.75,
        "shrinkA": 2,
        "shrinkB": 4,
    },
    zorder=7,
)


# =========================================================
# Axis formatting
# =========================================================
axis.set_xlabel(
    "Resource floor"
)

axis.set_ylabel(
    "Validation success rate (%)"
)

axis.set_xlim(
    0.475,
    1.005,
)

axis.set_ylim(
    45.70,
    49.65,
)

axis.set_xticks(
    resource_floors
)

axis.set_xticklabels(
    [
        "0.50",
        "0.75",
        "0.85",
        "0.90",
        "0.95",
        "0.98",
    ]
)

# Keep crowded right-side tick labels readable.
for label in axis.get_xticklabels():
    label.set_rotation(25)
    label.set_ha("right")

axis.grid(
    linestyle=":",
    linewidth=0.55,
    alpha=0.45,
)

axis.set_axisbelow(
    True
)

# No internal title or large explanatory note.
# The manuscript caption will provide the explanation.

figure.tight_layout(
    pad=0.55
)


# =========================================================
# Save PNG and PDF
# =========================================================
figure.savefig(
    png_path,
    dpi=600,
    format="png",
    bbox_inches="tight",
    pad_inches=0.04,
)

figure.savefig(
    pdf_path,
    format="pdf",
    bbox_inches="tight",
    pad_inches=0.04,
)

plt.close(
    figure
)


print("=" * 88)
print("FINAL UNCLUTTERED FIGURE 2 CREATED")
print("=" * 88)
print("Point labels simplified: True")
print("Overlapping labels removed: True")
print("Selected floor compactly highlighted: True")
print("Internal title removed: True")
print("Large note removed: True")
print("Single-column width: 3.5 inches")
print("PNG:", png_path)
print("PDF:", pdf_path)
print("=" * 88)
