from pathlib import Path

import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np


# =========================================================
# Official held-out validation results
# =========================================================
floor_labels = [
    "0.50",
    "0.75",
    "0.85",
    "0.90",
    "0.95",
    "0.98",
]

success_rates = np.array(
    [45.95, 47.57, 48.15, 48.50, 48.97, 49.41],
    dtype=float,
)

# Evenly spaced positions prevent right-side crowding
x_positions = np.arange(len(floor_labels))


# =========================================================
# Output directory
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
    / "Fig_2_Final_Small_Text_Resource_Floor_Sensitivity.png"
)

pdf_path = (
    output_directory
    / "Fig_2_Final_Small_Text_Resource_Floor_Sensitivity.pdf"
)


# =========================================================
# Small IEEE single-column fonts
# =========================================================
plt.rcParams.update(
    {
        "font.family": "serif",
        "font.serif": ["DejaVu Serif"],
        "font.size": 6.0,
        "axes.labelsize": 7.2,
        "xtick.labelsize": 6.2,
        "ytick.labelsize": 6.2,
        "axes.linewidth": 0.75,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
    }
)


figure, axis = plt.subplots(
    figsize=(3.5, 2.4)
)


# =========================================================
# Main colorful curve
# =========================================================
axis.plot(
    x_positions,
    success_rates,
    color="#1565C0",
    linewidth=1.45,
    zorder=2,
)

marker_colors = [
    "#42A5F5",
    "#26A69A",
    "#66BB6A",
    "#FFA726",
    "#AB47BC",
]

for index in range(5):
    axis.scatter(
        x_positions[index],
        success_rates[index],
        s=24,
        marker="o",
        facecolor=marker_colors[index],
        edgecolor="black",
        linewidth=0.45,
        zorder=4,
    )


# Selected floor marker
axis.scatter(
    x_positions[-1],
    success_rates[-1],
    s=68,
    marker="*",
    facecolor="#E53935",
    edgecolor="black",
    linewidth=0.65,
    zorder=5,
)


# =========================================================
# Small point labels
# =========================================================
label_offsets = [
    (7, 7),       # 45.95
    (-9, 7),      # 47.57
    (0, -10),     # 48.15
    (0, 7),       # 48.50
    (0, -10),     # 48.97
]

horizontal_alignment = [
    "left",
    "right",
    "center",
    "center",
    "center",
]

for index in range(5):
    axis.annotate(
        f"{success_rates[index]:.2f}",
        xy=(
            x_positions[index],
            success_rates[index],
        ),
        xytext=label_offsets[index],
        textcoords="offset points",
        ha=horizontal_alignment[index],
        va="center",
        fontsize=5.2,
        color="#222222",
        zorder=6,
    )


# =========================================================
# Small top-left selected-floor annotation
# =========================================================
axis.annotate(
    "Selected floor 0.98\n49.41% success",
    xy=(
        x_positions[-1],
        success_rates[-1],
    ),
    xycoords="data",
    xytext=(0.035, 0.955),
    textcoords="axes fraction",
    ha="left",
    va="top",
    fontsize=5.3,
    color="#C62828",
    bbox={
        "boxstyle": "round,pad=0.12",
        "facecolor": "white",
        "edgecolor": "#E53935",
        "linewidth": 0.50,
    },
    arrowprops={
        "arrowstyle": "->",
        "color": "#E53935",
        "linewidth": 0.60,
        "connectionstyle": "arc3,rad=-0.12",
    },
    zorder=7,
)


# =========================================================
# Axes
# =========================================================
axis.set_xlabel(
    "Resource floor"
)

axis.set_ylabel(
    "Validation success rate (%)"
)

axis.set_xticks(
    x_positions
)

axis.set_xticklabels(
    floor_labels
)

axis.set_xlim(
    -0.35,
    5.35,
)

axis.set_ylim(
    45.70,
    50.05,
)

axis.grid(
    axis="y",
    linestyle=":",
    linewidth=0.40,
    alpha=0.50,
)

axis.grid(
    axis="x",
    linestyle=":",
    linewidth=0.25,
    alpha=0.20,
)

axis.set_axisbelow(
    True
)

# No title inside the graph
figure.tight_layout(
    pad=0.35
)


# =========================================================
# Save PNG + PDF only
# =========================================================
figure.savefig(
    png_path,
    dpi=600,
    format="png",
    bbox_inches="tight",
    pad_inches=0.03,
)

figure.savefig(
    pdf_path,
    format="pdf",
    bbox_inches="tight",
    pad_inches=0.03,
)

plt.close(
    figure
)


print("=" * 88)
print("FINAL SMALL-TEXT FIGURE 2 CREATED")
print("=" * 88)
print("Annotation font size: 5.3 pt")
print("Point-label font size: 5.2 pt")
print("Top-left annotation: True")
print("Overlap removed: True")
print("PNG:", png_path)
print("PDF:", pdf_path)
print("=" * 88)
