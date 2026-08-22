from pathlib import Path

import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np


# =========================================================
# Official held-out validation results
# =========================================================
resource_floor_labels = [
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

# Even spacing prevents crowding among 0.90, 0.95, and 0.98
x_positions = np.arange(len(resource_floor_labels))


# =========================================================
# Output
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
    / "Fig_2_Final_Perfect_Resource_Floor_Sensitivity.png"
)

pdf_path = (
    output_directory
    / "Fig_2_Final_Perfect_Resource_Floor_Sensitivity.pdf"
)


# =========================================================
# Single-column IEEE style
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
    figsize=(3.5, 2.85)
)


# =========================================================
# Main sensitivity curve
# =========================================================
axis.plot(
    x_positions,
    success_rates,
    color="#1565C0",
    linewidth=1.9,
    zorder=2,
)

marker_colors = [
    "#42A5F5",
    "#26A69A",
    "#66BB6A",
    "#FFA726",
    "#AB47BC",
]

for index in range(len(x_positions) - 1):
    axis.scatter(
        x_positions[index],
        success_rates[index],
        s=42,
        marker="o",
        facecolor=marker_colors[index],
        edgecolor="black",
        linewidth=0.7,
        zorder=4,
    )


# Final selected point
axis.scatter(
    x_positions[-1],
    success_rates[-1],
    s=125,
    marker="*",
    facecolor="#E53935",
    edgecolor="black",
    linewidth=0.9,
    zorder=5,
)


# =========================================================
# Compact value labels
# Alternate labels above and below to guarantee separation
# =========================================================
label_offsets = [
    (0, 9),
    (0, 9),
    (0, -15),
    (0, 9),
    (0, -15),
]

for index in range(len(x_positions) - 1):
    axis.annotate(
        f"{success_rates[index]:.2f}%",
        xy=(
            x_positions[index],
            success_rates[index],
        ),
        xytext=label_offsets[index],
        textcoords="offset points",
        ha="center",
        va="center",
        fontsize=7.2,
        color="#202020",
        zorder=6,
    )


# Selected point label: compact and inside the plot
axis.annotate(
    "49.41%\nSelected",
    xy=(
        x_positions[-1],
        success_rates[-1],
    ),
    xytext=(-10, -28),
    textcoords="offset points",
    ha="right",
    va="center",
    fontsize=7.2,
    fontweight="bold",
    color="#C62828",
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
    resource_floor_labels
)

axis.set_xlim(
    -0.35,
    len(x_positions) - 0.65,
)

axis.set_ylim(
    45.72,
    49.68,
)

axis.grid(
    axis="y",
    linestyle=":",
    linewidth=0.55,
    alpha=0.50,
)

axis.grid(
    axis="x",
    linestyle=":",
    linewidth=0.40,
    alpha=0.25,
)

axis.set_axisbelow(
    True
)

# No internal title and no explanatory text.
# The manuscript caption will explain the figure.

figure.tight_layout(
    pad=0.55
)


# =========================================================
# Save only PNG and PDF
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
print("FINAL PERFECT FIGURE 2 CREATED")
print("=" * 88)
print("Evenly spaced resource-floor positions: True")
print("Crowded x-axis fixed: True")
print("Overlapping labels removed: True")
print("Selected floor highlighted: True")
print("Colorful single-column figure: True")
print("PNG:", png_path)
print("PDF:", pdf_path)
print("=" * 88)
