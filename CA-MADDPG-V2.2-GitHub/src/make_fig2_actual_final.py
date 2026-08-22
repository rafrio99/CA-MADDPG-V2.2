from pathlib import Path

import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np


# Official held-out validation results
floor_labels = ["0.50", "0.75", "0.85", "0.90", "0.95", "0.98"]

success = np.array(
    [45.95, 47.57, 48.15, 48.50, 48.97, 49.41],
    dtype=float,
)

x = np.arange(len(floor_labels))


# Output
out_dir = Path(
    "results/final_conference_package/"
    "figures_final_single_column"
)

out_dir.mkdir(parents=True, exist_ok=True)

png_path = out_dir / "Fig_2_Actual_Final_Resource_Floor_Sensitivity.png"
pdf_path = out_dir / "Fig_2_Actual_Final_Resource_Floor_Sensitivity.pdf"


# IEEE single-column settings
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


fig, ax = plt.subplots(figsize=(3.5, 2.9))


# Main curve
ax.plot(
    x,
    success,
    color="#1565C0",
    linewidth=1.9,
    zorder=2,
)


# Colorful markers
colors = [
    "#42A5F5",
    "#26A69A",
    "#66BB6A",
    "#FFA726",
    "#AB47BC",
]

for i in range(5):
    ax.scatter(
        x[i],
        success[i],
        s=42,
        marker="o",
        facecolor=colors[i],
        edgecolor="black",
        linewidth=0.7,
        zorder=4,
    )


# Selected point
ax.scatter(
    x[-1],
    success[-1],
    s=125,
    marker="*",
    facecolor="#E53935",
    edgecolor="black",
    linewidth=0.9,
    zorder=5,
)


# Small plain value labels, deliberately alternated
label_dy = [0.13, 0.16, -0.18, 0.16, -0.18]

for i in range(5):
    va = "bottom" if label_dy[i] > 0 else "top"

    ax.text(
        x[i],
        success[i] + label_dy[i],
        f"{success[i]:.2f}%",
        ha="center",
        va=va,
        fontsize=7,
        color="#202020",
        zorder=6,
    )


# Compact selected annotation in free top-right space
ax.annotate(
    "Selected floor = 0.98\nSuccess = 49.41%",
    xy=(x[-1], success[-1]),
    xytext=(4.05, 49.78),
    textcoords="data",
    ha="left",
    va="top",
    fontsize=7.1,
    fontweight="bold",
    color="#C62828",
    arrowprops={
        "arrowstyle": "->",
        "color": "#E53935",
        "linewidth": 0.8,
        "shrinkA": 2,
        "shrinkB": 4,
    },
    zorder=7,
)


# Axes
ax.set_xlabel("Resource floor")
ax.set_ylabel("Validation success rate (%)")

ax.set_xticks(x)
ax.set_xticklabels(floor_labels)

ax.set_xlim(-0.35, 5.35)
ax.set_ylim(45.70, 50.00)

ax.grid(
    axis="y",
    linestyle=":",
    linewidth=0.55,
    alpha=0.50,
)

ax.grid(
    axis="x",
    linestyle=":",
    linewidth=0.40,
    alpha=0.22,
)

ax.set_axisbelow(True)

# No title inside the figure.
# Caption will provide the title.

fig.tight_layout(pad=0.55)


# Save PNG + PDF
fig.savefig(
    png_path,
    dpi=600,
    format="png",
    bbox_inches="tight",
    pad_inches=0.04,
)

fig.savefig(
    pdf_path,
    format="pdf",
    bbox_inches="tight",
    pad_inches=0.04,
)

plt.close(fig)


print("=" * 88)
print("ACTUAL FINAL FIGURE 2 CREATED")
print("=" * 88)
print("Label boxes removed: True")
print("Large annotation box removed: True")
print("Evenly spaced x-axis: True")
print("Colorful single-column figure: True")
print("PNG:", png_path)
print("PDF:", pdf_path)
print("=" * 88)
