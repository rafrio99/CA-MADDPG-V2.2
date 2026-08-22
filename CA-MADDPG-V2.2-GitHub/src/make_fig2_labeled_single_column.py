from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np

# -----------------------------
# Output paths
# -----------------------------
OUTPUT_DIR = Path("results/final_conference_package/figures_final_single_column")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

jpg_path = OUTPUT_DIR / "Fig_2_Final_Labeled_Resource_Floor_Sensitivity.jpg"
pdf_path = OUTPUT_DIR / "Fig_2_Final_Labeled_Resource_Floor_Sensitivity.pdf"

# -----------------------------
# Data (official held-out validation results)
# -----------------------------
resource_floor = np.array([0.50, 0.75, 0.85, 0.90, 0.95, 0.98])
success_rate = np.array([45.946667, 47.566667, 48.153333, 48.500000, 48.973333, 49.413333])

selected_floor = 0.98
selected_success = 49.413333

# -----------------------------
# Figure style
# -----------------------------
plt.rcParams.update({
    "font.size": 10,
    "axes.labelsize": 10,
    "xtick.labelsize": 9,
    "ytick.labelsize": 9,
})

fig, ax = plt.subplots(figsize=(3.5, 3.0), dpi=300)

# Base line
ax.plot(
    resource_floor,
    success_rate,
    color="#1f77b4",
    linewidth=2.0,
    marker="o",
    markersize=5.5,
    markerfacecolor="#1f77b4",
    markeredgecolor="black",
    markeredgewidth=0.6,
    zorder=2,
)

# Highlight selected floor
ax.scatter(
    [selected_floor],
    [selected_success],
    s=120,
    marker="*",
    color="#d62728",
    edgecolors="black",
    linewidths=0.8,
    zorder=4,
)

# Point labels
label_offsets = {
    0.50: (0.010, 0.10),
    0.75: (0.010, 0.08),
    0.85: (0.010, 0.10),
    0.90: (0.010, -0.12),
    0.95: (-0.045, 0.10),
    0.98: (-0.075, -0.18),
}

for x, y in zip(resource_floor, success_rate):
    dx, dy = label_offsets[round(float(x), 2)]
    ax.text(
        x + dx,
        y + dy,
        f"{x:.2f}, {y:.2f}%",
        fontsize=8,
        ha="left",
        va="center",
        bbox=dict(boxstyle="round,pad=0.18", fc="white", ec="lightgray", alpha=0.95),
        zorder=5,
    )

# Selected annotation
ax.annotate(
    "Selected floor = 0.98",
    xy=(selected_floor, selected_success),
    xytext=(0.73, 49.15),
    textcoords="data",
    fontsize=8.5,
    color="#d62728",
    arrowprops=dict(arrowstyle="->", color="#d62728", lw=1.0),
    bbox=dict(boxstyle="round,pad=0.22", fc="#fff5f5", ec="#d62728", alpha=0.95),
    zorder=6,
)

# Axes
ax.set_xlabel("Resource floor")
ax.set_ylabel("Validation success rate (%)")

ax.set_xlim(0.48, 1.01)
ax.set_ylim(45.7, 49.6)

ax.set_xticks([0.50, 0.75, 0.85, 0.90, 0.95, 0.98])
ax.set_yticks([46.0, 46.5, 47.0, 47.5, 48.0, 48.5, 49.0, 49.5])

ax.grid(True, linestyle="--", linewidth=0.5, alpha=0.4)

for spine in ax.spines.values():
    spine.set_linewidth(0.8)

fig.tight_layout()

# Save
fig.savefig(jpg_path, format="jpg", dpi=300, bbox_inches="tight")
fig.savefig(pdf_path, format="pdf", bbox_inches="tight")
plt.close(fig)

print("=" * 82)
print("FINAL LABELED FIGURE 2 CREATED")
print("=" * 82)
print("JPG:", jpg_path)
print("PDF:", pdf_path)
print("Single-column width: 3.5 inches")
print("Point labels added: True")
print("Selected floor highlighted: True")
print("=" * 82)
