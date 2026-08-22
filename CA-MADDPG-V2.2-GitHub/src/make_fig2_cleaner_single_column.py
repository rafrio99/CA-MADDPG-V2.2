from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np

# -----------------------------
# Data
# -----------------------------
resource_floors = np.array([0.50, 0.75, 0.85, 0.90, 0.95, 0.98], dtype=float)
success_rates = np.array([45.95, 47.57, 48.15, 48.50, 48.97, 49.41], dtype=float)

selected_floor = 0.98
selected_success = 49.41

# -----------------------------
# Output
# -----------------------------
OUTPUT_DIR = Path("results/final_conference_package/figures_final_single_column")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

png_path = OUTPUT_DIR / "Fig_2_Clean_Single_Column_Resource_Floor_Sensitivity.png"
pdf_path = OUTPUT_DIR / "Fig_2_Clean_Single_Column_Resource_Floor_Sensitivity.pdf"

# -----------------------------
# Plot
# -----------------------------
plt.rcParams.update({
    "font.size": 9,
    "axes.labelsize": 11,
    "axes.titlesize": 11,
    "xtick.labelsize": 8,
    "ytick.labelsize": 8,
})

fig, ax = plt.subplots(figsize=(3.5, 3.0), dpi=300)

# line
ax.plot(
    resource_floors,
    success_rates,
    color="#1f77b4",
    linewidth=2.0,
    marker="o",
    markersize=5.5,
    markerfacecolor="#4da3ff",
    markeredgecolor="#1a1a1a",
    zorder=2,
)

# highlight selected floor
ax.scatter(
    [selected_floor],
    [selected_success],
    s=150,
    marker="*",
    color="#ff4d4d",
    edgecolors="black",
    linewidths=0.9,
    zorder=4,
)

# compact point labels with manual offsets
label_offsets = {
    0.50: (6, 6),
    0.75: (6, 6),
    0.85: (6, 6),
    0.90: (6, -14),
    0.95: (-42, 6),
    0.98: (-58, -20),
}

for x, y in zip(resource_floors, success_rates):
    dx, dy = label_offsets[float(x)]
    ax.annotate(
        f"{x:.2f}, {y:.2f}%",
        xy=(x, y),
        xytext=(dx, dy),
        textcoords="offset points",
        fontsize=7.5,
        color="black",
        bbox=dict(boxstyle="round,pad=0.18", fc="white", ec="#bfbfbf", alpha=0.92),
        zorder=5,
    )

# small selected-floor callout
ax.annotate(
    "Selected floor: 0.98",
    xy=(selected_floor, selected_success),
    xytext=(-78, -4),
    textcoords="offset points",
    fontsize=8,
    color="#c62828",
    bbox=dict(boxstyle="round,pad=0.22", fc="#fff5f5", ec="#d32f2f", alpha=0.95),
    arrowprops=dict(arrowstyle="->", color="#d32f2f", lw=0.9),
    zorder=6,
)

# axis formatting
ax.set_xlabel("Resource floor")
ax.set_ylabel("Validation success rate (%)")

ax.set_xlim(0.48, 1.01)
ax.set_ylim(45.7, 49.6)

ax.set_xticks([0.50, 0.75, 0.85, 0.90, 0.95, 0.98])
ax.set_xticklabels(["0.50", "0.75", "0.85", "0.90", "0.95", "0.98"])

ax.grid(True, linestyle="--", alpha=0.35, linewidth=0.6)
ax.set_axisbelow(True)

for spine in ax.spines.values():
    spine.set_linewidth(0.8)

plt.tight_layout()
fig.savefig(png_path, format="png", dpi=600, bbox_inches="tight")
fig.savefig(pdf_path, format="pdf", bbox_inches="tight")
plt.close(fig)

print("=" * 82)
print("CLEANER FIGURE 2 CREATED")
print("=" * 82)
print("PNG:", png_path)
print("PDF:", pdf_path)
print("Single-column width: 3.5 inches")
print("Label overlap reduced: True")
print("Selected floor highlighted: True")
print("=" * 82)
