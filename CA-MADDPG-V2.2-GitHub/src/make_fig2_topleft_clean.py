from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np

# -----------------------------
# Data
# -----------------------------
floors = [0.50, 0.75, 0.85, 0.90, 0.95, 0.98]
success = [45.95, 47.57, 48.15, 48.50, 48.97, 49.41]

selected_floor = 0.98
selected_success = 49.41

# Use evenly spaced x positions to avoid crowding
x = np.arange(len(floors))

# -----------------------------
# Output paths
# -----------------------------
out_dir = Path("results/final_conference_package/figures_final_single_column")
out_dir.mkdir(parents=True, exist_ok=True)

png_path = out_dir / "Fig_2_Final_TopLeft_Resource_Floor_Sensitivity.png"
pdf_path = out_dir / "Fig_2_Final_TopLeft_Resource_Floor_Sensitivity.pdf"

# -----------------------------
# Figure
# -----------------------------
plt.rcParams.update({
    "font.family": "serif",
    "font.size": 11
})

fig, ax = plt.subplots(figsize=(3.5, 3.0), dpi=300)

# Main line
ax.plot(x, success, linewidth=2.6)

# Colorful markers for first five points
marker_colors = ['#4dabf7', '#2bb3b1', '#74c365', '#ffb020', '#b34cc2']
for i in range(len(floors) - 1):
    ax.scatter(
        x[i], success[i],
        s=110,
        color=marker_colors[i],
        edgecolor='black',
        linewidth=1.0,
        zorder=3
    )

# Selected point: star
sel_idx = floors.index(selected_floor)
ax.scatter(
    x[sel_idx], success[sel_idx],
    s=420,
    marker='*',
    color='#ff4d4d',
    edgecolor='black',
    linewidth=1.2,
    zorder=5
)

# Compact labels for non-selected points only
label_offsets = {
    0: (-0.10, 0.18),  # 0.50
    1: (-0.12, 0.22),  # 0.75
    2: (-0.10, -0.22), # 0.85
    3: (-0.05, 0.22),  # 0.90
    4: (-0.06, -0.22), # 0.95
}

for i in range(len(floors) - 1):
    dx, dy = label_offsets[i]
    ax.text(
        x[i] + dx, success[i] + dy,
        f"{success[i]:.2f}%",
        fontsize=10,
        ha='center',
        va='center'
    )

# Top-left annotation box with arrow to selected point
annotation_text = "Selected floor = 0.98\nSuccess rate = 49.41%"
ax.annotate(
    annotation_text,
    xy=(x[sel_idx], success[sel_idx]),
    xycoords='data',
    xytext=(0.05, 0.94),
    textcoords='axes fraction',
    ha='left',
    va='top',
    fontsize=10.5,
    color='#d62f2f',
    fontweight='bold',
    bbox=dict(boxstyle='round,pad=0.28', fc='white', ec='#d62f2f', lw=1.2),
    arrowprops=dict(arrowstyle='->', color='#d62f2f', lw=1.5,
                    connectionstyle='arc3,rad=-0.15')
)

# Axes formatting
ax.set_xlabel("Resource floor", fontsize=13)
ax.set_ylabel("Validation success rate (%)", fontsize=13)

ax.set_xticks(x)
ax.set_xticklabels([f"{v:.2f}" for v in floors], rotation=0)

ax.set_ylim(45.7, 50.0)
ax.set_xlim(-0.3, len(floors) - 0.8)

ax.grid(True, linestyle='--', alpha=0.35)

for spine in ax.spines.values():
    spine.set_linewidth(1.0)

plt.tight_layout()

# Save only PNG + PDF
fig.savefig(png_path, bbox_inches="tight")
fig.savefig(pdf_path, bbox_inches="tight")
plt.close(fig)

print("=" * 88)
print("FINAL CLEAN TOP-LEFT FIGURE 2 CREATED")
print("=" * 88)
print("Top-left annotation box: True")
print("Arrow to selected point: True")
print("Evenly spaced x-axis: True")
print("Point crowding fixed: True")
print("PNG:", png_path)
print("PDF:", pdf_path)
print("=" * 88)
