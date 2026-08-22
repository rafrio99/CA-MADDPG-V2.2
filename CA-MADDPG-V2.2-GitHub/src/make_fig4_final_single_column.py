from pathlib import Path
import matplotlib.pyplot as plt

# =========================================================
# Official manuscript values (final conference-ready)
# x-axis  = Mean Latency (ms)
# y-axis  = Mean Energy (mJ)
# bubble  = Success Rate (%)
# =========================================================
methods = [
    {
        "name": "Latency heuristic",
        "latency": 1613.159708,
        "energy": 37113.174659,
        "success": 50.83,
        "reward": -206.420659,
    },
    {
        "name": "MADDPG three-seed mean",
        "latency": 1785.912933,
        "energy": 43354.071118,
        "success": 48.43,
        "reward": -224.221678,
    },
    {
        "name": "CA-MADDPG-V2.2 three-seed mean",
        "latency": 1684.140031,
        "energy": 38666.154115,
        "success": 49.36,
        "reward": -216.411850,
    },
    {
        "name": "Random allocation",
        "latency": 2601.074407,
        "energy": 86513.626403,
        "success": 38.49,
        "reward": -268.098128,
    },
]

# Manual label offsets to avoid overlap
label_offsets = {
    "Latency heuristic": (40, -1800),
    "MADDPG three-seed mean": (55, 2200),
    "CA-MADDPG-V2.2 three-seed mean": (-210, -2600),
    "Random allocation": (-230, 1800),
}

# Marker styles
marker_map = {
    "Latency heuristic": "o",
    "MADDPG three-seed mean": "s",
    "CA-MADDPG-V2.2 three-seed mean": "D",
    "Random allocation": "^",
}

# Output directory
output_dir = Path("results/final_conference_package/figures_final_single_column")
output_dir.mkdir(parents=True, exist_ok=True)

png_path = output_dir / "Fig_4_Final_Single_Column_Latency_Energy_Tradeoff.png"
pdf_path = output_dir / "Fig_4_Final_Single_Column_Latency_Energy_Tradeoff.pdf"

# ---------------------------------------------------------
# Plot
# ---------------------------------------------------------
plt.rcParams.update({
    "font.size": 9,
    "axes.labelsize": 10,
    "axes.titlesize": 10,
    "legend.fontsize": 8,
})

fig, ax = plt.subplots(figsize=(3.5, 3.9), dpi=300)

for item in methods:
    x = item["latency"]
    y = item["energy"]
    s = item["success"]
    reward = item["reward"]
    name = item["name"]

    bubble_size = 35 + (s - 35.0) * 9.0

    ax.scatter(
        x,
        y,
        s=bubble_size,
        marker=marker_map[name],
        edgecolors="black",
        linewidths=0.8,
        alpha=0.9,
        label=name,
        zorder=3,
    )

    dx, dy = label_offsets[name]
    label_text = (
        f"{name}\n"
        f"Success={s:.2f}%\n"
        f"Reward={reward:.2f}"
    )

    ax.annotate(
        label_text,
        xy=(x, y),
        xytext=(x + dx, y + dy),
        textcoords="data",
        fontsize=7.2,
        bbox=dict(boxstyle="round,pad=0.25", fc="white", ec="gray", lw=0.7),
        arrowprops=dict(arrowstyle="-", lw=0.7, color="gray"),
        zorder=4,
    )

# Highlight desirable region
ax.text(
    0.03,
    0.97,
    "Better region:\nlower latency,\nlower energy",
    transform=ax.transAxes,
    va="top",
    ha="left",
    fontsize=7.5,
    bbox=dict(boxstyle="round,pad=0.25", fc="white", ec="gray", lw=0.6),
)

ax.set_xlabel("Mean latency (ms)")
ax.set_ylabel("Mean energy (mJ)")
ax.set_title("Fig. 4. Latency-Energy Tradeoff")
ax.grid(True, linestyle="--", linewidth=0.5, alpha=0.5, zorder=0)

# Limits with padding
latencies = [m["latency"] for m in methods]
energies = [m["energy"] for m in methods]
ax.set_xlim(min(latencies) - 120, max(latencies) + 180)
ax.set_ylim(min(energies) - 5000, max(energies) + 6500)

# Compact legend
leg = ax.legend(
    loc="lower right",
    frameon=True,
    borderpad=0.4,
    handletextpad=0.5,
)
leg.get_frame().set_linewidth(0.7)

plt.tight_layout()
plt.savefig(png_path, bbox_inches="tight")
plt.savefig(pdf_path, bbox_inches="tight")
plt.close()

print("=" * 78)
print("FIGURE 4 FINAL SINGLE-COLUMN VERSION CREATED")
print("=" * 78)
print("PNG:", png_path)
print("PDF:", pdf_path)
print("Figure width: 3.5 inches")
print("Overlap fixed: True")
print("Single-column ready: True")
print("=" * 78)
