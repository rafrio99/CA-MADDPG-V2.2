from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


PACKAGE_DIRECTORY = Path(
    "results/final_conference_package"
)

TABLE_DIRECTORY = (
    PACKAGE_DIRECTORY / "tables"
)

OUTPUT_DIRECTORY = (
    PACKAGE_DIRECTORY
    / "figures_final_single_column"
)

OUTPUT_DIRECTORY.mkdir(
    parents=True,
    exist_ok=True,
)

PER_SEED_PATH = (
    TABLE_DIRECTORY
    / "Table_VI_Per_Seed_Robustness.csv"
)

OVERALL_PATH = (
    TABLE_DIRECTORY
    / "Table_II_Overall_Performance.csv"
)

for required_path in [
    PER_SEED_PATH,
    OVERALL_PATH,
]:
    if not required_path.exists():
        raise FileNotFoundError(
            f"Missing required file: {required_path}"
        )


per_seed_df = pd.read_csv(
    PER_SEED_PATH
)

overall_df = pd.read_csv(
    OVERALL_PATH
)


required_seed_columns = {
    "Training Seed",
    "Success Rate (%)",
    "Success Std (%)",
}

required_overall_columns = {
    "Method",
    "Success Rate (%)",
}

if not required_seed_columns.issubset(
    per_seed_df.columns
):
    raise RuntimeError(
        "Unexpected columns in the per-seed table."
    )

if not required_overall_columns.issubset(
    overall_df.columns
):
    raise RuntimeError(
        "Unexpected columns in the overall table."
    )


per_seed_df = (
    per_seed_df.sort_values(
        "Training Seed"
    )
    .reset_index(drop=True)
)

maddpg_rows = overall_df[
    overall_df["Method"] == "MADDPG"
]

if len(maddpg_rows) != 1:
    raise RuntimeError(
        "Exactly one MADDPG row was expected."
    )


maddpg_success = float(
    maddpg_rows.iloc[0][
        "Success Rate (%)"
    ]
)

success_values = per_seed_df[
    "Success Rate (%)"
].to_numpy(
    dtype=float
)

success_errors = per_seed_df[
    "Success Std (%)"
].to_numpy(
    dtype=float
)

seed_labels = [
    f"Seed {int(seed)}"
    for seed in per_seed_df[
        "Training Seed"
    ]
]

x_positions = np.arange(
    len(seed_labels)
)


plt.rcParams.update(
    {
        "font.family": "serif",
        "font.serif": [
            "DejaVu Serif",
        ],
        "font.size": 8.5,
        "axes.labelsize": 9,
        "xtick.labelsize": 8,
        "ytick.labelsize": 8,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
        "axes.linewidth": 0.8,
    }
)


# IEEE single-column width is approximately 3.5 inches.
figure, axis = plt.subplots(
    figsize=(3.5, 2.55)
)


axis.errorbar(
    x_positions,
    success_values,
    yerr=success_errors,
    fmt="o",
    markersize=6,
    markerfacecolor="white",
    markeredgecolor="black",
    markeredgewidth=1.1,
    color="black",
    capsize=4,
    elinewidth=1.0,
    capthick=1.0,
    linewidth=0,
    zorder=4,
)


axis.axhline(
    maddpg_success,
    color="black",
    linestyle="--",
    linewidth=1.0,
    zorder=2,
)


# Direct baseline label replaces the legend and
# prevents overlap in a single-column layout.
axis.text(
    0.08,
    maddpg_success + 0.18,
    (
        f"MADDPG mean = "
        f"{maddpg_success:.2f}%"
    ),
    ha="left",
    va="bottom",
    fontsize=7.5,
    bbox={
        "facecolor": "white",
        "edgecolor": "none",
        "pad": 1.0,
        "alpha": 0.9,
    },
    zorder=5,
)


for index, success_value in enumerate(
    success_values
):
    axis.annotate(
        f"{success_value:.2f}%",
        (
            x_positions[index],
            success_value,
        ),
        xytext=(0, 7),
        textcoords="offset points",
        ha="center",
        va="bottom",
        fontsize=8,
        fontweight="bold",
        zorder=6,
    )


axis.set_xticks(
    x_positions
)

axis.set_xticklabels(
    seed_labels
)

axis.set_ylabel(
    "Success rate (%)"
)


lower_limit = float(
    np.min(
        success_values
        - success_errors
    )
    - 0.7
)

upper_limit = float(
    np.max(
        success_values
        + success_errors
    )
    + 0.9
)

axis.set_ylim(
    lower_limit,
    upper_limit,
)

axis.set_xlim(
    -0.45,
    len(seed_labels) - 0.55,
)


axis.grid(
    axis="y",
    linestyle=":",
    linewidth=0.55,
    alpha=0.55,
)

axis.set_axisbelow(
    True
)


figure.tight_layout(
    pad=0.45
)


png_path = (
    OUTPUT_DIRECTORY
    / (
        "Fig_3_Final_Single_Column_"
        "Per_Seed_Robustness.png"
    )
)

pdf_path = (
    OUTPUT_DIRECTORY
    / (
        "Fig_3_Final_Single_Column_"
        "Per_Seed_Robustness.pdf"
    )
)


figure.savefig(
    png_path,
    dpi=600,
    bbox_inches="tight",
    pad_inches=0.03,
)

figure.savefig(
    pdf_path,
    bbox_inches="tight",
    pad_inches=0.03,
)

plt.close(
    figure
)


print("=" * 82)
print("FIGURE 3 SINGLE-COLUMN VERSION CREATED")
print("=" * 82)
print("Figure width: 3.5 inches")
print("Legend removed: True")
print(
    "Direct MADDPG baseline label:",
    f"{maddpg_success:.2f}%",
)
print("PNG:", png_path)
print("PDF:", pdf_path)
print("Previous figures were not modified.")
print("=" * 82)
print(
    "Figure 3 finalisation completed successfully."
)
print("=" * 82)
