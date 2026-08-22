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
    PACKAGE_DIRECTORY / "figures_revised"
)

OUTPUT_DIRECTORY.mkdir(
    parents=True,
    exist_ok=True,
)


OVERALL_PATH = (
    TABLE_DIRECTORY
    / "Table_II_Overall_Performance.csv"
)

SENSITIVITY_PATH = (
    TABLE_DIRECTORY
    / "Table_V_Resource_Sensitivity.csv"
)

PER_SEED_PATH = (
    TABLE_DIRECTORY
    / "Table_VI_Per_Seed_Robustness.csv"
)


for required_path in [
    OVERALL_PATH,
    SENSITIVITY_PATH,
    PER_SEED_PATH,
]:
    if not required_path.exists():
        raise FileNotFoundError(
            f"Missing required table: {required_path}"
        )


overall_df = pd.read_csv(
    OVERALL_PATH
)

sensitivity_df = pd.read_csv(
    SENSITIVITY_PATH
)

per_seed_df = pd.read_csv(
    PER_SEED_PATH
)


required_overall_columns = {
    "Method",
    "Mean Reward",
    "Reward Std",
    "Success Rate (%)",
    "Success Std (%)",
    "Latency (ms)",
    "Energy (mJ)",
}

required_sensitivity_columns = {
    "Resource Floor",
    "Mean Reward",
    "Success Rate (%)",
    "Success Std (%)",
    "Latency (ms)",
    "Energy (mJ)",
}

required_seed_columns = {
    "Training Seed",
    "Success Rate (%)",
    "Success Std (%)",
    "Latency (ms)",
    "Energy (mJ)",
}

if not required_overall_columns.issubset(
    overall_df.columns
):
    raise RuntimeError(
        "Overall-performance table has "
        "unexpected columns."
    )

if not required_sensitivity_columns.issubset(
    sensitivity_df.columns
):
    raise RuntimeError(
        "Sensitivity table has "
        "unexpected columns."
    )

if not required_seed_columns.issubset(
    per_seed_df.columns
):
    raise RuntimeError(
        "Per-seed table has "
        "unexpected columns."
    )


plt.rcParams.update(
    {
        "font.family": "serif",
        "font.serif": [
            "DejaVu Serif",
        ],
        "font.size": 8,
        "axes.labelsize": 8,
        "xtick.labelsize": 7,
        "ytick.labelsize": 7,
        "legend.fontsize": 7,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
        "axes.linewidth": 0.8,
    }
)


def save_figure(
    figure,
    filename,
):
    figure.tight_layout()

    png_path = (
        OUTPUT_DIRECTORY
        / f"{filename}.png"
    )

    pdf_path = (
        OUTPUT_DIRECTORY
        / f"{filename}.pdf"
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

    plt.close(figure)

    print("Saved:", png_path)
    print("Saved:", pdf_path)


# =========================================================
# Revised Figure 1:
# Overall success-rate comparison
# =========================================================

method_order = [
    "Latency heuristic",
    "CA-MADDPG-V2.2",
    "MADDPG",
    "CA-MADDPG-V2",
    "Random allocation",
]

missing_methods = [
    method
    for method in method_order
    if method not in set(
        overall_df["Method"]
    )
]

if missing_methods:
    raise RuntimeError(
        "Missing methods in overall table: "
        f"{missing_methods}"
    )

overall_ordered = (
    overall_df.set_index("Method")
    .loc[method_order]
    .reset_index()
)

success_values = overall_ordered[
    "Success Rate (%)"
].to_numpy(
    dtype=float
)

success_errors = overall_ordered[
    "Success Std (%)"
].to_numpy(
    dtype=float
)

figure, axis = plt.subplots(
    figsize=(7.16, 3.15)
)

bar_faces = [
    "0.82",
    "0.30",
    "0.68",
    "0.72",
    "0.88",
]

bar_hatches = [
    "//",
    "xx",
    "..",
    "\\\\",
    "--",
]

for index, method in enumerate(
    method_order
):
    proposed = (
        method
        == "CA-MADDPG-V2.2"
    )

    axis.bar(
        index,
        success_values[index],
        yerr=success_errors[index],
        capsize=3,
        width=0.67,
        color=bar_faces[index],
        edgecolor="black",
        linewidth=(
            1.6 if proposed else 0.8
        ),
        hatch=bar_hatches[index],
        error_kw={
            "elinewidth": 0.8,
            "capthick": 0.8,
        },
    )

    axis.text(
        index,
        (
            success_values[index]
            + success_errors[index]
            + 0.45
        ),
        f"{success_values[index]:.2f}%",
        ha="center",
        va="bottom",
        fontsize=7,
        fontweight=(
            "bold" if proposed else "normal"
        ),
    )

axis.set_ylabel(
    "Success rate (%)"
)

axis.set_xticks(
    np.arange(
        len(method_order)
    )
)

axis.set_xticklabels(
    method_order,
    rotation=12,
    ha="right",
)

for label in axis.get_xticklabels():
    if label.get_text() == (
        "CA-MADDPG-V2.2"
    ):
        label.set_fontweight(
            "bold"
        )

axis.set_ylim(
    0,
    float(
        np.max(
            success_values
            + success_errors
        )
        + 5.0
    ),
)

axis.grid(
    axis="y",
    linestyle=":",
    linewidth=0.6,
    alpha=0.55,
)

axis.set_axisbelow(
    True
)

save_figure(
    figure,
    "Fig_1_Revised_Success_Rate_Comparison",
)


# =========================================================
# Revised Figure 2:
# Resource-floor sensitivity
# =========================================================

sensitivity_sorted = (
    sensitivity_df.sort_values(
        "Resource Floor"
    )
    .reset_index(drop=True)
)

selected_rows = sensitivity_sorted[
    np.isclose(
        sensitivity_sorted[
            "Resource Floor"
        ],
        0.98,
    )
]

if selected_rows.empty:
    selected_row = (
        sensitivity_sorted.loc[
            sensitivity_sorted[
                "Success Rate (%)"
            ].idxmax()
        ]
    )
else:
    selected_row = (
        selected_rows.iloc[0]
    )

selected_floor = float(
    selected_row[
        "Resource Floor"
    ]
)

selected_success = float(
    selected_row[
        "Success Rate (%)"
    ]
)

figure, axis = plt.subplots(
    figsize=(3.5, 2.65)
)

axis.plot(
    sensitivity_sorted[
        "Resource Floor"
    ],
    sensitivity_sorted[
        "Success Rate (%)"
    ],
    marker="o",
    markersize=4.5,
    linewidth=1.2,
)

axis.axvline(
    selected_floor,
    linestyle="--",
    linewidth=0.9,
)

axis.scatter(
    [selected_floor],
    [selected_success],
    marker="*",
    s=85,
    edgecolor="black",
    linewidth=0.7,
    zorder=5,
)

axis.annotate(
    (
        f"Selected floor = "
        f"{selected_floor:.2f}\n"
        f"Success = "
        f"{selected_success:.2f}%"
    ),
    xy=(
        selected_floor,
        selected_success,
    ),
    xytext=(-68, -36),
    textcoords="offset points",
    fontsize=7,
    arrowprops={
        "arrowstyle": "->",
        "linewidth": 0.7,
    },
)

axis.set_xlabel(
    "Resource floor"
)

axis.set_ylabel(
    "Validation success rate (%)"
)

axis.set_xlim(
    0.48,
    1.005,
)

minimum_success = float(
    sensitivity_sorted[
        "Success Rate (%)"
    ].min()
)

maximum_success = float(
    sensitivity_sorted[
        "Success Rate (%)"
    ].max()
)

axis.set_ylim(
    minimum_success - 0.6,
    maximum_success + 0.65,
)

axis.grid(
    linestyle=":",
    linewidth=0.6,
    alpha=0.55,
)

axis.set_axisbelow(
    True
)

save_figure(
    figure,
    "Fig_2_Revised_Resource_Floor_Sensitivity",
)


# =========================================================
# Revised Figure 3:
# Per-seed robustness point plot
# =========================================================

per_seed_sorted = (
    per_seed_df.sort_values(
        "Training Seed"
    )
    .reset_index(drop=True)
)

seed_values = per_seed_sorted[
    "Success Rate (%)"
].to_numpy(
    dtype=float
)

seed_errors = per_seed_sorted[
    "Success Std (%)"
].to_numpy(
    dtype=float
)

seed_labels = [
    f"Seed {int(seed)}"
    for seed in per_seed_sorted[
        "Training Seed"
    ]
]

maddpg_rows = overall_df[
    overall_df["Method"]
    == "MADDPG"
]

if maddpg_rows.empty:
    raise RuntimeError(
        "MADDPG row is missing."
    )

maddpg_success = float(
    maddpg_rows.iloc[0][
        "Success Rate (%)"
    ]
)

figure, axis = plt.subplots(
    figsize=(3.5, 2.65)
)

x_positions = np.arange(
    len(seed_labels)
)

axis.errorbar(
    x_positions,
    seed_values,
    yerr=seed_errors,
    fmt="o",
    markersize=5,
    capsize=4,
    elinewidth=0.9,
    capthick=0.9,
    linewidth=0,
    label="CA-MADDPG-V2.2",
)

axis.axhline(
    maddpg_success,
    linestyle="--",
    linewidth=1.0,
    label=(
        f"MADDPG mean "
        f"({maddpg_success:.2f}%)"
    ),
)

for index, value in enumerate(
    seed_values
):
    axis.annotate(
        f"{value:.2f}%",
        (
            x_positions[index],
            value,
        ),
        xytext=(0, 7),
        textcoords="offset points",
        ha="center",
        va="bottom",
        fontsize=7,
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
        seed_values
        - seed_errors
    )
    - 1.2
)

upper_limit = float(
    np.max(
        seed_values
        + seed_errors
    )
    + 1.8
)

axis.set_ylim(
    lower_limit,
    upper_limit,
)

axis.grid(
    axis="y",
    linestyle=":",
    linewidth=0.6,
    alpha=0.55,
)

axis.legend(
    loc="lower right",
    frameon=True,
)

axis.set_axisbelow(
    True
)

save_figure(
    figure,
    "Fig_3_Revised_Per_Seed_Robustness",
)


# =========================================================
# Revised Figure 4:
# Latency-energy trade-off
# =========================================================

figure, axis = plt.subplots(
    figsize=(7.16, 3.35)
)

marker_mapping = {
    "Latency heuristic": "o",
    "CA-MADDPG-V2.2": "D",
    "MADDPG": "s",
    "CA-MADDPG-V2": "^",
    "Random allocation": "P",
}

annotation_offsets = {
    "Latency heuristic": (
        8,
        -25,
    ),
    "CA-MADDPG-V2.2": (
        9,
        9,
    ),
    "MADDPG": (
        9,
        14,
    ),
    "CA-MADDPG-V2": (
        9,
        -26,
    ),
    "Random allocation": (
        -92,
        -26,
    ),
}

for _, row in overall_ordered.iterrows():
    method = row[
        "Method"
    ]

    latency = float(
        row["Latency (ms)"]
    )

    energy = float(
        row["Energy (mJ)"]
    )

    success = float(
        row["Success Rate (%)"]
    )

    proposed = (
        method
        == "CA-MADDPG-V2.2"
    )

    axis.scatter(
        latency,
        energy,
        marker=marker_mapping[
            method
        ],
        s=(
            95 if proposed else 55
        ),
        facecolor=(
            "0.25" if proposed else "0.72"
        ),
        edgecolor="black",
        linewidth=(
            1.3 if proposed else 0.8
        ),
        zorder=4,
    )

    axis.annotate(
        (
            f"{method}\n"
            f"{success:.2f}% success"
        ),
        (
            latency,
            energy,
        ),
        xytext=annotation_offsets[
            method
        ],
        textcoords="offset points",
        fontsize=7,
        fontweight=(
            "bold" if proposed else "normal"
        ),
        annotation_clip=False,
    )

axis.set_xlabel(
    "Mean latency (ms)"
)

axis.set_ylabel(
    "Mean energy (mJ)"
)

axis.grid(
    linestyle=":",
    linewidth=0.6,
    alpha=0.55,
)

axis.set_axisbelow(
    True
)

axis.margins(
    x=0.13,
    y=0.19,
)

axis.text(
    0.02,
    0.96,
    "Lower-left region is preferable",
    transform=axis.transAxes,
    ha="left",
    va="top",
    fontsize=7,
    style="italic",
)

save_figure(
    figure,
    "Fig_4_Revised_Latency_Energy_Tradeoff",
)


summary_path = (
    OUTPUT_DIRECTORY
    / "figure_revision_summary.txt"
)

summary_path.write_text(
    "\n".join(
        [
            "REVISED MANUSCRIPT FIGURES",
            "==========================",
            "",
            "Fig. 1:",
            (
                "Overall success comparison with "
                "exact values, error bars, hatches, "
                "and proposed-model emphasis."
            ),
            "",
            "Fig. 2:",
            (
                "Held-out resource-floor sensitivity "
                "with the validation-selected 0.98 "
                "floor highlighted."
            ),
            "",
            "Fig. 3:",
            (
                "Per-seed point-and-error plot with "
                "MADDPG reference line."
            ),
            "",
            "Fig. 4:",
            (
                "Latency-energy trade-off with "
                "non-overlapping method labels and "
                "success-rate annotations."
            ),
            "",
            (
                "PDF files are recommended for "
                "LaTeX manuscripts."
            ),
            (
                "600-dpi PNG files are suitable "
                "for Microsoft Word."
            ),
        ]
    ),
    encoding="utf-8",
)

print("=" * 84)
print("REVISED MANUSCRIPT FIGURES CREATED")
print("=" * 84)
print("Output directory:", OUTPUT_DIRECTORY)
print("Revised figures:", 4)
print("PDF files:", 4)
print("PNG files:", 4)
print("Original figures were not modified.")
print("Summary:", summary_path)
print("=" * 84)
print(
    "Figure revision completed successfully."
)
print("=" * 84)
