from __future__ import annotations

import hashlib
import json
import shutil
from pathlib import Path

import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


V22_DIRECTORY = Path(
    "results/ca_maddpg_v22_three_seed"
)

METHOD_SUMMARY_PATH = (
    V22_DIRECTORY
    / "v22_official_method_summary.csv"
)

PAIRED_TEST_PATH = (
    V22_DIRECTORY
    / "v22_official_paired_tests.csv"
)

PER_SEED_PATH = (
    V22_DIRECTORY
    / "v22_per_seed_summary.csv"
)

ACTION_SUMMARY_PATH = (
    V22_DIRECTORY
    / "v22_action_summary.csv"
)

BENCHMARK_DECISION_PATH = (
    V22_DIRECTORY
    / "v22_benchmark_decision.json"
)

SENSITIVITY_PATH = Path(
    "results/v2_resource_projection_validation/"
    "v2_resource_projection_overall_summary.csv"
)

V21_SUMMARY_PATH = Path(
    "results/ca_maddpg_v21_seed42_10ep/"
    "v21_fixed_100_episode_summary.csv"
)

OUTPUT_DIRECTORY = Path(
    "results/final_conference_package"
)

TABLE_DIRECTORY = (
    OUTPUT_DIRECTORY / "tables"
)

FIGURE_DIRECTORY = (
    OUTPUT_DIRECTORY / "figures"
)

RAW_DIRECTORY = (
    OUTPUT_DIRECTORY / "official_raw_results"
)

for directory in [
    OUTPUT_DIRECTORY,
    TABLE_DIRECTORY,
    FIGURE_DIRECTORY,
    RAW_DIRECTORY,
]:
    directory.mkdir(
        parents=True,
        exist_ok=True,
    )


required_paths = [
    METHOD_SUMMARY_PATH,
    PAIRED_TEST_PATH,
    PER_SEED_PATH,
    ACTION_SUMMARY_PATH,
    BENCHMARK_DECISION_PATH,
    SENSITIVITY_PATH,
]

for required_path in required_paths:
    if not required_path.exists():
        raise FileNotFoundError(
            f"Missing required result file: "
            f"{required_path}"
        )


method_df = pd.read_csv(
    METHOD_SUMMARY_PATH
)

paired_df = pd.read_csv(
    PAIRED_TEST_PATH
)

per_seed_df = pd.read_csv(
    PER_SEED_PATH
)

action_df = pd.read_csv(
    ACTION_SUMMARY_PATH
)

sensitivity_df = pd.read_csv(
    SENSITIVITY_PATH
)

with BENCHMARK_DECISION_PATH.open(
    "r",
    encoding="utf-8",
) as file:
    decision = json.load(file)


# ---------------------------------------------------------
# Final integrity verification
# ---------------------------------------------------------

expected_training_seeds = {
    42,
    43,
    44,
}

actual_training_seeds = set(
    per_seed_df[
        "training_seed"
    ].astype(int).tolist()
)

if actual_training_seeds != expected_training_seeds:
    raise RuntimeError(
        "Unexpected training seeds: "
        f"{sorted(actual_training_seeds)}"
    )

if not bool(
    decision[
        "crossed_maddpg_three_seed_mean"
    ]
):
    raise RuntimeError(
        "V2.2 did not cross the MADDPG baseline."
    )

if not bool(
    decision[
        "significantly_crossed_maddpg"
    ]
):
    raise RuntimeError(
        "V2.2 improvement over MADDPG "
        "was not statistically significant."
    )

if bool(
    decision[
        "fixed_test_used_for_floor_selection"
    ]
):
    raise RuntimeError(
        "Test-set leakage detected in "
        "resource-floor selection."
    )

if not np.isclose(
    float(decision["resource_floor"]),
    0.98,
):
    raise RuntimeError(
        "Unexpected resource floor."
    )


# ---------------------------------------------------------
# Standard method naming and ordering
# ---------------------------------------------------------

method_order = [
    "Latency heuristic",
    "CA-MADDPG-V2.2 three-seed mean",
    "MADDPG three-seed mean",
    "CA-MADDPG-V2 three-seed mean",
    "Random allocation",
]

display_names = {
    "Latency heuristic": (
        "Latency heuristic"
    ),
    "CA-MADDPG-V2.2 three-seed mean": (
        "CA-MADDPG-V2.2"
    ),
    "MADDPG three-seed mean": (
        "MADDPG"
    ),
    "CA-MADDPG-V2 three-seed mean": (
        "CA-MADDPG-V2"
    ),
    "Random allocation": (
        "Random allocation"
    ),
}

available_methods = [
    method
    for method in method_order
    if method in set(
        method_df["method"]
    )
]

overall_df = (
    method_df[
        method_df["method"].isin(
            available_methods
        )
    ]
    .copy()
)

overall_df[
    "display_method"
] = overall_df[
    "method"
].map(display_names)

order_mapping = {
    method: index
    for index, method in enumerate(
        available_methods
    )
}

overall_df[
    "display_order"
] = overall_df[
    "method"
].map(order_mapping)

overall_df = overall_df.sort_values(
    "display_order"
).reset_index(drop=True)


# ---------------------------------------------------------
# Key conference claims
# ---------------------------------------------------------

def get_method_row(method_name: str) -> pd.Series:
    rows = method_df[
        method_df["method"]
        == method_name
    ]

    if len(rows) != 1:
        raise RuntimeError(
            f"Expected one row for "
            f"{method_name}, found {len(rows)}."
        )

    return rows.iloc[0]


v22_row = get_method_row(
    "CA-MADDPG-V2.2 three-seed mean"
)

maddpg_row = get_method_row(
    "MADDPG three-seed mean"
)

heuristic_row = get_method_row(
    "Latency heuristic"
)

v2_row = get_method_row(
    "CA-MADDPG-V2 three-seed mean"
)


success_improvement_pp = float(
    v22_row["success_rate_percent"]
    - maddpg_row["success_rate_percent"]
)

success_relative_improvement = float(
    100.0
    * success_improvement_pp
    / maddpg_row["success_rate_percent"]
)

reward_improvement = float(
    v22_row["mean_reward"]
    - maddpg_row["mean_reward"]
)

latency_reduction_ms = float(
    maddpg_row["mean_latency_ms"]
    - v22_row["mean_latency_ms"]
)

latency_reduction_percent = float(
    100.0
    * latency_reduction_ms
    / maddpg_row["mean_latency_ms"]
)

energy_reduction_mJ = float(
    maddpg_row["mean_energy_mJ"]
    - v22_row["mean_energy_mJ"]
)

energy_reduction_percent = float(
    100.0
    * energy_reduction_mJ
    / maddpg_row["mean_energy_mJ"]
)

maddpg_comparison = paired_df[
    paired_df[
        "comparison"
    ].str.endswith(
        "MADDPG three-seed mean"
    )
].iloc[0]

key_claims = {
    "proposed_model": "CA-MADDPG-V2.2",
    "training_seeds": [42, 43, 44],
    "fixed_evaluation_episodes_per_seed": 100,
    "resource_floor": 0.98,
    "resource_floor_selected_on": (
        "held-out validation seeds"
    ),
    "v22_success_rate_percent": float(
        v22_row["success_rate_percent"]
    ),
    "maddpg_success_rate_percent": float(
        maddpg_row["success_rate_percent"]
    ),
    "success_improvement_percentage_points": (
        success_improvement_pp
    ),
    "success_relative_improvement_percent": (
        success_relative_improvement
    ),
    "reward_improvement": (
        reward_improvement
    ),
    "latency_reduction_ms": (
        latency_reduction_ms
    ),
    "latency_reduction_percent": (
        latency_reduction_percent
    ),
    "energy_reduction_mJ": (
        energy_reduction_mJ
    ),
    "energy_reduction_percent": (
        energy_reduction_percent
    ),
    "success_paired_ttest_p": float(
        maddpg_comparison[
            "success_paired_ttest_p"
        ]
    ),
    "success_wilcoxon_p": float(
        maddpg_comparison[
            "success_wilcoxon_p"
        ]
    ),
    "success_cohen_dz": float(
        maddpg_comparison[
            "success_cohen_dz"
        ]
    ),
    "crossed_latency_heuristic": bool(
        decision[
            "crossed_latency_heuristic"
        ]
    ),
}


# ---------------------------------------------------------
# Table II: Overall performance
# ---------------------------------------------------------

overall_table = overall_df[
    [
        "display_method",
        "mean_reward",
        "std_reward",
        "success_rate_percent",
        "success_std_percent",
        "mean_latency_ms",
        "mean_energy_mJ",
    ]
].copy()

overall_table.columns = [
    "Method",
    "Mean Reward",
    "Reward Std",
    "Success Rate (%)",
    "Success Std (%)",
    "Latency (ms)",
    "Energy (mJ)",
]

overall_table.to_csv(
    TABLE_DIRECTORY
    / "Table_II_Overall_Performance.csv",
    index=False,
)


# ---------------------------------------------------------
# Table III: Ablation analysis
# ---------------------------------------------------------

ablation_methods = [
    "MADDPG three-seed mean",
    "CA-MADDPG-V2 three-seed mean",
    "CA-MADDPG-V2.2 three-seed mean",
]

ablation_df = method_df[
    method_df["method"].isin(
        ablation_methods
    )
].copy()

ablation_mapping = {
    "MADDPG three-seed mean": {
        "Method": "MADDPG",
        "Collaborative Attention": "No",
        "Resource Projection": "No",
    },
    "CA-MADDPG-V2 three-seed mean": {
        "Method": "CA-MADDPG-V2",
        "Collaborative Attention": "Yes",
        "Resource Projection": "No",
    },
    "CA-MADDPG-V2.2 three-seed mean": {
        "Method": "CA-MADDPG-V2.2",
        "Collaborative Attention": "Yes",
        "Resource Projection": "Yes",
    },
}

ablation_records = []

for method_name in ablation_methods:
    row = ablation_df[
        ablation_df["method"]
        == method_name
    ].iloc[0]

    description = ablation_mapping[
        method_name
    ]

    ablation_records.append(
        {
            **description,
            "Success Rate (%)": float(
                row[
                    "success_rate_percent"
                ]
            ),
            "Mean Reward": float(
                row["mean_reward"]
            ),
            "Latency (ms)": float(
                row["mean_latency_ms"]
            ),
            "Energy (mJ)": float(
                row["mean_energy_mJ"]
            ),
        }
    )

ablation_table = pd.DataFrame(
    ablation_records
)

ablation_table.to_csv(
    TABLE_DIRECTORY
    / "Table_III_Ablation_Analysis.csv",
    index=False,
)


# ---------------------------------------------------------
# Table IV: Statistical comparisons
# ---------------------------------------------------------

comparison_order = [
    "MADDPG three-seed mean",
    "Latency heuristic",
    "Random allocation",
    "CA-MADDPG-V2 three-seed mean",
]

statistical_records = []

for reference_name in comparison_order:
    rows = paired_df[
        paired_df[
            "comparison"
        ].str.endswith(
            reference_name
        )
    ]

    if rows.empty:
        continue

    row = rows.iloc[0]

    statistical_records.append(
        {
            "Reference": display_names.get(
                reference_name,
                reference_name,
            ),
            "Success Difference (pp)": float(
                row[
                    "success_difference_percentage_points"
                ]
            ),
            "Paired t-test p": float(
                row[
                    "success_paired_ttest_p"
                ]
            ),
            "Wilcoxon p": float(
                row[
                    "success_wilcoxon_p"
                ]
            ),
            "Cohen dz": float(
                row[
                    "success_cohen_dz"
                ]
            ),
            "Reward Difference": float(
                row[
                    "reward_difference"
                ]
            ),
            "Latency Difference (ms)": float(
                row[
                    "latency_difference_ms"
                ]
            ),
            "Energy Difference (mJ)": float(
                row[
                    "energy_difference_mJ"
                ]
            ),
        }
    )

statistical_table = pd.DataFrame(
    statistical_records
)

statistical_table.to_csv(
    TABLE_DIRECTORY
    / "Table_IV_Statistical_Comparisons.csv",
    index=False,
)


# ---------------------------------------------------------
# Table V: Resource-floor sensitivity
# ---------------------------------------------------------

sensitivity_table = sensitivity_df[
    [
        "resource_floor",
        "mean_reward",
        "success_rate_percent",
        "success_std_percent",
        "mean_latency_ms",
        "mean_energy_mJ",
        "mean_selected_cpu",
        "mean_selected_bandwidth",
    ]
].copy()

sensitivity_table = (
    sensitivity_table.sort_values(
        "resource_floor"
    )
)

sensitivity_table.columns = [
    "Resource Floor",
    "Mean Reward",
    "Success Rate (%)",
    "Success Std (%)",
    "Latency (ms)",
    "Energy (mJ)",
    "Selected CPU",
    "Selected Bandwidth",
]

sensitivity_table.to_csv(
    TABLE_DIRECTORY
    / "Table_V_Resource_Sensitivity.csv",
    index=False,
)


# ---------------------------------------------------------
# Table VI: Per-seed robustness
# ---------------------------------------------------------

per_seed_table = per_seed_df[
    [
        "training_seed",
        "saved_episode",
        "mean_reward",
        "success_rate_percent",
        "success_std_percent",
        "mean_latency_ms",
        "mean_energy_mJ",
    ]
].copy()

per_seed_table.columns = [
    "Training Seed",
    "Selected Episode",
    "Mean Reward",
    "Success Rate (%)",
    "Success Std (%)",
    "Latency (ms)",
    "Energy (mJ)",
]

per_seed_table.to_csv(
    TABLE_DIRECTORY
    / "Table_VI_Per_Seed_Robustness.csv",
    index=False,
)


# ---------------------------------------------------------
# Optional diagnostic V2.1 table
# ---------------------------------------------------------

if V21_SUMMARY_PATH.exists():
    v21_df = pd.read_csv(
        V21_SUMMARY_PATH
    )

    v21_df.to_csv(
        TABLE_DIRECTORY
        / "Diagnostic_CA_MADDPG_V21_Result.csv",
        index=False,
    )


# ---------------------------------------------------------
# Figure configuration
# ---------------------------------------------------------

plt.rcParams.update(
    {
        "font.size": 10,
        "axes.labelsize": 10,
        "axes.titlesize": 11,
        "xtick.labelsize": 9,
        "ytick.labelsize": 9,
        "legend.fontsize": 9,
    }
)


def save_figure(
    figure,
    stem_name: str,
):
    figure.tight_layout()

    figure.savefig(
        FIGURE_DIRECTORY
        / f"{stem_name}.png",
        dpi=600,
        bbox_inches="tight",
    )

    figure.savefig(
        FIGURE_DIRECTORY
        / f"{stem_name}.pdf",
        bbox_inches="tight",
    )

    plt.close(figure)


# ---------------------------------------------------------
# Figure 1: Success-rate comparison
# ---------------------------------------------------------

figure, axis = plt.subplots(
    figsize=(7.2, 4.2)
)

axis.bar(
    overall_df["display_method"],
    overall_df["success_rate_percent"],
    yerr=overall_df[
        "success_std_percent"
    ],
    capsize=4,
)

axis.set_ylabel(
    "Success rate (%)"
)

axis.set_title(
    "Fixed 100-Episode Performance Comparison"
)

axis.grid(
    axis="y",
    alpha=0.25,
)

axis.tick_params(
    axis="x",
    rotation=20,
)

save_figure(
    figure,
    "Fig_1_Success_Rate_Comparison",
)


# ---------------------------------------------------------
# Figure 2: Resource-floor sensitivity
# ---------------------------------------------------------

sensitivity_sorted = (
    sensitivity_df.sort_values(
        "resource_floor"
    )
)

figure, axis = plt.subplots(
    figsize=(6.6, 4.0)
)

axis.plot(
    sensitivity_sorted[
        "resource_floor"
    ],
    sensitivity_sorted[
        "success_rate_percent"
    ],
    marker="o",
)

axis.set_xlabel(
    "Resource floor"
)

axis.set_ylabel(
    "Validation success rate (%)"
)

axis.set_title(
    "Held-Out Resource-Floor Sensitivity"
)

axis.grid(
    alpha=0.25,
)

save_figure(
    figure,
    "Fig_2_Resource_Floor_Sensitivity",
)


# ---------------------------------------------------------
# Figure 3: Per-seed robustness
# ---------------------------------------------------------

figure, axis = plt.subplots(
    figsize=(6.2, 4.0)
)

seed_labels = [
    f"Seed {int(seed)}"
    for seed in per_seed_df[
        "training_seed"
    ]
]

axis.bar(
    seed_labels,
    per_seed_df[
        "success_rate_percent"
    ],
    yerr=per_seed_df[
        "success_std_percent"
    ],
    capsize=4,
)

axis.axhline(
    float(
        maddpg_row[
            "success_rate_percent"
        ]
    ),
    linestyle="--",
    label="MADDPG three-seed mean",
)

axis.set_ylabel(
    "Success rate (%)"
)

axis.set_title(
    "CA-MADDPG-V2.2 Per-Seed Robustness"
)

axis.legend()

axis.grid(
    axis="y",
    alpha=0.25,
)

save_figure(
    figure,
    "Fig_3_Per_Seed_Robustness",
)


# ---------------------------------------------------------
# Figure 4: Latency-energy trade-off
# ---------------------------------------------------------

figure, axis = plt.subplots(
    figsize=(6.5, 4.3)
)

axis.scatter(
    overall_df[
        "mean_latency_ms"
    ],
    overall_df[
        "mean_energy_mJ"
    ],
    s=65,
)

for _, row in overall_df.iterrows():
    axis.annotate(
        row["display_method"],
        (
            row["mean_latency_ms"],
            row["mean_energy_mJ"],
        ),
        xytext=(5, 5),
        textcoords="offset points",
        fontsize=8,
    )

axis.set_xlabel(
    "Mean latency (ms)"
)

axis.set_ylabel(
    "Mean energy (mJ)"
)

axis.set_title(
    "Latency-Energy Trade-Off"
)

axis.grid(
    alpha=0.25,
)

save_figure(
    figure,
    "Fig_4_Latency_Energy_Tradeoff",
)


# ---------------------------------------------------------
# IEEE-style LaTeX overall-results table
# ---------------------------------------------------------

latex_lines = [
    r"\begin{table*}[t]",
    r"\centering",
    r"\caption{Performance comparison over 100 fixed evaluation episodes.}",
    r"\label{tab:overall_performance}",
    r"\begin{tabular}{lrrrr}",
    r"\hline",
    (
        r"Method & Reward & Success (\%) "
        r"& Latency (ms) & Energy (mJ) \\"
    ),
    r"\hline",
]

for _, row in overall_df.iterrows():
    latex_lines.append(
        (
            f"{row['display_method']} & "
            f"{row['mean_reward']:.2f} & "
            f"{row['success_rate_percent']:.2f} "
            f"$\\pm$ "
            f"{row['success_std_percent']:.2f} & "
            f"{row['mean_latency_ms']:.2f} & "
            f"{row['mean_energy_mJ']:.2f} \\\\"
        )
    )

latex_lines.extend(
    [
        r"\hline",
        r"\end{tabular}",
        r"\end{table*}",
    ]
)

(
    TABLE_DIRECTORY
    / "Table_II_Overall_Performance.tex"
).write_text(
    "\n".join(latex_lines),
    encoding="utf-8",
)


# ---------------------------------------------------------
# Key claims and interpretation
# ---------------------------------------------------------

with (
    OUTPUT_DIRECTORY
    / "conference_key_claims.json"
).open(
    "w",
    encoding="utf-8",
) as file:
    json.dump(
        key_claims,
        file,
        indent=2,
    )

claim_text = f"""
FINAL CONFERENCE RESULT SUMMARY
================================

Proposed model:
CA-MADDPG-V2.2

Evaluation protocol:
- Training seeds: 42, 43, and 44
- Fixed evaluation episodes per model: 100
- Resource floor: 0.98
- Resource floor selected using held-out validation
- Fixed benchmark test episodes were not used for model selection

Main result:
- CA-MADDPG-V2.2 success rate:
  {key_claims['v22_success_rate_percent']:.2f}%
- MADDPG success rate:
  {key_claims['maddpg_success_rate_percent']:.2f}%
- Improvement:
  {key_claims['success_improvement_percentage_points']:.2f}
  percentage points
- Relative success improvement:
  {key_claims['success_relative_improvement_percent']:.2f}%

Efficiency improvement over MADDPG:
- Reward improvement:
  {key_claims['reward_improvement']:.2f}
- Latency reduction:
  {key_claims['latency_reduction_ms']:.2f} ms
  ({key_claims['latency_reduction_percent']:.2f}%)
- Energy reduction:
  {key_claims['energy_reduction_mJ']:.2f} mJ
  ({key_claims['energy_reduction_percent']:.2f}%)

Statistical evidence:
- Paired t-test p:
  {key_claims['success_paired_ttest_p']:.6e}
- Wilcoxon p:
  {key_claims['success_wilcoxon_p']:.6e}
- Paired Cohen dz:
  {key_claims['success_cohen_dz']:.4f}

Permitted manuscript claim:
CA-MADDPG-V2.2 significantly outperformed the
MADDPG learning baseline while reducing latency
and energy consumption.

Claim limitation:
CA-MADDPG-V2.2 did not outperform the latency
heuristic in success rate. Do not claim that the
proposed model outperformed every benchmark.
""".strip()

(
    OUTPUT_DIRECTORY
    / "conference_result_interpretation.txt"
).write_text(
    claim_text,
    encoding="utf-8",
)


# ---------------------------------------------------------
# Preserve raw official results
# ---------------------------------------------------------

source_files = required_paths + [
    V22_DIRECTORY
    / "v22_three_seed_episode_results.csv",
    V22_DIRECTORY
    / "v22_three_seed_episode_mean.csv",
    V22_DIRECTORY
    / "v22_three_seed_action_steps.csv",
]

copied_files = []

for source_path in source_files:
    if source_path.exists():
        destination_path = (
            RAW_DIRECTORY
            / source_path.name
        )

        shutil.copy2(
            source_path,
            destination_path,
        )

        copied_files.append(
            destination_path
        )


# ---------------------------------------------------------
# SHA-256 integrity manifest
# ---------------------------------------------------------

def sha256_file(path: Path) -> str:
    hash_object = hashlib.sha256()

    with path.open("rb") as file:
        while True:
            chunk = file.read(
                1024 * 1024
            )

            if not chunk:
                break

            hash_object.update(chunk)

    return hash_object.hexdigest()


manifest_records = []

for path in sorted(
    OUTPUT_DIRECTORY.rglob("*")
):
    if path.is_file():
        manifest_records.append(
            {
                "relative_path": str(
                    path.relative_to(
                        OUTPUT_DIRECTORY
                    )
                ),
                "size_bytes": (
                    path.stat().st_size
                ),
                "sha256": sha256_file(
                    path
                ),
            }
        )

manifest_path = (
    OUTPUT_DIRECTORY
    / "final_package_manifest.json"
)

with manifest_path.open(
    "w",
    encoding="utf-8",
) as file:
    json.dump(
        manifest_records,
        file,
        indent=2,
    )


print("=" * 92)
print("FINAL CONFERENCE RESULTS PACKAGE CREATED")
print("=" * 92)
print(
    "Output directory:",
    OUTPUT_DIRECTORY,
)
print(
    "Proposed model:",
    decision["model"],
)
print(
    "Resource floor:",
    decision["resource_floor"],
)
print(
    "Training seeds:",
    sorted(actual_training_seeds),
)
print(
    "V2.2 success rate:",
    f"{v22_row['success_rate_percent']:.2f}%",
)
print(
    "MADDPG success rate:",
    f"{maddpg_row['success_rate_percent']:.2f}%",
)
print(
    "Success improvement:",
    f"{success_improvement_pp:.2f} percentage points",
)
print(
    "Latency reduction:",
    f"{latency_reduction_ms:.2f} ms",
)
print(
    "Energy reduction:",
    f"{energy_reduction_mJ:.2f} mJ",
)
print(
    "Paired t-test p:",
    (
        f"{maddpg_comparison['success_paired_ttest_p']:.6e}"
    ),
)
print(
    "Significantly crossed MADDPG:",
    decision[
        "significantly_crossed_maddpg"
    ],
)
print(
    "Crossed latency heuristic:",
    decision[
        "crossed_latency_heuristic"
    ],
)
print(
    "Tables created:",
    len(
        list(
            TABLE_DIRECTORY.glob("*")
        )
    ),
)
print(
    "Figures created:",
    len(
        list(
            FIGURE_DIRECTORY.glob("*")
        )
    ),
)
print(
    "Raw official files preserved:",
    len(copied_files),
)
print(
    "Integrity manifest:",
    manifest_path,
)
print("=" * 92)
print(
    "Conference results package verification "
    "completed successfully."
)
print("=" * 92)
