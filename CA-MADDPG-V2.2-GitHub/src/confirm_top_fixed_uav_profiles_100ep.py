from pathlib import Path
import sys

import numpy as np
import pandas as pd
from scipy.stats import ttest_rel, wilcoxon

sys.path.insert(0, "src")

from uav_iot_env import UAVIOTParallelEnv


CONFIG_PATH = (
    "configs/ca_maddpg_v2_seed42_10ep.yaml"
)

BASELINE_PATH = Path(
    "results/robust_baseline_episode_comparison.csv"
)

OUTPUT_DIRECTORY = Path(
    "results/top_fixed_uav_100ep_confirmation"
)

PROFILES = [
    {
        "method": "Fixed uav_U3 resource 0.98",
        "selected_uav": "uav_U3",
        "resource_level": 0.98,
    },
    {
        "method": "Fixed uav_U1 resource 0.98",
        "selected_uav": "uav_U1",
        "resource_level": 0.98,
    },
    {
        "method": "Fixed uav_U2 resource 0.98",
        "selected_uav": "uav_U2",
        "resource_level": 0.98,
    },
]


def safe_wilcoxon(first, second):
    differences = np.asarray(
        first - second,
        dtype=np.float64,
    )

    if np.allclose(
        differences,
        0.0,
    ):
        return 1.0

    return float(
        wilcoxon(
            first,
            second,
            zero_method="wilcox",
            alternative="two-sided",
        ).pvalue
    )


def paired_cohen_dz(first, second):
    differences = np.asarray(
        first - second,
        dtype=np.float64,
    )

    standard_deviation = differences.std(
        ddof=1
    )

    if np.isclose(
        standard_deviation,
        0.0,
    ):
        return 0.0

    return float(
        differences.mean()
        / standard_deviation
    )


if not BASELINE_PATH.exists():
    raise FileNotFoundError(
        f"Missing baseline file: {BASELINE_PATH}"
    )

OUTPUT_DIRECTORY.mkdir(
    parents=True,
    exist_ok=True,
)

baseline_df = pd.read_csv(
    BASELINE_PATH
)

test_seeds = sorted(
    baseline_df[
        "test_seed"
    ].astype(int).unique().tolist()
)

if len(test_seeds) != 100:
    raise RuntimeError(
        "Expected 100 fixed test seeds, "
        f"found {len(test_seeds)}."
    )

reference_environment = UAVIOTParallelEnv(
    config_path=CONFIG_PATH
)

possible_agents = list(
    reference_environment.possible_agents
)

number_of_agents = len(
    possible_agents
)

action_dimension = (
    reference_environment.action_dimension
)

reference_environment.close()

for profile in PROFILES:
    if (
        profile["selected_uav"]
        not in possible_agents
    ):
        raise RuntimeError(
            "Unknown UAV in profile: "
            f"{profile['selected_uav']}"
        )

episode_records = []

print("=" * 96)
print("TOP FIXED-UAV PROFILE 100-EPISODE CONFIRMATION")
print("=" * 96)
print("Fixed test episodes:", len(test_seeds))
print("Profiles:", len(PROFILES))
print(
    "Total evaluation episodes:",
    len(test_seeds) * len(PROFILES),
)
print("=" * 96)

for profile in PROFILES:
    method = profile["method"]
    selected_uav = profile[
        "selected_uav"
    ]
    resource_level = float(
        profile["resource_level"]
    )

    selected_index = possible_agents.index(
        selected_uav
    )

    print(
        f"Evaluating {method}..."
    )

    for episode_index, test_seed in enumerate(
        test_seeds,
        start=1,
    ):
        environment = UAVIOTParallelEnv(
            config_path=CONFIG_PATH
        )

        agents = list(
            environment.possible_agents
        )

        observations, _ = environment.reset(
            seed=int(test_seed)
        )

        total_reward = 0.0
        successes = 0
        latencies = []
        energies = []
        steps = 0

        while environment.agents:
            joint_actions = np.zeros(
                (
                    number_of_agents,
                    action_dimension,
                ),
                dtype=np.float32,
            )

            joint_actions[
                selected_index,
                0,
            ] = 1.0

            joint_actions[:, 1] = (
                resource_level
            )

            joint_actions[:, 2] = (
                resource_level
            )

            action_dictionary = {
                agent: joint_actions[index]
                for index, agent in enumerate(
                    agents
                )
            }

            (
                next_observations,
                rewards,
                terminations,
                truncations,
                infos,
            ) = environment.step(
                action_dictionary
            )

            first_agent = agents[0]

            information = infos.get(
                first_agent,
                {},
            )

            total_reward += float(
                rewards.get(
                    first_agent,
                    0.0,
                )
            )

            successes += int(
                information.get(
                    "success",
                    False,
                )
            )

            latencies.append(
                float(
                    information.get(
                        "latency_ms",
                        0.0,
                    )
                )
            )

            energies.append(
                float(
                    information.get(
                        "energy_mJ",
                        0.0,
                    )
                )
            )

            observations = next_observations
            steps += 1

        environment.close()

        episode_records.append(
            {
                "method": method,
                "selected_uav": (
                    selected_uav
                ),
                "resource_level": (
                    resource_level
                ),
                "test_seed": int(
                    test_seed
                ),
                "reward": float(
                    total_reward
                ),
                "success_rate": float(
                    successes
                    / max(steps, 1)
                ),
                "mean_latency_ms": float(
                    np.mean(latencies)
                ),
                "mean_energy_mJ": float(
                    np.mean(energies)
                ),
            }
        )

        if episode_index % 20 == 0:
            print(
                f"  Completed "
                f"{episode_index}/"
                f"{len(test_seeds)} "
                "test episodes"
            )

profile_df = pd.DataFrame(
    episode_records
)

profile_summary_df = (
    profile_df.groupby(
        [
            "method",
            "selected_uav",
            "resource_level",
        ]
    )
    .agg(
        evaluation_episodes=(
            "test_seed",
            "count",
        ),
        mean_reward=(
            "reward",
            "mean",
        ),
        std_reward=(
            "reward",
            "std",
        ),
        mean_success_rate=(
            "success_rate",
            "mean",
        ),
        std_success_rate=(
            "success_rate",
            "std",
        ),
        mean_latency_ms=(
            "mean_latency_ms",
            "mean",
        ),
        std_latency_ms=(
            "mean_latency_ms",
            "std",
        ),
        mean_energy_mJ=(
            "mean_energy_mJ",
            "mean",
        ),
        std_energy_mJ=(
            "mean_energy_mJ",
            "std",
        ),
    )
    .reset_index()
)

profile_summary_df[
    "success_rate_percent"
] = (
    profile_summary_df[
        "mean_success_rate"
    ] * 100.0
)

profile_summary_df[
    "success_std_percent"
] = (
    profile_summary_df[
        "std_success_rate"
    ] * 100.0
)

reference_df = baseline_df[
    baseline_df["method"].isin(
        [
            "MADDPG three-seed mean",
            "Latency heuristic",
            "Random allocation",
        ]
    )
][
    [
        "test_seed",
        "method",
        "reward",
        "success_rate",
        "mean_latency_ms",
        "mean_energy_mJ",
    ]
].copy()

combined_df = pd.concat(
    [
        reference_df,
        profile_df[
            [
                "test_seed",
                "method",
                "reward",
                "success_rate",
                "mean_latency_ms",
                "mean_energy_mJ",
            ]
        ],
    ],
    ignore_index=True,
)

combined_summary_df = (
    combined_df.groupby(
        "method"
    )
    .agg(
        evaluation_episodes=(
            "test_seed",
            "count",
        ),
        mean_reward=(
            "reward",
            "mean",
        ),
        std_reward=(
            "reward",
            "std",
        ),
        mean_success_rate=(
            "success_rate",
            "mean",
        ),
        std_success_rate=(
            "success_rate",
            "std",
        ),
        mean_latency_ms=(
            "mean_latency_ms",
            "mean",
        ),
        mean_energy_mJ=(
            "mean_energy_mJ",
            "mean",
        ),
    )
    .reset_index()
)

combined_summary_df[
    "success_rate_percent"
] = (
    combined_summary_df[
        "mean_success_rate"
    ] * 100.0
)

combined_summary_df[
    "success_std_percent"
] = (
    combined_summary_df[
        "std_success_rate"
    ] * 100.0
)

paired_records = []

reference_methods = [
    "MADDPG three-seed mean",
    "Latency heuristic",
    "Random allocation",
]

for profile_method in profile_df[
    "method"
].unique():
    profile_data = profile_df[
        profile_df["method"]
        == profile_method
    ].sort_values(
        "test_seed"
    )

    for reference_method in (
        reference_methods
    ):
        reference_data = reference_df[
            reference_df["method"]
            == reference_method
        ].sort_values(
            "test_seed"
        )

        merged = profile_data.merge(
            reference_data,
            on="test_seed",
            suffixes=(
                "_profile",
                "_reference",
            ),
        )

        paired_records.append(
            {
                "comparison": (
                    f"{profile_method} vs "
                    f"{reference_method}"
                ),
                "success_difference_percentage_points": float(
                    100.0
                    * (
                        merged[
                            "success_rate_profile"
                        ].mean()
                        - merged[
                            "success_rate_reference"
                        ].mean()
                    )
                ),
                "success_paired_ttest_p": float(
                    ttest_rel(
                        merged[
                            "success_rate_profile"
                        ],
                        merged[
                            "success_rate_reference"
                        ],
                    ).pvalue
                ),
                "success_wilcoxon_p": (
                    safe_wilcoxon(
                        merged[
                            "success_rate_profile"
                        ],
                        merged[
                            "success_rate_reference"
                        ],
                    )
                ),
                "success_cohen_dz": (
                    paired_cohen_dz(
                        merged[
                            "success_rate_profile"
                        ],
                        merged[
                            "success_rate_reference"
                        ],
                    )
                ),
                "reward_difference": float(
                    merged[
                        "reward_profile"
                    ].mean()
                    - merged[
                        "reward_reference"
                    ].mean()
                ),
                "latency_difference_ms": float(
                    merged[
                        "mean_latency_ms_profile"
                    ].mean()
                    - merged[
                        "mean_latency_ms_reference"
                    ].mean()
                ),
                "energy_difference_mJ": float(
                    merged[
                        "mean_energy_mJ_profile"
                    ].mean()
                    - merged[
                        "mean_energy_mJ_reference"
                    ].mean()
                ),
            }
        )

# Direct profile-to-profile paired comparisons.
profile_methods = list(
    profile_df["method"].unique()
)

for first_index in range(
    len(profile_methods)
):
    for second_index in range(
        first_index + 1,
        len(profile_methods),
    ):
        first_method = profile_methods[
            first_index
        ]

        second_method = profile_methods[
            second_index
        ]

        first_data = profile_df[
            profile_df["method"]
            == first_method
        ].sort_values(
            "test_seed"
        )

        second_data = profile_df[
            profile_df["method"]
            == second_method
        ].sort_values(
            "test_seed"
        )

        merged = first_data.merge(
            second_data,
            on="test_seed",
            suffixes=(
                "_first",
                "_second",
            ),
        )

        paired_records.append(
            {
                "comparison": (
                    f"{first_method} vs "
                    f"{second_method}"
                ),
                "success_difference_percentage_points": float(
                    100.0
                    * (
                        merged[
                            "success_rate_first"
                        ].mean()
                        - merged[
                            "success_rate_second"
                        ].mean()
                    )
                ),
                "success_paired_ttest_p": float(
                    ttest_rel(
                        merged[
                            "success_rate_first"
                        ],
                        merged[
                            "success_rate_second"
                        ],
                    ).pvalue
                ),
                "success_wilcoxon_p": (
                    safe_wilcoxon(
                        merged[
                            "success_rate_first"
                        ],
                        merged[
                            "success_rate_second"
                        ],
                    )
                ),
                "success_cohen_dz": (
                    paired_cohen_dz(
                        merged[
                            "success_rate_first"
                        ],
                        merged[
                            "success_rate_second"
                        ],
                    )
                ),
                "reward_difference": float(
                    merged[
                        "reward_first"
                    ].mean()
                    - merged[
                        "reward_second"
                    ].mean()
                ),
                "latency_difference_ms": float(
                    merged[
                        "mean_latency_ms_first"
                    ].mean()
                    - merged[
                        "mean_latency_ms_second"
                    ].mean()
                ),
                "energy_difference_mJ": float(
                    merged[
                        "mean_energy_mJ_first"
                    ].mean()
                    - merged[
                        "mean_energy_mJ_second"
                    ].mean()
                ),
            }
        )

paired_df = pd.DataFrame(
    paired_records
)

episode_output = (
    OUTPUT_DIRECTORY
    / "top_fixed_uav_100ep_results.csv"
)

profile_summary_output = (
    OUTPUT_DIRECTORY
    / "top_fixed_uav_100ep_profile_summary.csv"
)

combined_summary_output = (
    OUTPUT_DIRECTORY
    / "top_fixed_uav_100ep_method_summary.csv"
)

paired_output = (
    OUTPUT_DIRECTORY
    / "top_fixed_uav_100ep_paired_tests.csv"
)

profile_df.to_csv(
    episode_output,
    index=False,
)

profile_summary_df.to_csv(
    profile_summary_output,
    index=False,
)

combined_summary_df.to_csv(
    combined_summary_output,
    index=False,
)

paired_df.to_csv(
    paired_output,
    index=False,
)

print()
print("=" * 125)
print("TOP FIXED-UAV 100-EPISODE PROFILE SUMMARY")
print("=" * 125)

print(
    profile_summary_df[
        [
            "method",
            "mean_reward",
            "std_reward",
            "success_rate_percent",
            "success_std_percent",
            "mean_latency_ms",
            "mean_energy_mJ",
        ]
    ]
    .sort_values(
        by=[
            "success_rate_percent",
            "mean_reward",
        ],
        ascending=[
            False,
            False,
        ],
    )
    .to_string(index=False)
)

print()
print("=" * 125)
print("COMBINED BENCHMARK SUMMARY")
print("=" * 125)

print(
    combined_summary_df[
        [
            "method",
            "mean_reward",
            "std_reward",
            "success_rate_percent",
            "success_std_percent",
            "mean_latency_ms",
            "mean_energy_mJ",
        ]
    ]
    .sort_values(
        by=[
            "success_rate_percent",
            "mean_reward",
        ],
        ascending=[
            False,
            False,
        ],
    )
    .to_string(index=False)
)

print()
print("=" * 125)
print("PAIRED STATISTICAL COMPARISONS")
print("=" * 125)

print(
    paired_df.to_string(
        index=False
    )
)

print()
print("=" * 125)
print("Saved episode results:", episode_output)
print("Saved profile summary:", profile_summary_output)
print("Saved method summary:", combined_summary_output)
print("Saved paired tests:", paired_output)
print("=" * 125)
