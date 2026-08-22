from __future__ import annotations

import gc
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from scipy.stats import ttest_rel, wilcoxon

sys.path.insert(0, "src")

from ca_maddpg_v2 import CAMADDPGV2
from uav_iot_env import UAVIOTParallelEnv


COMMON_ENV_CONFIG = (
    "configs/ca_maddpg_v2_seed42_10ep.yaml"
)

ROBUST_BASELINE_PATH = Path(
    "results/robust_baseline_episode_comparison.csv"
)

OUTPUT_DIRECTORY = Path(
    "results/ca_maddpg_v2_three_seed"
)

SEED_MODELS = [
    {
        "training_seed": 42,
        "config": (
            "configs/ca_maddpg_v2_seed42_10ep.yaml"
        ),
        "checkpoint": (
            "checkpoints/ca_maddpg_v2_seed42_10ep/"
            "ca_maddpg_v2_best.pt"
        ),
    },
    {
        "training_seed": 43,
        "config": (
            "configs/ca_maddpg_v2_seed43_10ep.yaml"
        ),
        "checkpoint": (
            "checkpoints/ca_maddpg_v2_seed43_10ep/"
            "ca_maddpg_v2_best.pt"
        ),
    },
    {
        "training_seed": 44,
        "config": (
            "configs/ca_maddpg_v2_seed44_10ep.yaml"
        ),
        "checkpoint": (
            "checkpoints/ca_maddpg_v2_seed44_10ep/"
            "ca_maddpg_v2_best.pt"
        ),
    },
]


def stack_observations(
    observation_dictionary,
    possible_agents,
    observation_dimension,
):
    zero_observation = np.zeros(
        observation_dimension,
        dtype=np.float32,
    )

    return np.stack(
        [
            np.asarray(
                observation_dictionary.get(
                    agent,
                    zero_observation,
                ),
                dtype=np.float32,
            )
            for agent in possible_agents
        ],
        axis=0,
    )


def load_model(
    config_path,
    checkpoint_path,
):
    environment = UAVIOTParallelEnv(
        config_path=COMMON_ENV_CONFIG
    )

    model = CAMADDPGV2(
        number_of_agents=len(
            environment.possible_agents
        ),
        observation_dimension=(
            environment.observation_dimension
        ),
        action_dimension=(
            environment.action_dimension
        ),
        config_path=config_path,
    )

    environment.close()

    checkpoint = model.load_checkpoint(
        checkpoint_path
    )

    return model, checkpoint


def evaluate_episode(
    model,
    test_seed,
):
    environment = UAVIOTParallelEnv(
        config_path=COMMON_ENV_CONFIG
    )

    possible_agents = list(
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
        joint_observations = (
            stack_observations(
                observations,
                possible_agents,
                environment.observation_dimension,
            )
        )

        joint_actions = model.select_actions(
            joint_observations,
            noise_scale=0.0,
        )

        action_dictionary = {
            agent: joint_actions[index]
            for index, agent in enumerate(
                possible_agents
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

        first_agent = possible_agents[0]

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

    return {
        "test_seed": int(test_seed),
        "reward": float(total_reward),
        "success_rate": float(
            successes / max(steps, 1)
        ),
        "mean_latency_ms": float(
            np.mean(latencies)
        ),
        "mean_energy_mJ": float(
            np.mean(energies)
        ),
    }


def safe_wilcoxon(
    first,
    second,
):
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


def paired_effect_size(
    first,
    second,
):
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


if not ROBUST_BASELINE_PATH.exists():
    raise FileNotFoundError(
        f"Missing baseline file: "
        f"{ROBUST_BASELINE_PATH}"
    )

for information in SEED_MODELS:
    checkpoint_path = Path(
        information["checkpoint"]
    )

    if not checkpoint_path.exists():
        raise FileNotFoundError(
            f"Missing checkpoint: "
            f"{checkpoint_path}"
        )

OUTPUT_DIRECTORY.mkdir(
    parents=True,
    exist_ok=True,
)

baseline_df = pd.read_csv(
    ROBUST_BASELINE_PATH
)

required_columns = {
    "test_seed",
    "method",
    "reward",
    "success_rate",
    "mean_latency_ms",
    "mean_energy_mJ",
}

missing_columns = (
    required_columns
    - set(baseline_df.columns)
)

if missing_columns:
    raise RuntimeError(
        "Baseline file is missing columns: "
        f"{sorted(missing_columns)}"
    )

test_seeds = sorted(
    baseline_df[
        "test_seed"
    ].astype(int).unique().tolist()
)

if len(test_seeds) != 100:
    raise RuntimeError(
        "Expected 100 fixed test episodes, "
        f"found {len(test_seeds)}."
    )

print("=" * 92)
print("CA-MADDPG-V2 OFFICIAL THREE-SEED BENCHMARK")
print("=" * 92)
print("Training seeds: 42, 43, 44")
print("Fixed test episodes:", len(test_seeds))
print("Evaluation episodes per model:", len(test_seeds))
print("=" * 92)

seed_episode_records = []
checkpoint_records = []

for information in SEED_MODELS:
    training_seed = int(
        information["training_seed"]
    )

    model, checkpoint = load_model(
        config_path=information["config"],
        checkpoint_path=(
            information["checkpoint"]
        ),
    )

    saved_episode = int(
        checkpoint["episode"]
    )

    checkpoint_records.append(
        {
            "training_seed": training_seed,
            "saved_episode": saved_episode,
            "checkpoint": (
                information["checkpoint"]
            ),
        }
    )

    print(
        f"Evaluating CA-MADDPG-V2 "
        f"training Seed {training_seed} "
        f"from Episode {saved_episode}..."
    )

    for episode_index, test_seed in enumerate(
        test_seeds,
        start=1,
    ):
        episode_result = evaluate_episode(
            model=model,
            test_seed=test_seed,
        )

        episode_result[
            "training_seed"
        ] = training_seed

        episode_result[
            "saved_episode"
        ] = saved_episode

        episode_result["method"] = (
            f"CA-MADDPG-V2 Seed "
            f"{training_seed}"
        )

        seed_episode_records.append(
            episode_result
        )

        if episode_index % 20 == 0:
            print(
                f"  Completed "
                f"{episode_index}/"
                f"{len(test_seeds)} "
                "test episodes"
            )

    del model
    del checkpoint

    gc.collect()
    torch.cuda.empty_cache()

seed_episode_df = pd.DataFrame(
    seed_episode_records
)

per_seed_summary_df = (
    seed_episode_df.groupby(
        [
            "training_seed",
            "saved_episode",
            "method",
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

per_seed_summary_df[
    "success_rate_percent"
] = (
    per_seed_summary_df[
        "mean_success_rate"
    ] * 100.0
)

per_seed_summary_df[
    "success_std_percent"
] = (
    per_seed_summary_df[
        "std_success_rate"
    ] * 100.0
)

three_seed_episode_df = (
    seed_episode_df.groupby(
        "test_seed"
    )
    .agg(
        reward=(
            "reward",
            "mean",
        ),
        success_rate=(
            "success_rate",
            "mean",
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

three_seed_episode_df["method"] = (
    "CA-MADDPG-V2 three-seed mean"
)

baseline_comparison_df = baseline_df[
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

combined_comparison_df = pd.concat(
    [
        baseline_comparison_df,
        three_seed_episode_df[
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

method_summary_df = (
    combined_comparison_df.groupby(
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

method_summary_df[
    "success_rate_percent"
] = (
    method_summary_df[
        "mean_success_rate"
    ] * 100.0
)

method_summary_df[
    "success_std_percent"
] = (
    method_summary_df[
        "std_success_rate"
    ] * 100.0
)

reference_methods = [
    "MADDPG three-seed mean",
    "Latency heuristic",
    "Random allocation",
]

paired_records = []

ca_data = three_seed_episode_df.sort_values(
    "test_seed"
)

for reference_method in reference_methods:
    reference_data = (
        baseline_comparison_df[
            baseline_comparison_df["method"]
            == reference_method
        ]
        .sort_values("test_seed")
    )

    merged = ca_data.merge(
        reference_data,
        on="test_seed",
        suffixes=(
            "_ca",
            "_reference",
        ),
    )

    paired_records.append(
        {
            "comparison": (
                "CA-MADDPG-V2 three-seed mean "
                f"vs {reference_method}"
            ),
            "success_difference_percentage_points": float(
                100.0
                * (
                    merged[
                        "success_rate_ca"
                    ].mean()
                    - merged[
                        "success_rate_reference"
                    ].mean()
                )
            ),
            "success_paired_ttest_p": float(
                ttest_rel(
                    merged[
                        "success_rate_ca"
                    ],
                    merged[
                        "success_rate_reference"
                    ],
                ).pvalue
            ),
            "success_wilcoxon_p": (
                safe_wilcoxon(
                    merged[
                        "success_rate_ca"
                    ],
                    merged[
                        "success_rate_reference"
                    ],
                )
            ),
            "success_cohen_dz": (
                paired_effect_size(
                    merged[
                        "success_rate_ca"
                    ],
                    merged[
                        "success_rate_reference"
                    ],
                )
            ),
            "reward_difference": float(
                merged[
                    "reward_ca"
                ].mean()
                - merged[
                    "reward_reference"
                ].mean()
            ),
            "reward_paired_ttest_p": float(
                ttest_rel(
                    merged[
                        "reward_ca"
                    ],
                    merged[
                        "reward_reference"
                    ],
                ).pvalue
            ),
            "latency_difference_ms": float(
                merged[
                    "mean_latency_ms_ca"
                ].mean()
                - merged[
                    "mean_latency_ms_reference"
                ].mean()
            ),
            "latency_paired_ttest_p": float(
                ttest_rel(
                    merged[
                        "mean_latency_ms_ca"
                    ],
                    merged[
                        "mean_latency_ms_reference"
                    ],
                ).pvalue
            ),
            "energy_difference_mJ": float(
                merged[
                    "mean_energy_mJ_ca"
                ].mean()
                - merged[
                    "mean_energy_mJ_reference"
                ].mean()
            ),
            "energy_paired_ttest_p": float(
                ttest_rel(
                    merged[
                        "mean_energy_mJ_ca"
                    ],
                    merged[
                        "mean_energy_mJ_reference"
                    ],
                ).pvalue
            ),
        }
    )

paired_df = pd.DataFrame(
    paired_records
)

ca_summary_row = method_summary_df[
    method_summary_df["method"]
    == "CA-MADDPG-V2 three-seed mean"
].iloc[0]

maddpg_summary_row = method_summary_df[
    method_summary_df["method"]
    == "MADDPG three-seed mean"
].iloc[0]

heuristic_summary_row = method_summary_df[
    method_summary_df["method"]
    == "Latency heuristic"
].iloc[0]

ca_success = float(
    ca_summary_row[
        "success_rate_percent"
    ]
)

maddpg_success = float(
    maddpg_summary_row[
        "success_rate_percent"
    ]
)

heuristic_success = float(
    heuristic_summary_row[
        "success_rate_percent"
    ]
)

maddpg_test_row = paired_df[
    paired_df["comparison"].str.endswith(
        "MADDPG three-seed mean"
    )
].iloc[0]

heuristic_test_row = paired_df[
    paired_df["comparison"].str.endswith(
        "Latency heuristic"
    )
].iloc[0]

decision = {
    "ca_maddpg_v2_success_rate_percent": (
        ca_success
    ),
    "maddpg_three_seed_success_rate_percent": (
        maddpg_success
    ),
    "latency_heuristic_success_rate_percent": (
        heuristic_success
    ),
    "crossed_maddpg_three_seed_mean": bool(
        ca_success > maddpg_success
    ),
    "significantly_crossed_maddpg": bool(
        ca_success > maddpg_success
        and maddpg_test_row[
            "success_paired_ttest_p"
        ] < 0.05
    ),
    "crossed_latency_heuristic": bool(
        ca_success > heuristic_success
    ),
    "significantly_crossed_heuristic": bool(
        ca_success > heuristic_success
        and heuristic_test_row[
            "success_paired_ttest_p"
        ] < 0.05
    ),
}

seed_episode_output = (
    OUTPUT_DIRECTORY
    / "ca_v2_three_seed_episode_results.csv"
)

per_seed_summary_output = (
    OUTPUT_DIRECTORY
    / "ca_v2_per_seed_summary.csv"
)

three_seed_mean_output = (
    OUTPUT_DIRECTORY
    / "ca_v2_three_seed_episode_mean.csv"
)

method_summary_output = (
    OUTPUT_DIRECTORY
    / "ca_v2_final_method_summary.csv"
)

paired_output = (
    OUTPUT_DIRECTORY
    / "ca_v2_final_paired_tests.csv"
)

checkpoint_output = (
    OUTPUT_DIRECTORY
    / "ca_v2_selected_checkpoints.csv"
)

decision_output = (
    OUTPUT_DIRECTORY
    / "ca_v2_benchmark_decision.json"
)

seed_episode_df.to_csv(
    seed_episode_output,
    index=False,
)

per_seed_summary_df.to_csv(
    per_seed_summary_output,
    index=False,
)

three_seed_episode_df.to_csv(
    three_seed_mean_output,
    index=False,
)

method_summary_df.to_csv(
    method_summary_output,
    index=False,
)

paired_df.to_csv(
    paired_output,
    index=False,
)

pd.DataFrame(
    checkpoint_records
).to_csv(
    checkpoint_output,
    index=False,
)

with decision_output.open(
    "w",
    encoding="utf-8",
) as file:
    json.dump(
        decision,
        file,
        indent=2,
    )

print()
print("=" * 120)
print("CA-MADDPG-V2 PER-SEED FIXED RESULTS")
print("=" * 120)

print(
    per_seed_summary_df[
        [
            "training_seed",
            "saved_episode",
            "mean_reward",
            "success_rate_percent",
            "success_std_percent",
            "mean_latency_ms",
            "mean_energy_mJ",
        ]
    ].to_string(index=False)
)

print()
print("=" * 120)
print("OFFICIAL THREE-SEED METHOD COMPARISON")
print("=" * 120)

print(
    method_summary_df[
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
print("=" * 120)
print("PAIRED STATISTICAL TESTS")
print("=" * 120)

print(
    paired_df.to_string(
        index=False
    )
)

print()
print("=" * 120)
print("BENCHMARK DECISION")
print("=" * 120)

for key, value in decision.items():
    print(f"{key}: {value}")

print()
print("=" * 120)
print("Saved per-seed episodes:", seed_episode_output)
print("Saved per-seed summary:", per_seed_summary_output)
print("Saved three-seed mean:", three_seed_mean_output)
print("Saved method summary:", method_summary_output)
print("Saved paired tests:", paired_output)
print("Saved checkpoints:", checkpoint_output)
print("Saved decision:", decision_output)
print("=" * 120)
