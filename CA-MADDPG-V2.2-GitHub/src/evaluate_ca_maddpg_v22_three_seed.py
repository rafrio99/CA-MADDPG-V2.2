from __future__ import annotations

import gc
import json
import sys
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from scipy.stats import ttest_rel, wilcoxon

sys.path.insert(0, "src")

from ca_maddpg_v22 import CAMADDPGV22
from uav_iot_env import UAVIOTParallelEnv


COMMON_ENV_CONFIG = (
    "configs/ca_maddpg_v2_seed42_10ep.yaml"
)

RESOURCE_FLOOR = 0.98

ROBUST_BASELINE_PATH = Path(
    "results/robust_baseline_episode_comparison.csv"
)

ORIGINAL_V2_PATH = Path(
    "results/ca_maddpg_v2_three_seed/"
    "ca_v2_three_seed_episode_mean.csv"
)

OUTPUT_DIRECTORY = Path(
    "results/ca_maddpg_v22_three_seed"
)

MODEL_INFORMATION = [
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


def safe_paired_ttest(
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
        ttest_rel(
            first,
            second,
        ).pvalue
    )


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


def paired_cohen_dz(
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


def load_model(
    config_path,
    checkpoint_path,
):
    environment = UAVIOTParallelEnv(
        config_path=COMMON_ENV_CONFIG
    )

    model = CAMADDPGV22(
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
        resource_floor=RESOURCE_FLOOR,
    )

    environment.close()

    checkpoint = model.load_checkpoint(
        checkpoint_path
    )

    return model, int(
        checkpoint["episode"]
    )


def evaluate_model(
    training_seed,
    config_path,
    checkpoint_path,
    test_seeds,
):
    model, saved_episode = load_model(
        config_path=config_path,
        checkpoint_path=checkpoint_path,
    )

    episode_records = []
    step_records = []

    print(
        f"Evaluating CA-MADDPG-V2.2 "
        f"training Seed {training_seed} "
        f"from Episode {saved_episode}..."
    )

    for episode_index, test_seed in enumerate(
        test_seeds,
        start=1,
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

            actions = model.select_actions(
                joint_observations,
                noise_scale=0.0,
            )

            selected_index = int(
                np.argmax(
                    actions[:, 0]
                )
            )

            selected_agent = (
                possible_agents[
                    selected_index
                ]
            )

            selected_cpu = float(
                actions[
                    selected_index,
                    1,
                ]
            )

            selected_bandwidth = float(
                actions[
                    selected_index,
                    2,
                ]
            )

            action_dictionary = {
                agent: actions[index]
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

            reward = float(
                rewards.get(
                    first_agent,
                    0.0,
                )
            )

            success = int(
                information.get(
                    "success",
                    False,
                )
            )

            latency = float(
                information.get(
                    "latency_ms",
                    0.0,
                )
            )

            energy = float(
                information.get(
                    "energy_mJ",
                    0.0,
                )
            )

            total_reward += reward
            successes += success
            latencies.append(latency)
            energies.append(energy)

            step_records.append(
                {
                    "training_seed": (
                        training_seed
                    ),
                    "saved_episode": (
                        saved_episode
                    ),
                    "test_seed": int(
                        test_seed
                    ),
                    "step": steps,
                    "selected_agent": (
                        selected_agent
                    ),
                    "selected_cpu": (
                        selected_cpu
                    ),
                    "selected_bandwidth": (
                        selected_bandwidth
                    ),
                    "selection_probability": float(
                        actions[
                            selected_index,
                            0,
                        ]
                    ),
                    "reward": reward,
                    "success": success,
                    "latency_ms": latency,
                    "energy_mJ": energy,
                }
            )

            observations = next_observations
            steps += 1

        environment.close()

        episode_records.append(
            {
                "training_seed": (
                    training_seed
                ),
                "saved_episode": (
                    saved_episode
                ),
                "test_seed": int(
                    test_seed
                ),
                "method": (
                    f"CA-MADDPG-V2.2 "
                    f"Seed {training_seed}"
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

    del model
    gc.collect()
    torch.cuda.empty_cache()

    return (
        pd.DataFrame(
            episode_records
        ),
        pd.DataFrame(
            step_records
        ),
    )


if not ROBUST_BASELINE_PATH.exists():
    raise FileNotFoundError(
        f"Missing baseline file: "
        f"{ROBUST_BASELINE_PATH}"
    )

for information in MODEL_INFORMATION:
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

print("=" * 100)
print("CA-MADDPG-V2.2 OFFICIAL THREE-SEED BENCHMARK")
print("=" * 100)
print("Training seeds: 42, 43, 44")
print("Resource floor:", RESOURCE_FLOOR)
print("Fixed test episodes per model:", len(test_seeds))
print(
    "Total model evaluation episodes:",
    len(test_seeds)
    * len(MODEL_INFORMATION),
)
print(
    "Resource floor was selected using "
    "held-out validation seeds."
)
print("=" * 100)

episode_frames = []
step_frames = []
checkpoint_records = []

for information in MODEL_INFORMATION:
    episode_frame, step_frame = evaluate_model(
        training_seed=int(
            information["training_seed"]
        ),
        config_path=(
            information["config"]
        ),
        checkpoint_path=(
            information["checkpoint"]
        ),
        test_seeds=test_seeds,
    )

    episode_frames.append(
        episode_frame
    )

    step_frames.append(
        step_frame
    )

    checkpoint_records.append(
        {
            "training_seed": (
                information[
                    "training_seed"
                ]
            ),
            "checkpoint": (
                information[
                    "checkpoint"
                ]
            ),
            "resource_floor": (
                RESOURCE_FLOOR
            ),
        }
    )

v22_episode_df = pd.concat(
    episode_frames,
    ignore_index=True,
)

v22_step_df = pd.concat(
    step_frames,
    ignore_index=True,
)

per_seed_summary_df = (
    v22_episode_df.groupby(
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
        mean_energy_mJ=(
            "mean_energy_mJ",
            "mean",
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
    v22_episode_df.groupby(
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
    "CA-MADDPG-V2.2 three-seed mean"
)

action_records = []

for training_seed, seed_steps in (
    v22_step_df.groupby(
        "training_seed"
    )
):
    selection_counts = Counter(
        seed_steps[
            "selected_agent"
        ].tolist()
    )

    dominant_agent, dominant_count = (
        selection_counts.most_common(1)[0]
    )

    action_records.append(
        {
            "training_seed": int(
                training_seed
            ),
            "dominant_agent": (
                dominant_agent
            ),
            "dominant_agent_share_percent": float(
                100.0
                * dominant_count
                / len(seed_steps)
            ),
            "unique_agents_selected": int(
                len(selection_counts)
            ),
            "mean_selected_cpu": float(
                seed_steps[
                    "selected_cpu"
                ].mean()
            ),
            "mean_selected_bandwidth": float(
                seed_steps[
                    "selected_bandwidth"
                ].mean()
            ),
            "minimum_selected_cpu": float(
                seed_steps[
                    "selected_cpu"
                ].min()
            ),
            "minimum_selected_bandwidth": float(
                seed_steps[
                    "selected_bandwidth"
                ].min()
            ),
        }
    )

action_summary_df = pd.DataFrame(
    action_records
)

reference_frames = [
    baseline_df[
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
]

if ORIGINAL_V2_PATH.exists():
    original_v2_df = pd.read_csv(
        ORIGINAL_V2_PATH
    )

    required_v2_columns = {
        "test_seed",
        "reward",
        "success_rate",
        "mean_latency_ms",
        "mean_energy_mJ",
    }

    missing_v2_columns = (
        required_v2_columns
        - set(
            original_v2_df.columns
        )
    )

    if missing_v2_columns:
        raise RuntimeError(
            "Original V2 result file is "
            "missing columns: "
            f"{sorted(missing_v2_columns)}"
        )

    original_v2_df = original_v2_df[
        [
            "test_seed",
            "reward",
            "success_rate",
            "mean_latency_ms",
            "mean_energy_mJ",
        ]
    ].copy()

    original_v2_df["method"] = (
        "CA-MADDPG-V2 three-seed mean"
    )

    reference_frames.append(
        original_v2_df
    )

reference_df = pd.concat(
    reference_frames,
    ignore_index=True,
)

combined_df = pd.concat(
    [
        reference_df,
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

paired_records = []

v22_reference = (
    three_seed_episode_df.sort_values(
        "test_seed"
    )
)

for reference_method in (
    reference_df[
        "method"
    ].unique()
):
    reference_method_df = (
        reference_df[
            reference_df["method"]
            == reference_method
        ]
        .sort_values(
            "test_seed"
        )
    )

    merged = v22_reference.merge(
        reference_method_df,
        on="test_seed",
        suffixes=(
            "_v22",
            "_reference",
        ),
    )

    paired_records.append(
        {
            "comparison": (
                "CA-MADDPG-V2.2 "
                "three-seed mean vs "
                f"{reference_method}"
            ),
            "success_difference_percentage_points": float(
                100.0
                * (
                    merged[
                        "success_rate_v22"
                    ].mean()
                    - merged[
                        "success_rate_reference"
                    ].mean()
                )
            ),
            "success_paired_ttest_p": (
                safe_paired_ttest(
                    merged[
                        "success_rate_v22"
                    ],
                    merged[
                        "success_rate_reference"
                    ],
                )
            ),
            "success_wilcoxon_p": (
                safe_wilcoxon(
                    merged[
                        "success_rate_v22"
                    ],
                    merged[
                        "success_rate_reference"
                    ],
                )
            ),
            "success_cohen_dz": (
                paired_cohen_dz(
                    merged[
                        "success_rate_v22"
                    ],
                    merged[
                        "success_rate_reference"
                    ],
                )
            ),
            "reward_difference": float(
                merged[
                    "reward_v22"
                ].mean()
                - merged[
                    "reward_reference"
                ].mean()
            ),
            "reward_paired_ttest_p": (
                safe_paired_ttest(
                    merged[
                        "reward_v22"
                    ],
                    merged[
                        "reward_reference"
                    ],
                )
            ),
            "latency_difference_ms": float(
                merged[
                    "mean_latency_ms_v22"
                ].mean()
                - merged[
                    "mean_latency_ms_reference"
                ].mean()
            ),
            "latency_paired_ttest_p": (
                safe_paired_ttest(
                    merged[
                        "mean_latency_ms_v22"
                    ],
                    merged[
                        "mean_latency_ms_reference"
                    ],
                )
            ),
            "energy_difference_mJ": float(
                merged[
                    "mean_energy_mJ_v22"
                ].mean()
                - merged[
                    "mean_energy_mJ_reference"
                ].mean()
            ),
            "energy_paired_ttest_p": (
                safe_paired_ttest(
                    merged[
                        "mean_energy_mJ_v22"
                    ],
                    merged[
                        "mean_energy_mJ_reference"
                    ],
                )
            ),
        }
    )

paired_df = pd.DataFrame(
    paired_records
)

v22_summary_row = method_summary_df[
    method_summary_df["method"]
    == "CA-MADDPG-V2.2 three-seed mean"
].iloc[0]

maddpg_summary_row = method_summary_df[
    method_summary_df["method"]
    == "MADDPG three-seed mean"
].iloc[0]

heuristic_summary_row = method_summary_df[
    method_summary_df["method"]
    == "Latency heuristic"
].iloc[0]

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

v22_success = float(
    v22_summary_row[
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

decision = {
    "model": (
        "CA-MADDPG-V2.2"
    ),
    "resource_floor": (
        RESOURCE_FLOOR
    ),
    "resource_floor_selection_source": (
        "held-out validation"
    ),
    "fixed_test_used_for_floor_selection": (
        False
    ),
    "v22_success_rate_percent": (
        v22_success
    ),
    "maddpg_three_seed_success_percent": (
        maddpg_success
    ),
    "latency_heuristic_success_percent": (
        heuristic_success
    ),
    "crossed_maddpg_three_seed_mean": bool(
        v22_success
        > maddpg_success
    ),
    "significantly_crossed_maddpg": bool(
        v22_success
        > maddpg_success
        and maddpg_test_row[
            "success_paired_ttest_p"
        ] < 0.05
    ),
    "crossed_latency_heuristic": bool(
        v22_success
        > heuristic_success
    ),
    "significantly_crossed_heuristic": bool(
        v22_success
        > heuristic_success
        and heuristic_test_row[
            "success_paired_ttest_p"
        ] < 0.05
    ),
}

original_v2_rows = method_summary_df[
    method_summary_df["method"]
    == "CA-MADDPG-V2 three-seed mean"
]

if not original_v2_rows.empty:
    original_v2_success = float(
        original_v2_rows.iloc[0][
            "success_rate_percent"
        ]
    )

    decision[
        "original_v2_success_percent"
    ] = original_v2_success

    decision[
        "improved_over_original_v2"
    ] = bool(
        v22_success
        > original_v2_success
    )

episode_output = (
    OUTPUT_DIRECTORY
    / "v22_three_seed_episode_results.csv"
)

step_output = (
    OUTPUT_DIRECTORY
    / "v22_three_seed_action_steps.csv"
)

per_seed_output = (
    OUTPUT_DIRECTORY
    / "v22_per_seed_summary.csv"
)

three_seed_mean_output = (
    OUTPUT_DIRECTORY
    / "v22_three_seed_episode_mean.csv"
)

action_summary_output = (
    OUTPUT_DIRECTORY
    / "v22_action_summary.csv"
)

method_summary_output = (
    OUTPUT_DIRECTORY
    / "v22_official_method_summary.csv"
)

paired_output = (
    OUTPUT_DIRECTORY
    / "v22_official_paired_tests.csv"
)

checkpoint_output = (
    OUTPUT_DIRECTORY
    / "v22_selected_checkpoints.csv"
)

decision_output = (
    OUTPUT_DIRECTORY
    / "v22_benchmark_decision.json"
)

v22_episode_df.to_csv(
    episode_output,
    index=False,
)

v22_step_df.to_csv(
    step_output,
    index=False,
)

per_seed_summary_df.to_csv(
    per_seed_output,
    index=False,
)

three_seed_episode_df.to_csv(
    three_seed_mean_output,
    index=False,
)

action_summary_df.to_csv(
    action_summary_output,
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
print("=" * 125)
print("CA-MADDPG-V2.2 PER-SEED FIXED RESULTS")
print("=" * 125)

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
    ].to_string(
        index=False
    )
)

print()
print("=" * 125)
print("V2.2 UAV-SELECTION AND RESOURCE SUMMARY")
print("=" * 125)

print(
    action_summary_df.to_string(
        index=False
    )
)

print()
print("=" * 125)
print("OFFICIAL THREE-SEED METHOD COMPARISON")
print("=" * 125)

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
    .to_string(
        index=False
    )
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
print("BENCHMARK DECISION")
print("=" * 125)

for key, value in decision.items():
    print(f"{key}: {value}")

print()
print("=" * 125)
print("Saved per-seed episodes:", episode_output)
print("Saved action steps:", step_output)
print("Saved per-seed summary:", per_seed_output)
print("Saved three-seed mean:", three_seed_mean_output)
print("Saved action summary:", action_summary_output)
print("Saved method summary:", method_summary_output)
print("Saved paired tests:", paired_output)
print("Saved checkpoints:", checkpoint_output)
print("Saved benchmark decision:", decision_output)
print("=" * 125)
