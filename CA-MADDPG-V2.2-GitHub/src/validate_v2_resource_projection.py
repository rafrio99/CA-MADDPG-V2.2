from __future__ import annotations

import gc
import json
import sys
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd
import torch

sys.path.insert(0, "src")

from ca_maddpg_v2 import CAMADDPGV2
from uav_iot_env import UAVIOTParallelEnv


ENV_CONFIG = (
    "configs/ca_maddpg_v2_seed42_10ep.yaml"
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

RESOURCE_FLOORS = [
    0.50,
    0.75,
    0.85,
    0.90,
    0.95,
    0.98,
]

# These seeds are separate from the fixed 100-episode
# benchmark seeds and are used only for model selection.
VALIDATION_SEEDS = list(
    range(710000, 710050)
)

OUTPUT_DIRECTORY = Path(
    "results/v2_resource_projection_validation"
)

OUTPUT_DIRECTORY.mkdir(
    parents=True,
    exist_ok=True,
)


def stack_observations(
    observations,
    agents,
    observation_dimension,
):
    zero_observation = np.zeros(
        observation_dimension,
        dtype=np.float32,
    )

    return np.stack(
        [
            np.asarray(
                observations.get(
                    agent,
                    zero_observation,
                ),
                dtype=np.float32,
            )
            for agent in agents
        ],
        axis=0,
    )


def load_model(
    config_path,
    checkpoint_path,
):
    environment = UAVIOTParallelEnv(
        config_path=ENV_CONFIG
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

    return model, int(
        checkpoint["episode"]
    )


def evaluate_episode(
    model,
    test_seed,
    resource_floor,
):
    environment = UAVIOTParallelEnv(
        config_path=ENV_CONFIG
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
    selected_agents = []
    selected_cpu_values = []
    selected_bandwidth_values = []
    steps = 0

    while environment.agents:
        joint_observations = (
            stack_observations(
                observations,
                agents,
                environment.observation_dimension,
            )
        )

        # Preserve the original V2 selection policy.
        actions = model.select_actions(
            joint_observations,
            noise_scale=0.0,
        )

        selected_index = int(
            np.argmax(actions[:, 0])
        )

        selected_agent = agents[
            selected_index
        ]

        # Safety/resource projection changes only CPU and
        # bandwidth. Selection probabilities are untouched.
        actions[:, 1] = np.maximum(
            actions[:, 1],
            resource_floor,
        )

        actions[:, 2] = np.maximum(
            actions[:, 2],
            resource_floor,
        )

        selected_agents.append(
            selected_agent
        )

        selected_cpu_values.append(
            float(
                actions[
                    selected_index,
                    1,
                ]
            )
        )

        selected_bandwidth_values.append(
            float(
                actions[
                    selected_index,
                    2,
                ]
            )
        )

        action_dictionary = {
            agent: actions[index]
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

    counts = Counter(
        selected_agents
    )

    dominant_agent, dominant_count = (
        counts.most_common(1)[0]
    )

    return {
        "reward": float(
            total_reward
        ),
        "success_rate": float(
            successes / max(steps, 1)
        ),
        "mean_latency_ms": float(
            np.mean(latencies)
        ),
        "mean_energy_mJ": float(
            np.mean(energies)
        ),
        "mean_selected_cpu": float(
            np.mean(selected_cpu_values)
        ),
        "mean_selected_bandwidth": float(
            np.mean(
                selected_bandwidth_values
            )
        ),
        "dominant_agent": dominant_agent,
        "dominant_agent_share_percent": float(
            100.0
            * dominant_count
            / max(len(selected_agents), 1)
        ),
        "unique_agents_selected": int(
            len(counts)
        ),
    }


for information in MODEL_INFORMATION:
    checkpoint_path = Path(
        information["checkpoint"]
    )

    if not checkpoint_path.exists():
        raise FileNotFoundError(
            f"Missing checkpoint: "
            f"{checkpoint_path}"
        )

print("=" * 100)
print("CA-MADDPG-V2 HELD-OUT RESOURCE-PROJECTION VALIDATION")
print("=" * 100)
print("Training seeds: 42, 43, 44")
print(
    "Validation episodes per model and floor:",
    len(VALIDATION_SEEDS),
)
print("Resource floors:", RESOURCE_FLOORS)
print(
    "Total validation episodes:",
    len(MODEL_INFORMATION)
    * len(RESOURCE_FLOORS)
    * len(VALIDATION_SEEDS),
)
print(
    "Fixed benchmark test seeds are not used."
)
print("=" * 100)

episode_records = []
checkpoint_records = []

for information in MODEL_INFORMATION:
    training_seed = int(
        information["training_seed"]
    )

    model, saved_episode = load_model(
        config_path=information["config"],
        checkpoint_path=(
            information["checkpoint"]
        ),
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

    for resource_floor in RESOURCE_FLOORS:
        print(
            f"Evaluating training Seed "
            f"{training_seed} | "
            f"floor={resource_floor:.2f}"
        )

        for episode_index, validation_seed in enumerate(
            VALIDATION_SEEDS,
            start=1,
        ):
            result = evaluate_episode(
                model=model,
                test_seed=validation_seed,
                resource_floor=resource_floor,
            )

            result.update(
                {
                    "training_seed": (
                        training_seed
                    ),
                    "saved_episode": (
                        saved_episode
                    ),
                    "validation_seed": int(
                        validation_seed
                    ),
                    "resource_floor": float(
                        resource_floor
                    ),
                }
            )

            episode_records.append(
                result
            )

            if episode_index % 25 == 0:
                print(
                    f"  Completed "
                    f"{episode_index}/"
                    f"{len(VALIDATION_SEEDS)}"
                )

    del model
    gc.collect()
    torch.cuda.empty_cache()

episode_df = pd.DataFrame(
    episode_records
)

per_seed_summary_df = (
    episode_df.groupby(
        [
            "training_seed",
            "saved_episode",
            "resource_floor",
        ]
    )
    .agg(
        validation_episodes=(
            "validation_seed",
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
        mean_selected_cpu=(
            "mean_selected_cpu",
            "mean",
        ),
        mean_selected_bandwidth=(
            "mean_selected_bandwidth",
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

overall_summary_df = (
    episode_df.groupby(
        "resource_floor"
    )
    .agg(
        total_validation_episodes=(
            "validation_seed",
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
        mean_selected_cpu=(
            "mean_selected_cpu",
            "mean",
        ),
        mean_selected_bandwidth=(
            "mean_selected_bandwidth",
            "mean",
        ),
    )
    .reset_index()
)

overall_summary_df[
    "success_rate_percent"
] = (
    overall_summary_df[
        "mean_success_rate"
    ] * 100.0
)

overall_summary_df[
    "success_std_percent"
] = (
    overall_summary_df[
        "std_success_rate"
    ] * 100.0
)

ranking_df = overall_summary_df.sort_values(
    by=[
        "success_rate_percent",
        "mean_reward",
        "mean_latency_ms",
        "mean_energy_mJ",
    ],
    ascending=[
        False,
        False,
        True,
        True,
    ],
).reset_index(drop=True)

best_row = ranking_df.iloc[0]

selected_floor = float(
    best_row["resource_floor"]
)

decision = {
    "selection_dataset": (
        "held_out_validation_seeds_710000_to_710049"
    ),
    "fixed_test_seeds_used_for_selection": False,
    "selected_resource_floor": (
        selected_floor
    ),
    "validation_success_rate_percent": float(
        best_row[
            "success_rate_percent"
        ]
    ),
    "validation_mean_reward": float(
        best_row["mean_reward"]
    ),
    "validation_mean_latency_ms": float(
        best_row["mean_latency_ms"]
    ),
    "validation_mean_energy_mJ": float(
        best_row["mean_energy_mJ"]
    ),
    "model_definition": (
        "CA-MADDPG-V2 with inference-time "
        "resource safety projection"
    ),
}

episode_output = (
    OUTPUT_DIRECTORY
    / "v2_resource_projection_validation_episodes.csv"
)

per_seed_output = (
    OUTPUT_DIRECTORY
    / "v2_resource_projection_per_seed_summary.csv"
)

overall_output = (
    OUTPUT_DIRECTORY
    / "v2_resource_projection_overall_summary.csv"
)

checkpoint_output = (
    OUTPUT_DIRECTORY
    / "v2_resource_projection_checkpoints.csv"
)

decision_output = (
    OUTPUT_DIRECTORY
    / "v2_resource_projection_decision.json"
)

episode_df.to_csv(
    episode_output,
    index=False,
)

per_seed_summary_df.to_csv(
    per_seed_output,
    index=False,
)

overall_summary_df.to_csv(
    overall_output,
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
print("PER-SEED VALIDATION SUMMARY")
print("=" * 125)

print(
    per_seed_summary_df[
        [
            "training_seed",
            "resource_floor",
            "mean_reward",
            "success_rate_percent",
            "success_std_percent",
            "mean_latency_ms",
            "mean_energy_mJ",
            "mean_selected_cpu",
            "mean_selected_bandwidth",
        ]
    ].to_string(index=False)
)

print()
print("=" * 125)
print("OVERALL RESOURCE-FLOOR RANKING")
print("=" * 125)

print(
    ranking_df[
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
    ].to_string(index=False)
)

print()
print("=" * 125)
print("VALIDATION DECISION")
print("=" * 125)

for key, value in decision.items():
    print(f"{key}: {value}")

print()
print("=" * 125)
print("Saved episode results:", episode_output)
print("Saved per-seed summary:", per_seed_output)
print("Saved overall summary:", overall_output)
print("Saved checkpoints:", checkpoint_output)
print("Saved decision:", decision_output)
print("=" * 125)
