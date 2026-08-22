from collections import Counter
from pathlib import Path
import sys

import numpy as np
import pandas as pd
import torch

sys.path.insert(0, "src")

from ca_maddpg_v2 import CAMADDPGV2
from uav_iot_env import UAVIOTParallelEnv


BASELINE_FILE = Path(
    "results/robust_baseline_episode_comparison.csv"
)

COMMON_ENV_CONFIG = (
    "configs/ca_maddpg_v2_seed42_10ep.yaml"
)

MODELS = [
    {
        "training_seed": 42,
        "config": "configs/ca_maddpg_v2_seed42_10ep.yaml",
        "checkpoint": (
            "checkpoints/ca_maddpg_v2_seed42_10ep/"
            "ca_maddpg_v2_best.pt"
        ),
    },
    {
        "training_seed": 43,
        "config": "configs/ca_maddpg_v2_seed43_10ep.yaml",
        "checkpoint": (
            "checkpoints/ca_maddpg_v2_seed43_10ep/"
            "ca_maddpg_v2_best.pt"
        ),
    },
    {
        "training_seed": 44,
        "config": "configs/ca_maddpg_v2_seed44_10ep.yaml",
        "checkpoint": (
            "checkpoints/ca_maddpg_v2_seed44_10ep/"
            "ca_maddpg_v2_best.pt"
        ),
    },
]

OUTPUT_DIRECTORY = Path(
    "results/ca_maddpg_v2_seed_diagnostic"
)

OUTPUT_DIRECTORY.mkdir(
    parents=True,
    exist_ok=True,
)


def stack_observations(
    observations,
    agents,
    dimension,
):
    zero_observation = np.zeros(
        dimension,
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


def load_model(information):
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
        config_path=information["config"],
    )

    environment.close()

    checkpoint = model.load_checkpoint(
        information["checkpoint"]
    )

    return model, int(checkpoint["episode"])


baseline = pd.read_csv(
    BASELINE_FILE
)

test_seeds = sorted(
    baseline["test_seed"]
    .astype(int)
    .unique()
)[:30]

step_records = []
summary_records = []

for information in MODELS:
    training_seed = information[
        "training_seed"
    ]

    model, saved_episode = load_model(
        information
    )

    selected_agents = []
    successes = []
    rewards = []
    latencies = []
    energies = []
    entropies = []
    gaps = []
    cpu_allocations = []
    bandwidth_allocations = []

    print(
        f"Diagnosing Seed {training_seed} "
        f"from Episode {saved_episode}..."
    )

    for test_seed in test_seeds:
        environment = UAVIOTParallelEnv(
            config_path=COMMON_ENV_CONFIG
        )

        agents = list(
            environment.possible_agents
        )

        observations, _ = environment.reset(
            seed=int(test_seed)
        )

        while environment.agents:
            joint_observations = (
                stack_observations(
                    observations,
                    agents,
                    environment.observation_dimension,
                )
            )

            actions = model.select_actions(
                joint_observations,
                noise_scale=0.0,
            )

            probabilities = np.clip(
                actions[:, 0],
                1e-12,
                1.0,
            )

            probabilities /= (
                probabilities.sum()
            )

            selected_index = int(
                np.argmax(probabilities)
            )

            selected_agent = agents[
                selected_index
            ]

            sorted_probabilities = np.sort(
                probabilities
            )

            entropy = float(
                -np.sum(
                    probabilities
                    * np.log(probabilities)
                )
                / np.log(len(probabilities))
            )

            gap = float(
                sorted_probabilities[-1]
                - sorted_probabilities[-2]
            )

            action_dictionary = {
                agent: actions[index]
                for index, agent in enumerate(
                    agents
                )
            }

            (
                next_observations,
                step_rewards,
                terminations,
                truncations,
                infos,
            ) = environment.step(
                action_dictionary
            )

            first_agent = agents[0]
            info = infos.get(
                first_agent,
                {},
            )

            selected_agents.append(
                selected_agent
            )

            successes.append(
                int(
                    info.get(
                        "success",
                        False,
                    )
                )
            )

            rewards.append(
                float(
                    step_rewards.get(
                        first_agent,
                        0.0,
                    )
                )
            )

            latencies.append(
                float(
                    info.get(
                        "latency_ms",
                        0.0,
                    )
                )
            )

            energies.append(
                float(
                    info.get(
                        "energy_mJ",
                        0.0,
                    )
                )
            )

            entropies.append(entropy)
            gaps.append(gap)

            cpu_allocations.append(
                float(
                    actions[
                        selected_index,
                        1,
                    ]
                )
            )

            bandwidth_allocations.append(
                float(
                    actions[
                        selected_index,
                        2,
                    ]
                )
            )

            step_records.append(
                {
                    "training_seed": training_seed,
                    "saved_episode": saved_episode,
                    "test_seed": int(test_seed),
                    "selected_agent": selected_agent,
                    "success": successes[-1],
                    "reward": rewards[-1],
                    "latency_ms": latencies[-1],
                    "energy_mJ": energies[-1],
                    "selection_entropy": entropy,
                    "selection_gap": gap,
                    "selected_cpu": (
                        cpu_allocations[-1]
                    ),
                    "selected_bandwidth": (
                        bandwidth_allocations[-1]
                    ),
                }
            )

            observations = next_observations

        environment.close()

    counts = Counter(
        selected_agents
    )

    dominant_agent, dominant_count = (
        counts.most_common(1)[0]
    )

    summary_records.append(
        {
            "training_seed": training_seed,
            "saved_episode": saved_episode,
            "dominant_agent": dominant_agent,
            "dominant_agent_share_percent": (
                100.0
                * dominant_count
                / len(selected_agents)
            ),
            "unique_agents_selected": len(
                counts
            ),
            "success_rate_percent": (
                100.0
                * np.mean(successes)
            ),
            "mean_reward": np.mean(
                rewards
            ),
            "mean_latency_ms": np.mean(
                latencies
            ),
            "mean_energy_mJ": np.mean(
                energies
            ),
            "mean_selection_entropy": (
                np.mean(entropies)
            ),
            "mean_selection_gap": np.mean(
                gaps
            ),
            "mean_selected_cpu": np.mean(
                cpu_allocations
            ),
            "mean_selected_bandwidth": (
                np.mean(
                    bandwidth_allocations
                )
            ),
        }
    )

    del model
    torch.cuda.empty_cache()


step_df = pd.DataFrame(
    step_records
)

summary_df = pd.DataFrame(
    summary_records
)

step_output = (
    OUTPUT_DIRECTORY
    / "ca_v2_seed_selection_steps.csv"
)

summary_output = (
    OUTPUT_DIRECTORY
    / "ca_v2_seed_selection_summary.csv"
)

step_df.to_csv(
    step_output,
    index=False,
)

summary_df.to_csv(
    summary_output,
    index=False,
)

print("=" * 130)
print("CA-MADDPG-V2 TRAINING-SEED SELECTION DIAGNOSTIC")
print("=" * 130)

print(
    summary_df.to_string(
        index=False
    )
)

print("=" * 130)
print("Saved step results:", step_output)
print("Saved summary:", summary_output)
print("=" * 130)
