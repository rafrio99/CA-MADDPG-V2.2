from pathlib import Path
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, "src")

from uav_iot_env import UAVIOTParallelEnv


CONFIG_PATH = (
    "configs/ca_maddpg_v2_seed42_10ep.yaml"
)

BASELINE_PATH = Path(
    "results/robust_baseline_episode_comparison.csv"
)

OUTPUT_DIRECTORY = Path(
    "results/fixed_uav_resource_diagnostic"
)

RESOURCE_LEVELS = [
    0.75,
    0.90,
    0.98,
]

NUMBER_OF_TEST_EPISODES = 30


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
    ].astype(int).unique()
)[:NUMBER_OF_TEST_EPISODES]

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

if action_dimension != 3:
    raise RuntimeError(
        "Expected action dimension 3, "
        f"received {action_dimension}."
    )

episode_records = []

print("=" * 100)
print("FIXED-UAV RESOURCE-PROFILE DIAGNOSTIC")
print("=" * 100)
print("UAVs:", number_of_agents)
print(
    "Fixed test episodes per profile:",
    len(test_seeds),
)
print(
    "Resource levels:",
    RESOURCE_LEVELS,
)
print(
    "Total profiles:",
    number_of_agents
    * len(RESOURCE_LEVELS),
)
print("=" * 100)

for resource_level in RESOURCE_LEVELS:
    for selected_index, selected_agent in enumerate(
        possible_agents
    ):
        print(
            f"Evaluating {selected_agent} | "
            f"CPU={resource_level:.2f} | "
            f"Bandwidth={resource_level:.2f}"
        )

        for test_seed in test_seeds:
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

                # One-hot UAV selection.
                joint_actions[
                    selected_index,
                    0,
                ] = 1.0

                # Equal resource profile is assigned to
                # every UAV, although only the selected
                # UAV should be executed by the environment.
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
                    "test_seed": int(
                        test_seed
                    ),
                    "selected_uav": (
                        selected_agent
                    ),
                    "resource_level": float(
                        resource_level
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

episode_df = pd.DataFrame(
    episode_records
)

summary_df = (
    episode_df.groupby(
        [
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
        success_rate=(
            "success_rate",
            "mean",
        ),
        success_std=(
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

summary_df[
    "success_rate_percent"
] = (
    summary_df["success_rate"]
    * 100.0
)

summary_df[
    "success_std_percent"
] = (
    summary_df["success_std"]
    * 100.0
)

summary_df = summary_df.sort_values(
    by=[
        "success_rate_percent",
        "mean_reward",
    ],
    ascending=[
        False,
        False,
    ],
).reset_index(drop=True)

episode_output = (
    OUTPUT_DIRECTORY
    / "fixed_uav_episode_results.csv"
)

summary_output = (
    OUTPUT_DIRECTORY
    / "fixed_uav_resource_summary.csv"
)

episode_df.to_csv(
    episode_output,
    index=False,
)

summary_df.to_csv(
    summary_output,
    index=False,
)

print()
print("=" * 120)
print("FIXED-UAV RESOURCE-PROFILE RANKING")
print("=" * 120)

print(
    summary_df[
        [
            "selected_uav",
            "resource_level",
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
print("BEST PROFILE FOR EACH UAV")
print("=" * 120)

best_per_uav = (
    summary_df.sort_values(
        by=[
            "selected_uav",
            "success_rate_percent",
            "mean_reward",
        ],
        ascending=[
            True,
            False,
            False,
        ],
    )
    .groupby(
        "selected_uav",
        as_index=False,
    )
    .first()
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
)

print(
    best_per_uav[
        [
            "selected_uav",
            "resource_level",
            "mean_reward",
            "success_rate_percent",
            "mean_latency_ms",
            "mean_energy_mJ",
        ]
    ].to_string(index=False)
)

print()
print("=" * 120)
print("DIRECT U1 VS U2 COMPARISON")
print("=" * 120)

u1_u2_df = summary_df[
    summary_df["selected_uav"].isin(
        [
            "uav_U1",
            "uav_U2",
        ]
    )
]

print(
    u1_u2_df[
        [
            "selected_uav",
            "resource_level",
            "mean_reward",
            "success_rate_percent",
            "mean_latency_ms",
            "mean_energy_mJ",
        ]
    ].to_string(index=False)
)

print()
print("=" * 120)
print(
    "Saved episode results:",
    episode_output,
)
print(
    "Saved summary:",
    summary_output,
)
print("=" * 120)
