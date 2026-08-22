from collections import Counter
from pathlib import Path
import sys

import numpy as np
import pandas as pd
import torch

sys.path.insert(0, "src")

from ca_maddpg_v2 import CAMADDPGV2
from ca_maddpg_v21 import CAMADDPGV21
from uav_iot_env import UAVIOTParallelEnv


ENV_CONFIG = "configs/ca_maddpg_v2_seed42_10ep.yaml"

V2_CONFIG = "configs/ca_maddpg_v2_seed42_10ep.yaml"
V2_CHECKPOINT = (
    "checkpoints/ca_maddpg_v2_seed42_10ep/"
    "ca_maddpg_v2_best.pt"
)

V21_CONFIG = "configs/ca_maddpg_v21_seed42_10ep.yaml"
V21_CHECKPOINT = (
    "checkpoints/ca_maddpg_v21_seed42_10ep/"
    "ca_maddpg_v21_best.pt"
)

BASELINE_FILE = Path(
    "results/robust_baseline_episode_comparison.csv"
)

OUTPUT_DIRECTORY = Path(
    "results/ca_v2_vs_v21_action_diagnostic"
)

OUTPUT_DIRECTORY.mkdir(
    parents=True,
    exist_ok=True,
)


def stack_observations(
    observation_dictionary,
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
                observation_dictionary.get(
                    agent,
                    zero_observation,
                ),
                dtype=np.float32,
            )
            for agent in agents
        ],
        axis=0,
    )


def create_models():
    environment = UAVIOTParallelEnv(
        config_path=ENV_CONFIG
    )

    number_of_agents = len(
        environment.possible_agents
    )

    observation_dimension = (
        environment.observation_dimension
    )

    action_dimension = (
        environment.action_dimension
    )

    environment.close()

    v2_model = CAMADDPGV2(
        number_of_agents=number_of_agents,
        observation_dimension=observation_dimension,
        action_dimension=action_dimension,
        config_path=V2_CONFIG,
    )

    v21_model = CAMADDPGV21(
        number_of_agents=number_of_agents,
        observation_dimension=observation_dimension,
        action_dimension=action_dimension,
        config_path=V21_CONFIG,
    )

    v2_checkpoint = v2_model.load_checkpoint(
        V2_CHECKPOINT
    )

    v21_checkpoint = v21_model.load_checkpoint(
        V21_CHECKPOINT
    )

    return (
        v2_model,
        v21_model,
        int(v2_checkpoint["episode"]),
        int(v21_checkpoint["episode"]),
    )


if not BASELINE_FILE.exists():
    raise FileNotFoundError(
        f"Missing baseline file: {BASELINE_FILE}"
    )

baseline = pd.read_csv(
    BASELINE_FILE
)

test_seeds = sorted(
    baseline["test_seed"]
    .astype(int)
    .unique()
)[:30]

(
    v2_model,
    v21_model,
    v2_saved_episode,
    v21_saved_episode,
) = create_models()

step_records = []

v2_agent_counter = Counter()
v21_agent_counter = Counter()

same_selection_steps = 0
total_steps = 0

print("=" * 86)
print("CA-MADDPG-V2 VS V2.1 ACTION DIAGNOSTIC")
print("=" * 86)
print("Test episodes:", len(test_seeds))
print("V2 saved episode:", v2_saved_episode)
print("V2.1 saved episode:", v21_saved_episode)
print("=" * 86)

for episode_index, test_seed in enumerate(
    test_seeds,
    start=1,
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

    step_index = 0

    while environment.agents:
        joint_observations = stack_observations(
            observations,
            agents,
            environment.observation_dimension,
        )

        v2_actions = v2_model.select_actions(
            joint_observations,
            noise_scale=0.0,
        )

        v21_actions = v21_model.select_actions(
            joint_observations,
            noise_scale=0.0,
        )

        v2_selected_index = int(
            np.argmax(v2_actions[:, 0])
        )

        v21_selected_index = int(
            np.argmax(v21_actions[:, 0])
        )

        v2_selected_agent = agents[
            v2_selected_index
        ]

        v21_selected_agent = agents[
            v21_selected_index
        ]

        same_selection = (
            v2_selected_agent
            == v21_selected_agent
        )

        same_selection_steps += int(
            same_selection
        )

        total_steps += 1

        v2_agent_counter[
            v2_selected_agent
        ] += 1

        v21_agent_counter[
            v21_selected_agent
        ] += 1

        v2_action_dictionary = {
            agent: v2_actions[index]
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
            v2_action_dictionary
        )

        first_agent = agents[0]

        information = infos.get(
            first_agent,
            {},
        )

        step_records.append(
            {
                "test_seed": int(test_seed),
                "step": step_index,
                "v2_selected_agent": (
                    v2_selected_agent
                ),
                "v21_selected_agent": (
                    v21_selected_agent
                ),
                "same_selected_agent": (
                    same_selection
                ),
                "v2_selected_cpu": float(
                    v2_actions[
                        v2_selected_index,
                        1,
                    ]
                ),
                "v21_selected_cpu": float(
                    v21_actions[
                        v21_selected_index,
                        1,
                    ]
                ),
                "v2_selected_bandwidth": float(
                    v2_actions[
                        v2_selected_index,
                        2,
                    ]
                ),
                "v21_selected_bandwidth": float(
                    v21_actions[
                        v21_selected_index,
                        2,
                    ]
                ),
                "v2_selection_probability": float(
                    v2_actions[
                        v2_selected_index,
                        0,
                    ]
                ),
                "v21_selection_probability": float(
                    v21_actions[
                        v21_selected_index,
                        0,
                    ]
                ),
                "v2_environment_success": int(
                    information.get(
                        "success",
                        False,
                    )
                ),
                "v2_environment_reward": float(
                    rewards.get(
                        first_agent,
                        0.0,
                    )
                ),
                "v2_environment_latency_ms": float(
                    information.get(
                        "latency_ms",
                        0.0,
                    )
                ),
                "v2_environment_energy_mJ": float(
                    information.get(
                        "energy_mJ",
                        0.0,
                    )
                ),
            }
        )

        observations = next_observations
        step_index += 1

    environment.close()

    if episode_index % 10 == 0:
        print(
            f"Completed {episode_index}/"
            f"{len(test_seeds)} episodes"
        )

step_df = pd.DataFrame(
    step_records
)

summary = {
    "test_episodes": len(test_seeds),
    "total_steps": total_steps,
    "same_selected_agent_percent": (
        100.0
        * same_selection_steps
        / max(total_steps, 1)
    ),
    "v2_dominant_agent": (
        v2_agent_counter.most_common(1)[0][0]
    ),
    "v2_dominant_share_percent": (
        100.0
        * v2_agent_counter.most_common(1)[0][1]
        / max(total_steps, 1)
    ),
    "v21_dominant_agent": (
        v21_agent_counter.most_common(1)[0][0]
    ),
    "v21_dominant_share_percent": (
        100.0
        * v21_agent_counter.most_common(1)[0][1]
        / max(total_steps, 1)
    ),
    "v2_unique_agents_selected": len(
        v2_agent_counter
    ),
    "v21_unique_agents_selected": len(
        v21_agent_counter
    ),
    "v2_mean_selected_cpu": float(
        step_df["v2_selected_cpu"].mean()
    ),
    "v21_mean_selected_cpu": float(
        step_df["v21_selected_cpu"].mean()
    ),
    "v2_mean_selected_bandwidth": float(
        step_df[
            "v2_selected_bandwidth"
        ].mean()
    ),
    "v21_mean_selected_bandwidth": float(
        step_df[
            "v21_selected_bandwidth"
        ].mean()
    ),
}

summary_df = pd.DataFrame(
    [summary]
)

step_output = (
    OUTPUT_DIRECTORY
    / "v2_vs_v21_action_steps.csv"
)

summary_output = (
    OUTPUT_DIRECTORY
    / "v2_vs_v21_action_summary.csv"
)

step_df.to_csv(
    step_output,
    index=False,
)

summary_df.to_csv(
    summary_output,
    index=False,
)

print()
print("=" * 105)
print("V2 VS V2.1 ACTION SUMMARY")
print("=" * 105)
print(
    summary_df.to_string(
        index=False
    )
)

print()
print("=" * 105)
print("V2 AGENT SELECTION COUNTS")
print("=" * 105)
print(dict(v2_agent_counter))

print()
print("=" * 105)
print("V2.1 AGENT SELECTION COUNTS")
print("=" * 105)
print(dict(v21_agent_counter))

print()
print("=" * 105)
print("Saved step results:", step_output)
print("Saved summary:", summary_output)
print("=" * 105)
