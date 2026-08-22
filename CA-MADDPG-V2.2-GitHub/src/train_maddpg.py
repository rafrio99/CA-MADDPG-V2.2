from __future__ import annotations

import argparse
import csv
import json
import random
import sys
import time
from pathlib import Path
from typing import Dict

import numpy as np
import torch
import yaml

from maddpg import MADDPG
from maddpg_components import ReplayBuffer
from uav_iot_env import UAVIOTParallelEnv


ROOT = Path(__file__).resolve().parents[1]


def set_random_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def observations_to_array(
    observation_dict: Dict[str, np.ndarray],
    agents: list[str],
) -> np.ndarray:
    return np.stack(
        [observation_dict[agent] for agent in agents]
    ).astype(np.float32)


def evaluate_policy(
    maddpg: MADDPG,
    number_of_episodes: int,
    seed: int,
) -> Dict[str, float]:
    environment = UAVIOTParallelEnv()

    episode_rewards = []
    success_rates = []
    mean_latencies = []
    mean_energies = []

    for evaluation_episode in range(number_of_episodes):
        observation_dict, _ = environment.reset(
            seed=seed + evaluation_episode
        )

        episode_reward = 0.0
        episode_successes = 0
        episode_steps = 0
        episode_latencies = []
        episode_energies = []

        while environment.agents:
            observation_array = observations_to_array(
                observation_dict,
                environment.possible_agents,
            )

            action_array = maddpg.select_actions(
                observation_array,
                noise_scale=0.0,
            )

            action_dict = {
                agent: action_array[index]
                for index, agent in enumerate(
                    environment.possible_agents
                )
            }

            (
                next_observation_dict,
                reward_dict,
                termination_dict,
                truncation_dict,
                info_dict,
            ) = environment.step(action_dict)

            first_agent = environment.possible_agents[0]
            information = info_dict[first_agent]

            episode_reward += reward_dict[first_agent]
            episode_successes += int(
                information["success"]
            )
            episode_steps += 1

            episode_latencies.append(
                information["latency_ms"]
            )
            episode_energies.append(
                information["energy_mJ"]
            )

            observation_dict = next_observation_dict

        episode_rewards.append(episode_reward)

        success_rates.append(
            episode_successes / max(episode_steps, 1)
        )

        mean_latencies.append(
            float(np.mean(episode_latencies))
        )

        mean_energies.append(
            float(np.mean(episode_energies))
        )

    environment.close()

    return {
        "evaluation_reward": float(
            np.mean(episode_rewards)
        ),
        "evaluation_success_rate": float(
            np.mean(success_rates)
        ),
        "evaluation_latency_ms": float(
            np.mean(mean_latencies)
        ),
        "evaluation_energy_mJ": float(
            np.mean(mean_energies)
        ),
    }


def save_training_row(
    csv_path: Path,
    row: Dict[str, object],
) -> None:
    file_exists = csv_path.exists()

    with csv_path.open(
        "a",
        newline="",
        encoding="utf-8",
    ) as file:
        writer = csv.DictWriter(
            file,
            fieldnames=list(row.keys()),
        )

        if not file_exists:
            writer.writeheader()

        writer.writerow(row)


def parse_arguments():
    parser = argparse.ArgumentParser(
        description="Train the MADDPG UAV–IoT baseline."
    )

    parser.add_argument(
        "--config",
        default="configs/maddpg_base.yaml",
        help="Path to the training configuration file.",
    )

    parser.add_argument(
        "--episodes",
        type=int,
        default=None,
        help="Override the number of training episodes.",
    )

    parser.add_argument(
        "--resume",
        default=None,
        help="Optional checkpoint from which to resume.",
    )

    return parser.parse_args()


def main():
    arguments = parse_arguments()

    config_path = ROOT / arguments.config

    with config_path.open(
        "r",
        encoding="utf-8",
    ) as file:
        config = yaml.safe_load(file)

    experiment_cfg = config["experiment"]
    training_cfg = config["training"]

    experiment_name = str(
        experiment_cfg["name"]
    )

    seed = int(
        experiment_cfg["seed"]
    )

    set_random_seed(seed)

    total_episodes = int(
        arguments.episodes
        if arguments.episodes is not None
        else training_cfg["episodes"]
    )

    batch_size = int(
        training_cfg["batch_size"]
    )

    replay_capacity = int(
        training_cfg["replay_capacity"]
    )

    warmup_steps = int(
        training_cfg["warmup_steps"]
    )

    checkpoint_interval = int(
        training_cfg["checkpoint_interval"]
    )

    evaluation_interval = int(
        training_cfg["evaluation_interval"]
    )

    evaluation_episodes = int(
        training_cfg.get("evaluation_episodes", 5)
    )

    log_interval = int(
        training_cfg.get("log_interval", 10)
    )

    noise_scale = float(
        training_cfg["exploration_noise"]
    )

    minimum_noise = float(
        training_cfg["minimum_noise"]
    )

    noise_decay = float(
        training_cfg["noise_decay"]
    )

    environment = UAVIOTParallelEnv(
        config_path=arguments.config
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

    maddpg = MADDPG(
        number_of_agents=number_of_agents,
        observation_dimension=observation_dimension,
        action_dimension=action_dimension,
        config_path=arguments.config,
    )

    replay_buffer = ReplayBuffer(
        capacity=replay_capacity,
        number_of_agents=number_of_agents,
        observation_dimension=observation_dimension,
        action_dimension=action_dimension,
        device=maddpg.device,
        seed=seed,
    )

    checkpoint_directory = (
        ROOT / "checkpoints" / experiment_name
    )

    log_directory = (
        ROOT / "logs" / experiment_name
    )

    result_directory = (
        ROOT / "results" / experiment_name
    )

    checkpoint_directory.mkdir(
        parents=True,
        exist_ok=True,
    )

    log_directory.mkdir(
        parents=True,
        exist_ok=True,
    )

    result_directory.mkdir(
        parents=True,
        exist_ok=True,
    )

    training_csv = (
        log_directory / "training_metrics.csv"
    )

    start_episode = 1
    total_environment_steps = 0
    best_evaluation_reward = -float("inf")

    if arguments.resume:
        checkpoint = maddpg.load_checkpoint(
            ROOT / arguments.resume
        )

        start_episode = int(
            checkpoint["episode"]
        ) + 1

        noise_scale = float(
            checkpoint.get(
                "noise_scale",
                noise_scale,
            )
        )

        extra_state = checkpoint.get(
            "extra_state",
            {},
        )

        total_environment_steps = int(
            extra_state.get(
                "total_environment_steps",
                0,
            )
        )

        best_evaluation_reward = float(
            extra_state.get(
                "best_evaluation_reward",
                -float("inf"),
            )
        )

        print(
            "Resumed from episode:",
            start_episode - 1,
        )

        print(
            "Replay buffer starts empty after resume."
        )

    print("=" * 70)
    print("Experiment:", experiment_name)
    print("Device:", maddpg.device)
    print("Agents:", number_of_agents)
    print("Observation dimension:", observation_dimension)
    print("Action dimension:", action_dimension)
    print("Training episodes:", total_episodes)
    print("Warm-up steps:", warmup_steps)
    print("Batch size:", batch_size)
    print("=" * 70)

    training_start_time = time.time()

    last_completed_episode = start_episode - 1

    try:
        for episode in range(
            start_episode,
            total_episodes + 1,
        ):
            episode_start_time = time.time()

            observation_dict, _ = environment.reset(
                seed=seed + episode
            )

            episode_reward = 0.0
            episode_successes = 0
            episode_steps = 0

            episode_latencies = []
            episode_energies = []
            episode_critic_losses = []
            episode_actor_losses = []

            while environment.agents:
                observation_array = observations_to_array(
                    observation_dict,
                    environment.possible_agents,
                )

                if total_environment_steps < warmup_steps:
                    action_array = np.stack(
                        [
                            environment.action_space(
                                agent
                            ).sample()
                            for agent in (
                                environment.possible_agents
                            )
                        ]
                    ).astype(np.float32)
                else:
                    action_array = maddpg.select_actions(
                        observation_array,
                        noise_scale=noise_scale,
                    )

                action_dict = {
                    agent: action_array[index]
                    for index, agent in enumerate(
                        environment.possible_agents
                    )
                }

                (
                    next_observation_dict,
                    reward_dict,
                    termination_dict,
                    truncation_dict,
                    info_dict,
                ) = environment.step(action_dict)

                next_observation_array = (
                    observations_to_array(
                        next_observation_dict,
                        environment.possible_agents,
                    )
                )

                reward_array = np.asarray(
                    [
                        reward_dict[agent]
                        for agent in (
                            environment.possible_agents
                        )
                    ],
                    dtype=np.float32,
                )

                done_array = np.asarray(
                    [
                        (
                            termination_dict[agent]
                            or truncation_dict[agent]
                        )
                        for agent in (
                            environment.possible_agents
                        )
                    ],
                    dtype=np.float32,
                )

                replay_buffer.add(
                    observations=observation_array,
                    actions=action_array,
                    rewards=reward_array,
                    next_observations=(
                        next_observation_array
                    ),
                    dones=done_array,
                )

                if (
                    total_environment_steps >= warmup_steps
                    and len(replay_buffer) >= batch_size
                ):
                    batch = replay_buffer.sample(
                        batch_size=batch_size
                    )

                    update_metrics = maddpg.update(
                        batch
                    )

                    episode_critic_losses.append(
                        update_metrics[
                            "critic_loss_mean"
                        ]
                    )

                    episode_actor_losses.append(
                        update_metrics[
                            "actor_loss_mean"
                        ]
                    )

                first_agent = (
                    environment.possible_agents[0]
                )

                information = info_dict[first_agent]

                episode_reward += (
                    reward_dict[first_agent]
                )

                episode_successes += int(
                    information["success"]
                )

                episode_latencies.append(
                    information["latency_ms"]
                )

                episode_energies.append(
                    information["energy_mJ"]
                )

                episode_steps += 1
                total_environment_steps += 1

                observation_dict = (
                    next_observation_dict
                )

            if total_environment_steps >= warmup_steps:
                noise_scale = max(
                    minimum_noise,
                    noise_scale * noise_decay,
                )

            success_rate = (
                episode_successes
                / max(episode_steps, 1)
            )

            mean_latency = float(
                np.mean(episode_latencies)
            )

            mean_energy = float(
                np.mean(episode_energies)
            )

            mean_critic_loss = (
                float(
                    np.mean(
                        episode_critic_losses
                    )
                )
                if episode_critic_losses
                else float("nan")
            )

            mean_actor_loss = (
                float(
                    np.mean(
                        episode_actor_losses
                    )
                )
                if episode_actor_losses
                else float("nan")
            )

            evaluation_metrics = {
                "evaluation_reward": float("nan"),
                "evaluation_success_rate": float("nan"),
                "evaluation_latency_ms": float("nan"),
                "evaluation_energy_mJ": float("nan"),
            }

            if (
                episode % evaluation_interval == 0
                or episode == total_episodes
            ):
                evaluation_metrics = evaluate_policy(
                    maddpg=maddpg,
                    number_of_episodes=(
                        evaluation_episodes
                    ),
                    seed=seed + 100_000 + episode,
                )

                if (
                    evaluation_metrics[
                        "evaluation_reward"
                    ]
                    > best_evaluation_reward
                ):
                    best_evaluation_reward = (
                        evaluation_metrics[
                            "evaluation_reward"
                        ]
                    )

                    maddpg.save_checkpoint(
                        checkpoint_directory
                        / "maddpg_best.pt",
                        episode=episode,
                        noise_scale=noise_scale,
                        extra_state={
                            "total_environment_steps": (
                                total_environment_steps
                            ),
                            "best_evaluation_reward": (
                                best_evaluation_reward
                            ),
                        },
                    )

            episode_duration = (
                time.time() - episode_start_time
            )

            training_row = {
                "episode": episode,
                "environment_steps": (
                    total_environment_steps
                ),
                "episode_reward": episode_reward,
                "success_rate": success_rate,
                "mean_latency_ms": mean_latency,
                "mean_energy_mJ": mean_energy,
                "critic_loss": mean_critic_loss,
                "actor_loss": mean_actor_loss,
                "noise_scale": noise_scale,
                "replay_buffer_size": len(
                    replay_buffer
                ),
                "evaluation_reward": (
                    evaluation_metrics[
                        "evaluation_reward"
                    ]
                ),
                "evaluation_success_rate": (
                    evaluation_metrics[
                        "evaluation_success_rate"
                    ]
                ),
                "evaluation_latency_ms": (
                    evaluation_metrics[
                        "evaluation_latency_ms"
                    ]
                ),
                "evaluation_energy_mJ": (
                    evaluation_metrics[
                        "evaluation_energy_mJ"
                    ]
                ),
                "episode_duration_seconds": (
                    episode_duration
                ),
            }

            save_training_row(
                training_csv,
                training_row,
            )

            if (
                episode % log_interval == 0
                or episode == 1
                or episode == total_episodes
            ):
                print(
                    f"Episode {episode:4d}/{total_episodes} | "
                    f"Reward {episode_reward:9.3f} | "
                    f"Success {100 * success_rate:6.2f}% | "
                    f"Noise {noise_scale:.4f} | "
                    f"Buffer {len(replay_buffer):6d} | "
                    f"Time {episode_duration:6.2f}s"
                )

            if episode % checkpoint_interval == 0:
                checkpoint_path = (
                    checkpoint_directory
                    / f"maddpg_episode_{episode:04d}.pt"
                )

                maddpg.save_checkpoint(
                    checkpoint_path,
                    episode=episode,
                    noise_scale=noise_scale,
                    extra_state={
                        "total_environment_steps": (
                            total_environment_steps
                        ),
                        "best_evaluation_reward": (
                            best_evaluation_reward
                        ),
                    },
                )

                maddpg.save_checkpoint(
                    checkpoint_directory
                    / "maddpg_latest.pt",
                    episode=episode,
                    noise_scale=noise_scale,
                    extra_state={
                        "total_environment_steps": (
                            total_environment_steps
                        ),
                        "best_evaluation_reward": (
                            best_evaluation_reward
                        ),
                    },
                )

                print(
                    "Checkpoint saved:",
                    checkpoint_path.name,
                )

            last_completed_episode = episode

    except KeyboardInterrupt:
        print(
            "\nTraining interrupted. Saving emergency checkpoint."
        )

        maddpg.save_checkpoint(
            checkpoint_directory
            / "maddpg_interrupted.pt",
            episode=last_completed_episode,
            noise_scale=noise_scale,
            extra_state={
                "total_environment_steps": (
                    total_environment_steps
                ),
                "best_evaluation_reward": (
                    best_evaluation_reward
                ),
            },
        )

        environment.close()
        sys.exit(130)

    final_checkpoint = (
        checkpoint_directory / "maddpg_final.pt"
    )

    maddpg.save_checkpoint(
        final_checkpoint,
        episode=last_completed_episode,
        noise_scale=noise_scale,
        extra_state={
            "total_environment_steps": (
                total_environment_steps
            ),
            "best_evaluation_reward": (
                best_evaluation_reward
            ),
        },
    )

    final_evaluation = evaluate_policy(
        maddpg=maddpg,
        number_of_episodes=10,
        seed=seed + 500_000,
    )

    total_training_seconds = (
        time.time() - training_start_time
    )

    summary = {
        "experiment": experiment_name,
        "device": str(maddpg.device),
        "episodes_completed": (
            last_completed_episode
        ),
        "total_environment_steps": (
            total_environment_steps
        ),
        "total_updates": maddpg.total_updates,
        "best_evaluation_reward": (
            best_evaluation_reward
        ),
        "final_evaluation": final_evaluation,
        "total_training_seconds": (
            total_training_seconds
        ),
        "final_checkpoint": str(
            final_checkpoint.relative_to(ROOT)
        ),
    }

    summary_path = (
        result_directory / "training_summary.json"
    )

    with summary_path.open(
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(
            summary,
            file,
            indent=2,
        )

    environment.close()

    print("=" * 70)
    print("Training completed successfully.")
    print("Episodes completed:", last_completed_episode)
    print("Environment steps:", total_environment_steps)
    print("MADDPG updates:", maddpg.total_updates)
    print(
        "Final evaluation reward:",
        round(
            final_evaluation[
                "evaluation_reward"
            ],
            4,
        ),
    )
    print(
        "Final evaluation success rate:",
        f"{100 * final_evaluation['evaluation_success_rate']:.2f}%",
    )
    print("Final checkpoint:", final_checkpoint)
    print("Training metrics:", training_csv)
    print("Summary:", summary_path)
    print("=" * 70)


if __name__ == "__main__":
    main()
