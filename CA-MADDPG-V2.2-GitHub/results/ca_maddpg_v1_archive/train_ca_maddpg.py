from __future__ import annotations

import argparse
import csv
import json
import random
import time
from collections import deque
from pathlib import Path
from typing import Any

import numpy as np
import torch
import yaml

from ca_maddpg import CAMADDPG
from uav_iot_env import UAVIOTParallelEnv


ROOT = Path(__file__).resolve().parents[1]


class MultiAgentReplayBuffer:
    """Replay buffer for joint multi-agent transitions."""

    def __init__(
        self,
        capacity: int,
        number_of_agents: int,
        observation_dimension: int,
        action_dimension: int,
        seed: int,
    ):
        if capacity <= 0:
            raise ValueError(
                "Replay-buffer capacity must be positive."
            )

        self.capacity = int(capacity)
        self.number_of_agents = int(number_of_agents)
        self.observation_dimension = int(
            observation_dimension
        )
        self.action_dimension = int(
            action_dimension
        )

        self.rng = np.random.default_rng(seed)

        self.observations = np.zeros(
            (
                self.capacity,
                self.number_of_agents,
                self.observation_dimension,
            ),
            dtype=np.float32,
        )

        self.actions = np.zeros(
            (
                self.capacity,
                self.number_of_agents,
                self.action_dimension,
            ),
            dtype=np.float32,
        )

        self.rewards = np.zeros(
            (
                self.capacity,
                self.number_of_agents,
            ),
            dtype=np.float32,
        )

        self.next_observations = np.zeros(
            (
                self.capacity,
                self.number_of_agents,
                self.observation_dimension,
            ),
            dtype=np.float32,
        )

        self.dones = np.zeros(
            (
                self.capacity,
                self.number_of_agents,
            ),
            dtype=np.float32,
        )

        self.position = 0
        self.size = 0

    def add(
        self,
        observations: np.ndarray,
        actions: np.ndarray,
        rewards: np.ndarray,
        next_observations: np.ndarray,
        dones: np.ndarray,
    ) -> None:
        index = self.position

        self.observations[index] = observations
        self.actions[index] = actions
        self.rewards[index] = rewards
        self.next_observations[index] = (
            next_observations
        )
        self.dones[index] = dones

        self.position = (
            self.position + 1
        ) % self.capacity

        self.size = min(
            self.size + 1,
            self.capacity,
        )

    def sample(
        self,
        batch_size: int,
    ) -> dict[str, np.ndarray]:
        if self.size < batch_size:
            raise RuntimeError(
                "Not enough transitions in replay buffer."
            )

        indices = self.rng.choice(
            self.size,
            size=batch_size,
            replace=False,
        )

        return {
            "observations": self.observations[
                indices
            ],
            "actions": self.actions[indices],
            "rewards": self.rewards[indices],
            "next_observations": (
                self.next_observations[indices]
            ),
            "dones": self.dones[indices],
        }

    def __len__(self) -> int:
        return self.size


def resolve_path(path: str | Path) -> Path:
    resolved = Path(path)

    if not resolved.is_absolute():
        resolved = ROOT / resolved

    return resolved


def load_config(
    config_path: str | Path,
) -> dict[str, Any]:
    path = resolve_path(config_path)

    if not path.exists():
        raise FileNotFoundError(
            f"Configuration file not found: {path}"
        )

    with path.open(
        "r",
        encoding="utf-8",
    ) as file:
        return yaml.safe_load(file)


def set_global_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def stack_observations(
    observation_dictionary: dict,
    possible_agents: list[str],
    observation_dimension: int,
) -> np.ndarray:
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


def evaluate_policy(
    model: CAMADDPG,
    config_path: str,
    number_of_episodes: int,
    seed: int,
) -> dict[str, float]:
    episode_rewards = []
    episode_success_rates = []
    episode_latencies = []
    episode_energies = []

    previous_actor_states = [
        actor.training
        for actor in model.actors
    ]

    for actor in model.actors:
        actor.eval()

    for evaluation_index in range(
        number_of_episodes
    ):
        env = UAVIOTParallelEnv(
            config_path=config_path
        )

        possible_agents = list(
            env.possible_agents
        )

        observations, _ = env.reset(
            seed=seed + evaluation_index
        )

        total_reward = 0.0
        successes = 0
        latencies = []
        energies = []
        steps = 0

        while env.agents:
            joint_observations = (
                stack_observations(
                    observations,
                    possible_agents,
                    env.observation_dimension,
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
            ) = env.step(action_dictionary)

            first_agent = possible_agents[0]
            information = infos.get(
                first_agent,
                {},
            )

            total_reward += float(
                rewards.get(first_agent, 0.0)
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

            steps += 1
            observations = next_observations

        env.close()

        episode_rewards.append(total_reward)

        episode_success_rates.append(
            successes / max(steps, 1)
        )

        episode_latencies.append(
            float(np.mean(latencies))
            if latencies
            else 0.0
        )

        episode_energies.append(
            float(np.mean(energies))
            if energies
            else 0.0
        )

    for actor, was_training in zip(
        model.actors,
        previous_actor_states,
    ):
        actor.train(was_training)

    return {
        "evaluation_reward": float(
            np.mean(episode_rewards)
        ),
        "evaluation_reward_std": float(
            np.std(
                episode_rewards,
                ddof=1,
            )
            if len(episode_rewards) > 1
            else 0.0
        ),
        "evaluation_success_rate": float(
            np.mean(episode_success_rates)
        ),
        "evaluation_success_std": float(
            np.std(
                episode_success_rates,
                ddof=1,
            )
            if len(episode_success_rates) > 1
            else 0.0
        ),
        "evaluation_latency_ms": float(
            np.mean(episode_latencies)
        ),
        "evaluation_energy_mJ": float(
            np.mean(episode_energies)
        ),
    }


def write_metrics_csv(
    path: Path,
    records: list[dict[str, Any]],
) -> None:
    if not records:
        return

    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    fieldnames = [
        "episode",
        "environment_steps",
        "episode_reward",
        "success_rate",
        "mean_latency_ms",
        "mean_energy_mJ",
        "critic_loss",
        "actor_loss",
        "noise_scale",
        "replay_buffer_size",
        "evaluation_reward",
        "evaluation_success_rate",
        "evaluation_latency_ms",
        "evaluation_energy_mJ",
        "episode_duration_seconds",
    ]

    with path.open(
        "w",
        newline="",
        encoding="utf-8",
    ) as file:
        writer = csv.DictWriter(
            file,
            fieldnames=fieldnames,
        )

        writer.writeheader()

        for record in records:
            writer.writerow(
                {
                    field: record.get(
                        field,
                        np.nan,
                    )
                    for field in fieldnames
                }
            )


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Train Collaborative Attention-"
            "Enhanced MADDPG."
        )
    )

    parser.add_argument(
        "--config",
        type=str,
        default=(
            "configs/"
            "ca_maddpg_seed42_30ep.yaml"
        ),
    )

    parser.add_argument(
        "--episodes",
        type=int,
        default=None,
    )

    parser.add_argument(
        "--resume",
        type=str,
        default=None,
    )

    arguments = parser.parse_args()

    config_path = arguments.config
    config = load_config(config_path)

    experiment_config = config.get(
        "experiment",
        {},
    )

    training_config = config.get(
        "training",
        {},
    )

    experiment_name = experiment_config.get(
        "name",
        "ca_maddpg_experiment",
    )

    seed = int(
        experiment_config.get(
            "seed",
            42,
        )
    )

    set_global_seed(seed)

    total_episodes = int(
        arguments.episodes
        if arguments.episodes is not None
        else training_config.get(
            "episodes",
            30,
        )
    )

    batch_size = int(
        training_config.get(
            "batch_size",
            256,
        )
    )

    replay_capacity = int(
        training_config.get(
            "replay_capacity",
            20000,
        )
    )

    warmup_steps = int(
        training_config.get(
            "warmup_steps",
            500,
        )
    )

    exploration_noise = float(
        training_config.get(
            "exploration_noise",
            0.20,
        )
    )

    minimum_noise = float(
        training_config.get(
            "minimum_noise",
            0.05,
        )
    )

    noise_decay = float(
        training_config.get(
            "noise_decay",
            0.9995,
        )
    )

    checkpoint_interval = int(
        training_config.get(
            "checkpoint_interval",
            10,
        )
    )

    evaluation_interval = int(
        training_config.get(
            "evaluation_interval",
            10,
        )
    )

    evaluation_episodes = int(
        training_config.get(
            "evaluation_episodes",
            10,
        )
    )

    log_interval = int(
        training_config.get(
            "log_interval",
            5,
        )
    )

    checkpoint_directory = resolve_path(
        Path("checkpoints")
        / experiment_name
    )

    log_directory = resolve_path(
        Path("logs")
        / experiment_name
    )

    results_directory = resolve_path(
        Path("results")
        / experiment_name
    )

    checkpoint_directory.mkdir(
        parents=True,
        exist_ok=True,
    )

    log_directory.mkdir(
        parents=True,
        exist_ok=True,
    )

    results_directory.mkdir(
        parents=True,
        exist_ok=True,
    )

    training_metrics_path = (
        log_directory
        / "training_metrics.csv"
    )

    summary_path = (
        results_directory
        / "training_summary.json"
    )

    env = UAVIOTParallelEnv(
        config_path=config_path
    )

    possible_agents = list(
        env.possible_agents
    )

    number_of_agents = len(
        possible_agents
    )

    observation_dimension = (
        env.observation_dimension
    )

    action_dimension = (
        env.action_dimension
    )

    model = CAMADDPG(
        number_of_agents=number_of_agents,
        observation_dimension=(
            observation_dimension
        ),
        action_dimension=action_dimension,
        config_path=config_path,
    )

    replay_buffer = MultiAgentReplayBuffer(
        capacity=replay_capacity,
        number_of_agents=number_of_agents,
        observation_dimension=(
            observation_dimension
        ),
        action_dimension=action_dimension,
        seed=seed,
    )

    start_episode = 1
    total_environment_steps = 0
    noise_scale = exploration_noise

    best_evaluation_reward = -float("inf")
    best_evaluation_success = -float("inf")

    if arguments.resume is not None:
        checkpoint = model.load_checkpoint(
            arguments.resume
        )

        start_episode = int(
            checkpoint["episode"]
        ) + 1

        noise_scale = float(
            checkpoint.get(
                "noise_scale",
                exploration_noise,
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

        best_evaluation_success = float(
            extra_state.get(
                "best_evaluation_success",
                -float("inf"),
            )
        )

        print(
            "Resumed from episode:",
            checkpoint["episode"],
        )

        print(
            "Replay buffer starts empty "
            "after resume."
        )

    print("=" * 72)
    print("CA-MADDPG TRAINING")
    print("=" * 72)
    print("Experiment:", experiment_name)
    print("Device:", model.device)
    print(
        "GPU:",
        torch.cuda.get_device_name(0)
        if torch.cuda.is_available()
        else "CPU",
    )
    print("Agents:", number_of_agents)
    print(
        "Observation dimension:",
        observation_dimension,
    )
    print(
        "Action dimension:",
        action_dimension,
    )
    print(
        "Training episodes:",
        total_episodes,
    )
    print(
        "Warm-up steps:",
        warmup_steps,
    )
    print("Batch size:", batch_size)
    print(
        "Replay capacity:",
        replay_capacity,
    )
    print("=" * 72)

    training_records = []
    training_start_time = time.perf_counter()

    recent_rewards = deque(
        maxlen=10
    )

    for episode in range(
        start_episode,
        total_episodes + 1,
    ):
        episode_start_time = (
            time.perf_counter()
        )

        observations, _ = env.reset(
            seed=seed + episode - 1
        )

        episode_reward = 0.0
        episode_successes = 0
        episode_latencies = []
        episode_energies = []

        episode_critic_losses = []
        episode_actor_losses = []

        episode_steps = 0

        while env.agents:
            joint_observations = (
                stack_observations(
                    observations,
                    possible_agents,
                    observation_dimension,
                )
            )

            if (
                total_environment_steps
                < warmup_steps
            ):
                joint_actions = np.random.uniform(
                    low=0.0,
                    high=1.0,
                    size=(
                        number_of_agents,
                        action_dimension,
                    ),
                ).astype(np.float32)

            else:
                joint_actions = (
                    model.select_actions(
                        joint_observations,
                        noise_scale=noise_scale,
                    )
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
            ) = env.step(action_dictionary)

            joint_next_observations = (
                stack_observations(
                    next_observations,
                    possible_agents,
                    observation_dimension,
                )
            )

            reward_array = np.asarray(
                [
                    rewards.get(agent, 0.0)
                    for agent in possible_agents
                ],
                dtype=np.float32,
            )

            done_array = np.asarray(
                [
                    float(
                        terminations.get(
                            agent,
                            False,
                        )
                        or truncations.get(
                            agent,
                            False,
                        )
                    )
                    for agent in possible_agents
                ],
                dtype=np.float32,
            )

            replay_buffer.add(
                observations=joint_observations,
                actions=joint_actions,
                rewards=reward_array,
                next_observations=(
                    joint_next_observations
                ),
                dones=done_array,
            )

            first_agent = possible_agents[0]

            information = infos.get(
                first_agent,
                {},
            )

            episode_reward += float(
                rewards.get(
                    first_agent,
                    0.0,
                )
            )

            episode_successes += int(
                information.get(
                    "success",
                    False,
                )
            )

            episode_latencies.append(
                float(
                    information.get(
                        "latency_ms",
                        0.0,
                    )
                )
            )

            episode_energies.append(
                float(
                    information.get(
                        "energy_mJ",
                        0.0,
                    )
                )
            )

            total_environment_steps += 1
            episode_steps += 1

            observations = next_observations

            if (
                total_environment_steps
                >= warmup_steps
                and len(replay_buffer)
                >= batch_size
            ):
                batch = replay_buffer.sample(
                    batch_size
                )

                update_metrics = model.update(
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
            else np.nan
        )

        mean_actor_loss = (
            float(
                np.mean(
                    episode_actor_losses
                )
            )
            if episode_actor_losses
            else np.nan
        )

        evaluation_metrics = {}

        if (
            episode % evaluation_interval == 0
            or episode == total_episodes
        ):
            evaluation_metrics = (
                evaluate_policy(
                    model=model,
                    config_path=config_path,
                    number_of_episodes=(
                        evaluation_episodes
                    ),
                    seed=(
                        500000
                        + seed * 1000
                        + episode
                    ),
                )
            )

            current_success = (
                evaluation_metrics[
                    "evaluation_success_rate"
                ]
            )

            current_reward = (
                evaluation_metrics[
                    "evaluation_reward"
                ]
            )

            better_checkpoint = (
                current_success
                > best_evaluation_success
                or (
                    np.isclose(
                        current_success,
                        best_evaluation_success,
                    )
                    and current_reward
                    > best_evaluation_reward
                )
            )

            if better_checkpoint:
                best_evaluation_success = (
                    current_success
                )

                best_evaluation_reward = (
                    current_reward
                )

                model.save_checkpoint(
                    path=(
                        checkpoint_directory
                        / "ca_maddpg_best.pt"
                    ),
                    episode=episode,
                    noise_scale=noise_scale,
                    extra_state={
                        "total_environment_steps": (
                            total_environment_steps
                        ),
                        "best_evaluation_reward": (
                            best_evaluation_reward
                        ),
                        "best_evaluation_success": (
                            best_evaluation_success
                        ),
                    },
                )

        if (
            episode % checkpoint_interval == 0
            or episode == total_episodes
        ):
            model.save_checkpoint(
                path=(
                    checkpoint_directory
                    / (
                        "ca_maddpg_episode_"
                        f"{episode:04d}.pt"
                    )
                ),
                episode=episode,
                noise_scale=noise_scale,
                extra_state={
                    "total_environment_steps": (
                        total_environment_steps
                    ),
                    "best_evaluation_reward": (
                        best_evaluation_reward
                    ),
                    "best_evaluation_success": (
                        best_evaluation_success
                    ),
                },
            )

        model.save_checkpoint(
            path=(
                checkpoint_directory
                / "ca_maddpg_latest.pt"
            ),
            episode=episode,
            noise_scale=noise_scale,
            extra_state={
                "total_environment_steps": (
                    total_environment_steps
                ),
                "best_evaluation_reward": (
                    best_evaluation_reward
                ),
                "best_evaluation_success": (
                    best_evaluation_success
                ),
            },
        )

        episode_duration = (
            time.perf_counter()
            - episode_start_time
        )

        record = {
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
                evaluation_metrics.get(
                    "evaluation_reward",
                    np.nan,
                )
            ),
            "evaluation_success_rate": (
                evaluation_metrics.get(
                    "evaluation_success_rate",
                    np.nan,
                )
            ),
            "evaluation_latency_ms": (
                evaluation_metrics.get(
                    "evaluation_latency_ms",
                    np.nan,
                )
            ),
            "evaluation_energy_mJ": (
                evaluation_metrics.get(
                    "evaluation_energy_mJ",
                    np.nan,
                )
            ),
            "episode_duration_seconds": (
                episode_duration
            ),
        }

        training_records.append(record)
        recent_rewards.append(
            episode_reward
        )

        write_metrics_csv(
            training_metrics_path,
            training_records,
        )

        if (
            episode == 1
            or episode % log_interval == 0
            or episode == total_episodes
        ):
            print(
                f"Episode {episode:4d}/"
                f"{total_episodes} | "
                f"Reward {episode_reward:9.3f} | "
                f"Success "
                f"{100 * success_rate:6.2f}% | "
                f"Noise {noise_scale:.4f} | "
                f"Buffer {len(replay_buffer):5d} | "
                f"Updates {model.total_updates:6d} | "
                f"Time {episode_duration:7.2f}s"
            )

            if evaluation_metrics:
                print(
                    "  Evaluation | "
                    f"Reward "
                    f"{evaluation_metrics['evaluation_reward']:.3f} | "
                    f"Success "
                    f"{100 * evaluation_metrics['evaluation_success_rate']:.2f}% | "
                    f"Latency "
                    f"{evaluation_metrics['evaluation_latency_ms']:.2f} ms | "
                    f"Energy "
                    f"{evaluation_metrics['evaluation_energy_mJ']:.2f} mJ"
                )

    env.close()

    final_evaluation = evaluate_policy(
        model=model,
        config_path=config_path,
        number_of_episodes=(
            evaluation_episodes
        ),
        seed=900000 + seed,
    )

    final_checkpoint = (
        checkpoint_directory
        / "ca_maddpg_final.pt"
    )

    model.save_checkpoint(
        path=final_checkpoint,
        episode=total_episodes,
        noise_scale=noise_scale,
        extra_state={
            "total_environment_steps": (
                total_environment_steps
            ),
            "best_evaluation_reward": (
                best_evaluation_reward
            ),
            "best_evaluation_success": (
                best_evaluation_success
            ),
            "final_evaluation": (
                final_evaluation
            ),
        },
    )

    total_training_seconds = (
        time.perf_counter()
        - training_start_time
    )

    summary = {
        "experiment": experiment_name,
        "model": "CA-MADDPG",
        "device": str(model.device),
        "gpu": (
            torch.cuda.get_device_name(0)
            if torch.cuda.is_available()
            else "CPU"
        ),
        "training_seed": seed,
        "episodes_completed": (
            total_episodes
        ),
        "total_environment_steps": (
            total_environment_steps
        ),
        "total_updates": (
            model.total_updates
        ),
        "batch_size": batch_size,
        "replay_capacity": replay_capacity,
        "warmup_steps": warmup_steps,
        "best_evaluation_reward": (
            best_evaluation_reward
        ),
        "best_evaluation_success_rate": (
            best_evaluation_success
        ),
        "final_evaluation": (
            final_evaluation
        ),
        "total_training_seconds": (
            total_training_seconds
        ),
        "final_checkpoint": str(
            final_checkpoint.relative_to(ROOT)
        ),
        "training_metrics": str(
            training_metrics_path.relative_to(
                ROOT
            )
        ),
    }

    with summary_path.open(
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(
            summary,
            file,
            indent=2,
        )

    print("=" * 72)
    print(
        "CA-MADDPG training completed successfully."
    )
    print(
        "Episodes completed:",
        total_episodes,
    )
    print(
        "Environment steps:",
        total_environment_steps,
    )
    print(
        "CA-MADDPG updates:",
        model.total_updates,
    )
    print(
        "Final evaluation reward:",
        f"{final_evaluation['evaluation_reward']:.4f}",
    )
    print(
        "Final evaluation success rate:",
        (
            f"{100 * final_evaluation['evaluation_success_rate']:.2f}%"
        ),
    )
    print(
        "Final evaluation latency:",
        (
            f"{final_evaluation['evaluation_latency_ms']:.2f} ms"
        ),
    )
    print(
        "Final evaluation energy:",
        (
            f"{final_evaluation['evaluation_energy_mJ']:.2f} mJ"
        ),
    )
    print(
        "Final checkpoint:",
        final_checkpoint,
    )
    print(
        "Training metrics:",
        training_metrics_path,
    )
    print(
        "Summary:",
        summary_path,
    )
    print("=" * 72)


if __name__ == "__main__":
    main()
