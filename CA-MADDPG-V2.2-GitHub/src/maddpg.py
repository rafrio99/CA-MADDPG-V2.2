from __future__ import annotations

from pathlib import Path
from typing import Dict, Sequence

import numpy as np
import torch
import torch.nn.functional as F
import yaml
from torch import nn
from torch.optim import Adam

from maddpg_components import (
    Actor,
    Critic,
    hard_update,
    soft_update,
)


ROOT = Path(__file__).resolve().parents[1]


class MADDPG:
    """Multi-Agent Deep Deterministic Policy Gradient manager."""

    def __init__(
        self,
        number_of_agents: int,
        observation_dimension: int,
        action_dimension: int,
        config_path: str = "configs/maddpg_base.yaml",
    ):
        with open(ROOT / config_path, "r", encoding="utf-8") as file:
            self.config = yaml.safe_load(file)

        experiment_cfg = self.config["experiment"]
        training_cfg = self.config["training"]

        self.number_of_agents = int(number_of_agents)
        self.observation_dimension = int(observation_dimension)
        self.action_dimension = int(action_dimension)

        self.gamma = float(training_cfg["gamma"])
        self.tau = float(training_cfg["tau"])

        self.actor_learning_rate = float(
            training_cfg["actor_learning_rate"]
        )

        self.critic_learning_rate = float(
            training_cfg["critic_learning_rate"]
        )

        self.hidden_dimensions: Sequence[int] = tuple(
            int(value)
            for value in training_cfg["hidden_dimensions"]
        )

        self.gradient_clip_norm = float(
            training_cfg.get("gradient_clip_norm", 10.0)
        )

        requested_device = str(
            experiment_cfg.get("device", "cuda")
        ).lower()

        gpu_id = int(
            experiment_cfg.get("gpu_id", 0)
        )

        if requested_device == "cuda" and torch.cuda.is_available():
            self.device = torch.device(f"cuda:{gpu_id}")
        else:
            self.device = torch.device("cpu")

        self.global_observation_dimension = (
            self.number_of_agents
            * self.observation_dimension
        )

        self.joint_action_dimension = (
            self.number_of_agents
            * self.action_dimension
        )

        self.actors = nn.ModuleList(
            [
                Actor(
                    observation_dimension=self.observation_dimension,
                    action_dimension=self.action_dimension,
                    hidden_dimensions=self.hidden_dimensions,
                )
                for _ in range(self.number_of_agents)
            ]
        ).to(self.device)

        self.target_actors = nn.ModuleList(
            [
                Actor(
                    observation_dimension=self.observation_dimension,
                    action_dimension=self.action_dimension,
                    hidden_dimensions=self.hidden_dimensions,
                )
                for _ in range(self.number_of_agents)
            ]
        ).to(self.device)

        self.critics = nn.ModuleList(
            [
                Critic(
                    global_observation_dimension=(
                        self.global_observation_dimension
                    ),
                    joint_action_dimension=(
                        self.joint_action_dimension
                    ),
                    hidden_dimensions=self.hidden_dimensions,
                )
                for _ in range(self.number_of_agents)
            ]
        ).to(self.device)

        self.target_critics = nn.ModuleList(
            [
                Critic(
                    global_observation_dimension=(
                        self.global_observation_dimension
                    ),
                    joint_action_dimension=(
                        self.joint_action_dimension
                    ),
                    hidden_dimensions=self.hidden_dimensions,
                )
                for _ in range(self.number_of_agents)
            ]
        ).to(self.device)

        self.actor_optimizers = [
            Adam(
                actor.parameters(),
                lr=self.actor_learning_rate,
            )
            for actor in self.actors
        ]

        self.critic_optimizers = [
            Adam(
                critic.parameters(),
                lr=self.critic_learning_rate,
            )
            for critic in self.critics
        ]

        for agent_index in range(self.number_of_agents):
            hard_update(
                self.target_actors[agent_index],
                self.actors[agent_index],
            )

            hard_update(
                self.target_critics[agent_index],
                self.critics[agent_index],
            )

        self.total_updates = 0

    def select_actions(
        self,
        observations: np.ndarray,
        noise_scale: float = 0.0,
    ) -> np.ndarray:
        observations = np.asarray(
            observations,
            dtype=np.float32,
        )

        expected_shape = (
            self.number_of_agents,
            self.observation_dimension,
        )

        if observations.shape != expected_shape:
            raise ValueError(
                f"Observation shape {observations.shape} does not "
                f"match {expected_shape}."
            )

        observation_tensor = torch.as_tensor(
            observations,
            dtype=torch.float32,
            device=self.device,
        )

        actions = []

        with torch.no_grad():
            for agent_index, actor in enumerate(self.actors):
                action = actor(
                    observation_tensor[agent_index]
                )

                actions.append(
                    action.cpu().numpy()
                )

        action_array = np.asarray(
            actions,
            dtype=np.float32,
        )

        if noise_scale > 0.0:
            exploration_noise = np.random.normal(
                loc=0.0,
                scale=noise_scale,
                size=action_array.shape,
            ).astype(np.float32)

            action_array = action_array + exploration_noise

        return np.clip(
            action_array,
            0.0,
            1.0,
        ).astype(np.float32)

    def update(
        self,
        batch: Dict[str, torch.Tensor],
    ) -> Dict[str, object]:
        observations = batch["observations"]
        actions = batch["actions"]
        rewards = batch["rewards"]
        next_observations = batch["next_observations"]
        dones = batch["dones"]

        batch_size = observations.shape[0]

        global_observations = observations.reshape(
            batch_size,
            self.global_observation_dimension,
        )

        global_next_observations = next_observations.reshape(
            batch_size,
            self.global_observation_dimension,
        )

        joint_actions = actions.reshape(
            batch_size,
            self.joint_action_dimension,
        )

        with torch.no_grad():
            next_action_list = [
                self.target_actors[agent_index](
                    next_observations[:, agent_index, :]
                )
                for agent_index in range(self.number_of_agents)
            ]

            joint_next_actions = torch.cat(
                next_action_list,
                dim=-1,
            )

        critic_losses = []
        actor_losses = []

        for agent_index in range(self.number_of_agents):
            with torch.no_grad():
                target_q_value = self.target_critics[
                    agent_index
                ](
                    global_next_observations,
                    joint_next_actions,
                ).squeeze(-1)

                target_value = (
                    rewards[:, agent_index]
                    + self.gamma
                    * (1.0 - dones[:, agent_index])
                    * target_q_value
                )

            current_q_value = self.critics[
                agent_index
            ](
                global_observations,
                joint_actions,
            ).squeeze(-1)

            critic_loss = F.mse_loss(
                current_q_value,
                target_value,
            )

            self.critic_optimizers[
                agent_index
            ].zero_grad(set_to_none=True)

            critic_loss.backward()

            torch.nn.utils.clip_grad_norm_(
                self.critics[agent_index].parameters(),
                self.gradient_clip_norm,
            )

            self.critic_optimizers[
                agent_index
            ].step()

            for parameter in self.critics[
                agent_index
            ].parameters():
                parameter.requires_grad_(False)

            predicted_action_list = []

            for other_agent_index in range(
                self.number_of_agents
            ):
                predicted_action = self.actors[
                    other_agent_index
                ](
                    observations[:, other_agent_index, :]
                )

                if other_agent_index != agent_index:
                    predicted_action = predicted_action.detach()

                predicted_action_list.append(
                    predicted_action
                )

            joint_predicted_actions = torch.cat(
                predicted_action_list,
                dim=-1,
            )

            actor_loss = -self.critics[
                agent_index
            ](
                global_observations,
                joint_predicted_actions,
            ).mean()

            self.actor_optimizers[
                agent_index
            ].zero_grad(set_to_none=True)

            actor_loss.backward()

            torch.nn.utils.clip_grad_norm_(
                self.actors[agent_index].parameters(),
                self.gradient_clip_norm,
            )

            self.actor_optimizers[
                agent_index
            ].step()

            for parameter in self.critics[
                agent_index
            ].parameters():
                parameter.requires_grad_(True)

            critic_losses.append(
                float(critic_loss.detach().cpu().item())
            )

            actor_losses.append(
                float(actor_loss.detach().cpu().item())
            )

        self.update_target_networks()
        self.total_updates += 1

        return {
            "critic_loss_mean": float(
                np.mean(critic_losses)
            ),
            "actor_loss_mean": float(
                np.mean(actor_losses)
            ),
            "critic_losses": critic_losses,
            "actor_losses": actor_losses,
            "total_updates": self.total_updates,
        }

    def update_target_networks(self) -> None:
        for agent_index in range(self.number_of_agents):
            soft_update(
                self.target_actors[agent_index],
                self.actors[agent_index],
                self.tau,
            )

            soft_update(
                self.target_critics[agent_index],
                self.critics[agent_index],
                self.tau,
            )

    def save_checkpoint(
        self,
        checkpoint_path: str | Path,
        episode: int,
        noise_scale: float,
        extra_state: Dict | None = None,
    ) -> None:
        checkpoint_path = Path(checkpoint_path)
        checkpoint_path.parent.mkdir(
            parents=True,
            exist_ok=True,
        )

        checkpoint = {
            "episode": int(episode),
            "noise_scale": float(noise_scale),
            "total_updates": int(self.total_updates),
            "actors": self.actors.state_dict(),
            "target_actors": self.target_actors.state_dict(),
            "critics": self.critics.state_dict(),
            "target_critics": self.target_critics.state_dict(),
            "actor_optimizers": [
                optimizer.state_dict()
                for optimizer in self.actor_optimizers
            ],
            "critic_optimizers": [
                optimizer.state_dict()
                for optimizer in self.critic_optimizers
            ],
            "extra_state": extra_state or {},
        }

        torch.save(
            checkpoint,
            checkpoint_path,
        )

    def load_checkpoint(
        self,
        checkpoint_path: str | Path,
    ) -> Dict:
        checkpoint = torch.load(
            checkpoint_path,
            map_location=self.device,
        )

        self.actors.load_state_dict(
            checkpoint["actors"]
        )

        self.target_actors.load_state_dict(
            checkpoint["target_actors"]
        )

        self.critics.load_state_dict(
            checkpoint["critics"]
        )

        self.target_critics.load_state_dict(
            checkpoint["target_critics"]
        )

        for optimizer, optimizer_state in zip(
            self.actor_optimizers,
            checkpoint["actor_optimizers"],
        ):
            optimizer.load_state_dict(
                optimizer_state
            )

        for optimizer, optimizer_state in zip(
            self.critic_optimizers,
            checkpoint["critic_optimizers"],
        ):
            optimizer.load_state_dict(
                optimizer_state
            )

        self.total_updates = int(
            checkpoint.get("total_updates", 0)
        )

        return checkpoint


if __name__ == "__main__":
    number_of_agents = 10
    observation_dimension = 29
    action_dimension = 3

    maddpg = MADDPG(
        number_of_agents=number_of_agents,
        observation_dimension=observation_dimension,
        action_dimension=action_dimension,
    )

    sample_observations = np.random.rand(
        number_of_agents,
        observation_dimension,
    ).astype(np.float32)

    sample_actions = maddpg.select_actions(
        sample_observations,
        noise_scale=0.1,
    )

    actor_parameter_count = sum(
        parameter.numel()
        for parameter in maddpg.actors.parameters()
    )

    critic_parameter_count = sum(
        parameter.numel()
        for parameter in maddpg.critics.parameters()
    )

    print("Device:", maddpg.device)
    print("Number of actors:", len(maddpg.actors))
    print("Number of critics:", len(maddpg.critics))
    print("Selected-action shape:", sample_actions.shape)
    print("Actor parameters:", actor_parameter_count)
    print("Critic parameters:", critic_parameter_count)
