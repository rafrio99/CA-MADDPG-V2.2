from __future__ import annotations

import copy
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F
import yaml
from torch import nn

from ca_maddpg_components import (
    CollaborativeAttentionActor,
    CollaborativeAttentionCritic,
)


ROOT = Path(__file__).resolve().parents[1]


class CAMADDPG:
    """
    Collaborative Attention-Enhanced MADDPG.

    Each UAV actor uses all UAV observations through collaborative
    multi-head attention. Each centralised critic evaluates the joint
    observation-action state using attention pooling.
    """

    def __init__(
        self,
        number_of_agents: int,
        observation_dimension: int,
        action_dimension: int,
        config_path: str = "configs/ca_maddpg_seed42_30ep.yaml",
        device: str | torch.device | None = None,
    ):
        self.number_of_agents = int(number_of_agents)
        self.observation_dimension = int(observation_dimension)
        self.action_dimension = int(action_dimension)

        config_file = Path(config_path)

        if not config_file.is_absolute():
            config_file = ROOT / config_file

        if not config_file.exists():
            raise FileNotFoundError(
                f"Configuration file not found: {config_file}"
            )

        with config_file.open(
            "r",
            encoding="utf-8",
        ) as file:
            self.config = yaml.safe_load(file)

        training_config = self.config.get(
            "training",
            {},
        )

        model_config = self.config.get(
            "model",
            {},
        )

        if device is None:
            self.device = torch.device(
                "cuda:0"
                if torch.cuda.is_available()
                else "cpu"
            )
        else:
            self.device = torch.device(device)

        self.gamma = float(
            training_config.get("gamma", 0.99)
        )

        self.tau = float(
            training_config.get("tau", 0.005)
        )

        self.actor_learning_rate = float(
            training_config.get(
                "actor_learning_rate",
                0.0001,
            )
        )

        self.critic_learning_rate = float(
            training_config.get(
                "critic_learning_rate",
                0.001,
            )
        )

        self.gradient_clip_norm = float(
            training_config.get(
                "gradient_clip_norm",
                10.0,
            )
        )

        embedding_dimension = int(
            model_config.get(
                "attention_embedding_dimension",
                128,
            )
        )

        attention_heads = int(
            model_config.get(
                "attention_heads",
                4,
            )
        )

        attention_layers = int(
            model_config.get(
                "attention_layers",
                1,
            )
        )

        attention_dropout = float(
            model_config.get(
                "attention_dropout",
                0.10,
            )
        )

        hidden_dimensions = tuple(
            int(value)
            for value in training_config.get(
                "hidden_dimensions",
                [256, 256],
            )
        )

        self.actors = nn.ModuleList(
            [
                CollaborativeAttentionActor(
                    number_of_agents=(
                        self.number_of_agents
                    ),
                    observation_dimension=(
                        self.observation_dimension
                    ),
                    action_dimension=(
                        self.action_dimension
                    ),
                    agent_index=agent_index,
                    embedding_dimension=(
                        embedding_dimension
                    ),
                    number_of_heads=attention_heads,
                    number_of_layers=attention_layers,
                    dropout=attention_dropout,
                    hidden_dimensions=hidden_dimensions,
                )
                for agent_index in range(
                    self.number_of_agents
                )
            ]
        ).to(self.device)

        self.critics = nn.ModuleList(
            [
                CollaborativeAttentionCritic(
                    number_of_agents=(
                        self.number_of_agents
                    ),
                    observation_dimension=(
                        self.observation_dimension
                    ),
                    action_dimension=(
                        self.action_dimension
                    ),
                    embedding_dimension=(
                        embedding_dimension
                    ),
                    number_of_heads=attention_heads,
                    number_of_layers=attention_layers,
                    dropout=attention_dropout,
                    hidden_dimensions=hidden_dimensions,
                )
                for _ in range(
                    self.number_of_agents
                )
            ]
        ).to(self.device)

        self.target_actors = copy.deepcopy(
            self.actors
        ).to(self.device)

        self.target_critics = copy.deepcopy(
            self.critics
        ).to(self.device)

        for target_actor in self.target_actors:
            target_actor.requires_grad_(False)

        for target_critic in self.target_critics:
            target_critic.requires_grad_(False)

        self.actor_optimisers = [
            torch.optim.Adam(
                actor.parameters(),
                lr=self.actor_learning_rate,
            )
            for actor in self.actors
        ]

        self.critic_optimisers = [
            torch.optim.Adam(
                critic.parameters(),
                lr=self.critic_learning_rate,
            )
            for critic in self.critics
        ]

        self.total_updates = 0

    @staticmethod
    def _as_tensor(
        value: Any,
        device: torch.device,
    ) -> torch.Tensor:
        if isinstance(value, torch.Tensor):
            return value.to(
                device=device,
                dtype=torch.float32,
            )

        return torch.as_tensor(
            value,
            dtype=torch.float32,
            device=device,
        )

    def _prepare_batch(
        self,
        batch: Any,
    ) -> tuple[
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
    ]:
        if isinstance(batch, dict):
            observations = batch["observations"]
            actions = batch["actions"]
            rewards = batch["rewards"]
            next_observations = batch[
                "next_observations"
            ]
            dones = batch["dones"]

        elif isinstance(batch, (tuple, list)):
            if len(batch) != 5:
                raise ValueError(
                    "Replay-buffer batch must contain "
                    "five elements."
                )

            (
                observations,
                actions,
                rewards,
                next_observations,
                dones,
            ) = batch

        else:
            raise TypeError(
                "Unsupported replay-buffer batch format."
            )

        observations = self._as_tensor(
            observations,
            self.device,
        )

        actions = self._as_tensor(
            actions,
            self.device,
        )

        rewards = self._as_tensor(
            rewards,
            self.device,
        )

        next_observations = self._as_tensor(
            next_observations,
            self.device,
        )

        dones = self._as_tensor(
            dones,
            self.device,
        )

        if rewards.ndim == 3:
            rewards = rewards.squeeze(-1)

        if dones.ndim == 3:
            dones = dones.squeeze(-1)

        expected_observation_shape = (
            observations.shape[0],
            self.number_of_agents,
            self.observation_dimension,
        )

        expected_action_shape = (
            actions.shape[0],
            self.number_of_agents,
            self.action_dimension,
        )

        if (
            tuple(observations.shape)
            != expected_observation_shape
        ):
            raise ValueError(
                "Unexpected observation batch shape: "
                f"{tuple(observations.shape)}"
            )

        if (
            tuple(actions.shape)
            != expected_action_shape
        ):
            raise ValueError(
                "Unexpected action batch shape: "
                f"{tuple(actions.shape)}"
            )

        return (
            observations,
            actions,
            rewards,
            next_observations,
            dones,
        )

    def select_actions(
        self,
        observations: np.ndarray,
        noise_scale: float = 0.0,
    ) -> np.ndarray:
        observation_tensor = self._as_tensor(
            observations,
            self.device,
        )

        if observation_tensor.ndim != 2:
            raise ValueError(
                "Observations must have shape "
                "[agents, observation_dimension]."
            )

        observation_tensor = (
            observation_tensor.unsqueeze(0)
        )

        previous_training_states = [
            actor.training
            for actor in self.actors
        ]

        for actor in self.actors:
            actor.eval()

        with torch.no_grad():
            action_tensor = torch.stack(
                [
                    actor(observation_tensor)
                    for actor in self.actors
                ],
                dim=1,
            )

        for actor, was_training in zip(
            self.actors,
            previous_training_states,
        ):
            actor.train(was_training)

        actions = (
            action_tensor.squeeze(0)
            .detach()
            .cpu()
            .numpy()
        )

        if noise_scale > 0.0:
            actions = actions + np.random.normal(
                loc=0.0,
                scale=float(noise_scale),
                size=actions.shape,
            ).astype(np.float32)

        actions = np.clip(
            actions,
            0.0,
            1.0,
        ).astype(np.float32)

        return actions

    def _calculate_target_actions(
        self,
        next_observations: torch.Tensor,
    ) -> torch.Tensor:
        previous_states = [
            actor.training
            for actor in self.target_actors
        ]

        for actor in self.target_actors:
            actor.eval()

        with torch.no_grad():
            target_actions = torch.stack(
                [
                    target_actor(
                        next_observations
                    )
                    for target_actor in (
                        self.target_actors
                    )
                ],
                dim=1,
            )

        for actor, was_training in zip(
            self.target_actors,
            previous_states,
        ):
            actor.train(was_training)

        return target_actions

    def update(
        self,
        batch: Any,
    ) -> dict[str, Any]:
        (
            observations,
            replay_actions,
            rewards,
            next_observations,
            dones,
        ) = self._prepare_batch(batch)

        target_actions = (
            self._calculate_target_actions(
                next_observations
            )
        )

        critic_losses: list[float] = []
        actor_losses: list[float] = []

        # Update all centralised critics.
        for agent_index in range(
            self.number_of_agents
        ):
            critic = self.critics[agent_index]
            target_critic = self.target_critics[
                agent_index
            ]

            critic_optimiser = (
                self.critic_optimisers[
                    agent_index
                ]
            )

            with torch.no_grad():
                target_critic.eval()

                target_q_value = target_critic(
                    next_observations,
                    target_actions,
                )

                target_reward = rewards[
                    :,
                    agent_index,
                ].unsqueeze(-1)

                target_done = dones[
                    :,
                    agent_index,
                ].unsqueeze(-1)

                target_value = (
                    target_reward
                    + self.gamma
                    * (1.0 - target_done)
                    * target_q_value
                )

            current_q_value = critic(
                observations,
                replay_actions,
            )

            critic_loss = F.mse_loss(
                current_q_value,
                target_value,
            )

            critic_optimiser.zero_grad(
                set_to_none=True
            )

            critic_loss.backward()

            torch.nn.utils.clip_grad_norm_(
                critic.parameters(),
                max_norm=self.gradient_clip_norm,
            )

            critic_optimiser.step()

            critic_losses.append(
                float(
                    critic_loss.detach().item()
                )
            )

        # Update each decentralised attention actor.
        for agent_index in range(
            self.number_of_agents
        ):
            actor = self.actors[agent_index]
            critic = self.critics[agent_index]

            actor_optimiser = (
                self.actor_optimisers[
                    agent_index
                ]
            )

            policy_actions = []

            for policy_agent_index in range(
                self.number_of_agents
            ):
                policy_actor = self.actors[
                    policy_agent_index
                ]

                if (
                    policy_agent_index
                    == agent_index
                ):
                    policy_action = policy_actor(
                        observations
                    )
                else:
                    with torch.no_grad():
                        policy_action = (
                            policy_actor(
                                observations
                            )
                        )

                policy_actions.append(
                    policy_action
                )

            joint_policy_actions = torch.stack(
                policy_actions,
                dim=1,
            )

            for parameter in critic.parameters():
                parameter.requires_grad_(False)

            actor_loss = -critic(
                observations,
                joint_policy_actions,
            ).mean()

            actor_optimiser.zero_grad(
                set_to_none=True
            )

            actor_loss.backward()

            torch.nn.utils.clip_grad_norm_(
                actor.parameters(),
                max_norm=self.gradient_clip_norm,
            )

            actor_optimiser.step()

            for parameter in critic.parameters():
                parameter.requires_grad_(True)

            actor_losses.append(
                float(
                    actor_loss.detach().item()
                )
            )

        self.soft_update_targets()
        self.total_updates += 1

        return {
            "critic_losses": critic_losses,
            "actor_losses": actor_losses,
            "critic_loss_mean": float(
                np.mean(critic_losses)
            ),
            "actor_loss_mean": float(
                np.mean(actor_losses)
            ),
            "total_updates": self.total_updates,
        }

    @torch.no_grad()
    def soft_update_targets(self) -> None:
        for actor, target_actor in zip(
            self.actors,
            self.target_actors,
        ):
            for parameter, target_parameter in zip(
                actor.parameters(),
                target_actor.parameters(),
            ):
                target_parameter.data.mul_(
                    1.0 - self.tau
                )

                target_parameter.data.add_(
                    self.tau * parameter.data
                )

        for critic, target_critic in zip(
            self.critics,
            self.target_critics,
        ):
            for parameter, target_parameter in zip(
                critic.parameters(),
                target_critic.parameters(),
            ):
                target_parameter.data.mul_(
                    1.0 - self.tau
                )

                target_parameter.data.add_(
                    self.tau * parameter.data
                )

    @torch.no_grad()
    def hard_update_targets(self) -> None:
        for actor, target_actor in zip(
            self.actors,
            self.target_actors,
        ):
            target_actor.load_state_dict(
                actor.state_dict()
            )

        for critic, target_critic in zip(
            self.critics,
            self.target_critics,
        ):
            target_critic.load_state_dict(
                critic.state_dict()
            )

    def save_checkpoint(
        self,
        path: str | Path,
        episode: int,
        noise_scale: float,
        extra_state: dict[str, Any] | None = None,
    ) -> None:
        checkpoint_path = Path(path)

        if not checkpoint_path.is_absolute():
            checkpoint_path = ROOT / checkpoint_path

        checkpoint_path.parent.mkdir(
            parents=True,
            exist_ok=True,
        )

        checkpoint = {
            "model_name": "CA-MADDPG",
            "episode": int(episode),
            "noise_scale": float(noise_scale),
            "total_updates": int(
                self.total_updates
            ),
            "actors": [
                actor.state_dict()
                for actor in self.actors
            ],
            "critics": [
                critic.state_dict()
                for critic in self.critics
            ],
            "target_actors": [
                actor.state_dict()
                for actor in self.target_actors
            ],
            "target_critics": [
                critic.state_dict()
                for critic in self.target_critics
            ],
            "actor_optimisers": [
                optimiser.state_dict()
                for optimiser in (
                    self.actor_optimisers
                )
            ],
            "critic_optimisers": [
                optimiser.state_dict()
                for optimiser in (
                    self.critic_optimisers
                )
            ],
            "extra_state": (
                extra_state
                if extra_state is not None
                else {}
            ),
        }

        torch.save(
            checkpoint,
            checkpoint_path,
        )

    def load_checkpoint(
        self,
        path: str | Path,
    ) -> dict[str, Any]:
        checkpoint_path = Path(path)

        if not checkpoint_path.is_absolute():
            checkpoint_path = ROOT / checkpoint_path

        checkpoint = torch.load(
            checkpoint_path,
            map_location=self.device,
            weights_only=False,
        )

        if checkpoint.get(
            "model_name"
        ) != "CA-MADDPG":
            raise ValueError(
                "The checkpoint is not a "
                "CA-MADDPG checkpoint."
            )

        for model, state_dict in zip(
            self.actors,
            checkpoint["actors"],
        ):
            model.load_state_dict(state_dict)

        for model, state_dict in zip(
            self.critics,
            checkpoint["critics"],
        ):
            model.load_state_dict(state_dict)

        for model, state_dict in zip(
            self.target_actors,
            checkpoint["target_actors"],
        ):
            model.load_state_dict(state_dict)

        for model, state_dict in zip(
            self.target_critics,
            checkpoint["target_critics"],
        ):
            model.load_state_dict(state_dict)

        for optimiser, state_dict in zip(
            self.actor_optimisers,
            checkpoint["actor_optimisers"],
        ):
            optimiser.load_state_dict(
                state_dict
            )

        for optimiser, state_dict in zip(
            self.critic_optimisers,
            checkpoint[
                "critic_optimisers"
            ],
        ):
            optimiser.load_state_dict(
                state_dict
            )

        self.total_updates = int(
            checkpoint.get(
                "total_updates",
                0,
            )
        )

        return checkpoint


if __name__ == "__main__":
    number_of_agents = 10
    observation_dimension = 29
    action_dimension = 3

    model = CAMADDPG(
        number_of_agents=number_of_agents,
        observation_dimension=(
            observation_dimension
        ),
        action_dimension=action_dimension,
    )

    observations = np.random.default_rng(
        42
    ).random(
        (
            number_of_agents,
            observation_dimension,
        ),
        dtype=np.float32,
    )

    actions = model.select_actions(
        observations,
        noise_scale=0.0,
    )

    actor_parameter_count = sum(
        parameter.numel()
        for actor in model.actors
        for parameter in actor.parameters()
    )

    critic_parameter_count = sum(
        parameter.numel()
        for critic in model.critics
        for parameter in critic.parameters()
    )

    print("Model: CA-MADDPG")
    print("Device:", model.device)
    print(
        "Number of actors:",
        len(model.actors),
    )
    print(
        "Number of critics:",
        len(model.critics),
    )
    print(
        "Selected-action shape:",
        actions.shape,
    )
    print(
        "Action minimum:",
        round(float(actions.min()), 6),
    )
    print(
        "Action maximum:",
        round(float(actions.max()), 6),
    )
    print(
        "Total actor parameters:",
        actor_parameter_count,
    )
    print(
        "Total critic parameters:",
        critic_parameter_count,
    )
