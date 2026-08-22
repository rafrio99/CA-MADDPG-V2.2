from __future__ import annotations

import copy
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F
import yaml

from ca_maddpg_v2_components import (
    SharedJointAttentionActor,
    SharedTeamAttentionCritic,
)


ROOT = Path(__file__).resolve().parents[1]


class CAMADDPGV2:
    """
    Shared Collaborative Attention-Enhanced MADDPG Version 2.

    A single joint actor produces UAV-selection probabilities and
    resource-allocation actions for the complete UAV team. A shared
    cooperative critic evaluates the full joint state-action pair.
    """

    def __init__(
        self,
        number_of_agents: int,
        observation_dimension: int,
        action_dimension: int,
        config_path: str = (
            "configs/ca_maddpg_v2_seed42_10ep.yaml"
        ),
        device: str | torch.device | None = None,
    ):
        self.number_of_agents = int(
            number_of_agents
        )

        self.observation_dimension = int(
            observation_dimension
        )

        self.action_dimension = int(
            action_dimension
        )

        config_file = Path(config_path)

        if not config_file.is_absolute():
            config_file = ROOT / config_file

        if not config_file.exists():
            raise FileNotFoundError(
                f"Configuration not found: {config_file}"
            )

        with config_file.open(
            "r",
            encoding="utf-8",
        ) as file:
            self.config = yaml.safe_load(file)

        model_config = self.config.get(
            "model",
            {},
        )

        training_config = self.config.get(
            "training",
            {},
        )

        if device is None:
            self.device = torch.device(
                "cuda:0"
                if torch.cuda.is_available()
                else "cpu"
            )
        else:
            self.device = torch.device(
                device
            )

        self.gamma = float(
            training_config.get(
                "gamma",
                0.99,
            )
        )

        self.tau = float(
            training_config.get(
                "tau",
                0.0025,
            )
        )

        self.actor_learning_rate = float(
            training_config.get(
                "actor_learning_rate",
                0.00005,
            )
        )

        self.critic_learning_rate = float(
            training_config.get(
                "critic_learning_rate",
                0.0005,
            )
        )

        self.gradient_clip_norm = float(
            training_config.get(
                "gradient_clip_norm",
                5.0,
            )
        )

        # Positive coefficient means the optimiser reduces
        # selection entropy and creates a clearer UAV ranking.
        self.selection_entropy_coefficient = float(
            training_config.get(
                "selection_entropy_coefficient",
                0.01,
            )
        )

        embedding_dimension = int(
            model_config.get(
                "attention_embedding_dimension",
                64,
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
                0.05,
            )
        )

        self.selection_temperature = float(
            model_config.get(
                "selection_temperature",
                0.25,
            )
        )

        self.minimum_resource_fraction = float(
            model_config.get(
                "minimum_resource_fraction",
                0.50,
            )
        )

        hidden_dimensions = tuple(
            int(value)
            for value in model_config.get(
                "hidden_dimensions",
                [128, 128],
            )
        )

        self.actor = SharedJointAttentionActor(
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
            selection_temperature=(
                self.selection_temperature
            ),
            minimum_resource_fraction=(
                self.minimum_resource_fraction
            ),
            hidden_dimensions=hidden_dimensions,
        ).to(self.device)

        self.critic = SharedTeamAttentionCritic(
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
        ).to(self.device)

        self.target_actor = copy.deepcopy(
            self.actor
        ).to(self.device)

        self.target_critic = copy.deepcopy(
            self.critic
        ).to(self.device)

        self.target_actor.requires_grad_(
            False
        )

        self.target_critic.requires_grad_(
            False
        )

        self.actor_optimiser = (
            torch.optim.Adam(
                self.actor.parameters(),
                lr=self.actor_learning_rate,
            )
        )

        self.critic_optimiser = (
            torch.optim.Adam(
                self.critic.parameters(),
                lr=self.critic_learning_rate,
            )
        )

        self.total_updates = 0

    @staticmethod
    def _as_tensor(
        value: Any,
        device: torch.device,
    ) -> torch.Tensor:
        if isinstance(
            value,
            torch.Tensor,
        ):
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
            observations = batch[
                "observations"
            ]

            actions = batch["actions"]
            rewards = batch["rewards"]

            next_observations = batch[
                "next_observations"
            ]

            dones = batch["dones"]

        elif isinstance(
            batch,
            (tuple, list),
        ):
            if len(batch) != 5:
                raise ValueError(
                    "Batch must contain five elements."
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
                "Unsupported batch format."
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
                "Unexpected observation shape: "
                f"{tuple(observations.shape)}"
            )

        if (
            tuple(actions.shape)
            != expected_action_shape
        ):
            raise ValueError(
                "Unexpected action shape: "
                f"{tuple(actions.shape)}"
            )

        return (
            observations,
            actions,
            rewards,
            next_observations,
            dones,
        )

    @staticmethod
    def _softmax_numpy(
        logits: np.ndarray,
    ) -> np.ndarray:
        shifted = logits - np.max(
            logits
        )

        exponentials = np.exp(
            shifted
        )

        return exponentials / np.sum(
            exponentials
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

        previous_training_state = (
            self.actor.training
        )

        self.actor.eval()

        with torch.no_grad():
            action_tensor = self.actor(
                observation_tensor
            )

        self.actor.train(
            previous_training_state
        )

        actions = (
            action_tensor.detach()
            .cpu()
            .numpy()
            .astype(np.float32)
        )

        if noise_scale > 0.0:
            # Preserve the sum-to-one selection structure.
            selection_probabilities = np.clip(
                actions[:, 0],
                1e-8,
                1.0,
            )

            noisy_selection_logits = (
                np.log(
                    selection_probabilities
                )
                + np.random.normal(
                    loc=0.0,
                    scale=float(
                        noise_scale
                    ),
                    size=(
                        self.number_of_agents
                    ),
                )
            )

            actions[:, 0] = (
                self._softmax_numpy(
                    noisy_selection_logits
                )
            )

            resource_noise = (
                np.random.normal(
                    loc=0.0,
                    scale=float(
                        noise_scale
                    ),
                    size=(
                        self.number_of_agents,
                        2,
                    ),
                ).astype(np.float32)
            )

            actions[:, 1:] = np.clip(
                actions[:, 1:]
                + resource_noise,
                self.minimum_resource_fraction,
                1.0,
            )

        actions[:, 0] = (
            actions[:, 0]
            / max(
                float(
                    actions[:, 0].sum()
                ),
                1e-8,
            )
        )

        actions[:, 1:] = np.clip(
            actions[:, 1:],
            self.minimum_resource_fraction,
            1.0,
        )

        return actions.astype(
            np.float32
        )

    def update(
        self,
        batch: Any,
    ) -> dict[str, float]:
        (
            observations,
            replay_actions,
            rewards,
            next_observations,
            dones,
        ) = self._prepare_batch(
            batch
        )

        # The environment uses cooperative rewards.
        # Mean is robust even when all agents receive
        # identical values.
        team_rewards = rewards.mean(
            dim=1,
            keepdim=True,
        )

        team_dones = dones.max(
            dim=1,
            keepdim=True,
        ).values

        self.target_actor.eval()
        self.target_critic.eval()

        with torch.no_grad():
            target_actions = (
                self.target_actor(
                    next_observations
                )
            )

            target_q_values = (
                self.target_critic(
                    next_observations,
                    target_actions,
                )
            )

            target_values = (
                team_rewards
                + self.gamma
                * (
                    1.0
                    - team_dones
                )
                * target_q_values
            )

        current_q_values = self.critic(
            observations,
            replay_actions,
        )

        critic_loss = F.mse_loss(
            current_q_values,
            target_values,
        )

        self.critic_optimiser.zero_grad(
            set_to_none=True
        )

        critic_loss.backward()

        torch.nn.utils.clip_grad_norm_(
            self.critic.parameters(),
            max_norm=(
                self.gradient_clip_norm
            ),
        )

        self.critic_optimiser.step()

        for parameter in (
            self.critic.parameters()
        ):
            parameter.requires_grad_(
                False
            )

        (
            policy_actions,
            actor_diagnostics,
        ) = self.actor(
            observations,
            return_attention=True,
        )

        policy_q_values = self.critic(
            observations,
            policy_actions,
        )

        selection_probabilities = (
            actor_diagnostics[
                "selection_probabilities"
            ]
        )

        selection_entropy = -(
            selection_probabilities
            * torch.log(
                selection_probabilities
                + 1e-8
            )
        ).sum(
            dim=1
        ).mean()

        mean_selection_gap = (
            actor_diagnostics[
                "selection_gap"
            ].mean()
        )

        actor_policy_loss = -(
            policy_q_values.mean()
        )

        actor_loss = (
            actor_policy_loss
            + self.selection_entropy_coefficient
            * selection_entropy
        )

        self.actor_optimiser.zero_grad(
            set_to_none=True
        )

        actor_loss.backward()

        torch.nn.utils.clip_grad_norm_(
            self.actor.parameters(),
            max_norm=(
                self.gradient_clip_norm
            ),
        )

        self.actor_optimiser.step()

        for parameter in (
            self.critic.parameters()
        ):
            parameter.requires_grad_(
                True
            )

        self.soft_update_targets()

        self.total_updates += 1

        return {
            "critic_loss": float(
                critic_loss.detach().item()
            ),
            "actor_loss": float(
                actor_loss.detach().item()
            ),
            "actor_policy_loss": float(
                actor_policy_loss.detach().item()
            ),
            "selection_entropy": float(
                selection_entropy.detach().item()
            ),
            "mean_selection_gap": float(
                mean_selection_gap.detach().item()
            ),
            "total_updates": int(
                self.total_updates
            ),
        }

    @torch.no_grad()
    def soft_update_targets(
        self,
    ) -> None:
        for parameter, target_parameter in zip(
            self.actor.parameters(),
            self.target_actor.parameters(),
        ):
            target_parameter.data.mul_(
                1.0 - self.tau
            )

            target_parameter.data.add_(
                self.tau
                * parameter.data
            )

        for parameter, target_parameter in zip(
            self.critic.parameters(),
            self.target_critic.parameters(),
        ):
            target_parameter.data.mul_(
                1.0 - self.tau
            )

            target_parameter.data.add_(
                self.tau
                * parameter.data
            )

    @torch.no_grad()
    def hard_update_targets(
        self,
    ) -> None:
        self.target_actor.load_state_dict(
            self.actor.state_dict()
        )

        self.target_critic.load_state_dict(
            self.critic.state_dict()
        )

    def save_checkpoint(
        self,
        path: str | Path,
        episode: int,
        noise_scale: float,
        extra_state: dict[str, Any]
        | None = None,
    ) -> None:
        checkpoint_path = Path(path)

        if not checkpoint_path.is_absolute():
            checkpoint_path = (
                ROOT / checkpoint_path
            )

        checkpoint_path.parent.mkdir(
            parents=True,
            exist_ok=True,
        )

        checkpoint = {
            "model_name": "CA-MADDPG-V2",
            "episode": int(episode),
            "noise_scale": float(
                noise_scale
            ),
            "total_updates": int(
                self.total_updates
            ),
            "actor": (
                self.actor.state_dict()
            ),
            "critic": (
                self.critic.state_dict()
            ),
            "target_actor": (
                self.target_actor.state_dict()
            ),
            "target_critic": (
                self.target_critic.state_dict()
            ),
            "actor_optimiser": (
                self.actor_optimiser.state_dict()
            ),
            "critic_optimiser": (
                self.critic_optimiser.state_dict()
            ),
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
            checkpoint_path = (
                ROOT / checkpoint_path
            )

        checkpoint = torch.load(
            checkpoint_path,
            map_location=self.device,
            weights_only=False,
        )

        if checkpoint.get(
            "model_name"
        ) != "CA-MADDPG-V2":
            raise ValueError(
                "Checkpoint is not a "
                "CA-MADDPG-V2 checkpoint."
            )

        self.actor.load_state_dict(
            checkpoint["actor"]
        )

        self.critic.load_state_dict(
            checkpoint["critic"]
        )

        self.target_actor.load_state_dict(
            checkpoint[
                "target_actor"
            ]
        )

        self.target_critic.load_state_dict(
            checkpoint[
                "target_critic"
            ]
        )

        self.actor_optimiser.load_state_dict(
            checkpoint[
                "actor_optimiser"
            ]
        )

        self.critic_optimiser.load_state_dict(
            checkpoint[
                "critic_optimiser"
            ]
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
    batch_size = 8

    rng = np.random.default_rng(
        42
    )

    model = CAMADDPGV2(
        number_of_agents=number_of_agents,
        observation_dimension=(
            observation_dimension
        ),
        action_dimension=action_dimension,
    )

    observations = rng.random(
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

    batch = {
        "observations": rng.random(
            (
                batch_size,
                number_of_agents,
                observation_dimension,
            ),
            dtype=np.float32,
        ),
        "actions": rng.random(
            (
                batch_size,
                number_of_agents,
                action_dimension,
            ),
            dtype=np.float32,
        ),
        "rewards": rng.normal(
            loc=-1.0,
            scale=0.5,
            size=(
                batch_size,
                number_of_agents,
            ),
        ).astype(np.float32),
        "next_observations": rng.random(
            (
                batch_size,
                number_of_agents,
                observation_dimension,
            ),
            dtype=np.float32,
        ),
        "dones": np.zeros(
            (
                batch_size,
                number_of_agents,
            ),
            dtype=np.float32,
        ),
    }

    metrics = model.update(
        batch
    )

    print("=" * 76)
    print("CA-MADDPG-V2 MANAGER TEST")
    print("=" * 76)
    print("Device:", model.device)
    print(
        "Action shape:",
        actions.shape,
    )
    print(
        "Selection sum:",
        f"{actions[:, 0].sum():.6f}",
    )
    print(
        "CPU minimum:",
        f"{actions[:, 1].min():.6f}",
    )
    print(
        "Bandwidth minimum:",
        f"{actions[:, 2].min():.6f}",
    )
    print(
        "Critic loss:",
        f"{metrics['critic_loss']:.6f}",
    )
    print(
        "Actor loss:",
        f"{metrics['actor_loss']:.6f}",
    )
    print(
        "Selection entropy:",
        f"{metrics['selection_entropy']:.6f}",
    )
    print(
        "Mean selection gap:",
        f"{metrics['mean_selection_gap']:.6f}",
    )
    print(
        "Total updates:",
        model.total_updates,
    )
    print("=" * 76)

    assert actions.shape == (
        number_of_agents,
        action_dimension,
    )

    assert np.isclose(
        actions[:, 0].sum(),
        1.0,
        atol=1e-6,
    )

    assert actions[:, 1:].min() >= 0.50

    assert np.isfinite(
        metrics["critic_loss"]
    )

    assert np.isfinite(
        metrics["actor_loss"]
    )

    print(
        "CA-MADDPG-V2 manager test "
        "passed successfully."
    )
