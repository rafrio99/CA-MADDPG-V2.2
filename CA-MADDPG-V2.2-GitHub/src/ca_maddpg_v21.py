from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F

from ca_maddpg_v2 import CAMADDPGV2


ROOT = Path(__file__).resolve().parents[1]


class CAMADDPGV21(CAMADDPGV2):
    """
    CA-MADDPG Version 2.1.

    This version retains the shared collaborative-attention
    architecture from Version 2 and adds resource-allocation
    stabilisation to prevent low-CPU/low-bandwidth collapse.
    """

    CHECKPOINT_MODEL_NAME = "CA-MADDPG-V2.1"

    def __init__(
        self,
        number_of_agents: int,
        observation_dimension: int,
        action_dimension: int,
        config_path: str = (
            "configs/"
            "ca_maddpg_v21_seed42_10ep.yaml"
        ),
        device: str | torch.device | None = None,
    ):
        super().__init__(
            number_of_agents=number_of_agents,
            observation_dimension=(
                observation_dimension
            ),
            action_dimension=action_dimension,
            config_path=config_path,
            device=device,
        )

        model_config = self.config.get(
            "model",
            {},
        )

        training_config = self.config.get(
            "training",
            {},
        )

        self.minimum_cpu_fraction = float(
            model_config.get(
                "minimum_cpu_fraction",
                model_config.get(
                    "minimum_resource_fraction",
                    0.75,
                ),
            )
        )

        self.minimum_bandwidth_fraction = float(
            model_config.get(
                "minimum_bandwidth_fraction",
                model_config.get(
                    "minimum_resource_fraction",
                    0.75,
                ),
            )
        )

        self.target_cpu_fraction = float(
            model_config.get(
                "target_cpu_fraction",
                0.90,
            )
        )

        self.target_bandwidth_fraction = float(
            model_config.get(
                "target_bandwidth_fraction",
                0.90,
            )
        )

        self.resource_adequacy_coefficient = float(
            training_config.get(
                "resource_adequacy_coefficient",
                0.05,
            )
        )

        if not (
            0.0
            <= self.minimum_cpu_fraction
            <= self.target_cpu_fraction
            <= 1.0
        ):
            raise ValueError(
                "CPU fractions must satisfy "
                "0 <= minimum <= target <= 1."
            )

        if not (
            0.0
            <= self.minimum_bandwidth_fraction
            <= self.target_bandwidth_fraction
            <= 1.0
        ):
            raise ValueError(
                "Bandwidth fractions must satisfy "
                "0 <= minimum <= target <= 1."
            )

        if self.resource_adequacy_coefficient < 0.0:
            raise ValueError(
                "Resource adequacy coefficient "
                "must be non-negative."
            )

    def select_actions(
        self,
        observations: np.ndarray,
        noise_scale: float = 0.0,
    ) -> np.ndarray:
        actions = super().select_actions(
            observations=observations,
            noise_scale=noise_scale,
        )

        actions[:, 1] = np.clip(
            actions[:, 1],
            self.minimum_cpu_fraction,
            1.0,
        )

        actions[:, 2] = np.clip(
            actions[:, 2],
            self.minimum_bandwidth_fraction,
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
        ) = self._prepare_batch(batch)

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
            target_actions = self.target_actor(
                next_observations
            )

            target_q_values = self.target_critic(
                next_observations,
                target_actions,
            )

            target_values = (
                team_rewards
                + self.gamma
                * (1.0 - team_dones)
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

        critic_gradient_norm = (
            torch.nn.utils.clip_grad_norm_(
                self.critic.parameters(),
                max_norm=self.gradient_clip_norm,
            )
        )

        self.critic_optimiser.step()

        for parameter in self.critic.parameters():
            parameter.requires_grad_(False)

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

        cpu_actions = policy_actions[
            :,
            :,
            1,
        ]

        bandwidth_actions = policy_actions[
            :,
            :,
            2,
        ]

        # Differentiable resource values associated with
        # the UAV-selection probabilities.
        expected_selected_cpu = torch.sum(
            selection_probabilities
            * cpu_actions,
            dim=1,
        )

        expected_selected_bandwidth = torch.sum(
            selection_probabilities
            * bandwidth_actions,
            dim=1,
        )

        cpu_shortfall = F.relu(
            self.target_cpu_fraction
            - expected_selected_cpu
        )

        bandwidth_shortfall = F.relu(
            self.target_bandwidth_fraction
            - expected_selected_bandwidth
        )

        cpu_adequacy_loss = (
            cpu_shortfall.pow(2).mean()
        )

        bandwidth_adequacy_loss = (
            bandwidth_shortfall.pow(2).mean()
        )

        resource_adequacy_loss = (
            cpu_adequacy_loss
            + bandwidth_adequacy_loss
        )

        actor_policy_loss = -(
            policy_q_values.mean()
        )

        actor_loss = (
            actor_policy_loss
            + self.selection_entropy_coefficient
            * selection_entropy
            + self.resource_adequacy_coefficient
            * resource_adequacy_loss
        )

        self.actor_optimiser.zero_grad(
            set_to_none=True
        )

        actor_loss.backward()

        actor_gradient_norm = (
            torch.nn.utils.clip_grad_norm_(
                self.actor.parameters(),
                max_norm=self.gradient_clip_norm,
            )
        )

        self.actor_optimiser.step()

        for parameter in self.critic.parameters():
            parameter.requires_grad_(True)

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
            "resource_adequacy_loss": float(
                resource_adequacy_loss.detach().item()
            ),
            "cpu_adequacy_loss": float(
                cpu_adequacy_loss.detach().item()
            ),
            "bandwidth_adequacy_loss": float(
                bandwidth_adequacy_loss
                .detach()
                .item()
            ),
            "expected_selected_cpu": float(
                expected_selected_cpu
                .mean()
                .detach()
                .item()
            ),
            "expected_selected_bandwidth": float(
                expected_selected_bandwidth
                .mean()
                .detach()
                .item()
            ),
            "critic_gradient_norm": float(
                torch.as_tensor(
                    critic_gradient_norm
                )
                .detach()
                .item()
            ),
            "actor_gradient_norm": float(
                torch.as_tensor(
                    actor_gradient_norm
                )
                .detach()
                .item()
            ),
            "total_updates": int(
                self.total_updates
            ),
        }

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
            "model_name": (
                self.CHECKPOINT_MODEL_NAME
            ),
            "episode": int(episode),
            "noise_scale": float(
                noise_scale
            ),
            "total_updates": int(
                self.total_updates
            ),
            "actor": self.actor.state_dict(),
            "critic": self.critic.state_dict(),
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
            "resource_settings": {
                "minimum_cpu_fraction": (
                    self.minimum_cpu_fraction
                ),
                "minimum_bandwidth_fraction": (
                    self.minimum_bandwidth_fraction
                ),
                "target_cpu_fraction": (
                    self.target_cpu_fraction
                ),
                "target_bandwidth_fraction": (
                    self.target_bandwidth_fraction
                ),
                "resource_adequacy_coefficient": (
                    self.resource_adequacy_coefficient
                ),
            },
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
        ) != self.CHECKPOINT_MODEL_NAME:
            raise ValueError(
                "Checkpoint is not a "
                "CA-MADDPG-V2.1 checkpoint."
            )

        self.actor.load_state_dict(
            checkpoint["actor"]
        )

        self.critic.load_state_dict(
            checkpoint["critic"]
        )

        self.target_actor.load_state_dict(
            checkpoint["target_actor"]
        )

        self.target_critic.load_state_dict(
            checkpoint["target_critic"]
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
    batch_size = 16

    rng = np.random.default_rng(42)
    torch.manual_seed(42)

    model = CAMADDPGV21(
        number_of_agents=number_of_agents,
        observation_dimension=(
            observation_dimension
        ),
        action_dimension=action_dimension,
        config_path=(
            "configs/"
            "ca_maddpg_v21_seed42_10ep.yaml"
        ),
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
        "actions": np.concatenate(
            [
                rng.dirichlet(
                    np.ones(number_of_agents),
                    size=batch_size,
                ).astype(
                    np.float32
                )[:, :, None],
                rng.uniform(
                    low=0.75,
                    high=1.0,
                    size=(
                        batch_size,
                        number_of_agents,
                        2,
                    ),
                ).astype(np.float32),
            ],
            axis=2,
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

    actor_parameter_before = next(
        model.actor.parameters()
    ).detach().clone()

    metrics = model.update(batch)

    actor_parameter_after = next(
        model.actor.parameters()
    ).detach().clone()

    actor_changed = not torch.equal(
        actor_parameter_before,
        actor_parameter_after,
    )

    # Compute the deterministic reference action after
    # the learning update. The saved checkpoint contains
    # these updated actor parameters.
    reference_actions = model.select_actions(
        observations,
        noise_scale=0.0,
    )

    checkpoint_path = Path(
        "checkpoints/"
        "ca_maddpg_v21_seed42_10ep/"
        "ca_maddpg_v21_manager_test.pt"
    )

    model.save_checkpoint(
        path=checkpoint_path,
        episode=1,
        noise_scale=0.20,
        extra_state={
            "test": "manager_update",
        },
    )

    restored_model = CAMADDPGV21(
        number_of_agents=number_of_agents,
        observation_dimension=(
            observation_dimension
        ),
        action_dimension=action_dimension,
        config_path=(
            "configs/"
            "ca_maddpg_v21_seed42_10ep.yaml"
        ),
    )

    checkpoint = restored_model.load_checkpoint(
        checkpoint_path
    )

    restored_actions = (
        restored_model.select_actions(
            observations,
            noise_scale=0.0,
        )
    )

    actions_match = np.allclose(
        reference_actions,
        restored_actions,
        atol=1e-6,
    )

    print("=" * 78)
    print("CA-MADDPG-V2.1 MANAGER TEST")
    print("=" * 78)
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
        "Expected selected CPU:",
        f"{metrics['expected_selected_cpu']:.6f}",
    )
    print(
        "Expected selected bandwidth:",
        f"{metrics['expected_selected_bandwidth']:.6f}",
    )
    print(
        "Resource adequacy loss:",
        f"{metrics['resource_adequacy_loss']:.8f}",
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
        "Selection gap:",
        f"{metrics['mean_selection_gap']:.6f}",
    )
    print(
        "Actor parameters changed:",
        actor_changed,
    )
    print(
        "Loaded episode:",
        checkpoint["episode"],
    )
    print(
        "Loaded updates:",
        restored_model.total_updates,
    )
    print(
        "Checkpoint actions match:",
        actions_match,
    )
    print(
        "Checkpoint saved:",
        checkpoint_path,
    )
    print("=" * 78)

    assert actions.shape == (
        number_of_agents,
        action_dimension,
    )

    assert np.isclose(
        actions[:, 0].sum(),
        1.0,
        atol=1e-6,
    )

    assert actions[:, 1].min() >= 0.75
    assert actions[:, 2].min() >= 0.75

    assert np.isfinite(
        metrics["critic_loss"]
    )

    assert np.isfinite(
        metrics["actor_loss"]
    )

    assert np.isfinite(
        metrics["resource_adequacy_loss"]
    )

    assert actor_changed
    assert restored_model.total_updates == 1
    assert actions_match

    print(
        "CA-MADDPG-V2.1 manager test "
        "passed successfully."
    )
