from __future__ import annotations

from typing import Dict, Sequence

import numpy as np
import torch
from torch import nn


def build_mlp(
    input_dimension: int,
    hidden_dimensions: Sequence[int],
    output_dimension: int,
    output_activation: nn.Module | None = None,
) -> nn.Sequential:
    layers = []
    current_dimension = input_dimension

    for hidden_dimension in hidden_dimensions:
        layers.extend(
            [
                nn.Linear(current_dimension, hidden_dimension),
                nn.ReLU(),
            ]
        )
        current_dimension = hidden_dimension

    layers.append(
        nn.Linear(current_dimension, output_dimension)
    )

    if output_activation is not None:
        layers.append(output_activation)

    return nn.Sequential(*layers)


class Actor(nn.Module):
    """Decentralised actor for one UAV agent."""

    def __init__(
        self,
        observation_dimension: int,
        action_dimension: int,
        hidden_dimensions: Sequence[int] = (256, 256),
    ):
        super().__init__()

        self.network = build_mlp(
            input_dimension=observation_dimension,
            hidden_dimensions=hidden_dimensions,
            output_dimension=action_dimension,
            output_activation=nn.Sigmoid(),
        )

    def forward(self, observation: torch.Tensor) -> torch.Tensor:
        return self.network(observation)


class Critic(nn.Module):
    """Centralised critic using all UAV observations and actions."""

    def __init__(
        self,
        global_observation_dimension: int,
        joint_action_dimension: int,
        hidden_dimensions: Sequence[int] = (256, 256),
    ):
        super().__init__()

        critic_input_dimension = (
            global_observation_dimension
            + joint_action_dimension
        )

        self.network = build_mlp(
            input_dimension=critic_input_dimension,
            hidden_dimensions=hidden_dimensions,
            output_dimension=1,
        )

    def forward(
        self,
        global_observation: torch.Tensor,
        joint_action: torch.Tensor,
    ) -> torch.Tensor:
        critic_input = torch.cat(
            [global_observation, joint_action],
            dim=-1,
        )

        return self.network(critic_input)


class ReplayBuffer:
    """Fixed-size replay buffer for multi-agent transitions."""

    def __init__(
        self,
        capacity: int,
        number_of_agents: int,
        observation_dimension: int,
        action_dimension: int,
        device: torch.device,
        seed: int = 42,
    ):
        if capacity <= 0:
            raise ValueError("Replay-buffer capacity must be positive.")

        self.capacity = int(capacity)
        self.number_of_agents = int(number_of_agents)
        self.observation_dimension = int(observation_dimension)
        self.action_dimension = int(action_dimension)
        self.device = device

        self.observations = np.empty(
            (
                self.capacity,
                self.number_of_agents,
                self.observation_dimension,
            ),
            dtype=np.float32,
        )

        self.actions = np.empty(
            (
                self.capacity,
                self.number_of_agents,
                self.action_dimension,
            ),
            dtype=np.float32,
        )

        self.rewards = np.empty(
            (self.capacity, self.number_of_agents),
            dtype=np.float32,
        )

        self.next_observations = np.empty_like(
            self.observations
        )

        self.dones = np.empty(
            (self.capacity, self.number_of_agents),
            dtype=np.float32,
        )

        self.position = 0
        self.size = 0
        self.random_generator = np.random.default_rng(seed)

    def __len__(self) -> int:
        return self.size

    def add(
        self,
        observations: np.ndarray,
        actions: np.ndarray,
        rewards: np.ndarray,
        next_observations: np.ndarray,
        dones: np.ndarray,
    ) -> None:
        expected_observation_shape = (
            self.number_of_agents,
            self.observation_dimension,
        )

        expected_action_shape = (
            self.number_of_agents,
            self.action_dimension,
        )

        expected_vector_shape = (
            self.number_of_agents,
        )

        observations = np.asarray(
            observations,
            dtype=np.float32,
        )

        actions = np.asarray(
            actions,
            dtype=np.float32,
        )

        rewards = np.asarray(
            rewards,
            dtype=np.float32,
        )

        next_observations = np.asarray(
            next_observations,
            dtype=np.float32,
        )

        dones = np.asarray(
            dones,
            dtype=np.float32,
        )

        if observations.shape != expected_observation_shape:
            raise ValueError(
                f"Observation shape {observations.shape} does not "
                f"match {expected_observation_shape}."
            )

        if next_observations.shape != expected_observation_shape:
            raise ValueError(
                f"Next-observation shape "
                f"{next_observations.shape} does not match "
                f"{expected_observation_shape}."
            )

        if actions.shape != expected_action_shape:
            raise ValueError(
                f"Action shape {actions.shape} does not "
                f"match {expected_action_shape}."
            )

        if rewards.shape != expected_vector_shape:
            raise ValueError(
                f"Reward shape {rewards.shape} does not "
                f"match {expected_vector_shape}."
            )

        if dones.shape != expected_vector_shape:
            raise ValueError(
                f"Done shape {dones.shape} does not "
                f"match {expected_vector_shape}."
            )

        self.observations[self.position] = observations
        self.actions[self.position] = actions
        self.rewards[self.position] = rewards
        self.next_observations[self.position] = next_observations
        self.dones[self.position] = dones

        self.position = (
            self.position + 1
        ) % self.capacity

        self.size = min(
            self.size + 1,
            self.capacity,
        )

    def sample(self, batch_size: int) -> Dict[str, torch.Tensor]:
        if self.size < batch_size:
            raise ValueError(
                f"Requested batch size {batch_size}, but the "
                f"buffer contains only {self.size} transitions."
            )

        indices = self.random_generator.integers(
            low=0,
            high=self.size,
            size=batch_size,
        )

        return {
            "observations": torch.as_tensor(
                self.observations[indices],
                dtype=torch.float32,
                device=self.device,
            ),
            "actions": torch.as_tensor(
                self.actions[indices],
                dtype=torch.float32,
                device=self.device,
            ),
            "rewards": torch.as_tensor(
                self.rewards[indices],
                dtype=torch.float32,
                device=self.device,
            ),
            "next_observations": torch.as_tensor(
                self.next_observations[indices],
                dtype=torch.float32,
                device=self.device,
            ),
            "dones": torch.as_tensor(
                self.dones[indices],
                dtype=torch.float32,
                device=self.device,
            ),
        }


@torch.no_grad()
def hard_update(
    target_network: nn.Module,
    source_network: nn.Module,
) -> None:
    target_network.load_state_dict(
        source_network.state_dict()
    )


@torch.no_grad()
def soft_update(
    target_network: nn.Module,
    source_network: nn.Module,
    tau: float,
) -> None:
    for target_parameter, source_parameter in zip(
        target_network.parameters(),
        source_network.parameters(),
    ):
        target_parameter.data.mul_(1.0 - tau)
        target_parameter.data.add_(
            source_parameter.data,
            alpha=tau,
        )


if __name__ == "__main__":
    device = torch.device(
        "cuda:0" if torch.cuda.is_available() else "cpu"
    )

    number_of_agents = 10
    observation_dimension = 29
    action_dimension = 3

    actor = Actor(
        observation_dimension=observation_dimension,
        action_dimension=action_dimension,
    ).to(device)

    critic = Critic(
        global_observation_dimension=(
            number_of_agents * observation_dimension
        ),
        joint_action_dimension=(
            number_of_agents * action_dimension
        ),
    ).to(device)

    observations = torch.rand(
        4,
        observation_dimension,
        device=device,
    )

    global_observations = torch.rand(
        4,
        number_of_agents * observation_dimension,
        device=device,
    )

    joint_actions = torch.rand(
        4,
        number_of_agents * action_dimension,
        device=device,
    )

    actor_output = actor(observations)
    critic_output = critic(
        global_observations,
        joint_actions,
    )

    print("Device:", device)
    print("Actor output shape:", tuple(actor_output.shape))
    print("Critic output shape:", tuple(critic_output.shape))
