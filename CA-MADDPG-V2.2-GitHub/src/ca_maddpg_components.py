from __future__ import annotations

from typing import Sequence

import torch
from torch import nn


def build_mlp(
    input_dimension: int,
    hidden_dimensions: Sequence[int],
    output_dimension: int,
    output_activation: nn.Module | None = None,
) -> nn.Sequential:
    """Create a feed-forward neural network."""

    layers: list[nn.Module] = []
    current_dimension = int(input_dimension)

    for hidden_dimension in hidden_dimensions:
        layers.extend(
            [
                nn.Linear(
                    current_dimension,
                    int(hidden_dimension),
                ),
                nn.ReLU(),
            ]
        )

        current_dimension = int(hidden_dimension)

    layers.append(
        nn.Linear(
            current_dimension,
            int(output_dimension),
        )
    )

    if output_activation is not None:
        layers.append(output_activation)

    return nn.Sequential(*layers)


class CollaborativeAttentionBlock(nn.Module):
    """Multi-head self-attention block for UAV collaboration."""

    def __init__(
        self,
        embedding_dimension: int,
        number_of_heads: int,
        dropout: float = 0.10,
        feedforward_multiplier: int = 4,
    ):
        super().__init__()

        if embedding_dimension % number_of_heads != 0:
            raise ValueError(
                "The embedding dimension must be divisible "
                "by the number of attention heads."
            )

        self.self_attention = nn.MultiheadAttention(
            embed_dim=embedding_dimension,
            num_heads=number_of_heads,
            dropout=dropout,
            batch_first=True,
        )

        self.attention_dropout = nn.Dropout(dropout)
        self.attention_normalisation = nn.LayerNorm(
            embedding_dimension
        )

        feedforward_dimension = (
            embedding_dimension
            * feedforward_multiplier
        )

        self.feedforward = nn.Sequential(
            nn.Linear(
                embedding_dimension,
                feedforward_dimension,
            ),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(
                feedforward_dimension,
                embedding_dimension,
            ),
            nn.Dropout(dropout),
        )

        self.feedforward_normalisation = nn.LayerNorm(
            embedding_dimension
        )

    def forward(
        self,
        tokens: torch.Tensor,
        return_attention: bool = False,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        attention_output, attention_weights = (
            self.self_attention(
                query=tokens,
                key=tokens,
                value=tokens,
                need_weights=return_attention,
                average_attn_weights=False,
            )
        )

        tokens = self.attention_normalisation(
            tokens
            + self.attention_dropout(
                attention_output
            )
        )

        feedforward_output = self.feedforward(
            tokens
        )

        tokens = self.feedforward_normalisation(
            tokens + feedforward_output
        )

        return tokens, attention_weights


class CollaborativeObservationEncoder(nn.Module):
    """
    Encode all UAV observations and exchange information
    through collaborative multi-head self-attention.
    """

    def __init__(
        self,
        number_of_agents: int,
        input_dimension: int,
        embedding_dimension: int = 128,
        number_of_heads: int = 4,
        number_of_layers: int = 1,
        dropout: float = 0.10,
    ):
        super().__init__()

        if number_of_agents <= 0:
            raise ValueError(
                "The number of agents must be positive."
            )

        if number_of_layers <= 0:
            raise ValueError(
                "The number of attention layers must be positive."
            )

        self.number_of_agents = int(
            number_of_agents
        )

        self.input_dimension = int(
            input_dimension
        )

        self.embedding_dimension = int(
            embedding_dimension
        )

        self.input_projection = nn.Sequential(
            nn.Linear(
                self.input_dimension,
                self.embedding_dimension,
            ),
            nn.GELU(),
            nn.LayerNorm(
                self.embedding_dimension
            ),
        )

        self.agent_identity_embedding = nn.Embedding(
            self.number_of_agents,
            self.embedding_dimension,
        )

        self.attention_layers = nn.ModuleList(
            [
                CollaborativeAttentionBlock(
                    embedding_dimension=(
                        self.embedding_dimension
                    ),
                    number_of_heads=number_of_heads,
                    dropout=dropout,
                )
                for _ in range(number_of_layers)
            ]
        )

    def forward(
        self,
        joint_inputs: torch.Tensor,
        return_attention: bool = False,
    ) -> tuple[
        torch.Tensor,
        list[torch.Tensor],
    ]:
        if joint_inputs.ndim != 3:
            raise ValueError(
                "Joint inputs must have shape "
                "[batch, agents, features]."
            )

        batch_size, number_of_agents, input_dimension = (
            joint_inputs.shape
        )

        if number_of_agents != self.number_of_agents:
            raise ValueError(
                f"Expected {self.number_of_agents} agents, "
                f"but received {number_of_agents}."
            )

        if input_dimension != self.input_dimension:
            raise ValueError(
                f"Expected input dimension "
                f"{self.input_dimension}, but received "
                f"{input_dimension}."
            )

        tokens = self.input_projection(
            joint_inputs
        )

        agent_indices = torch.arange(
            self.number_of_agents,
            device=joint_inputs.device,
        ).unsqueeze(0).expand(
            batch_size,
            -1,
        )

        tokens = (
            tokens
            + self.agent_identity_embedding(
                agent_indices
            )
        )

        all_attention_weights: list[
            torch.Tensor
        ] = []

        for attention_layer in self.attention_layers:
            tokens, attention_weights = (
                attention_layer(
                    tokens,
                    return_attention=return_attention,
                )
            )

            if (
                return_attention
                and attention_weights is not None
            ):
                all_attention_weights.append(
                    attention_weights
                )

        return tokens, all_attention_weights


class AttentionPooling(nn.Module):
    """Learn a weighted summary of all UAV representations."""

    def __init__(
        self,
        embedding_dimension: int,
    ):
        super().__init__()

        self.score_network = nn.Sequential(
            nn.Linear(
                embedding_dimension,
                embedding_dimension,
            ),
            nn.Tanh(),
            nn.Linear(
                embedding_dimension,
                1,
            ),
        )

    def forward(
        self,
        tokens: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        attention_scores = self.score_network(
            tokens
        ).squeeze(-1)

        pooling_weights = torch.softmax(
            attention_scores,
            dim=-1,
        )

        context = torch.sum(
            tokens
            * pooling_weights.unsqueeze(-1),
            dim=1,
        )

        return context, pooling_weights


class CollaborativeAttentionActor(nn.Module):
    """
    Decentralised actor enhanced by collaborative UAV context.

    The actor combines:
    1. The attended representation of its own UAV.
    2. An attention-pooled representation of all UAVs.
    """

    def __init__(
        self,
        number_of_agents: int,
        observation_dimension: int,
        action_dimension: int,
        agent_index: int,
        embedding_dimension: int = 128,
        number_of_heads: int = 4,
        number_of_layers: int = 1,
        dropout: float = 0.10,
        hidden_dimensions: Sequence[int] = (
            256,
            256,
        ),
    ):
        super().__init__()

        if not 0 <= agent_index < number_of_agents:
            raise ValueError(
                "Agent index is outside the valid range."
            )

        self.number_of_agents = int(
            number_of_agents
        )

        self.observation_dimension = int(
            observation_dimension
        )

        self.action_dimension = int(
            action_dimension
        )

        self.agent_index = int(
            agent_index
        )

        self.encoder = CollaborativeObservationEncoder(
            number_of_agents=self.number_of_agents,
            input_dimension=self.observation_dimension,
            embedding_dimension=embedding_dimension,
            number_of_heads=number_of_heads,
            number_of_layers=number_of_layers,
            dropout=dropout,
        )

        self.context_pooling = AttentionPooling(
            embedding_dimension
        )

        self.policy_network = build_mlp(
            input_dimension=(
                2 * embedding_dimension
            ),
            hidden_dimensions=hidden_dimensions,
            output_dimension=self.action_dimension,
            output_activation=nn.Sigmoid(),
        )

    def forward(
        self,
        joint_observations: torch.Tensor,
        return_attention: bool = False,
    ):
        remove_batch_dimension = False

        if joint_observations.ndim == 2:
            joint_observations = (
                joint_observations.unsqueeze(0)
            )

            remove_batch_dimension = True

        encoded_tokens, self_attention_weights = (
            self.encoder(
                joint_observations,
                return_attention=return_attention,
            )
        )

        collaborative_context, pooling_weights = (
            self.context_pooling(
                encoded_tokens
            )
        )

        local_agent_representation = (
            encoded_tokens[
                :,
                self.agent_index,
                :,
            ]
        )

        actor_input = torch.cat(
            [
                local_agent_representation,
                collaborative_context,
            ],
            dim=-1,
        )

        action = self.policy_network(
            actor_input
        )

        if remove_batch_dimension:
            action = action.squeeze(0)
            pooling_weights = pooling_weights.squeeze(0)

            self_attention_weights = [
                weights.squeeze(0)
                for weights in (
                    self_attention_weights
                )
            ]

        if return_attention:
            return action, {
                "self_attention_weights": (
                    self_attention_weights
                ),
                "pooling_weights": pooling_weights,
            }

        return action


class CollaborativeAttentionCritic(nn.Module):
    """
    Centralised critic using attention over all UAV
    observation-action pairs.
    """

    def __init__(
        self,
        number_of_agents: int,
        observation_dimension: int,
        action_dimension: int,
        embedding_dimension: int = 128,
        number_of_heads: int = 4,
        number_of_layers: int = 1,
        dropout: float = 0.10,
        hidden_dimensions: Sequence[int] = (
            256,
            256,
        ),
    ):
        super().__init__()

        self.number_of_agents = int(
            number_of_agents
        )

        self.observation_dimension = int(
            observation_dimension
        )

        self.action_dimension = int(
            action_dimension
        )

        joint_agent_input_dimension = (
            self.observation_dimension
            + self.action_dimension
        )

        self.encoder = CollaborativeObservationEncoder(
            number_of_agents=self.number_of_agents,
            input_dimension=(
                joint_agent_input_dimension
            ),
            embedding_dimension=embedding_dimension,
            number_of_heads=number_of_heads,
            number_of_layers=number_of_layers,
            dropout=dropout,
        )

        self.context_pooling = AttentionPooling(
            embedding_dimension
        )

        self.value_network = build_mlp(
            input_dimension=embedding_dimension,
            hidden_dimensions=hidden_dimensions,
            output_dimension=1,
        )

    def forward(
        self,
        joint_observations: torch.Tensor,
        joint_actions: torch.Tensor,
        return_attention: bool = False,
    ):
        remove_batch_dimension = False

        if joint_observations.ndim == 2:
            joint_observations = (
                joint_observations.unsqueeze(0)
            )

            joint_actions = (
                joint_actions.unsqueeze(0)
            )

            remove_batch_dimension = True

        expected_observation_shape = (
            joint_observations.shape[0],
            self.number_of_agents,
            self.observation_dimension,
        )

        expected_action_shape = (
            joint_actions.shape[0],
            self.number_of_agents,
            self.action_dimension,
        )

        if (
            tuple(joint_observations.shape)
            != expected_observation_shape
        ):
            raise ValueError(
                "Unexpected joint-observation shape: "
                f"{tuple(joint_observations.shape)}."
            )

        if (
            tuple(joint_actions.shape)
            != expected_action_shape
        ):
            raise ValueError(
                "Unexpected joint-action shape: "
                f"{tuple(joint_actions.shape)}."
            )

        joint_agent_inputs = torch.cat(
            [
                joint_observations,
                joint_actions,
            ],
            dim=-1,
        )

        encoded_tokens, self_attention_weights = (
            self.encoder(
                joint_agent_inputs,
                return_attention=return_attention,
            )
        )

        collaborative_context, pooling_weights = (
            self.context_pooling(
                encoded_tokens
            )
        )

        q_value = self.value_network(
            collaborative_context
        )

        if remove_batch_dimension:
            q_value = q_value.squeeze(0)
            pooling_weights = pooling_weights.squeeze(0)

            self_attention_weights = [
                weights.squeeze(0)
                for weights in (
                    self_attention_weights
                )
            ]

        if return_attention:
            return q_value, {
                "self_attention_weights": (
                    self_attention_weights
                ),
                "pooling_weights": pooling_weights,
            }

        return q_value


if __name__ == "__main__":
    device = torch.device(
        "cuda:0"
        if torch.cuda.is_available()
        else "cpu"
    )

    batch_size = 4
    number_of_agents = 10
    observation_dimension = 29
    action_dimension = 3

    actor = CollaborativeAttentionActor(
        number_of_agents=number_of_agents,
        observation_dimension=observation_dimension,
        action_dimension=action_dimension,
        agent_index=0,
        embedding_dimension=128,
        number_of_heads=4,
        number_of_layers=1,
        dropout=0.10,
    ).to(device)

    critic = CollaborativeAttentionCritic(
        number_of_agents=number_of_agents,
        observation_dimension=observation_dimension,
        action_dimension=action_dimension,
        embedding_dimension=128,
        number_of_heads=4,
        number_of_layers=1,
        dropout=0.10,
    ).to(device)

    actor.eval()
    critic.eval()

    observations = torch.rand(
        batch_size,
        number_of_agents,
        observation_dimension,
        device=device,
    )

    actions = torch.rand(
        batch_size,
        number_of_agents,
        action_dimension,
        device=device,
    )

    with torch.no_grad():
        actor_output, actor_attention = actor(
            observations,
            return_attention=True,
        )

        critic_output, critic_attention = critic(
            observations,
            actions,
            return_attention=True,
        )

    actor_parameter_count = sum(
        parameter.numel()
        for parameter in actor.parameters()
    )

    critic_parameter_count = sum(
        parameter.numel()
        for parameter in critic.parameters()
    )

    print("Device:", device)
    print(
        "Actor output shape:",
        tuple(actor_output.shape),
    )
    print(
        "Critic output shape:",
        tuple(critic_output.shape),
    )
    print(
        "Actor pooling shape:",
        tuple(
            actor_attention[
                "pooling_weights"
            ].shape
        ),
    )
    print(
        "Critic pooling shape:",
        tuple(
            critic_attention[
                "pooling_weights"
            ].shape
        ),
    )
    print(
        "Actor self-attention shape:",
        tuple(
            actor_attention[
                "self_attention_weights"
            ][0].shape
        ),
    )
    print(
        "Critic self-attention shape:",
        tuple(
            critic_attention[
                "self_attention_weights"
            ][0].shape
        ),
    )
    print(
        "Actor parameters:",
        actor_parameter_count,
    )
    print(
        "Critic parameters:",
        critic_parameter_count,
    )
