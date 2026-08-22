from __future__ import annotations

from typing import Sequence

import torch
from torch import nn


def build_mlp(
    input_dimension: int,
    hidden_dimensions: Sequence[int],
    output_dimension: int,
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
                nn.GELU(),
            ]
        )

        current_dimension = int(
            hidden_dimension
        )

    layers.append(
        nn.Linear(
            current_dimension,
            int(output_dimension),
        )
    )

    return nn.Sequential(*layers)


class CollaborativeAttentionBlock(nn.Module):
    """Residual multi-head self-attention block."""

    def __init__(
        self,
        embedding_dimension: int,
        number_of_heads: int,
        dropout: float,
        feedforward_multiplier: int = 4,
    ):
        super().__init__()

        if (
            embedding_dimension
            % number_of_heads
            != 0
        ):
            raise ValueError(
                "Embedding dimension must be divisible "
                "by the number of attention heads."
            )

        self.self_attention = nn.MultiheadAttention(
            embed_dim=embedding_dimension,
            num_heads=number_of_heads,
            dropout=dropout,
            batch_first=True,
        )

        self.attention_dropout = nn.Dropout(
            dropout
        )

        self.attention_normalisation = (
            nn.LayerNorm(
                embedding_dimension
            )
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

        self.feedforward_normalisation = (
            nn.LayerNorm(
                embedding_dimension
            )
        )

    def forward(
        self,
        tokens: torch.Tensor,
        return_attention: bool = False,
    ) -> tuple[
        torch.Tensor,
        torch.Tensor | None,
    ]:
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

        tokens = (
            self.feedforward_normalisation(
                tokens
                + feedforward_output
            )
        )

        return tokens, attention_weights


class JointAttentionEncoder(nn.Module):
    """
    Encode all UAV observations jointly using one shared
    attention encoder.
    """

    def __init__(
        self,
        number_of_agents: int,
        input_dimension: int,
        embedding_dimension: int,
        number_of_heads: int,
        number_of_layers: int,
        dropout: float,
    ):
        super().__init__()

        if number_of_agents <= 0:
            raise ValueError(
                "Number of agents must be positive."
            )

        if number_of_layers <= 0:
            raise ValueError(
                "Number of attention layers must be positive."
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

        self.agent_identity_embedding = (
            nn.Embedding(
                self.number_of_agents,
                self.embedding_dimension,
            )
        )

        self.attention_layers = nn.ModuleList(
            [
                CollaborativeAttentionBlock(
                    embedding_dimension=(
                        self.embedding_dimension
                    ),
                    number_of_heads=(
                        number_of_heads
                    ),
                    dropout=dropout,
                )
                for _ in range(
                    number_of_layers
                )
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

        (
            batch_size,
            number_of_agents,
            input_dimension,
        ) = joint_inputs.shape

        if (
            number_of_agents
            != self.number_of_agents
        ):
            raise ValueError(
                f"Expected {self.number_of_agents} "
                f"agents, received "
                f"{number_of_agents}."
            )

        if (
            input_dimension
            != self.input_dimension
        ):
            raise ValueError(
                f"Expected input dimension "
                f"{self.input_dimension}, "
                f"received {input_dimension}."
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

        attention_history: list[
            torch.Tensor
        ] = []

        for attention_layer in (
            self.attention_layers
        ):
            tokens, attention_weights = (
                attention_layer(
                    tokens,
                    return_attention=(
                        return_attention
                    ),
                )
            )

            if (
                return_attention
                and attention_weights
                is not None
            ):
                attention_history.append(
                    attention_weights
                )

        return tokens, attention_history


class AttentionPooling(nn.Module):
    """Learn a cooperative global UAV representation."""

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
    ) -> tuple[
        torch.Tensor,
        torch.Tensor,
    ]:
        scores = self.score_network(
            tokens
        ).squeeze(-1)

        weights = torch.softmax(
            scores,
            dim=-1,
        )

        context = torch.sum(
            tokens
            * weights.unsqueeze(-1),
            dim=1,
        )

        return context, weights


class SharedJointAttentionActor(nn.Module):
    """
    One shared actor jointly produces:

    1. A softmax UAV-selection probability for every UAV.
    2. A CPU allocation fraction for every UAV.
    3. A bandwidth allocation fraction for every UAV.
    """

    def __init__(
        self,
        number_of_agents: int,
        observation_dimension: int,
        action_dimension: int = 3,
        embedding_dimension: int = 64,
        number_of_heads: int = 4,
        number_of_layers: int = 1,
        dropout: float = 0.05,
        selection_temperature: float = 0.25,
        minimum_resource_fraction: float = 0.50,
        hidden_dimensions: Sequence[int] = (
            128,
            128,
        ),
    ):
        super().__init__()

        if action_dimension != 3:
            raise ValueError(
                "CA-MADDPG-V2 currently expects "
                "three action outputs per UAV."
            )

        if selection_temperature <= 0.0:
            raise ValueError(
                "Selection temperature must be positive."
            )

        if not (
            0.0
            <= minimum_resource_fraction
            < 1.0
        ):
            raise ValueError(
                "Minimum resource fraction must be "
                "inside [0, 1)."
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

        self.selection_temperature = float(
            selection_temperature
        )

        self.minimum_resource_fraction = (
            float(
                minimum_resource_fraction
            )
        )

        self.encoder = JointAttentionEncoder(
            number_of_agents=(
                self.number_of_agents
            ),
            input_dimension=(
                self.observation_dimension
            ),
            embedding_dimension=(
                embedding_dimension
            ),
            number_of_heads=number_of_heads,
            number_of_layers=number_of_layers,
            dropout=dropout,
        )

        self.selection_head = build_mlp(
            input_dimension=(
                embedding_dimension
            ),
            hidden_dimensions=(
                hidden_dimensions
            ),
            output_dimension=1,
        )

        self.resource_head = build_mlp(
            input_dimension=(
                embedding_dimension
            ),
            hidden_dimensions=(
                hidden_dimensions
            ),
            output_dimension=2,
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

        encoded_tokens, attention_history = (
            self.encoder(
                joint_observations,
                return_attention=(
                    return_attention
                ),
            )
        )

        selection_logits = (
            self.selection_head(
                encoded_tokens
            ).squeeze(-1)
        )

        selection_probabilities = (
            torch.softmax(
                selection_logits
                / self.selection_temperature,
                dim=1,
            )
        )

        raw_resources = torch.sigmoid(
            self.resource_head(
                encoded_tokens
            )
        )

        resource_range = (
            1.0
            - self.minimum_resource_fraction
        )

        resource_actions = (
            self.minimum_resource_fraction
            + resource_range
            * raw_resources
        )

        joint_actions = torch.cat(
            [
                selection_probabilities.unsqueeze(
                    -1
                ),
                resource_actions,
            ],
            dim=-1,
        )

        sorted_probabilities = torch.sort(
            selection_probabilities,
            dim=1,
            descending=True,
        ).values

        selection_gap = (
            sorted_probabilities[:, 0]
            - sorted_probabilities[:, 1]
        )

        if remove_batch_dimension:
            joint_actions = (
                joint_actions.squeeze(0)
            )

            selection_probabilities = (
                selection_probabilities.squeeze(
                    0
                )
            )

            selection_logits = (
                selection_logits.squeeze(0)
            )

            selection_gap = (
                selection_gap.squeeze(0)
            )

            attention_history = [
                weights.squeeze(0)
                for weights in (
                    attention_history
                )
            ]

        if return_attention:
            return joint_actions, {
                "selection_logits": (
                    selection_logits
                ),
                "selection_probabilities": (
                    selection_probabilities
                ),
                "selection_gap": (
                    selection_gap
                ),
                "self_attention_weights": (
                    attention_history
                ),
            }

        return joint_actions


class SharedTeamAttentionCritic(nn.Module):
    """
    One shared cooperative critic evaluates the complete
    UAV team action.
    """

    def __init__(
        self,
        number_of_agents: int,
        observation_dimension: int,
        action_dimension: int = 3,
        embedding_dimension: int = 64,
        number_of_heads: int = 4,
        number_of_layers: int = 1,
        dropout: float = 0.05,
        hidden_dimensions: Sequence[int] = (
            128,
            128,
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

        critic_input_dimension = (
            self.observation_dimension
            + self.action_dimension
        )

        self.encoder = JointAttentionEncoder(
            number_of_agents=(
                self.number_of_agents
            ),
            input_dimension=(
                critic_input_dimension
            ),
            embedding_dimension=(
                embedding_dimension
            ),
            number_of_heads=number_of_heads,
            number_of_layers=number_of_layers,
            dropout=dropout,
        )

        self.attention_pooling = (
            AttentionPooling(
                embedding_dimension
            )
        )

        self.value_network = build_mlp(
            input_dimension=(
                embedding_dimension
            ),
            hidden_dimensions=(
                hidden_dimensions
            ),
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
                "Unexpected observation shape: "
                f"{tuple(joint_observations.shape)}"
            )

        if (
            tuple(joint_actions.shape)
            != expected_action_shape
        ):
            raise ValueError(
                "Unexpected action shape: "
                f"{tuple(joint_actions.shape)}"
            )

        critic_inputs = torch.cat(
            [
                joint_observations,
                joint_actions,
            ],
            dim=-1,
        )

        encoded_tokens, attention_history = (
            self.encoder(
                critic_inputs,
                return_attention=(
                    return_attention
                ),
            )
        )

        team_context, pooling_weights = (
            self.attention_pooling(
                encoded_tokens
            )
        )

        team_q_value = self.value_network(
            team_context
        )

        if remove_batch_dimension:
            team_q_value = (
                team_q_value.squeeze(0)
            )

            pooling_weights = (
                pooling_weights.squeeze(0)
            )

            attention_history = [
                weights.squeeze(0)
                for weights in (
                    attention_history
                )
            ]

        if return_attention:
            return team_q_value, {
                "pooling_weights": (
                    pooling_weights
                ),
                "self_attention_weights": (
                    attention_history
                ),
            }

        return team_q_value


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

    actor = SharedJointAttentionActor(
        number_of_agents=number_of_agents,
        observation_dimension=(
            observation_dimension
        ),
        action_dimension=action_dimension,
        embedding_dimension=64,
        number_of_heads=4,
        number_of_layers=1,
        dropout=0.05,
        selection_temperature=0.25,
        minimum_resource_fraction=0.50,
    ).to(device)

    critic = SharedTeamAttentionCritic(
        number_of_agents=number_of_agents,
        observation_dimension=(
            observation_dimension
        ),
        action_dimension=action_dimension,
        embedding_dimension=64,
        number_of_heads=4,
        number_of_layers=1,
        dropout=0.05,
    ).to(device)

    actor.eval()
    critic.eval()

    observations = torch.rand(
        batch_size,
        number_of_agents,
        observation_dimension,
        device=device,
    )

    with torch.no_grad():
        actions, actor_diagnostics = actor(
            observations,
            return_attention=True,
        )

        q_values, critic_diagnostics = (
            critic(
                observations,
                actions,
                return_attention=True,
            )
        )

    actor_parameters = sum(
        parameter.numel()
        for parameter in actor.parameters()
    )

    critic_parameters = sum(
        parameter.numel()
        for parameter in critic.parameters()
    )

    selection_sums = actions[
        :,
        :,
        0,
    ].sum(dim=1)

    print("=" * 76)
    print("CA-MADDPG-V2 COMPONENT TEST")
    print("=" * 76)
    print("Device:", device)
    print(
        "Actor output shape:",
        tuple(actions.shape),
    )
    print(
        "Critic output shape:",
        tuple(q_values.shape),
    )
    print(
        "Selection probability sums:",
        selection_sums.detach().cpu().numpy(),
    )
    print(
        "Selection probability minimum:",
        f"{actions[:, :, 0].min().item():.6f}",
    )
    print(
        "Selection probability maximum:",
        f"{actions[:, :, 0].max().item():.6f}",
    )
    print(
        "CPU range:",
        f"{actions[:, :, 1].min().item():.6f}",
        "to",
        f"{actions[:, :, 1].max().item():.6f}",
    )
    print(
        "Bandwidth range:",
        f"{actions[:, :, 2].min().item():.6f}",
        "to",
        f"{actions[:, :, 2].max().item():.6f}",
    )
    print(
        "Selection gap shape:",
        tuple(
            actor_diagnostics[
                "selection_gap"
            ].shape
        ),
    )
    print(
        "Actor attention shape:",
        tuple(
            actor_diagnostics[
                "self_attention_weights"
            ][0].shape
        ),
    )
    print(
        "Critic attention shape:",
        tuple(
            critic_diagnostics[
                "self_attention_weights"
            ][0].shape
        ),
    )
    print(
        "Critic pooling shape:",
        tuple(
            critic_diagnostics[
                "pooling_weights"
            ].shape
        ),
    )
    print(
        "Actor parameters:",
        actor_parameters,
    )
    print(
        "Critic parameters:",
        critic_parameters,
    )
    print("=" * 76)

    assert actions.shape == (
        batch_size,
        number_of_agents,
        action_dimension,
    )

    assert q_values.shape == (
        batch_size,
        1,
    )

    assert torch.allclose(
        selection_sums,
        torch.ones_like(
            selection_sums
        ),
        atol=1e-6,
    )

    assert (
        actions[:, :, 1:].min().item()
        >= 0.50
    )

    assert (
        actions.max().item()
        <= 1.0
    )

    print(
        "CA-MADDPG-V2 component test "
        "passed successfully."
    )
