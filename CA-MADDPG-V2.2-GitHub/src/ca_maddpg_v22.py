from __future__ import annotations

from typing import Any

import numpy as np

from ca_maddpg_v2 import CAMADDPGV2


class CAMADDPGV22(CAMADDPGV2):
    """
    CA-MADDPG Version 2.2.

    Version 2.2 preserves the trained CA-MADDPG-V2
    UAV-selection policy and applies an inference-time
    safety projection only to CPU and bandwidth actions.
    """

    MODEL_NAME = "CA-MADDPG-V2.2"

    def __init__(
        self,
        number_of_agents: int,
        observation_dimension: int,
        action_dimension: int,
        config_path: str,
        resource_floor: float = 0.98,
        device: Any = None,
    ):
        super().__init__(
            number_of_agents=number_of_agents,
            observation_dimension=observation_dimension,
            action_dimension=action_dimension,
            config_path=config_path,
            device=device,
        )

        self.resource_floor = float(
            resource_floor
        )

        if not (
            0.0
            <= self.resource_floor
            <= 1.0
        ):
            raise ValueError(
                "resource_floor must be "
                "between 0 and 1."
            )

    def project_actions(
        self,
        actions: np.ndarray,
    ) -> np.ndarray:
        projected_actions = np.asarray(
            actions,
            dtype=np.float32,
        ).copy()

        if (
            projected_actions.ndim != 2
            or projected_actions.shape[1] != 3
        ):
            raise ValueError(
                "Expected actions with shape "
                "(number_of_agents, 3)."
            )

        # Column 0 contains learned UAV-selection
        # probabilities and must remain unchanged.
        projected_actions[:, 1] = np.maximum(
            projected_actions[:, 1],
            self.resource_floor,
        )

        projected_actions[:, 2] = np.maximum(
            projected_actions[:, 2],
            self.resource_floor,
        )

        projected_actions[:, 1:] = np.clip(
            projected_actions[:, 1:],
            self.resource_floor,
            1.0,
        )

        return projected_actions

    def select_actions(
        self,
        observations: np.ndarray,
        noise_scale: float = 0.0,
    ) -> np.ndarray:
        original_actions = super().select_actions(
            observations=observations,
            noise_scale=noise_scale,
        )

        return self.project_actions(
            original_actions
        )
