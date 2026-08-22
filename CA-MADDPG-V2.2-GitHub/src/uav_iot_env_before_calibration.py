from __future__ import annotations

from pathlib import Path
from typing import Dict

import gymnasium as gym
import numpy as np
import pandas as pd
import yaml
from pettingzoo import ParallelEnv


ROOT = Path(__file__).resolve().parents[1]


class UAVIOTParallelEnv(ParallelEnv):
    metadata = {
        "name": "uav_iot_resource_management_v0",
        "render_modes": [],
        "is_parallelizable": True,
    }

    def __init__(self, config_path: str = "configs/maddpg_base.yaml"):
        super().__init__()

        with open(ROOT / config_path, "r", encoding="utf-8") as file:
            self.config = yaml.safe_load(file)

        env_cfg = self.config["environment"]
        data_cfg = self.config["data"]

        self.tasks = pd.read_csv(ROOT / data_cfg["task_file"])
        self.uav_profiles = pd.read_csv(ROOT / data_cfg["uav_file"])

        self.task_features = env_cfg["task_features"]
        self.agent_features = env_cfg["agent_features"]
        self.episode_length = int(env_cfg["episode_length"])
        self.shared_reward = bool(env_cfg["shared_reward"])
        self.reward_weights = env_cfg["reward_weights"]

        self.uav_profiles = self.uav_profiles.sort_values(
            "closest_uav_id",
            key=lambda series: series.str.extract(r"(\d+)")[0].astype(int),
        ).reset_index(drop=True)

        self.uav_ids = self.uav_profiles["closest_uav_id"].tolist()
        self.possible_agents = [f"uav_{uav_id}" for uav_id in self.uav_ids]
        self.agents = self.possible_agents.copy()

        self.agent_to_index = {
            agent: index for index, agent in enumerate(self.possible_agents)
        }

        self.task_min = self.tasks[self.task_features].min().to_numpy(dtype=np.float32)
        self.task_max = self.tasks[self.task_features].max().to_numpy(dtype=np.float32)

        self.agent_min = (
            self.uav_profiles[self.agent_features].min().to_numpy(dtype=np.float32)
        )
        self.agent_max = (
            self.uav_profiles[self.agent_features].max().to_numpy(dtype=np.float32)
        )

        self.task_scale = np.maximum(self.task_max - self.task_min, 1e-8)
        self.agent_scale = np.maximum(self.agent_max - self.agent_min, 1e-8)

        self.action_dimension = int(env_cfg["action"]["dimension"])
        self.observation_dimension = (
            len(self.task_features)
            + len(self.agent_features)
            + 4
        )

        self._action_spaces = {
            agent: gym.spaces.Box(
                low=0.0,
                high=1.0,
                shape=(self.action_dimension,),
                dtype=np.float32,
            )
            for agent in self.possible_agents
        }

        self._observation_spaces = {
            agent: gym.spaces.Box(
                low=0.0,
                high=1.0,
                shape=(self.observation_dimension,),
                dtype=np.float32,
            )
            for agent in self.possible_agents
        }

        self.initial_energy_mAh = self.uav_profiles[
            "mec_uav_remaining_energy_mAh"
        ].to_numpy(dtype=np.float64)

        self.remaining_energy_mAh = self.initial_energy_mAh.copy()
        self.assignment_counts = np.zeros(len(self.possible_agents), dtype=np.float64)

        self.current_index = 0
        self.start_index = 0
        self.step_count = 0
        self.np_random = np.random.default_rng(42)

    def observation_space(self, agent: str):
        return self._observation_spaces[agent]

    def action_space(self, agent: str):
        return self._action_spaces[agent]

    @staticmethod
    def _normalise(values, minimum, scale):
        normalised = (values - minimum) / scale
        return np.clip(normalised, 0.0, 1.0).astype(np.float32)

    @staticmethod
    def _haversine_distance_m(lat1, lon1, lat2, lon2):
        earth_radius_m = 6_371_000.0

        lat1 = np.radians(lat1)
        lon1 = np.radians(lon1)
        lat2 = np.radians(lat2)
        lon2 = np.radians(lon2)

        dlat = lat2 - lat1
        dlon = lon2 - lon1

        value = (
            np.sin(dlat / 2.0) ** 2
            + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2.0) ** 2
        )

        return float(
            2.0 * earth_radius_m * np.arctan2(np.sqrt(value), np.sqrt(1.0 - value))
        )

    def _current_task(self):
        return self.tasks.iloc[self.current_index]

    def _local_distance(self, agent_index: int):
        task = self._current_task()
        profile = self.uav_profiles.iloc[agent_index]

        return self._haversine_distance_m(
            task["Location_Lat"],
            task["Location_Lon"],
            profile["mec_uav_lat"],
            profile["mec_uav_lon"],
        )

    def _observation(self, agent: str):
        agent_index = self.agent_to_index[agent]
        task = self._current_task()
        profile = self.uav_profiles.iloc[agent_index]

        task_values = task[self.task_features].to_numpy(dtype=np.float32)
        agent_values = profile[self.agent_features].to_numpy(dtype=np.float32)

        task_vector = self._normalise(
            task_values,
            self.task_min,
            self.task_scale,
        )

        agent_vector = self._normalise(
            agent_values,
            self.agent_min,
            self.agent_scale,
        )

        local_distance_ratio = np.clip(
            self._local_distance(agent_index) / 20_000.0,
            0.0,
            1.0,
        )

        remaining_energy_ratio = np.clip(
            self.remaining_energy_mAh[agent_index]
            / max(self.initial_energy_mAh[agent_index], 1e-8),
            0.0,
            1.0,
        )

        total_assignments = max(self.assignment_counts.sum(), 1.0)
        assignment_ratio = np.clip(
            self.assignment_counts[agent_index] / total_assignments,
            0.0,
            1.0,
        )

        episode_progress = np.clip(
            self.step_count / max(self.episode_length, 1),
            0.0,
            1.0,
        )

        dynamic_vector = np.asarray(
            [
                local_distance_ratio,
                remaining_energy_ratio,
                assignment_ratio,
                episode_progress,
            ],
            dtype=np.float32,
        )

        return np.concatenate(
            [task_vector, agent_vector, dynamic_vector]
        ).astype(np.float32)

    def reset(self, seed=None, options=None):
        if seed is not None:
            self.np_random = np.random.default_rng(seed)

        self.agents = self.possible_agents.copy()

        maximum_start = max(
            0,
            len(self.tasks) - self.episode_length - 1,
        )

        self.start_index = int(
            self.np_random.integers(0, maximum_start + 1)
        )

        self.current_index = self.start_index
        self.step_count = 0

        self.remaining_energy_mAh = self.initial_energy_mAh.copy()
        self.assignment_counts = np.zeros(
            len(self.possible_agents),
            dtype=np.float64,
        )

        observations = {
            agent: self._observation(agent)
            for agent in self.agents
        }

        infos = {
            agent: {
                "start_index": self.start_index,
            }
            for agent in self.agents
        }

        return observations, infos

    def step(self, actions: Dict[str, np.ndarray]):
        active_agents = self.agents.copy()

        if not active_agents:
            raise RuntimeError("step() called after the episode ended.")

        for agent in active_agents:
            if agent not in actions:
                raise KeyError(f"Missing action for {agent}")

        clipped_actions = {
            agent: np.clip(
                np.asarray(actions[agent], dtype=np.float32),
                0.0,
                1.0,
            )
            for agent in active_agents
        }

        selection_scores = np.asarray(
            [clipped_actions[agent][0] for agent in active_agents]
        )

        selected_index = int(np.argmax(selection_scores))
        selected_agent = active_agents[selected_index]
        selected_action = clipped_actions[selected_agent]

        cpu_fraction = max(float(selected_action[1]), 0.05)
        bandwidth_fraction = max(float(selected_action[2]), 0.05)

        task = self._current_task()
        profile = self.uav_profiles.iloc[selected_index]

        local_distance_m = self._local_distance(selected_index)

        distance_factor = 1.0 / (
            1.0 + local_distance_m / 5_000.0
        )

        task_uplink_rate = float(task["uplink_data_rate_Mbps"])
        uav_uplink_rate = float(profile["uplink_rate_Mbps"])

        effective_uplink_rate_Mbps = max(
            min(task_uplink_rate, uav_uplink_rate)
            * bandwidth_fraction
            * distance_factor,
            0.01,
        )

        input_bits = float(task["task_size_MB"]) * 8_000_000.0

        transmission_time_s = (
            input_bits
            / (effective_uplink_rate_Mbps * 1_000_000.0)
        )

        allocated_cpu_cycles_per_second = max(
            float(profile["mec_uav_cpu_GHz"])
            * 1_000_000_000.0
            * cpu_fraction,
            1.0,
        )

        processing_time_s = (
            float(task["task_cpu_demand_cycles"])
            / allocated_cpu_cycles_per_second
        )

        downlink_bits = input_bits * 0.05

        effective_downlink_rate_Mbps = max(
            float(profile["downlink_rate_Mbps"])
            * bandwidth_fraction
            * distance_factor,
            0.01,
        )

        downlink_time_s = (
            downlink_bits
            / (effective_downlink_rate_Mbps * 1_000_000.0)
        )

        total_latency_ms = (
            transmission_time_s
            + processing_time_s
            + downlink_time_s
        ) * 1_000.0

        communication_energy_mJ = (
            float(task["uplink_power_transfer_mW"])
            * (transmission_time_s + downlink_time_s)
        )

        compute_power_watts = (
            15.0 + 35.0 * cpu_fraction**3
        )

        computation_energy_mJ = (
            compute_power_watts
            * processing_time_s
            * 1_000.0
        )

        total_energy_mJ = (
            communication_energy_mJ
            + computation_energy_mJ
        )

        delay_deadline_ms = max(
            float(task["delay_deadline_ms"]),
            1e-8,
        )

        energy_deadline_mJ = max(
            float(task["energy_deadline_mJ"]),
            1e-8,
        )

        delay_violation = float(
            total_latency_ms > delay_deadline_ms
        )

        energy_violation = float(
            total_energy_mJ > energy_deadline_mJ
        )

        battery_voltage = 11.1
        consumed_mAh = (
            total_energy_mJ / 1_000.0
        ) / (battery_voltage * 3.6)

        self.remaining_energy_mAh[selected_index] = max(
            0.0,
            self.remaining_energy_mAh[selected_index]
            - consumed_mAh,
        )

        resource_overload = float(
            self.remaining_energy_mAh[selected_index] <= 0.0
        )

        success = float(
            delay_violation == 0.0
            and energy_violation == 0.0
            and resource_overload == 0.0
        )

        self.assignment_counts[selected_index] += 1.0

        mean_load = np.mean(self.assignment_counts)
        load_imbalance = float(
            np.std(self.assignment_counts)
            / max(mean_load, 1e-8)
        )
        load_imbalance = min(load_imbalance, 2.0)

        latency_ratio = min(
            total_latency_ms / delay_deadline_ms,
            5.0,
        )

        energy_ratio = min(
            total_energy_mJ / energy_deadline_mJ,
            5.0,
        )

        reward = (
            self.reward_weights["success"] * success
            - self.reward_weights["latency"] * latency_ratio
            - self.reward_weights["energy"] * energy_ratio
            - self.reward_weights["deadline_violation"]
            * max(delay_violation, energy_violation)
            - self.reward_weights["resource_overload"]
            * resource_overload
            - self.reward_weights["load_balance"]
            * load_imbalance
        )

        rewards = {
            agent: float(reward)
            if self.shared_reward
            else float(reward if agent == selected_agent else 0.0)
            for agent in active_agents
        }

        infos = {
            agent: {
                "selected_agent": selected_agent,
                "selected": agent == selected_agent,
                "task_id": str(task["task_id"]),
                "latency_ms": float(total_latency_ms),
                "energy_mJ": float(total_energy_mJ),
                "success": bool(success),
                "delay_violation": bool(delay_violation),
                "energy_violation": bool(energy_violation),
                "local_distance_m": float(local_distance_m),
                "cpu_fraction": float(cpu_fraction),
                "bandwidth_fraction": float(bandwidth_fraction),
            }
            for agent in active_agents
        }

        self.step_count += 1
        self.current_index += 1

        episode_finished = (
            self.step_count >= self.episode_length
            or self.current_index >= len(self.tasks)
        )

        terminations = {
            agent: False for agent in active_agents
        }

        truncations = {
            agent: episode_finished for agent in active_agents
        }

        if episode_finished:
            observations = {
                agent: np.zeros(
                    self.observation_dimension,
                    dtype=np.float32,
                )
                for agent in active_agents
            }
            self.agents = []
        else:
            observations = {
                agent: self._observation(agent)
                for agent in active_agents
            }

        return (
            observations,
            rewards,
            terminations,
            truncations,
            infos,
        )

    def render(self):
        return None

    def close(self):
        return None


if __name__ == "__main__":
    environment = UAVIOTParallelEnv()
    observations, infos = environment.reset(seed=42)

    print("Agents:", len(environment.agents))
    print(
        "Observation dimension:",
        environment.observation_dimension,
    )
    print(
        "Action dimension:",
        environment.action_dimension,
    )

    for step in range(5):
        random_actions = {
            agent: environment.action_space(agent).sample()
            for agent in environment.agents
        }

        observations, rewards, terminations, truncations, infos = (
            environment.step(random_actions)
        )

        first_agent = environment.possible_agents[0]

        print(
            f"Step {step + 1}:",
            "reward =",
            round(rewards[first_agent], 4),
            "selected =",
            infos[first_agent]["selected_agent"],
            "success =",
            infos[first_agent]["success"],
        )
