from pathlib import Path
import json

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data"

INPUT_FILE = DATA_DIR / "AEC_IoT_TaskNet_sorted.csv"
TASK_OUTPUT = DATA_DIR / "processed_tasks.csv"
UAV_OUTPUT = DATA_DIR / "uav_profiles.csv"
METADATA_OUTPUT = DATA_DIR / "preprocessing_metadata.json"


def main():
    df = pd.read_csv(INPUT_FILE, parse_dates=["task_gen_time"])
    df = df.sort_values("task_gen_time").reset_index(drop=True)

    # Encode the three task categories.
    task_types = sorted(df["task_type"].unique().tolist())
    task_type_map = {name: index for index, name in enumerate(task_types)}
    df["task_type_code"] = df["task_type"].map(task_type_map)

    # Convert task-generation time into cyclical daily features.
    seconds = (
        df["task_gen_time"].dt.hour * 3600
        + df["task_gen_time"].dt.minute * 60
        + df["task_gen_time"].dt.second
    )
    df["time_sin"] = np.sin(2 * np.pi * seconds / 86400)
    df["time_cos"] = np.cos(2 * np.pi * seconds / 86400)

    # Approximate the computational workload required by each task.
    df["task_cpu_demand_cycles"] = (
        df["task_size_MB"] * 8_000_000 * df["cpu_cycles_per_bit"]
    )

    # Remove columns with no variation.
    constant_columns = [
        column for column in df.columns
        if df[column].nunique(dropna=False) <= 1
    ]
    processed_df = df.drop(columns=constant_columns)

    # Save one fixed resource profile for each of the ten UAV agents.
    uav_columns = [
        "closest_uav_id",
        "mec_uav_area_name",
        "mec_uav_lat",
        "mec_uav_lon",
        "mec_uav_cpu_GHz",
        "mec_uav_remaining_energy_mAh",
        "mec_uav_obstacle_distance_m",
        "mec_uav_temperature_c",
        "mec_uav_humidity_pct",
        "mec_uav_wind_speed_mps",
        "mec_uav_visibility_km",
        "mec_uav_precipitation_mmh",
        "bandwidth_MHz_y",
        "uplink_rate_Mbps",
        "downlink_rate_Mbps",
        "aec_storage_GB",
        "aec_ram_GB",
    ]

    uav_profiles = (
        df[uav_columns]
        .drop_duplicates(subset=["closest_uav_id"])
        .sort_values("closest_uav_id")
        .reset_index(drop=True)
    )

    processed_df.to_csv(TASK_OUTPUT, index=False)
    uav_profiles.to_csv(UAV_OUTPUT, index=False)

    metadata = {
        "input_rows": int(len(df)),
        "processed_columns": int(processed_df.shape[1]),
        "number_of_uavs": int(uav_profiles.shape[0]),
        "task_type_mapping": task_type_map,
        "constant_columns_removed": constant_columns,
        "start_time": str(df["task_gen_time"].min()),
        "end_time": str(df["task_gen_time"].max()),
    }

    with open(METADATA_OUTPUT, "w", encoding="utf-8") as file:
        json.dump(metadata, file, indent=2)

    print("Preprocessing script created successfully.")
    print("Input shape:", df.shape)
    print("Processed task shape:", processed_df.shape)
    print("UAV profile shape:", uav_profiles.shape)
    print("Constant columns removed:", constant_columns)


if __name__ == "__main__":
    main()
