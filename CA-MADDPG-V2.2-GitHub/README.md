# CA-MADDPG-V2.2: Attention-Assisted Multi-Agent Task Offloading in UAV-Assisted IoT Edge Networks

This repository contains the implementation of **CA-MADDPG-V2.2**, an
attention-assisted multi-agent reinforcement learning framework for adaptive
task offloading and resource allocation in UAV-assisted IoT edge networks.

CA-MADDPG-V2.2 combines contextual multi-agent representation learning,
centralized training with decentralized execution, and inference-time resource
safety projection. The framework jointly determines UAV selection, CPU
allocation, and bandwidth allocation under dynamic resource constraints.

## Key Features

- Shared joint-attention encoding for contextual multi-agent representations
- Decentralized actor networks for agent-level decision-making
- A centralized critic for coordinated training
- Joint UAV selection, CPU allocation, and bandwidth allocation
- Inference-time safety projection for resource-feasible decisions
- Controlled three-seed training and fixed-episode evaluation

## Framework

The decision process consists of the following stages:

1. UAV agents observe their local task and resource states.
2. The joint-attention encoder generates contextual representations.
3. Decentralized actors produce UAV-selection and resource-allocation actions.
4. The safety projection refines the continuous resource decisions.
5. The environment returns task success, latency, energy consumption, reward,
   and next-state feedback.

## Environment

Experiments are conducted in the **AEC-IoT TaskNet Environment**, which models:

- UAV-assisted IoT edge computing
- Multi-agent task offloading
- CPU resource allocation
- Bandwidth allocation
- Dynamic resource constraints

## Repository Structure

```text
CA-MADDPG-V2.2/
|-- LICENSE
`-- CA-MADDPG-V2.2-GitHub/
    |-- configs/          # Experiment configuration files
    |-- figures/          # Generated figures
    |-- results/          # Evaluation outputs and statistical results
    |-- src/              # Source code
    |-- .gitignore
    |-- README.md
    |-- environment.yml
    `-- requirements.txt
```

## Compared Methods

CA-MADDPG-V2.2 is evaluated against:

- MADDPG
- CA-MADDPG-V2
- Random Allocation
- Latency-Oriented Heuristic

Plain CA-MADDPG and CA-MADDPG-V2.1 are not included in the final controlled
comparison because complete official multi-seed results were unavailable.

## Evaluation Metrics

The evaluation reports:

- Cumulative reward
- Task success rate
- Completion latency
- Energy consumption

## Experimental Protocol

| Setting | Value |
|---|---|
| Training seeds | 42, 43, 44 |
| Evaluated checkpoint | Episode 10 |
| Fixed test episodes per model | 100 |
| Decision steps per episode | 100 |
| Validation resource floors | 0.50, 0.75, 0.85, 0.90, 0.95, 0.98 |
| Validation episodes per floor | 150 |
| Selected resource floor | 0.98 |
| GPU | NVIDIA GeForce RTX 3090 (24 GB) |

## Installation

### 1. Clone the repository

```bash
git clone https://github.com/rafrio99/CA-MADDPG-V2.2.git
cd CA-MADDPG-V2.2/CA-MADDPG-V2.2-GitHub
```

### 2. Install the dependencies

Using `venv` and `pip`:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

Alternatively, recreate the Conda environment from the supplied file:

```bash
conda env create -f environment.yml
```

Activate the environment using the environment name declared in
`environment.yml`.

## Usage and Reproducibility

Experiment configurations are provided in `configs/`, implementation files are
provided in `src/`, and the verified evaluation outputs are retained in
`results/`. Use the configuration corresponding to the required method, seed,
and evaluation setting.

For reproduction of the reported evaluation, retain the fixed protocol shown
above: train with seeds 42, 43, and 44; evaluate the Episode 10 checkpoint over
100 fixed test episodes per model; and use 100 decision steps per episode. The
resource-floor selection uses 150 validation episodes for each candidate floor
and selects a floor of 0.98.

## Verified Results

The following aggregate results were obtained under the controlled evaluation
protocol. Higher reward and success are better, whereas lower latency and
energy are better.

| Method | Reward (higher is better) | Success (%) (higher is better) | Latency (ms) (lower is better) | Energy (mJ) (lower is better) |
|---|---:|---:|---:|---:|
| Latency-Oriented Heuristic | -206.42 | 50.83 | 1613.16 | 37113.17 |
| **CA-MADDPG-V2.2** | **-216.41** | **49.36** | **1684.14** | **38666.15** |
| MADDPG | -224.22 | 48.43 | 1785.91 | 43354.07 |
| CA-MADDPG-V2 | -244.67 | 45.72 | 1925.81 | 29940.52 |
| Random Allocation | -268.10 | 38.49 | 2601.07 | 86513.63 |

CA-MADDPG-V2.2 improves upon conventional MADDPG and random allocation across
the four reported metrics. However, it is not the best method for every metric:
the latency-oriented heuristic achieves higher success, lower latency, and lower
energy than CA-MADDPG-V2.2, while CA-MADDPG-V2 records the lowest energy in the
table. The contribution of CA-MADDPG-V2.2 should therefore be interpreted as a
learning-based approach that provides balanced multi-objective decision-making,
rather than universal superiority over every comparator.

## Limitations

- The reported findings are specific to the AEC-IoT TaskNet environment and the
  controlled evaluation protocol described above.
- The latency-oriented heuristic remains stronger on its specialized objective.
- Results should not be interpreted as evidence of universal superiority under
  different environments or experimental settings.

## Citation

If you use this repository in your research, please cite the accompanying work:

> **CA-MADDPG-V2.2: Attention-Assisted Multi-Agent Task Offloading in
> UAV-Assisted IoT Edge Networks**

Complete bibliographic information will be added after publication.

## License

This project is licensed under the MIT License. See the [LICENSE](../LICENSE)
file for details.

## Contact

For questions, reproducibility issues, or bug reports, please open an issue in
this GitHub repository.
