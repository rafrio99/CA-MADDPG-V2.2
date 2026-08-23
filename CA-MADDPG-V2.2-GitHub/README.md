# CA-MADDPG-V2.2: Attention-Assisted Multi-Agent Task Offloading in UAV-Assisted IoT Edge Networks

This repository contains the implementation of **CA-MADDPG-V2.2**, an attention-assisted multi-agent reinforcement learning framework for adaptive task offloading and resource allocation in UAV-assisted IoT edge networks.

The proposed framework integrates:
- A shared joint-attention encoder for multi-agent contextual representation learning
- Decentralized actor networks
- A centralized critic network
- An inference-time resource safety projection mechanism

CA-MADDPG-V2.2 aims to improve coordinated decision-making among UAV agents under dynamic resource constraints by jointly optimizing UAV selection, CPU allocation, and bandwidth allocation.

---

## Overview

UAV-assisted IoT edge networks require efficient task offloading and resource allocation due to limited computation, communication, and energy resources.

CA-MADDPG-V2.2 extends the MADDPG framework by introducing:

1. **Context-aware joint-attention encoding**
   - Captures dependencies among distributed UAV observations.
   - Provides contextual information for coordinated multi-agent decisions.

2. **Multi-agent actor-critic learning**
   - Decentralized actors generate individual UAV decisions.
   - A centralized critic evaluates joint observations and actions during training.

3. **Inference-time safety projection**
   - Refines continuous resource allocation outputs.
   - Improves feasibility of CPU and bandwidth allocation decisions.

---

## Framework

The overall decision process consists of:

1. UAV agents observe local task and resource states.
2. The joint-attention encoder generates contextual representations.
3. Decentralized actors produce UAV selection and resource allocation actions.
4. Safety projection refines continuous resource decisions.
5. The environment returns task success, latency, energy consumption, reward, and next-state feedback.

---

## Environment

Experiments are conducted in the:

**AEC-IoT TaskNet Environment**

The environment models:
- UAV-assisted IoT edge computing
- Multi-agent task offloading
- CPU resource allocation
- Bandwidth allocation
- Dynamic resource constraints

---

## Compared Methods

CA-MADDPG-V2.2 is evaluated against:

- MADDPG
- CA-MADDPG-V2
- Random Allocation
- Latency-Oriented Heuristic

Evaluation metrics include:

- Task success rate
- Completion latency
- Energy consumption
- Cumulative reward

---

## Experimental Settings

| Setting | Value |
|---|---|
| Training seeds | 42, 43, 44 |
| Test episodes | 100 |
| Decision steps per episode | 100 |
| Validation resource floors | 0.50–0.98 |
| Selected resource floor | 0.98 |
| GPU | NVIDIA GeForce RTX 3090 |

---

## Installation

### 1. Clone repository

```bash
git clone https://github.com/your_username/CA-MADDPG-V2.2.git
cd CA-MADDPG-V2.2
