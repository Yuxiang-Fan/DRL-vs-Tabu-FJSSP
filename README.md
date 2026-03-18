# Energy-Aware Flexible Job Shop Scheduling: A Comparative Study of DRL and Tabu Search

This repository provides an implementation and comparative analysis of optimization algorithms for the Multi-Objective Flexible Job Shop Scheduling Problem (FJSSP). The study focuses on the simultaneous minimization of the **Makespan** (total completion time) and **Total Energy Consumption (TEC)** in manufacturing environments.

The project evaluates two distinct methodological approaches to address this NP-hard problem: a Deep Reinforcement Learning (DRL) framework based on Masked Proximal Policy Optimization (PPO), and a traditional Tabu Search (TS) heuristic.

## 🧠 Objective Function

In alignment with green manufacturing principles, the model accounts for both processing energy and machine idle energy. To address the dimensional disparity between time and energy, Z-score normalization is applied to construct the joint objective function:

$$Minimize \quad J = 0.25 \times Z_{makespan} + 0.75 \times Z_{TEC}$$

Static normalization parameters ($\mu$ and $\sigma$) are pre-computed and utilized throughout the evaluation to maintain the consistency of multi-objective weights across different stages of training and testing.

## 🛠️ Implementation Details

### 1. Deep Reinforcement Learning Solver
The DRL approach is implemented in `src/masked_ppo_fjssp.py` with the following characteristics:
* **Action Masking**: Utilizes the `MaskablePPO` framework from `sb3-contrib`. This mechanism is designed to enforce physical scheduling constraints (such as machine availability and operation precedence) by masking invalid actions during the agent's decision-making process.
* **Reward Shaping**: A dense reward structure is employed, utilizing normalized incremental costs for Makespan and TEC at each step to assist in network convergence.
* **Observation Management**: Value clipping is applied to state features to maintain gradient stability during the exploration phase.

### 2. Heuristic Algorithm Baseline
The Tabu Search implementation in `src/tabu_search_baseline.py` serves as a performance baseline:
* **Search Strategy**: Incorporates neighborhood structures based on critical paths and adaptive perturbation mechanisms.
* **Benchmarking**: Evaluation is conducted using uniform datasets and energy metrics consistent with the DRL model to facilitate comparative analysis.

## 📁 Project Structure

```text
FJSSP_Energy_Optimization/
├── data/                      # Datasets and operational parameters
│   ├── energy_data/           # Energy-related data files (Processing/Idle energy)
│   └── 车间数据.docx           # Standard FJSSP dataset (MK01-MK11)
├── src/                       # Source code directory
│   ├── masked_ppo_fjssp.py    # DRL (PPO) training and inference script
│   └── tabu_search_baseline.py# Tabu Search benchmark implementation
└── README.md                  # Project documentation
