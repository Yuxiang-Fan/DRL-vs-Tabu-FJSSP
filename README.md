# Energy-Aware Flexible Job Shop Scheduling: DRL vs. Tabu Search

This repository presents a comparative study of optimization algorithms for the **Multi-Objective Flexible Job Shop Scheduling Problem (FJSSP)**. The project aims to simultaneously minimize the **Makespan** (total completion time) and **Total Energy Consumption (TEC)** of the workshop. 

To solve this NP-Hard problem, this project implements and compares two distinct approaches: a state-of-the-art **Deep Reinforcement Learning (Masked PPO)** algorithm and a classic operational research **Heuristic Algorithm (Tabu Search)**.

## 🎯 Objective Function

In the context of green manufacturing, both processing energy and machine idle energy are considered. Because time and energy have different dimensions, static Z-score normalization is applied to construct a strict joint objective function:

$Minimize \quad J=0.25\times Z_{makespan}+0.75\times Z_{TEC}$

*Note: Static Z-scores (pre-computed $\mu$ and $\sigma$) are used instead of dynamic normalization to ensure the physical meaning of the multi-objective weights remains absolutely stable throughout the entire training and evaluation lifecycle.*

## 🚀 Key Features & Technical Highlights

### 1. Deep Reinforcement Learning Solver
Located in: `src/masked_ppo_fjssp.py`
* **Action Masking**: Utilizes `MaskablePPO` and `ActionMasker` from `sb3-contrib`. By masking the logits of invalid actions in the computational graph, the agent is **100% guaranteed to satisfy complex physical scheduling constraints** (e.g., machine availability, operation precedence), solving a major pain point of traditional RL in combinatorial optimization.
* **Dense Reward Shaping**: Breaks down the sparse terminal reward into step-by-step decisions by calculating the "normalized incremental cost" of Makespan and TEC for each action, significantly accelerating network convergence.
* **Observation Clipping**: Applies extreme value clipping `[-10, 10]` to the state features fed into the neural network, effectively preventing gradient explosion during early exploration.

### 2. Heuristic Algorithm Baseline
Located in: `src/tabu_search_baseline.py`
* **Tabu Search**: Serves as a powerful operational research baseline, featuring Critical Path neighborhood structures and adaptive perturbation mechanisms.
* **Fair Benchmarking**: Shares the exact same dataset, energy evaluation metrics, and static Z-score parameters as the DRL model to ensure rigorous comparative experiments.

## 📁 Project Structure

```text
FJSSP_Energy_Optimization/
├── data/                      # Datasets and operational parameters
│   ├── energy_data/           # Excel files containing processing and idle energy data
│   └── 车间数据.docx           # Standard FJSSP dataset (MK01-MK11)
├── src/                       # Source code directory
│   ├── masked_ppo_fjssp.py    # DRL (PPO) training and inference script
│   └── tabu_search_baseline.py# Tabu Search benchmark script
└── README.md                  # Project documentation