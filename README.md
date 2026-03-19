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
```

---

# 能源感知型柔性车间调度：DRL 与禁忌搜索的对比研究

本项目针对多目标柔性作业车间调度问题 (FJSSP) 提供了优化算法的实现与对比分析。研究重点是在制造业环境中同时最小化**完工时间 (Makespan)** 和**总能耗 (TEC)**。

项目评估了两种不同的方法论来解决这一 NP-hard 问题：基于掩码近端策略优化 (Masked PPO) 的深度强化学习 (DRL) 框架，以及传统的禁忌搜索 (TS) 启发式算法。

## 🧠 目标函数

为了符合绿色制造原则，模型同时考虑了加工能耗和机器空闲能耗。为了解决时间与能量之间的维度差异，采用 Z-score 标准化构建联合目标函数：

$$Minimize \quad J = 0.25 \times Z_{makespan} + 0.75 \times Z_{TEC}$$

静态标准化参数 ($\mu$ 和 $\sigma$) 经过预计算，并在整个评估过程中使用，以保持不同训练和测试阶段多目标权重的一致性。

## 🛠️ 实现细节

### 1. 深度强化学习求解器
DRL 方法在 `src/masked_ppo_fjssp.py` 中实现，具有以下特点：
* **动作掩码 (Action Masking)**：利用 `sb3-contrib` 中的 `MaskablePPO` 框架。该机制旨在通过在智能体决策过程中遮蔽无效动作，强制执行物理调度约束（如机器可用性和工序先后顺序）。
* **奖励塑形 (Reward Shaping)**：采用稠密奖励结构，利用每一步 Makespan 和 TEC 的归一化增量成本来辅助网络收敛。
* **观测管理 (Observation Management)**：对状态特征进行数值裁剪（Value clipping），以保持探索阶段的梯度稳定性。

### 2. 启发式算法基准
在 `src/tabu_search_baseline.py` 中实现的禁忌搜索作为性能基准：
* **搜索策略**：结合了基于关键路径的邻域结构和自适应扰动机制。
* **基准测试**：使用统一的数据集和与 DRL 模型一致的能量指标进行评估，以便进行对比分析。

## 📁 项目结构

```text
FJSSP_Energy_Optimization/
├── data/                      # 数据集与运行参数
│   ├── energy_data/           # 能源相关数据文件（加工/空闲能耗）
│   └── 车间数据.docx           # 标准 FJSSP 数据集 (MK01-MK11)
├── src/                       # 源代码目录
│   ├── masked_ppo_fjssp.py    # DRL (PPO) 训练与推理脚本
│   └── tabu_search_baseline.py# 禁忌搜索基准实现
└── README.md                  # 项目文档
```
