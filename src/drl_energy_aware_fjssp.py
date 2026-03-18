import os
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

import numpy as np
import pandas as pd
import re
from docx import Document
import gymnasium as gym
from gymnasium import spaces

# ===== 核心修改 1：引入 MaskablePPO 和 ActionMasker =====
from sb3_contrib import MaskablePPO
from sb3_contrib.common.wrappers import ActionMasker
# ========================================================

import torch
import time
from multiprocessing import Pool
import matplotlib.pyplot as plt


# ==========================================
# 1. 数据读取与解析模块
# ==========================================
def read_docx(docx_path):
    doc = Document(docx_path)
    lines = []
    for para in doc.paragraphs:
        text = para.text.strip()
        if text:
            text = re.sub(r'\s+', ' ', text)
            lines.append(text)
    return lines


def parse_fjssp_data(data_lines):
    instances = []
    current_instance = None
    for line in data_lines:
        if line.startswith('MK'):
            if current_instance:
                instances.append(current_instance)
            current_instance = {'name': line, 'jobs': []}
            continue
        numbers = list(map(float, line.split()))
        if not numbers:
            continue
        if len(numbers) == 3:
            current_instance['num_jobs'] = int(numbers[0])
            current_instance['num_machines'] = int(numbers[1])
            current_instance['avg_machines_per_op'] = numbers[2]
        else:
            job = []
            idx = 0
            num_ops = int(numbers[idx])
            idx += 1
            for _ in range(num_ops):
                num_machines = int(numbers[idx])
                idx += 1
                machines = []
                for _ in range(num_machines):
                    machine_id = int(numbers[idx]) - 1
                    processing_time = float(numbers[idx + 1])
                    machines.append((machine_id, processing_time))
                    idx += 2
                job.append(machines)
            current_instance['jobs'].append(job)
    if current_instance:
        instances.append(current_instance)
    return instances


def load_energy_data(instance_name, num_jobs, max_ops, num_machines):
    # 修改路径 1: 能耗数据文件路径
    energy_file = f'./energy_data/energy_data_{instance_name}.xlsx'
    try:
        df = pd.read_excel(energy_file, engine='openpyxl')
    except FileNotFoundError:
        print(f"错误：能耗数据文件 {energy_file} 未找到。")
        raise

    PDk = np.zeros((num_jobs, max_ops), dtype=float)
    PWk = np.zeros(num_machines, dtype=float)

    for _, row in df.iterrows():
        job_id = int(row['Job_ID']) - 1
        op_id = int(row['Operation_ID']) - 1
        if 0 <= job_id < num_jobs and 0 <= op_id < max_ops:
            PDk[job_id, op_id] = float(row['Processing_Energy'])

    machine_energy = df[['Machine_ID', 'Idle_Energy']].drop_duplicates().set_index('Machine_ID')
    for machine_id in range(num_machines):
        try:
            PWk[machine_id] = float(machine_energy.loc[machine_id + 1, 'Idle_Energy'])
        except KeyError:
            PWk[machine_id] = 0.5  # 默认值

    return PDk, PWk


# ==========================================
# 2. 强化学习环境定义
# ==========================================
class FJSSPEnv(gym.Env):
    def __init__(self, instance, normalization_params):
        super(FJSSPEnv, self).__init__()
        self.instance = instance
        self.jobs = instance['jobs']
        self.num_jobs = instance['num_jobs']
        self.num_machines = instance['num_machines']
        self.max_ops = max(len(job) for job in self.jobs)
        self.max_proc_time = max(
            proc_time for job in self.jobs for op in job for _, proc_time in op
        )

        self.mu_m = float(normalization_params['mu_m'])
        self.sigma_m = float(normalization_params['sigma_m'])
        self.mu_t = float(normalization_params['mu_t'])
        self.sigma_t = float(normalization_params['sigma_t'])

        self.action_mapping = []
        for job_idx in range(self.num_jobs):
            for op_idx in range(len(self.jobs[job_idx])):
                for machine_id, _ in self.jobs[job_idx][op_idx]:
                    self.action_mapping.append((job_idx, op_idx, machine_id))
        max_actions = len(self.action_mapping)
        self.action_space = spaces.Discrete(max_actions)

        obs_dim = 2 + self.num_machines + self.num_jobs + self.num_jobs * self.max_ops * 4
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(obs_dim,), dtype=np.float32
        )

        self.PDk, self.PWk = load_energy_data(instance['name'], self.num_jobs, self.max_ops, self.num_machines)
        self.schedule = []
        self.reset()

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.makespan = 0.0
        self.TEC = 0.0
        self.machine_busy_until = [0.0] * self.num_machines
        self.job_op_idx = [0] * self.num_jobs
        self.completed_jobs = set()
        self.active_tasks = []
        self.schedule = []

        self.op_status = np.zeros((self.num_jobs, self.max_ops), dtype=int)
        for job_idx in range(self.num_jobs):
            self.op_status[job_idx, 0] = 2
            for op_idx in range(1, len(self.jobs[job_idx])):
                self.op_status[job_idx, op_idx] = 1
            for op_idx in range(len(self.jobs[job_idx]), self.max_ops):
                self.op_status[job_idx, op_idx] = 0

        return self._get_obs(), {}

    def _get_obs(self):
        norm_makespan = (self.makespan - self.mu_m) / self.sigma_m
        norm_tec = (self.TEC - self.mu_t) / self.sigma_t

        # 优化 1：加入极端值截断防御 (Clipping)
        # 训练初期模型可能会瞎探索导致 makespan 极大，截断可以防止特征爆炸导致梯度 NaN
        norm_makespan = np.clip(norm_makespan, -10.0, 10.0)
        norm_tec = np.clip(norm_tec, -10.0, 10.0)

        machine_status = np.array([
            (t - self.makespan) / self.max_proc_time if t > self.makespan else 0.0
            for t in self.machine_busy_until
        ])
        job_progress = np.array([
            self.job_op_idx[j] / len(self.jobs[j]) for j in range(self.num_jobs)
        ])
        op_status_one_hot = np.zeros((self.num_jobs, self.max_ops, 4))
        for j in range(self.num_jobs):
            for o in range(self.max_ops):
                status = self.op_status[j, o]
                op_status_one_hot[j, o, status] = 1

        obs = np.concatenate([
            [norm_makespan, norm_tec],
            machine_status,
            job_progress,
            op_status_one_hot.flatten()
        ]).astype(np.float32)
        return obs

    def _get_action_mask(self):
        mask = np.zeros(len(self.action_mapping), dtype=np.int8)
        idle_jobs = [j for j in range(self.num_jobs) if j not in self.completed_jobs]

        job_last_end_time = [0.0] * self.num_jobs
        for j, o, _, _, e in self.active_tasks:
            if o == self.job_op_idx[j] - 1:
                job_last_end_time[j] = max(job_last_end_time[j], e)

        for action_idx, (job_idx, op_idx, machine_id) in enumerate(self.action_mapping):
            if (job_idx in idle_jobs and
                    self.job_op_idx[job_idx] == op_idx and
                    self.op_status[job_idx, op_idx] == 2 and
                    self.machine_busy_until[machine_id] <= self.makespan):
                if op_idx == 0 or job_last_end_time[job_idx] <= self.makespan:
                    mask[action_idx] = 1

        if np.sum(mask) == 0:
            for action_idx, (job_idx, op_idx, _) in enumerate(self.action_mapping):
                if job_idx in idle_jobs and self.job_op_idx[job_idx] == op_idx:
                    mask[action_idx] = 1
        return mask

    def step(self, action):
        mask = self._get_action_mask()
        if mask[action] == 0:
            raise ValueError(f"严重错误：MaskablePPO 采样到了非法动作 {action}！")

        # 记录执行动作前的状态，用于计算密集奖励
        old_makespan = self.makespan
        old_tec = self.TEC

        job_idx, op_idx, machine_id = self.action_mapping[action]
        proc_time = float(next(pt for mid, pt in self.jobs[job_idx][op_idx] if mid == machine_id))

        job_last_end_time = 0.0
        for j, o, _, _, e in self.active_tasks:
            if j == job_idx and o == op_idx - 1:
                job_last_end_time = max(job_last_end_time, e)

        start_time = max(float(self.makespan), float(self.machine_busy_until[machine_id]), float(job_last_end_time))
        end_time = start_time + proc_time

        self.machine_busy_until[machine_id] = end_time
        self.active_tasks.append((job_idx, op_idx, machine_id, start_time, end_time))
        self.TEC += proc_time * self.PDk[job_idx, op_idx]
        self.op_status[job_idx, op_idx] = 3

        self.schedule.append({
            'Start_Time': start_time, 'Job_ID': job_idx + 1,
            'Operation_ID': op_idx + 1, 'Machine_ID': machine_id + 1, 'End_Time': end_time
        })

        self.job_op_idx[job_idx] += 1
        self.op_status[job_idx, op_idx] = 0
        if self.job_op_idx[job_idx] < len(self.jobs[job_idx]):
            self.op_status[job_idx, self.job_op_idx[job_idx]] = 2
        else:
            self.completed_jobs.add(job_idx)

        self.active_tasks = [(j, o, m, s, e) for j, o, m, s, e in self.active_tasks if e > self.makespan]

        idle_jobs = [j for j in range(self.num_jobs) if j not in self.completed_jobs]
        available_machines = [m for m in range(self.num_machines) if self.machine_busy_until[m] <= self.makespan]

        job_last_end_time_array = [0.0] * self.num_jobs
        for j, o, _, _, e in self.active_tasks:
            if o == self.job_op_idx[j] - 1:
                job_last_end_time_array[j] = max(job_last_end_time_array[j], e)

        can_schedule = False
        for j in idle_jobs:
            if job_last_end_time_array[j] <= self.makespan and any(
                    self.machine_busy_until[mid] <= self.makespan for mid, _ in self.jobs[j][self.job_op_idx[j]]
            ):
                can_schedule = True
                break

        if not (can_schedule and available_machines):
            next_times = [t for t in self.machine_busy_until if t > self.makespan]
            next_times.extend(
                [job_last_end_time_array[j] for j in idle_jobs if job_last_end_time_array[j] > self.makespan])

            if next_times:
                old_makespan_jump = self.makespan
                self.makespan = float(min(next_times))
                for m in range(self.num_machines):
                    if self.machine_busy_until[m] <= old_makespan_jump:
                        self.TEC += (self.makespan - old_makespan_jump) * self.PWk[m]
            else:
                self.makespan += 1.0

        done = len(self.completed_jobs) == self.num_jobs
        if done and self.schedule:
            self.makespan = float(max(record['End_Time'] for record in self.schedule))

        # 优化 2：计算密集奖励 (Dense Reward)
        # 将总目标分解到每一步：计算当前步造成的 Makespan 和 TEC 的“归一化增量”
        # 这样模型每走一步都能立刻得到反馈，而不是等到最后一步瞎猜
        delta_makespan_norm = (self.makespan - old_makespan) / self.sigma_m
        delta_tec_norm = (self.TEC - old_tec) / self.sigma_t

        # 如果是最后一步结束，我们额外给予一个“全局 Z-score 结算”作为补充纠偏
        if done:
            final_z_makespan = (self.makespan - self.mu_m) / self.sigma_m
            final_z_tec = (self.TEC - self.mu_t) / self.sigma_t
            step_reward = - (0.25 * final_z_makespan + 0.75 * final_z_tec)
        else:
            # 增量惩罚（最小化时间的流逝和能耗的增加）
            step_reward = - (0.25 * delta_makespan_norm + 0.75 * delta_tec_norm)

        info = {}
        return self._get_obs(), step_reward, done, False, info


def mask_fn(env: gym.Env) -> np.ndarray:
    return env.unwrapped._get_action_mask()


# ==========================================
# 3. 评估与记录模块
# ==========================================
def evaluate_and_save(env, model, instance_name, iteration, results_base_path):
    best_score = float('inf')
    best_makespan, best_tec, best_schedule = None, None, None

    u_env = env.unwrapped

    for _ in range(20):
        obs, _ = env.reset()
        done = False
        step_count = 0
        while not done:
            action_masks = mask_fn(env)
            action, _ = model.predict(obs, action_masks=action_masks, deterministic=True)
            obs, reward, done, _, info = env.step(action)
            step_count += 1
            if step_count > 1000:
                done = True

        makespan = float(max(record['End_Time'] for record in u_env.schedule)) if u_env.schedule else 0.0
        z_makespan = (makespan - u_env.mu_m) / u_env.sigma_m
        z_tec = (u_env.TEC - u_env.mu_t) / u_env.sigma_t
        score = 0.25 * z_makespan + 0.75 * z_tec

        if score < best_score:
            best_score = score
            best_makespan = makespan
            best_tec = u_env.TEC
            best_schedule = u_env.schedule.copy()

    print(f"{instance_name} 迭代 {iteration}: 最优 Makespan={best_makespan:.2f}, 最优 TEC={best_tec:.2f}")

    instance_result_path = os.path.join(results_base_path, instance_name, str(iteration))
    os.makedirs(instance_result_path, exist_ok=True)

    if best_schedule:
        path_df = pd.DataFrame(best_schedule)
        path_excel_path = os.path.join(instance_result_path, f'ppo_best_path_{instance_name}_{iteration}.xlsx')
        try:
            path_df.to_excel(path_excel_path, index=False, engine='openpyxl')
        except Exception:
            pass

        metrics_df = pd.DataFrame({
            'Instance': [instance_name], 'Iteration': [iteration], 'Makespan': [best_makespan], 'TEC': [best_tec]
        })
        metrics_excel_path = os.path.join(instance_result_path, f'ppo_best_metrics_{instance_name}_{iteration}.xlsx')
        try:
            metrics_df.to_excel(metrics_excel_path, index=False, engine='openpyxl')
        except Exception:
            pass

    return best_makespan, best_tec, best_schedule


# ==========================================
# 4. 训练逻辑主干
# ==========================================
def train_instance(args):
    instance, normalization_params, docx_path, excel_path = args
    instance_name = instance['name']
    episodes = 50000

    raw_env = FJSSPEnv(instance, normalization_params[instance_name])
    env = ActionMasker(raw_env, mask_fn)

    model = MaskablePPO(
        policy="MlpPolicy",
        env=env,
        learning_rate=3e-4,
        gamma=0.99,
        gae_lambda=0.95,
        clip_range=0.2,
        ent_coef=0.02,
        n_steps=2048,
        batch_size=128,
        n_epochs=10,
        device="cpu",
        verbose=0
    )

    # 修改路径 2: 结果保存根目录和图表保存目录
    results_base_path = '.'
    reward_plot_path = '.'
    os.makedirs(reward_plot_path, exist_ok=True)

    rewards = []
    u_env = env.unwrapped

    for ep in range(episodes):
        model.learn(total_timesteps=2048)

        obs, _ = env.reset()
        done = False
        episode_reward = 0
        step_count = 0
        while not done:
            action_masks = mask_fn(env)
            action, _ = model.predict(obs, action_masks=action_masks)
            obs, reward, done, _, info = env.step(action)
            episode_reward += reward
            step_count += 1
            if step_count > 1000:
                done = True

        rewards.append(episode_reward)

        if (ep + 1) % 10 == 0:
            print(f"{instance_name} Episode {ep + 1}/{episodes}, 奖励: {episode_reward:.2f}, 步数: {step_count}")

        if (ep + 1) % 5000 == 0:
            evaluate_and_save(env, model, instance_name, ep + 1, results_base_path)

        if ep >= 100:
            mean_reward = np.mean(rewards[-100:])
            std_reward = np.std(rewards[-100:])
            if std_reward / (abs(mean_reward) + 1e-8) < 0.005:
                print(f"{instance_name} 在第 {ep + 1} 轮训练判定收敛并终止")
                break

    plt.figure(figsize=(10, 6))
    plt.plot(range(1, len(rewards) + 1), rewards, label='Reward')
    plt.xlabel('Episode')
    plt.ylabel('Reward')
    plt.title(f'Reward Curve for {instance_name}')
    plt.legend()
    plt.grid(True)
    plot_path = os.path.join(reward_plot_path, f'{instance_name}_reward.png')
    try:
        plt.savefig(plot_path)
    except Exception:
        pass
    plt.close()

    final_makespan, final_tec, final_schedule = evaluate_and_save(env, model, instance_name, episodes,
                                                                  results_base_path)

    if final_schedule:
        path_df = pd.DataFrame(final_schedule)
        final_path_excel_path = os.path.join(os.path.dirname(excel_path), f'ppo_best_path_{instance_name}.xlsx')
        try:
            path_df.to_excel(final_path_excel_path, index=False, engine='openpyxl')
        except Exception as e:
            # 修改路径 3: 异常备份路径
            backup_path = f'D:\\pycharm community\\machine learning test\\FJSP\\ppo_best_path_{instance_name}_backup.xlsx'
            path_df.to_excel(backup_path, index=False, engine='openpyxl')

    return (instance_name, final_makespan, final_tec)


# ==========================================
# 5. 主程序入口
# ==========================================
def main(docx_path, excel_path):
    data_lines = read_docx(docx_path)
    instances = parse_fjssp_data(data_lines)

    normalization_params = {
        'MK01': {'mu_m': 58.43, 'sigma_m': 6.2789, 'mu_t': 419.465, 'sigma_t': 47.331},
        'MK02': {'mu_m': 47.36, 'sigma_m': 10.1208, 'mu_t': 404.119, 'sigma_t': 107.490},
        'MK03': {'mu_m': 255.86, 'sigma_m': 18.0159, 'mu_t': 3011.279, 'sigma_t': 210.933},
        'MK04': {'mu_m': 93.10, 'sigma_m': 12.1543, 'mu_t': 824.529, 'sigma_t': 109.316},
        'MK05': {'mu_m': 218.91, 'sigma_m': 15.9297, 'mu_t': 2258.841, 'sigma_t': 179.013},
        'MK06': {'mu_m': 96.15, 'sigma_m': 14.2238, 'mu_t': 1112.054, 'sigma_t': 174.178},
        'MK07': {'mu_m': 260.73, 'sigma_m': 27.5562, 'mu_t': 2330.053, 'sigma_t': 254.372},
        'MK08': {'mu_m': 531.08, 'sigma_m': 35.5648, 'mu_t': 5800.536, 'sigma_t': 400.599},
        'MK09': {'mu_m': 387.50, 'sigma_m': 30.1606, 'mu_t': 4319.576, 'sigma_t': 356.662},
        'MK10': {'mu_m': 338.23, 'sigma_m': 47.1202, 'mu_t': 5842.201, 'sigma_t': 741.054},
        'MK11': {'mu_m': 52.42, 'sigma_m': 5.02, 'mu_t': 1019.71, 'sigma_t': 37.61}
    }

    with Pool(processes=10) as pool:
        args = [(instance, normalization_params, docx_path, excel_path) for instance in instances if
                instance['name'] in normalization_params]
        results = pool.map(train_instance, args)

    df = pd.DataFrame(results, columns=['实例', '最优 Makespan', '最优 TEC'])
    try:
        df.to_excel(excel_path, index=False, engine='openpyxl')
        print(f"\n结果已保存到 {excel_path}")
    except PermissionError:
        # 修改路径 4: 结果备份路径
        backup_excel_path = 'ppo_results_backup.xlsx'
        df.to_excel(backup_excel_path, index=False, engine='openpyxl')
        print(f"\n结果已保存到备用路径 {backup_excel_path}")


if __name__ == "__main__":
    # 修改路径 5: 主输入输出文件路径
    docx_path = '车间数据.docx'
    excel_path = 'ppo_best_results.xlsx'
    main(docx_path, excel_path)