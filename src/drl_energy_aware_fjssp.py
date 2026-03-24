import os
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

import numpy as np
import pandas as pd
import re
from docx import Document
import gymnasium as gym
from gymnasium import spaces
from sb3_contrib import MaskablePPO
from sb3_contrib.common.wrappers import ActionMasker
import torch
import time
from multiprocessing import Pool
import matplotlib.pyplot as plt

def read_docx(docx_path):
    """从 Word 文档中提取文本行并清洗空白字符"""
    doc = Document(docx_path)
    lines = []
    for para in doc.paragraphs:
        text = para.text.strip()
        if text:
            text = re.sub(r'\s+', ' ', text)
            lines.append(text)
    return lines

def parse_fjssp_data(data_lines):
    """
    解析柔性作业车间调度问题 (FJSSP) 的标准数据集。
    返回实例列表，包含机器数、工件数、工序及对应加工时间和可选机器。
    """
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
        
        # 解析全局参数行
        if len(numbers) == 3:
            current_instance['num_jobs'] = int(numbers[0])
            current_instance['num_machines'] = int(numbers[1])
            current_instance['avg_machines_per_op'] = numbers[2]
        else:
            # 解析具体工件的工序信息
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
    """
    加载能耗数据，包括机器加工能耗和空闲能耗。
    """
    energy_file = f'./energy_data/energy_data_{instance_name}.xlsx'
    try:
        df = pd.read_excel(energy_file, engine='openpyxl')
    except FileNotFoundError:
        print(f"数据读取失败：缺失能耗文件 {energy_file}")
        raise

    process_energy = np.zeros((num_jobs, max_ops), dtype=float)
    idle_energy = np.zeros(num_machines, dtype=float)

    for _, row in df.iterrows():
        job_id = int(row['Job_ID']) - 1
        op_id = int(row['Operation_ID']) - 1
        if 0 <= job_id < num_jobs and 0 <= op_id < max_ops:
            process_energy[job_id, op_id] = float(row['Processing_Energy'])

    machine_energy_df = df[['Machine_ID', 'Idle_Energy']].drop_duplicates().set_index('Machine_ID')
    for machine_id in range(num_machines):
        try:
            idle_energy[machine_id] = float(machine_energy_df.loc[machine_id + 1, 'Idle_Energy'])
        except KeyError:
            idle_energy[machine_id] = 0.5  # 缺失值的默认空闲能耗惩罚

    return process_energy, idle_energy


class FJSSPEnv(gym.Env):
    """
    柔性作业车间调度问题的自定义强化学习环境。
    优化目标：联合最小化最大完工时间与总能耗。
    """
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

        # 接收预计算的Z-score标准化参数，用于对齐量纲
        self.mu_m = float(normalization_params['mu_m'])
        self.sigma_m = float(normalization_params['sigma_m'])
        self.mu_t = float(normalization_params['mu_t'])
        self.sigma_t = float(normalization_params['sigma_t'])

        # 构建动作空间映射：(工件ID, 工序ID, 分配的机器ID)
        self.action_mapping = []
        for job_idx in range(self.num_jobs):
            for op_idx in range(len(self.jobs[job_idx])):
                for machine_id, _ in self.jobs[job_idx][op_idx]:
                    self.action_mapping.append((job_idx, op_idx, machine_id))
                    
        max_actions = len(self.action_mapping)
        self.action_space = spaces.Discrete(max_actions)

        # 观测空间维度设计：全局目标 + 机器状态 + 工件进度 + 工序状态独热编码
        obs_dim = 2 + self.num_machines + self.num_jobs + self.num_jobs * self.max_ops * 4
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(obs_dim,), dtype=np.float32
        )

        self.process_energy, self.idle_energy = load_energy_data(
            instance['name'], self.num_jobs, self.max_ops, self.num_machines
        )
        self.schedule = []
        self.reset()

    def reset(self, seed=None, options=None):
        """初始化回合状态"""
        super().reset(seed=seed)
        self.current_makespan = 0.0
        self.current_tec = 0.0
        self.machine_busy_until = [0.0] * self.num_machines
        self.job_op_idx = [0] * self.num_jobs
        self.completed_jobs = set()
        self.active_tasks = []
        self.schedule = []

        # 工序状态矩阵: 0-未激活, 1-等待中(前置工序未完成), 2-可调度, 3-已完成/加工中
        self.op_status = np.zeros((self.num_jobs, self.max_ops), dtype=int)
        for job_idx in range(self.num_jobs):
            self.op_status[job_idx, 0] = 2
            for op_idx in range(1, len(self.jobs[job_idx])):
                self.op_status[job_idx, op_idx] = 1
            for op_idx in range(len(self.jobs[job_idx]), self.max_ops):
                self.op_status[job_idx, op_idx] = 0

        return self._get_obs(), {}

    def _get_obs(self):
        """构建并返回当前环境的观测向量"""
        norm_makespan = (self.current_makespan - self.mu_m) / self.sigma_m
        norm_tec = (self.current_tec - self.mu_t) / self.sigma_t

        # 引入数值裁剪以防止探索初期梯度爆炸
        norm_makespan = np.clip(norm_makespan, -10.0, 10.0)
        norm_tec = np.clip(norm_tec, -10.0, 10.0)

        machine_status = np.array([
            (t - self.current_makespan) / self.max_proc_time if t > self.current_makespan else 0.0
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
        """动态生成合法动作掩码，过滤违反时序约束和物理约束的动作"""
        mask = np.zeros(len(self.action_mapping), dtype=np.int8)
        idle_jobs = [j for j in range(self.num_jobs) if j not in self.completed_jobs]

        job_last_end_time = [0.0] * self.num_jobs
        for j, o, _, _, e in self.active_tasks:
            if o == self.job_op_idx[j] - 1:
                job_last_end_time[j] = max(job_last_end_time[j], e)

        for action_idx, (job_idx, op_idx, machine_id) in enumerate(self.action_mapping):
            is_job_ready = job_idx in idle_jobs and self.job_op_idx[job_idx] == op_idx
            is_op_schedulable = self.op_status[job_idx, op_idx] == 2
            is_machine_available = self.machine_busy_until[machine_id] <= self.current_makespan
            
            if is_job_ready and is_op_schedulable and is_machine_available:
                if op_idx == 0 or job_last_end_time[job_idx] <= self.current_makespan:
                    mask[action_idx] = 1

        # 死锁兜底机制：若无立即可用动作，放宽机器可用性限制以推进时间
        if np.sum(mask) == 0:
            for action_idx, (job_idx, op_idx, _) in enumerate(self.action_mapping):
                if job_idx in idle_jobs and self.job_op_idx[job_idx] == op_idx:
                    mask[action_idx] = 1
        return mask

    def step(self, action):
        """执行调度动作并推进环境状态"""
        mask = self._get_action_mask()
        if mask[action] == 0:
            raise ValueError(f"策略异常：采样到被掩码屏蔽的非法动作 {action}。")

        old_makespan = self.current_makespan
        old_tec = self.current_tec

        job_idx, op_idx, machine_id = self.action_mapping[action]
        proc_time = float(next(pt for mid, pt in self.jobs[job_idx][op_idx] if mid == machine_id))

        # 计算工序实际开工时间，需满足前置工序完工及机器空闲双重约束
        job_last_end_time = 0.0
        for j, o, _, _, e in self.active_tasks:
            if j == job_idx and o == op_idx - 1:
                job_last_end_time = max(job_last_end_time, e)

        start_time = max(float(self.current_makespan), float(self.machine_busy_until[machine_id]), float(job_last_end_time))
        end_time = start_time + proc_time

        # 更新系统状态
        self.machine_busy_until[machine_id] = end_time
        self.active_tasks.append((job_idx, op_idx, machine_id, start_time, end_time))
        self.current_tec += proc_time * self.process_energy[job_idx, op_idx]
        self.op_status[job_idx, op_idx] = 3

        self.schedule.append({
            'Start_Time': start_time, 'Job_ID': job_idx + 1,
            'Operation_ID': op_idx + 1, 'Machine_ID': machine_id + 1, 'End_Time': end_time
        })

        # 推进工件进度
        self.job_op_idx[job_idx] += 1
        self.op_status[job_idx, op_idx] = 0
        if self.job_op_idx[job_idx] < len(self.jobs[job_idx]):
            self.op_status[job_idx, self.job_op_idx[job_idx]] = 2
        else:
            self.completed_jobs.add(job_idx)

        # 清理已完成的活跃任务
        self.active_tasks = [(j, o, m, s, e) for j, o, m, s, e in self.active_tasks if e > self.current_makespan]

        # 时间推进逻辑
        idle_jobs = [j for j in range(self.num_jobs) if j not in self.completed_jobs]
        available_machines = [m for m in range(self.num_machines) if self.machine_busy_until[m] <= self.current_makespan]

        job_last_end_time_array = [0.0] * self.num_jobs
        for j, o, _, _, e in self.active_tasks:
            if o == self.job_op_idx[j] - 1:
                job_last_end_time_array[j] = max(job_last_end_time_array[j], e)

        can_schedule = False
        for j in idle_jobs:
            if job_last_end_time_array[j] <= self.current_makespan and any(
                    self.machine_busy_until[mid] <= self.current_makespan for mid, _ in self.jobs[j][self.job_op_idx[j]]
            ):
                can_schedule = True
                break

        if not (can_schedule and available_machines):
            next_times = [t for t in self.machine_busy_until if t > self.current_makespan]
            next_times.extend(
                [job_last_end_time_array[j] for j in idle_jobs if job_last_end_time_array[j] > self.current_makespan])

            if next_times:
                old_makespan_jump = self.current_makespan
                self.current_makespan = float(min(next_times))
                # 累加机器待机能耗
                for m in range(self.num_machines):
                    if self.machine_busy_until[m] <= old_makespan_jump:
                        self.current_tec += (self.current_makespan - old_makespan_jump) * self.idle_energy[m]
            else:
                self.current_makespan += 1.0

        done = len(self.completed_jobs) == self.num_jobs
        if done and self.schedule:
            self.current_makespan = float(max(record['End_Time'] for record in self.schedule))

        # 稠密奖励设计：基于归一化增量的步进惩罚
        delta_makespan_norm = (self.current_makespan - old_makespan) / self.sigma_m
        delta_tec_norm = (self.current_tec - old_tec) / self.sigma_t

        if done:
            # 终点全局Z-score校准
            final_z_makespan = (self.current_makespan - self.mu_m) / self.sigma_m
            final_z_tec = (self.current_tec - self.mu_t) / self.sigma_t
            step_reward = - (0.25 * final_z_makespan + 0.75 * final_z_tec)
        else:
            step_reward = - (0.25 * delta_makespan_norm + 0.75 * delta_tec_norm)

        return self._get_obs(), step_reward, done, False, {}

def mask_fn(env: gym.Env) -> np.ndarray:
    """包装器所需的掩码提取函数"""
    return env.unwrapped._get_action_mask()

def evaluate_and_save(env, model, instance_name, iteration, results_base_path):
    """评估当前策略模型并保存调度甘特图数据"""
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
            obs, reward, done, _, _ = env.step(action)
            step_count += 1
            if step_count > 1000:
                done = True

        makespan = float(max(record['End_Time'] for record in u_env.schedule)) if u_env.schedule else 0.0
        z_makespan = (makespan - u_env.mu_m) / u_env.sigma_m
        z_tec = (u_env.current_tec - u_env.mu_t) / u_env.sigma_t
        score = 0.25 * z_makespan + 0.75 * z_tec

        if score < best_score:
            best_score = score
            best_makespan = makespan
            best_tec = u_env.current_tec
            best_schedule = u_env.schedule.copy()

    print(f"[{instance_name}] 迭代 {iteration}: 最佳完工时间={best_makespan:.2f}, 最佳能耗={best_tec:.2f}")

    instance_result_path = os.path.join(results_base_path, instance_name, str(iteration))
    os.makedirs(instance_result_path, exist_ok=True)

    if best_schedule:
        try:
            pd.DataFrame(best_schedule).to_excel(
                os.path.join(instance_result_path, f'ppo_schedule_{instance_name}_{iteration}.xlsx'), 
                index=False, engine='openpyxl'
            )
            pd.DataFrame({
                'Instance': [instance_name], 'Iteration': [iteration], 
                'Makespan': [best_makespan], 'TEC': [best_tec]
            }).to_excel(
                os.path.join(instance_result_path, f'ppo_metrics_{instance_name}_{iteration}.xlsx'), 
                index=False, engine='openpyxl'
            )
        except Exception as e:
            print(f"文件保存异常: {e}")

    return best_makespan, best_tec, best_schedule

def train_instance(args):
    """单实例训练进程逻辑"""
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

    results_base_path = './results'
    reward_plot_path = './plots'
    os.makedirs(reward_plot_path, exist_ok=True)

    rewards = []
    
    for ep in range(episodes):
        model.learn(total_timesteps=2048)

        obs, _ = env.reset()
        done = False
        episode_reward = 0
        step_count = 0
        while not done:
            action_masks = mask_fn(env)
            action, _ = model.predict(obs, action_masks=action_masks)
            obs, reward, done, _, _ = env.step(action)
            episode_reward += reward
            step_count += 1
            if step_count > 1000:
                done = True

        rewards.append(episode_reward)

        if (ep + 1) % 10 == 0:
            print(f"[{instance_name}] Episode {ep + 1}/{episodes} | 奖励: {episode_reward:.2f} | 调度步数: {step_count}")

        if (ep + 1) % 5000 == 0:
            evaluate_and_save(env, model, instance_name, ep + 1, results_base_path)

        # 早停机制检测收敛
        if ep >= 100:
            mean_reward = np.mean(rewards[-100:])
            std_reward = np.std(rewards[-100:])
            if std_reward / (abs(mean_reward) + 1e-8) < 0.005:
                print(f"[{instance_name}] 判定收敛，于第 {ep + 1} 轮提前终止。")
                break

    plt.figure(figsize=(10, 6))
    plt.plot(range(1, len(rewards) + 1), rewards, label='Reward')
    plt.xlabel('Episode')
    plt.ylabel('Cumulative Reward')
    plt.title(f'Convergence Curve: {instance_name}')
    plt.legend()
    plt.grid(True)
    try:
        plt.savefig(os.path.join(reward_plot_path, f'{instance_name}_convergence.png'))
    except Exception:
        pass
    plt.close()

    final_makespan, final_tec, final_schedule = evaluate_and_save(
        env, model, instance_name, episodes, results_base_path
    )

    if final_schedule:
        try:
            pd.DataFrame(final_schedule).to_excel(
                os.path.join(os.path.dirname(excel_path), f'ppo_final_schedule_{instance_name}.xlsx'), 
                index=False, engine='openpyxl'
            )
        except Exception:
            backup_path = f'./ppo_final_schedule_{instance_name}_backup.xlsx'
            pd.DataFrame(final_schedule).to_excel(backup_path, index=False, engine='openpyxl')

    return (instance_name, final_makespan, final_tec)

def main(docx_path, excel_path):
    data_lines = read_docx(docx_path)
    instances = parse_fjssp_data(data_lines)

    # 预设的环境静态标准化参数，用于对齐Makespan和TEC的量纲
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

    # 使用多进程池并行训练各个测试实例
    with Pool(processes=10) as pool:
        args = [(instance, normalization_params, docx_path, excel_path) for instance in instances if
                instance['name'] in normalization_params]
        results = pool.map(train_instance, args)

    df = pd.DataFrame(results, columns=['Instance', 'Best Makespan', 'Best TEC'])
    try:
        df.to_excel(excel_path, index=False, engine='openpyxl')
        print(f"\n全局寻优结果已汇总至: {excel_path}")
    except PermissionError:
        backup_excel_path = './ppo_results_backup.xlsx'
        df.to_excel(backup_excel_path, index=False, engine='openpyxl')
        print(f"\n文件被占用，结果已保存至备用路径: {backup_excel_path}")

if __name__ == "__main__":
    main('车间数据.docx', 'ppo_evaluation_results.xlsx')
