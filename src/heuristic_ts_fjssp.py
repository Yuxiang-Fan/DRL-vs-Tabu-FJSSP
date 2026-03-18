import random
import copy
import time
from docx import Document
import re
import pandas as pd
import numpy as np
from concurrent.futures import ThreadPoolExecutor

# 参数定义
CHECK_INTERVAL = 10
MAX_ITERATIONS = 10000

# Z-score 参数（仅保留 MK11）
# Z-score 参数（对齐 PPO 的所有 MK 实例）
Z_SCORE_PARAMS = {
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

# 调度结果缓存
schedule_cache = {}

# 从 Word 文档读取数据
def read_docx(docx_path):
    doc = Document(docx_path)
    lines = []
    for para in doc.paragraphs:
        text = para.text.strip()
        if text:
            text = re.sub(r'\s+', ' ', text)
            lines.append(text)
    return lines

# 解析 FJSSP 数据集（仅处理 MK11）
# 解析 FJSSP 数据集（处理所有 MK 数据）
def parse_fjssp_data(data_lines):
    instances = []
    current_instance = None
    for line in data_lines:
        if line.startswith('MK'):
            if current_instance:
                instances.append(current_instance)
            current_instance = {'name': line.strip(), 'jobs': []}
            continue
        if not current_instance:
            continue
        try:
            numbers = list(map(float, line.split()))
        except ValueError:
            continue
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

# 从 Excel 读取能耗数据
def load_energy_data(excel_path, instance_name, num_jobs, max_ops, num_machines):
    try:
        df = pd.read_excel(excel_path, engine='openpyxl')
    except FileNotFoundError:
        print(f"错误：能耗数据文件 {excel_path} 未找到。")
        raise
    except Exception as e:
        print(f"读取 {excel_path} 时出错：{e}")
        raise
    PDk = {}
    PWk = np.zeros(num_machines, dtype=float)
    for _, row in df.iterrows():
        job_id = int(row['Job_ID']) - 1
        op_id = int(row['Operation_ID']) - 1
        machine_id = int(row['Machine_ID1']) - 1
        proc_energy = float(row['Processing_Energy']) if not pd.isna(row['Processing_Energy']) else 0.0
        if 0 <= job_id < num_jobs and 0 <= op_id < max_ops and 0 <= machine_id < num_machines:
            PDk[(job_id, op_id, machine_id)] = proc_energy
    idle_data = df[['Machine_ID2', 'Idle_Energy']].dropna().drop_duplicates().set_index('Machine_ID2')
    for machine_id in range(num_machines):
        try:
            PWk[machine_id] = float(idle_data.loc[machine_id + 1, 'Idle_Energy'])
        except KeyError:
            print(f"警告：在 {instance_name} 中未找到机器 {machine_id + 1} 的空闲能耗数据。使用默认值 0.5。")
            PWk[machine_id] = 0.5
    return PDk, PWk

# 初始化解
def initialize_solution(job_data, num_machines):
    solution = [[] for _ in range(len(job_data))]
    machine_load = [0] * num_machines
    job_indices = sorted(range(len(job_data)), key=lambda i: sum(min(op, key=lambda x: x[1])[1] for op in job_data[i]), reverse=True)
    for job_idx in job_indices:
        job = job_data[job_idx]
        for op in job:
            best_machine, best_duration = min(op, key=lambda x: machine_load[x[0]] + x[1])
            solution[job_idx].append((best_machine, best_duration))
            machine_load[best_machine] += best_duration
    return solution

# 计算关键路径
def find_critical_path(solution, job_data, num_machines):
    machine_times = [0] * num_machines
    job_times = [0] * len(solution)
    critical_ops = []
    for job_idx, job in enumerate(solution):
        for op_idx, (machine, duration) in enumerate(job):
            start = max(job_times[job_idx], machine_times[machine])
            end = start + duration
            job_times[job_idx] = end
            machine_times[machine] = end
            if end == max(machine_times):
                critical_ops.append((job_idx, op_idx))
    return machine_times, critical_ops

# 计算 makespan 和 TEC 以及它们的 Z-score
def evaluate_schedule_makespan(solution, num_jobs, num_machines, PDk, PWk, z_params):
    solution_key = tuple(map(tuple, solution))
    if solution_key in schedule_cache:
        return schedule_cache[solution_key]
    machine_times = np.zeros(num_machines)
    job_op_idx = np.zeros(num_jobs, dtype=int)
    job_times = np.zeros(num_jobs)
    processing_energy = 0.0
    operations = [(job_idx, op_idx, machine, duration)
                  for job_idx, job in enumerate(solution)
                  for op_idx, (machine, duration) in enumerate(job)]
    completed_ops = 0
    total_ops = len(operations)
    while completed_ops < total_ops:
        earliest_start = float('inf')
        next_op = None
        for job_idx, op_idx, machine, duration in operations:
            if op_idx == job_op_idx[job_idx]:
                start_time = job_times[job_idx]
                machine_available_time = machine_times[machine]
                actual_start = max(start_time, machine_available_time)
                if actual_start < earliest_start:
                    earliest_start = actual_start
                    next_op = (job_idx, op_idx, machine, duration)
        if next_op is None:
            print(f"无法调度更多工序，当前完成工序数: {completed_ops}/{total_ops}")
            raise ValueError("无法找到可执行的工序，可能存在死锁或数据错误")
        job_idx, op_idx, machine, duration = next_op
        start_time = job_times[job_idx]
        machine_available_time = machine_times[machine]
        actual_start = max(start_time, machine_available_time)
        end_time = actual_start + duration
        proc_energy = PDk.get((job_idx, op_idx, machine), 0.0)
        processing_energy += proc_energy * duration
        machine_times[machine] = end_time
        job_times[job_idx] = end_time
        job_op_idx[job_idx] += 1
        operations.remove((job_idx, op_idx, machine, duration))
        completed_ops += 1
    makespan = np.max(machine_times)
    idle_energy = np.sum(PWk * (makespan - machine_times).clip(min=0))
    TEC = processing_energy + idle_energy
    z_makespan = (makespan - z_params['mu_m']) / z_params['sigma_m']
    z_TEC = (TEC - z_params['mu_t']) / z_params['sigma_t']
    print(f"Makespan: {makespan}, TEC: {TEC}, Z-makespan: {z_makespan}, Z-TEC: {z_TEC}")
    result = (makespan, TEC, z_makespan, z_TEC)
    schedule_cache[solution_key] = result
    return result

# 生成邻域解
def generate_neighbor(solution, job_data, num_machines):
    neighbor = [row[:] for row in solution]
    machine_times, critical_ops = find_critical_path(neighbor, job_data, num_machines)
    num_changes = random.randint(1, 3)
    for _ in range(num_changes):
        if random.random() < 0.5 and critical_ops:
            job_idx, op_idx = random.choice(critical_ops)
        else:
            job_idx = random.randint(0, len(solution) - 1)
            op_idx = random.randint(0, len(solution[job_idx]) - 1)
        current_machine, _ = neighbor[job_idx][op_idx]
        available_options = job_data[job_idx][op_idx]
        valid_options = [opt for opt in available_options if opt[0] != current_machine]
        if valid_options:
            machine_load = [machine_times[m] for m in range(num_machines)]
            best_option = min(valid_options, key=lambda x: machine_load[x[0]] + x[1])
            neighbor[job_idx][op_idx] = best_option
    return neighbor

# 扰动机制
def perturb_solution(solution, job_data, num_machines, iteration, max_iterations):
    perturbed = [row[:] for row in solution]
    perturb_scale = int(5 + 5 * (1 - iteration / max_iterations))
    machine_times, _ = find_critical_path(perturbed, job_data, num_machines)
    machine_loads = sorted(enumerate(machine_times), key=lambda x: x[1], reverse=True)
    high_load_machines = [m[0] for m in machine_loads[:num_machines // 2]]
    for _ in range(min(perturb_scale, len(solution))):
        job_idx = random.randint(0, len(solution) - 1)
        op_idx = random.randint(0, len(solution[job_idx]) - 1)
        if perturbed[job_idx][op_idx][0] in high_load_machines or random.random() < 0.3:
            available_options = job_data[job_idx][op_idx]
            new_machine, new_duration = random.choice(available_options)
            perturbed[job_idx][op_idx] = (new_machine, new_duration)
    return perturbed

# 动态参数调整
def get_dynamic_params(instance_name, num_jobs, num_machines):
    scale = num_jobs * num_machines
    if scale > 100:
        return 100, 2000
    return 50, 1000

# 并行评估函数
def evaluate_neighbor(args):
    solution, num_jobs, num_machines, PDk, PWk, z_params = args
    return evaluate_schedule_makespan(solution, num_jobs, num_machines, PDk, PWk, z_params)

# 禁忌搜索
def tabu_search(job_data, num_jobs, num_machines, PDk, PWk, z_params, instance_name):
    start_time = time.time()
    current_solution = initialize_solution(job_data, num_machines)
    best_solution = copy.deepcopy(current_solution)
    best_makespan, best_TEC, best_z_makespan, best_z_TEC = evaluate_schedule_makespan(
        best_solution, num_jobs, num_machines, PDk, PWk, z_params
    )
    best_objective = 0.25 * best_z_makespan + 0.75 * best_z_TEC
    tabu_list = []
    tabu_tenure, MAX_NO_IMPROVE = get_dynamic_params(instance_name, num_jobs, num_machines)
    no_improve_count = 0
    for iteration in range(MAX_ITERATIONS):
        if time.time() - start_time > 1800:
            print(f"实例 {instance_name} 超过1800秒，提前终止")
            break
        if no_improve_count > MAX_NO_IMPROVE // 2:
            current_solution = perturb_solution(current_solution, job_data, num_machines, iteration, MAX_ITERATIONS)
            no_improve_count = 0
            print(f"迭代 {iteration}, 执行扰动，当前最佳目标: {best_objective}")
        neighbors = [generate_neighbor(current_solution, job_data, num_machines) for _ in range(4)]
        with ThreadPoolExecutor(max_workers=4) as executor:
            results = list(
                executor.map(evaluate_neighbor, [(n, num_jobs, num_machines, PDk, PWk, z_params) for n in neighbors]))
        best_neighbor_idx = min(range(len(results)), key=lambda i: 0.25 * results[i][2] + 0.75 * results[i][3])
        neighbor_makespan, neighbor_TEC, neighbor_z_makespan, neighbor_z_TEC = results[best_neighbor_idx]
        neighbor_objective = 0.25 * neighbor_z_makespan + 0.75 * neighbor_z_TEC
        neighbor = neighbors[best_neighbor_idx]
        move = (neighbor_objective, tuple(map(tuple, neighbor)))
        if move not in tabu_list or neighbor_objective < best_objective:
            current_solution = neighbor
            current_objective = neighbor_objective
            if neighbor_objective < best_objective:
                best_solution = copy.deepcopy(neighbor)
                best_makespan = neighbor_makespan
                best_TEC = neighbor_TEC
                best_z_makespan = neighbor_z_makespan
                best_z_TEC = neighbor_z_TEC
                best_objective = neighbor_objective
                no_improve_count = 0
            else:
                no_improve_count += 1
            tabu_list.append(move)
            if len(tabu_list) > tabu_tenure:
                tabu_list.pop(0)
        if no_improve_count >= MAX_NO_IMPROVE:
            break
        if iteration % CHECK_INTERVAL == 0:
            print(f"迭代 {iteration}, 当前最佳目标: {best_objective}")
    return best_solution, best_makespan, best_TEC, best_objective

# 主函数
def main(docx_path):
    start_program = time.time()
    data_lines = read_docx(docx_path)
    instances = parse_fjssp_data(data_lines)
    results = {}

    for instance in instances:
        instance_name = instance['name']
        num_jobs = instance['num_jobs']
        num_machines = instance['num_machines']
        job_data = instance['jobs']
        max_ops = max(len(job) for job in job_data)

        # 动态加载对应实例的能耗 Excel 文件，并使用绝对路径
        energy_excel_path = f'D:\\pycharm community\\machine learning test\\FJSP\\energy_data\\energy_data_{instance_name}.xlsx'

        try:
            PDk, PWk = load_energy_data(energy_excel_path, instance_name, num_jobs, max_ops, num_machines)
        except Exception as e:
            print(f"跳过实例 {instance_name}，因为加载能耗数据失败: {e}")
            continue

        z_params = Z_SCORE_PARAMS.get(instance_name)
        if z_params is None:
            print(f"错误：未找到实例 {instance_name} 的 Z-score 参数。")
            continue

        print(f"\n处理 {instance_name}...")
        print(f"作业数: {num_jobs}, 机器数: {num_machines}")
        start_time = time.time()

        best_solution, best_makespan, best_TEC, best_objective = tabu_search(
            job_data, num_jobs, num_machines, PDk, PWk, z_params, instance_name
        )

        end_time = time.time()
        results[instance_name] = {
            "best_solution": best_solution,
            "best_makespan": best_makespan,
            "best_TEC": best_TEC,
            "best_objective": best_objective,
            "runtime": end_time - start_time
        }

    print("\n=== 所有实例结果 ===")
    for instance_name, result in results.items():
        print(f"\n实例: {instance_name}")
        print(f"最佳 makespan: {result['best_makespan']}")
        print(f"最佳 TEC: {result['best_TEC']}")
        print(f"最佳目标函数值 (Z-score 后): {result['best_objective']}")
        print(f"运行时间: {result['runtime']:.2f} 秒")

    total_time = time.time() - start_program
    print(f"\n总程序运行时间: {total_time:.2f} 秒")
    return results


if __name__ == "__main__":
    # 更新为你的实际文件路径
    docx_path = "D:\\pycharm community\\machine learning test\\FJSP\\车间数据.docx"
    main(docx_path)