import random
import copy
import time
import re
import pandas as pd
import numpy as np
from docx import Document
from concurrent.futures import ThreadPoolExecutor

# 全局超参数定义
CHECK_INTERVAL = 10
MAX_ITERATIONS = 10000
NEIGHBORHOOD_SIZE = 10  # 增大邻域搜索规模，以配合动作禁忌机制

# 预计算的 Z-score 静态标准化参数
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

schedule_cache = {}

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

def load_energy_data(excel_path, instance_name, num_jobs, max_ops, num_machines):
    try:
        df = pd.read_excel(excel_path, engine='openpyxl')
    except FileNotFoundError:
        print(f"数据异常：缺失能耗文件 {excel_path}")
        raise
        
    process_energy = {}
    idle_energy = np.zeros(num_machines, dtype=float)
    
    for _, row in df.iterrows():
        job_id = int(row['Job_ID']) - 1
        op_id = int(row['Operation_ID']) - 1
        machine_id = int(row['Machine_ID1']) - 1
        proc_energy = float(row['Processing_Energy']) if not pd.isna(row['Processing_Energy']) else 0.0
        if 0 <= job_id < num_jobs and 0 <= op_id < max_ops and 0 <= machine_id < num_machines:
            process_energy[(job_id, op_id, machine_id)] = proc_energy
            
    idle_data = df[['Machine_ID2', 'Idle_Energy']].dropna().drop_duplicates().set_index('Machine_ID2')
    for machine_id in range(num_machines):
        try:
            idle_energy[machine_id] = float(idle_data.loc[machine_id + 1, 'Idle_Energy'])
        except KeyError:
            idle_energy[machine_id] = 0.5
            
    return process_energy, idle_energy

def initialize_solution(job_data, num_machines):
    solution = [[] for _ in range(len(job_data))]
    machine_load = [0] * num_machines
    job_indices = sorted(
        range(len(job_data)), 
        key=lambda i: sum(min(op, key=lambda x: x[1])[1] for op in job_data[i]), 
        reverse=True
    )
    for job_idx in job_indices:
        job = job_data[job_idx]
        for op in job:
            best_machine, best_duration = min(op, key=lambda x: machine_load[x[0]] + x[1])
            solution[job_idx].append((best_machine, best_duration))
            machine_load[best_machine] += best_duration
    return solution

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

def evaluate_schedule(solution, num_jobs, num_machines, process_energy, idle_energy, z_params):
    solution_key = tuple(map(tuple, solution))
    if solution_key in schedule_cache:
        return schedule_cache[solution_key]
        
    machine_times = np.zeros(num_machines)
    job_op_idx = np.zeros(num_jobs, dtype=int)
    job_times = np.zeros(num_jobs)
    total_processing_energy = 0.0
    
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
            raise ValueError("调度死锁：无法找到拓扑排序中的下一个可执行工序。")
            
        job_idx, op_idx, machine, duration = next_op
        actual_start = max(job_times[job_idx], machine_times[machine])
        end_time = actual_start + duration
        
        proc_energy = process_energy.get((job_idx, op_idx, machine), 0.0)
        total_processing_energy += proc_energy * duration
        
        machine_times[machine] = end_time
        job_times[job_idx] = end_time
        job_op_idx[job_idx] += 1
        
        operations.remove((job_idx, op_idx, machine, duration))
        completed_ops += 1
        
    makespan = np.max(machine_times)
    total_idle_energy = np.sum(idle_energy * (makespan - machine_times).clip(min=0))
    TEC = total_processing_energy + total_idle_energy
    
    z_makespan = (makespan - z_params['mu_m']) / z_params['sigma_m']
    z_TEC = (TEC - z_params['mu_t']) / z_params['sigma_t']
    
    result = (makespan, TEC, z_makespan, z_TEC)
    schedule_cache[solution_key] = result
    return result

def generate_neighborhood(solution, job_data, num_machines):
    """
    邻域生成：执行动作，并显式记录该动作以及它会触发的禁忌属性。
    返回：新解, 应用的动作列表, 需加入禁忌表的属性列表
    """
    neighbor = [row[:] for row in solution]
    machine_times, critical_ops = find_critical_path(neighbor, job_data, num_machines)
    
    num_changes = random.randint(1, 3)
    applied_moves = []
    tabu_attributes = []
    
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
            new_machine = best_option[0]
            
            # 记录本次操作：把 (job_idx, op_idx) 分配到了 new_machine
            applied_moves.append((job_idx, op_idx, new_machine))
            # 记录禁忌红线：未来一段时间内，不准把 (job_idx, op_idx) 移回 current_machine
            tabu_attributes.append((job_idx, op_idx, current_machine))
            
    return neighbor, applied_moves, tabu_attributes

def adaptive_perturbation(solution, job_data, num_machines, iteration, max_iterations):
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

def get_dynamic_params(num_jobs, num_machines):
    scale = num_jobs * num_machines
    if scale > 100:
        return 100, 2000
    return 50, 1000

def evaluate_neighbor_wrapper(args):
    neighbor, applied_moves, tabu_attrs, num_jobs, num_machines, process_energy, idle_energy, z_params = args
    makespan, TEC, z_makespan, z_TEC = evaluate_schedule(neighbor, num_jobs, num_machines, process_energy, idle_energy, z_params)
    return neighbor, applied_moves, tabu_attrs, makespan, TEC, z_makespan, z_TEC

def tabu_search_solver(job_data, num_jobs, num_machines, process_energy, idle_energy, z_params, instance_name):
    start_time = time.time()
    current_solution = initialize_solution(job_data, num_machines)
    best_solution = copy.deepcopy(current_solution)
    
    best_makespan, best_TEC, best_z_makespan, best_z_TEC = evaluate_schedule(
        best_solution, num_jobs, num_machines, process_energy, idle_energy, z_params
    )
    best_objective = 0.25 * best_z_makespan + 0.75 * best_z_TEC
    
    tabu_list = []  # 此时禁忌表存储的是动作属性：(job_idx, op_idx, machine_idx)
    tabu_tenure, MAX_NO_IMPROVE = get_dynamic_params(num_jobs, num_machines)
    no_improve_count = 0
    
    for iteration in range(MAX_ITERATIONS):
        if time.time() - start_time > 1800:
            print(f"[{instance_name}] 达到计算时间上限 (1800s)，提前终止搜索。")
            break
            
        if no_improve_count > MAX_NO_IMPROVE // 2:
            current_solution = adaptive_perturbation(current_solution, job_data, num_machines, iteration, MAX_ITERATIONS)
            no_improve_count = 0
            
        # 并行生成并评估邻域解集
        neighbors_data = [generate_neighborhood(current_solution, job_data, num_machines) for _ in range(NEIGHBORHOOD_SIZE)]
        with ThreadPoolExecutor(max_workers=4) as executor:
            eval_args = [(n[0], n[1], n[2], num_jobs, num_machines, process_energy, idle_energy, z_params) for n in neighbors_data]
            results = list(executor.map(evaluate_neighbor_wrapper, eval_args))
            
        # 解析候选解并按目标函数值从小到大排序
        candidates = []
        for res in results:
            neighbor, applied_moves, tabu_attrs, makespan, TEC, z_m, z_T = res
            obj = 0.25 * z_m + 0.75 * z_T
            candidates.append({
                'solution': neighbor, 'applied_moves': applied_moves, 'tabu_attrs': tabu_attrs,
                'makespan': makespan, 'TEC': TEC, 'obj': obj
            })
        candidates.sort(key=lambda x: x['obj'])
        
        # 遍历排序后的候选解，结合禁忌表与特赦准则进行选择
        accepted_candidate = None
        for candidate in candidates:
            # 检查当前动作是否触碰了禁忌红线
            is_tabu = any(move in tabu_list for move in candidate['applied_moves'])
            
            # 【核心逻辑生效处】特赦准则：未被禁忌，或者被禁忌但打破了历史记录
            if not is_tabu or candidate['obj'] < best_objective:
                accepted_candidate = candidate
                # 如果触发了特赦，可以打印出来供观察
                # if is_tabu and candidate['obj'] < best_objective:
                #     print(f"迭代 {iteration}: 触发特赦准则！破局目标值: {candidate['obj']:.4f}")
                break
                
        # 兜底逻辑：如果所有解都被禁忌且无一满足特赦（极其罕见），强制接受最好的那个
        if accepted_candidate is None:
            accepted_candidate = candidates[0]
            
        # 状态流转
        current_solution = accepted_candidate['solution']
        current_objective = accepted_candidate['obj']
        
        # 更新全局最优
        if current_objective < best_objective:
            best_solution = copy.deepcopy(current_solution)
            best_makespan = accepted_candidate['makespan']
            best_TEC = accepted_candidate['TEC']
            best_objective = current_objective
            no_improve_count = 0
        else:
            no_improve_count += 1
            
        # 更新禁忌表：将当前解附带的禁忌属性加入列表
        for attr in accepted_candidate['tabu_attrs']:
            tabu_list.append(attr)
        while len(tabu_list) > tabu_tenure:
            tabu_list.pop(0)  # FIFO淘汰最早进入的禁忌属性
            
        if no_improve_count >= MAX_NO_IMPROVE:
            print(f"[{instance_name}] 连续 {MAX_NO_IMPROVE} 步无改进，触发早停机制。")
            break
            
        if iteration % CHECK_INTERVAL == 0:
            print(f"[{instance_name}] Iteration {iteration} | Best Obj: {best_objective:.4f} | No Improve: {no_improve_count}")
            
    return best_solution, best_makespan, best_TEC, best_objective

def main(docx_path):
    start_program = time.time()
    data_lines = read_docx(docx_path)
    instances = parse_fjssp_data(data_lines)
    results_summary = {}

    for instance in instances:
        instance_name = instance['name']
        num_jobs = instance['num_jobs']
        num_machines = instance['num_machines']
        job_data = instance['jobs']
        max_ops = max(len(job) for job in job_data)

        energy_excel_path = f'./energy_data/energy_data_{instance_name}.xlsx'

        try:
            process_energy, idle_energy = load_energy_data(energy_excel_path, instance_name, num_jobs, max_ops, num_machines)
        except Exception as e:
            continue

        z_params = Z_SCORE_PARAMS.get(instance_name)
        if z_params is None:
            continue

        print(f"\n================ 启动求解实例: {instance_name} ================")
        print(f"参数规模 - 作业数: {num_jobs}, 机器数: {num_machines}")
        
        start_time = time.time()
        best_solution, best_makespan, best_TEC, best_objective = tabu_search_solver(
            job_data, num_jobs, num_machines, process_energy, idle_energy, z_params, instance_name
        )
        end_time = time.time()
        
        results_summary[instance_name] = {
            "Makespan": best_makespan,
            "TEC": best_TEC,
            "Objective": best_objective,
            "Runtime_sec": end_time - start_time
        }

    print("\n================ 全局寻优结果汇总 ================")
    summary_df = pd.DataFrame.from_dict(results_summary, orient='index')
    print(summary_df.to_string())

    total_time = time.time() - start_program
    print(f"\n总计运行时长: {total_time:.2f} 秒")

if __name__ == "__main__":
    main("车间数据.docx")
