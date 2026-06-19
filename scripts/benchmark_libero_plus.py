import subprocess
import json
import re
import os
import sys

# Configuration
N_EPISODES = 20  # Publishing-level (set to 1 for smoke test)
MAX_TASKS_PER_AXIS = 10  # Maximum tasks to evaluate per axis

# The 4 valid axes and their corresponding identifying string inside the LIBERO-plus task names
AXES_MAPPING = {
    'Camera': '_view_0_0',  # Specifically _view_ with 0_0 to avoid initstate overlap
    'Robot': '_initstate_',
    'Language': '_language_',
    'Noise': '_noise_'
}

# Add third_party path so we can import libero
libero_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'third_party', 'LIBERO-plus'))
sys.path.append(libero_path)

try:
    from libero.libero.benchmark.libero_suite_task_map import libero_task_map
except ImportError as e:
    print(f"Error importing libero_task_map: {e}")
    print("Please ensure LIBERO-plus is properly installed in third_party/LIBERO-plus")
    sys.exit(1)

def run_evaluation(model_name, pretrained_path, suite_name, task_id, task_string_name, n_episodes=20):
    """Runs lerobot_eval.py and returns the success rate."""
    output_dir = f"outputs/eval/full_{model_name}_{task_string_name}"
    cmd = [
        "/home/dhruv/miniforge3/envs/qwen-vla/bin/python", "-m", "lerobot.scripts.lerobot_eval",
        f"--policy.path={pretrained_path}",
        "--policy.expert_width_multiplier=0.5",
        "--env.type=libero",
        f"--env.task={suite_name}",
        f"--env.task_ids=[{task_id}]",
        f"--eval.n_episodes={n_episodes}",
        f"--eval.batch_size={n_episodes}",
        "--eval.use_async_envs=false",
        f"--output_dir={output_dir}"
    ]
    
    print(f"  [Running] {' '.join(cmd)}", flush=True)
    # Pipe stdout to a temp file
    log_file = f"{output_dir}_temp.log"
    os.system(f"mkdir -p outputs/eval")
    
    env = os.environ.copy()
    env["PYTHONPATH"] = libero_path + ":" + env.get("PYTHONPATH", "")
    
    with open(log_file, "w") as f:
        result = subprocess.run(cmd, stdout=f, stderr=subprocess.STDOUT, text=True, env=env)
        
    with open(log_file, "r") as f:
        log_content = f.read()
    
    if result.returncode != 0:
        print(f"  [Error] Evaluation failed for {task_string_name}", flush=True)
        print(log_content[-1000:], flush=True) # Print last 1000 chars of error
        return 0.0

    # Parse success rate from the console output
    match = re.search(r"'success_rate':\s*([0-9.]+)", log_content)
    if match:
        return float(match.group(1))
        
    return 0.0

def get_suite_and_id(task_string):
    """Returns (suite_name, task_id_integer) for a given task string."""
    for suite, tasks in libero_task_map.items():
        if task_string in tasks:
            return suite, tasks.index(task_string)
    raise ValueError(f"Task {task_string} not found in libero_task_map!")

def main():
    print(f"--- LIBERO-plus Benchmarking Script ({N_EPISODES} episodes/task) ---")
    
    # 1. Gather up to MAX_TASKS_PER_AXIS tasks for each axis
    tasks_list = libero_task_map['libero_spatial']
    selected_tasks = {axis: [] for axis in AXES_MAPPING}
    
    for axis, keyword in AXES_MAPPING.items():
        matches = [t for t in tasks_list if keyword in t]
        if matches:
            selected_tasks[axis] = matches[:MAX_TASKS_PER_AXIS]
        else:
            # Fallback across all suites
            for suite, s_tasks in libero_task_map.items():
                suite_matches = [t for t in s_tasks if keyword in t]
                if suite_matches:
                    selected_tasks[axis] = suite_matches[:MAX_TASKS_PER_AXIS]
                    break
            
            # If still nothing, fallback to first generic task (safety net)
            if not selected_tasks[axis]:
                selected_tasks[axis] = [tasks_list[0]]
                
        print(f"Axis [{axis}] has {len(selected_tasks[axis])} tasks scheduled.")
    
    models = {
        "Baseline": "HuggingFaceVLA/smolvla_libero"
    }
    
    results = {model: {} for model in models}
    
    # 2. Run evaluations
    for model_name, path in models.items():
        print(f"\nEvaluating Model: {model_name}")
        total_success = 0.0
        
        for axis, tasks in selected_tasks.items():
            print(f"-> Testing Axis [{axis}]")
            axis_success_rates = []
            
            for task in tasks:
                suite_name, task_id = get_suite_and_id(task)
                success_rate = run_evaluation(model_name, path, suite_name, task_id, task, n_episodes=N_EPISODES)
                axis_success_rates.append(success_rate)
                print(f"     Task: {task} | Success: {success_rate}%")
            
            # Average success rate for the axis
            avg_axis_success = sum(axis_success_rates) / len(axis_success_rates)
            results[model_name][axis] = avg_axis_success
            total_success += avg_axis_success
            print(f"   => Average Success Rate for [{axis}]: {avg_axis_success:.1f}%")
        
        results[model_name]["Total"] = total_success / len(selected_tasks)
    
    # 3. Print the final Markdown Table
    print("\n\n LIBERO-Plus Benchmark Results")
    print("| Model | Camera | Robot | Language | Noise | Total |")
    print("|---|---|---|---|---|---|")
    for model_name, scores in results.items():
        row = f"| {model_name} | {scores.get('Camera', 0.0):.1f} | {scores.get('Robot', 0.0):.1f} | {scores.get('Language', 0.0):.1f} | {scores.get('Noise', 0.0):.1f} | {scores.get('Total', 0.0):.1f} |"
        print(row)
        
    print("\nBenchmarking complete!")

if __name__ == "__main__":
    main()
