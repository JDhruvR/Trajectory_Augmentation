import subprocess
import json
import re
import os
import sys

# Configuration
N_EPISODES = 10  # Publishing-level config
BATCH_SIZE = 10  # Max parallel physics environments per task

SUITES = {
    "libero_spatial_env": 10,
    "libero_object_env": 10,
    "libero_goal_env": 10,
    "libero_10_env": 10
}

MODELS = {
    "Baseline": "HuggingFaceVLA/smolvla_libero",
    "Checkpt-20k": "outputs/train/smolvla_libero_all_aug/checkpoints/020000/pretrained_model"
}

def run_evaluation(model_name, pretrained_path, suite_name, task_id, n_episodes):
    """Runs lerobot_eval.py for a SINGLE task and returns the success rate."""
    output_dir = f"outputs/eval/pro_{model_name}_{suite_name}_task{task_id}"
    
    # We must format the absolute path if it's a local checkpoint
    if pretrained_path.startswith("outputs/"):
        pretrained_path = os.path.abspath(pretrained_path)

    cmd = [
        "/home/dhruv/miniforge3/envs/qwen-vla/bin/python", "-m", "lerobot.scripts.lerobot_eval",
        f"--policy.path={pretrained_path}",
        "--policy.expert_width_multiplier=0.5",
        "--env.type=libero",
        f"--env.task={suite_name}",
        f"--env.task_ids=[{task_id}]",
        f"--eval.n_episodes={n_episodes}",
        f"--eval.batch_size={min(n_episodes, BATCH_SIZE)}",
        "--eval.use_async_envs=false",
        f"--output_dir={output_dir}"
    ]
    
    print(f"  [Running {suite_name} Task {task_id}] {' '.join(cmd)}", flush=True)
    
    # Pipe stdout to a temp file
    log_file = f"{output_dir}_temp.log"
    os.system(f"mkdir -p outputs/eval")
    
    env = os.environ.copy()
    
    with open(log_file, "w") as f:
        result = subprocess.run(cmd, stdout=f, stderr=subprocess.STDOUT, text=True, env=env)
        
    with open(log_file, "r") as f:
        log_content = f.read()
    
    if result.returncode != 0:
        print(f"  [Error] Evaluation failed for {suite_name} Task {task_id}", flush=True)
        print(log_content[-1000:], flush=True) # Print last 1000 chars of error
        return 0.0

    matches = re.findall(r"'success_rate':\s*([0-9.]+)", log_content)
    if matches:
        return float(matches[-1])
        
    return 0.0

def main():
    print(f"--- LIBERO-PRO GPU-Safe Benchmarking Script ({N_EPISODES} episodes/task) ---")
    
    results = {model: {} for model in MODELS}
    
    for model_name, path in MODELS.items():
        print(f"\nEvaluating Model: {model_name}")
        total_success = 0.0
        
        for suite, num_tasks in SUITES.items():
            print(f"-> Testing Suite [{suite}]")
            suite_task_scores = []
            
            # Evaluate each task sequentially to avoid GPU OOM
            for task_id in range(num_tasks):
                score = run_evaluation(model_name, path, suite, task_id, n_episodes=N_EPISODES)
                suite_task_scores.append(score)
                print(f"     Task {task_id} Success: {score}%")
            
            suite_avg = sum(suite_task_scores) / len(suite_task_scores)
            results[model_name][suite] = suite_avg
            total_success += suite_avg
            print(f"   => Average Success Rate for [{suite}]: {suite_avg:.1f}%")
        
        results[model_name]["Total"] = total_success / len(SUITES)
    
    # Print the final Markdown Table
    print("\n\n LIBERO-PRO Benchmark Results")
    header = "| Model | " + " | ".join([s.replace("libero_", "").replace("_env", "").capitalize() for s in SUITES.keys()]) + " | Total |"
    print(header)
    print("|---" * (len(SUITES) + 2) + "|")
    
    for model_name, scores in results.items():
        row = f"| {model_name} | "
        for suite in SUITES.keys():
            row += f"{scores.get(suite, 0.0):.1f} | "
        row += f"{scores.get('Total', 0.0):.1f} |"
        print(row)
        
    print("\nBenchmarking complete!")

if __name__ == "__main__":
    main()
