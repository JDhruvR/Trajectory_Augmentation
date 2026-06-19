import subprocess
import time
import re
import os

def get_gpu_memory():
    """Runs nvidia-smi and returns the GPU memory usage in MB."""
    try:
        result = subprocess.run(
            ['nvidia-smi', '--query-gpu=memory.used', '--format=csv,nounits,noheader'],
            capture_output=True, text=True
        )
        return int(result.stdout.strip())
    except:
        return 0

def test_batch_size(batch_size):
    print(f"\nTesting Batch Size: {batch_size}")
    
    cmd = [
        "/home/dhruv/miniforge3/envs/qwen-vla/bin/python", "-m", "lerobot.scripts.lerobot_eval",
        "--policy.path=HuggingFaceVLA/smolvla_libero",
        "--policy.expert_width_multiplier=0.5",
        "--env.type=libero",
        "--env.task=libero_10_env",
        "--env.task_ids=[0]",
        f"--eval.n_episodes={batch_size}",
        f"--eval.batch_size={batch_size}",
        "--eval.use_async_envs=false",
        f"--output_dir=outputs/eval/smoke_test_batch_{batch_size}"
    ]
    
    # Start the process
    process = subprocess.Popen(
        cmd, 
        stdout=subprocess.DEVNULL, 
        stderr=subprocess.DEVNULL,
        env=os.environ.copy()
    )
    
    max_mem = 0
    start_time = time.time()
    
    # Monitor memory while process is running (up to 30 seconds to get peak mem of env building)
    while process.poll() is None:
        mem = get_gpu_memory()
        if mem > max_mem:
            max_mem = mem
            
        time.sleep(1.0)
        
        # Kill it after 30 seconds, we just want to see peak memory of environment loading and first steps
        if time.time() - start_time > 45:
            process.kill()
            break
            
    print(f"Max GPU Memory Used: {max_mem} MB")
    
if __name__ == "__main__":
    for bs in [1, 5, 10, 20]:
        test_batch_size(bs)
