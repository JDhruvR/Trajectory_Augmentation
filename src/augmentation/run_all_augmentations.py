import os
import subprocess
import time

SUITES = [
    "libero_object",
    "libero_10"
]

# # to be done later:
#     "libero_object",
#     "libero_10"

# done
# "libero_spatial",
#     "libero_goal"


BASE_TARGET_DIR = "/home/dhruv/Trajectory_Augmentation/data/LIBERO-datasets"
BASE_OUTPUT_DIR = "/home/dhruv/Trajectory_Augmentation/data/LIBERO-datasets-augmented"

def main():
    print("Starting massive parallel dataset augmentation for all LIBERO suites...")
    start_time = time.time()
    
    for suite in SUITES:
        target_dir = os.path.join(BASE_TARGET_DIR, suite)
        output_dir = os.path.join(BASE_OUTPUT_DIR, suite)
        
        if not os.path.exists(target_dir):
            print(f"Skipping {suite}, directory not found: {target_dir}")
            continue
            
        print(f"\n{'='*50}")
        print(f"Launching augmentation pipeline for: {suite.upper()}")
        print(f"{'='*50}")
        
        # We run the command synchronously. Since generate_augmented_dataset.py uses 
        # multiprocessing internally to max out the CPU for each suite, we process one suite at a time.
        cmd = [
            "/home/dhruv/miniforge3/envs/qwen-vla/bin/python",
            "-u",
            "src/augmentation/generate_augmented_dataset.py",
            "--target_dir", target_dir,
            "--output_dir", output_dir,
            "--num_augmentations", "3"
        ]
        
        # Pass EGL environment variables for headless rendering
        env = os.environ.copy()
        env["MUJOCO_GL"] = "egl"
        env["EGL_DEVICE_ID"] = "0"
        
        try:
            subprocess.run(cmd, env=env, check=True)
            print(f"Successfully finished augmenting {suite}.")
        except subprocess.CalledProcessError as e:
            print(f"Error occurred while processing {suite}. Exited with code {e.returncode}")
            
    total_time = time.time() - start_time
    print(f"\nAll suites completed! Total time: {total_time / 3600:.2f} hours.")

if __name__ == "__main__":
    main()
