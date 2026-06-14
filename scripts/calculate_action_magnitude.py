import os
import glob
import h5py
import numpy as np

DATA_DIR = "/home/dhruv/Trajectory_Augmentation/data/LIBERO-datasets"
SUITES = ["libero_goal", "libero_object", "libero_spatial", "libero_10"]

total_demos = 0
all_magnitudes = []
gripper_closed_violations = 0

print("Analyzing first 30 timesteps of 5 demos per task...")

for suite in SUITES:
    suite_dir = os.path.join(DATA_DIR, suite)
    if not os.path.exists(suite_dir):
        print(f"Skipping {suite}, directory not found.")
        continue
        
    hdf5_files = glob.glob(os.path.join(suite_dir, "*.hdf5"))
    for hdf5_file in hdf5_files:
        task_name = os.path.basename(hdf5_file)
        
        with h5py.File(hdf5_file, 'r') as f:
            demo_keys = list(f['data'].keys())
            
            # Select first 5 demos
            for i in range(min(5, len(demo_keys))):
                demo_key = demo_keys[i]
                actions = f['data'][demo_key]['actions'][:]
                
                # We only look at the first 30 steps
                # Some demos might be shorter than 30 steps, so we take min
                num_steps = min(30, len(actions))
                actions_30 = actions[:num_steps]
                
                # Check gripper (index 6). 
                # Usually -1 is open, 1 is closed. We verify it is not closed (> 0)
                gripper_actions = actions_30[:, 6]
                if np.any(gripper_actions > 0):
                    print(f"VIOLATION: {suite}/{task_name} {demo_key} has gripper closed in first 30 steps! Max val: {np.max(gripper_actions)}")
                    gripper_closed_violations += 1
                
                # Calculate L2 norm of the first 6 elements (pos + ori)
                pos_ori_actions = actions_30[:, :6]
                magnitudes = np.linalg.norm(pos_ori_actions, axis=1)
                
                all_magnitudes.extend(magnitudes)
                total_demos += 1

print("\n--- RESULTS ---")
print(f"Total demos analyzed: {total_demos}")
print(f"Total steps analyzed: {len(all_magnitudes)}")
print(f"Gripper closed violations (first 30 steps): {gripper_closed_violations}")

if len(all_magnitudes) > 0:
    avg_mag = np.mean(all_magnitudes)
    std_mag = np.std(all_magnitudes)
    print(f"Average action magnitude (L2 norm of pos+ori): {avg_mag:.6f}")
    print(f"Std dev of action magnitude: {std_mag:.6f}")
    
    # Save the average magnitude to a file so other scripts can use it
    with open("average_action_magnitude.txt", "w") as f:
        f.write(str(avg_mag))
