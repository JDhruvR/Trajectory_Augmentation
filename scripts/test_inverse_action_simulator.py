#!/usr/bin/env python3
"""
Simulator-Based Inverse Action Test

Tests trajectory reversibility using MuJoCo physics simulation (robosuite).
Applies forward action, then inverse action, and measures error in state recovery.

Usage:
    python test_inverse_action_simulator.py [--demo_idx 0] [--num_steps 50]

Output:
    - libero_inverse_action_sim_eef8.txt: Per-step results
    - libero_inverse_action_sim_eef8_stats.txt: Statistics summary
    - libero_inverse_action_sim_eef8_diff_hist.png: Error histogram
    - libero_inverse_action_sim_recovery.png: Recovery visualization
"""

import numpy as np
import h5py
from pathlib import Path
import matplotlib.pyplot as plt
import argparse
from typing import Dict, Tuple

import robosuite as suite
from robosuite.utils.transform_utils import quat2axisangle, axisangle2quat


# ============================================================================
# CONFIGURATION
# ============================================================================

LIBERO_REPO = Path("/home/dhruv/Trajectory_Augmentation/third_party/LIBERO")
DEMO_PATH = Path("/home/dhruv/Trajectory_Augmentation/data/LIBERO-datasets/libero_goal/open_the_middle_drawer_of_the_cabinet_demo.hdf5")
OUTPUT_DIR = Path("/home/dhruv/Trajectory_Augmentation/report_mds")

OUTPUT_DIR.mkdir(exist_ok=True, parents=True)

print(f"LIBERO repo: {LIBERO_REPO}")
print(f"Demo file: {DEMO_PATH}")
print(f"Output dir: {OUTPUT_DIR}")


# ============================================================================
# HELPER FUNCTIONS
# ============================================================================

def rot_mat_to_quat(rot_mat: np.ndarray) -> np.ndarray:
    """Convert 3x3 rotation matrix to quaternion [x, y, z, w]"""
    trace = rot_mat.trace()
    if trace > 0:
        S = np.sqrt(trace + 1.0) * 2
        w = 0.25 * S
        x = (rot_mat[2, 1] - rot_mat[1, 2]) / S
        y = (rot_mat[0, 2] - rot_mat[2, 0]) / S
        z = (rot_mat[1, 0] - rot_mat[0, 1]) / S
    elif (rot_mat[0, 0] > rot_mat[1, 1]) and (rot_mat[0, 0] > rot_mat[2, 2]):
        S = np.sqrt(1.0 + rot_mat[0, 0] - rot_mat[1, 1] - rot_mat[2, 2]) * 2
        w = (rot_mat[2, 1] - rot_mat[1, 2]) / S
        x = 0.25 * S
        y = (rot_mat[0, 1] + rot_mat[1, 0]) / S
        z = (rot_mat[0, 2] + rot_mat[2, 0]) / S
    elif rot_mat[1, 1] > rot_mat[2, 2]:
        S = np.sqrt(1.0 + rot_mat[1, 1] - rot_mat[0, 0] - rot_mat[2, 2]) * 2
        w = (rot_mat[0, 2] - rot_mat[2, 0]) / S
        x = (rot_mat[0, 1] + rot_mat[1, 0]) / S
        y = 0.25 * S
        z = (rot_mat[1, 2] + rot_mat[2, 1]) / S
    else:
        S = np.sqrt(1.0 + rot_mat[2, 2] - rot_mat[0, 0] - rot_mat[1, 1]) * 2
        w = (rot_mat[1, 0] - rot_mat[0, 1]) / S
        x = (rot_mat[0, 2] + rot_mat[2, 0]) / S
        y = (rot_mat[1, 2] + rot_mat[2, 1]) / S
        z = 0.25 * S
    
    return np.array([x, y, z, w])


def extract_eef_state(env) -> np.ndarray:
    """
    Extract 8D EEF state from robosuite environment
    
    Returns:
        eef_state (8,): [pos_x, pos_y, pos_z, ori_x, ori_y, ori_z, grip_l, grip_r]
    """
    robot = env.robots[0]
    sim = env.sim
    
    # End-effector position (3D) - from MuJoCo site xpos
    eef_site_id = robot.eef_site_id
    eef_pos = sim.data.site_xpos[eef_site_id]  # (3,)
    
    # End-effector orientation - convert rotation matrix to axis-angle
    eef_rot_mat = sim.data.site_xmat[eef_site_id].reshape((3, 3))  # 9-element vec -> 3x3 mat
    eef_quat = rot_mat_to_quat(eef_rot_mat)
    eef_ori = quat2axisangle(eef_quat)  # (3,)
    
    # Gripper state (2D) - get from joint positions
    gripper_pos = []
    for joint_name in ['gripper0_finger_joint1', 'gripper0_finger_joint2']:
        joint_id = None
        for i, name in enumerate(sim.model.joint_names):
            if name == joint_name:
                joint_id = i
                break
        if joint_id is not None:
            joint_qpos_idx = sim.model.jnt_qposadr[joint_id]
            gripper_pos.append(sim.data.qpos[joint_qpos_idx])
    
    if len(gripper_pos) < 2:
        gripper_pos = [0.0, 0.0]
    
    gripper_state = np.array(gripper_pos[:2], dtype=np.float64)  # (2,)
    
    # Concatenate
    eef_state = np.concatenate([eef_pos, eef_ori, gripper_state])
    
    return eef_state  # (8,)


def reset_to_state(env, target_state: np.ndarray) -> None:
    """
    Reset environment to a specific MuJoCo state (usually 79D or 110D)
    
    Args:
        env: Robosuite environment
        target_state: Full MuJoCo state
    """
    sim = env.sim
    
    # Use the robust flattened state setter which handles time/qpos/qvel properly
    sim.set_state_from_flattened(target_state)
    sim.forward()


def load_hdf5_trajectory(filepath: str, demo_idx: int = 0) -> Dict:
    """
    Load trajectory from HDF5 demo file
    
    Args:
        filepath: Path to HDF5 file
        demo_idx: Demo index to load
    
    Returns:
        dict: Contains states_110d, actions, ee_pos, ee_ori, gripper_states
    """
    with h5py.File(filepath, 'r') as f:
        demo_key = f"data/demo_{demo_idx}"
        
        states_110d = f[f"{demo_key}/states"][:]  # (T, 110)
        actions = f[f"{demo_key}/actions"][:]      # (T-1, 7)
        
        # Load observations for reference
        ee_pos = f[f"{demo_key}/obs/ee_pos"][:]        # (T, 3)
        ee_ori = f[f"{demo_key}/obs/ee_ori"][:]        # (T, 3)
        gripper_states = f[f"{demo_key}/obs/gripper_states"][:]  # (T, 2)
    
    return {
        'states_110d': states_110d,
        'actions': actions,
        'ee_pos': ee_pos,
        'ee_ori': ee_ori,
        'gripper_states': gripper_states,
    }


def create_environment(demo_path: str):
    """Create robosuite environment using metadata from the demo file"""
    import json
    import h5py
    import os
    import sys
    
    # Need to make sure LIBERO_REPO is in sys.path
    if str(LIBERO_REPO) not in sys.path:
        sys.path.insert(0, str(LIBERO_REPO))
        
    import libero.libero.envs  # registers the libero envs

    with h5py.File(demo_path, 'r') as f:
        env_args = json.loads(f['data'].attrs['env_args'])
    
    # Construct BDDL path manually
    task_name = Path(demo_path).stem.replace("_demo", "")
    dataset_type = demo_path.split("/")[-2]  # e.g., libero_goal
    bddl_path = str(LIBERO_REPO / "libero" / "libero" / "bddl_files" / dataset_type / f"{task_name}.bddl")
    
    print(f"  Using BDDL: {bddl_path}")
    
    env = suite.make(
        env_name="Libero_Tabletop_Manipulation",
        bddl_file_name=bddl_path,
        robots=env_args["env_kwargs"].get("robots", ["Panda"]),
        has_renderer=False,
        has_offscreen_renderer=False,
        use_camera_obs=False,
        control_freq=env_args["env_kwargs"].get("control_freq", 20),
        horizon=200,
    )
    
    return env


# ============================================================================
# MAIN TEST FUNCTION
# ============================================================================

def test_inverse_action_simulator(env, trajectory_data: Dict, demo_idx: int = 0,
                                   max_steps: int = 50) -> Dict:
    """
    Test inverse action reversibility using simulator
    """
    
    states_110d = trajectory_data['states_110d']
    actions = trajectory_data['actions']
    
    num_steps = min(len(states_110d) - 1, max_steps)
    
    # Storage for results
    results = {
        's1': [],
        'a1': [],
        's2': [],
        's1_reconstructed': [],
        'error': [],
        'error_pos': [],
        'error_ori': [],
        'error_grip': [],
    }
    
    print(f"\\n=== Testing Inverse Actions (Simulator) ===")
    print(f"Demo index: {demo_idx}")
    print(f"Testing {num_steps} steps...\\n")
    
    # Test each step
    for step in range(num_steps):
        try:
            # Step 1: Reset to s1 (full 110D state)
            s1_110d = states_110d[step]
            reset_to_state(env, s1_110d)
            s1_eef = extract_eef_state(env)
            
            # Step 2: Apply forward action
            a1 = actions[step]  # (7,)
            a1_padded = np.concatenate([a1[:6], [a1[6], a1[6]]])  # (8,)
            env.step(a1_padded)
            s2_eef = extract_eef_state(env)
            
            # Step 3: Apply inverse action
            inverse_action = a1.copy()
            inverse_action[:6] *= -1
            inverse_action[6] = 0  # Zero out gripper for inverse
            inverse_action_padded = np.concatenate([inverse_action[:6], [inverse_action[6], inverse_action[6]]])
            
            env.step(inverse_action_padded)
            s1_reconstructed_eef = extract_eef_state(env)
            
            # Step 4: Compute errors
            total_error = np.linalg.norm(s1_eef - s1_reconstructed_eef)
            pos_error = np.linalg.norm(s1_eef[:3] - s1_reconstructed_eef[:3])
            ori_error = np.linalg.norm(s1_eef[3:6] - s1_reconstructed_eef[3:6])
            grip_error = np.linalg.norm(s1_eef[6:8] - s1_reconstructed_eef[6:8])
            
            # Store results
            results['s1'].append(s1_eef)
            results['a1'].append(a1)
            results['s2'].append(s2_eef)
            results['s1_reconstructed'].append(s1_reconstructed_eef)
            results['error'].append(total_error)
            results['error_pos'].append(pos_error)
            results['error_ori'].append(ori_error)
            results['error_grip'].append(grip_error)
            
            if step % 10 == 0:
                print(f"  Step {step:3d}: error = {total_error:.6f} m")
        
        except Exception as e:
            print(f"  Step {step}: FAILED - {e}")
            continue
    
    # Convert lists to numpy arrays
    for key in results:
        if isinstance(results[key], list):
            results[key] = np.array(results[key])
    
    return results


# ============================================================================
# STATISTICS COMPUTATION
# ============================================================================

def compute_statistics(results: Dict) -> Dict:
    """Compute statistical summaries"""
    
    errors = results['error']
    errors_pos = results['error_pos']
    errors_ori = results['error_ori']
    errors_grip = results['error_grip']
    
    # Get initial states for normalization
    s1_states = np.array(results['s1'])
    s1_norms = np.linalg.norm(s1_states, axis=1)
    
    stats = {
        'num_steps': len(errors),
        'mean_error': np.mean(errors),
        'std_error': np.std(errors),
        'median_error': np.median(errors),
        'max_error': np.max(errors),
        'min_error': np.min(errors),
        'percentile_95': np.percentile(errors, 95),
        'percentile_99': np.percentile(errors, 99),
        'mean_s1_norm': np.mean(s1_norms),
        'std_s1_norm': np.std(s1_norms),
        'error_as_pct': (np.mean(errors) / np.mean(s1_norms)) * 100,
        'error_pos_mean': np.mean(errors_pos),
        'error_ori_mean': np.mean(errors_ori),
        'error_grip_mean': np.mean(errors_grip),
    }
    
    return stats


# ============================================================================
# RESULTS SAVING
# ============================================================================

def save_results_to_file(results: Dict, stats: Dict, output_dir: Path,
                         prefix: str = "libero_inverse_action_sim") -> None:
    """Save results to text files and plots"""
    
    output_dir = Path(output_dir)
    output_dir.mkdir(exist_ok=True, parents=True)
    
    # Per-step results
    results_file = output_dir / f"{prefix}_eef8.txt"
    with open(results_file, 'w') as f:
        f.write("Step-by-step simulator inverse action test results\\n")
        f.write("=" * 100 + "\\n\\n")
        f.write("Format: step | ‖s1‖ | ‖error‖ | ‖Δpos‖ | ‖Δori‖ | ‖Δgrip‖\\n")
        f.write("-" * 100 + "\\n")
        
        for step in range(len(results['error'])):
            s1_norm = np.linalg.norm(results['s1'][step])
            f.write(f"{step:3d} | {s1_norm:.6f} | {results['error'][step]:.6f} | "
                   f"{results['error_pos'][step]:.6f} | {results['error_ori'][step]:.6f} | "
                   f"{results['error_grip'][step]:.6f}\\n")
    
    print(f"✓ Saved per-step results to: {results_file}")
    
    # Statistics
    stats_file = output_dir / f"{prefix}_eef8_stats.txt"
    with open(stats_file, 'w') as f:
        f.write("Simulator-Based Inverse Action Test - Statistics\\n")
        f.write("=" * 80 + "\\n\\n")
        f.write(f"Number of steps tested: {stats['num_steps']}\\n")
        f.write(f"Mean ‖s1‖: {stats['mean_s1_norm']:.6f} m\\n")
        f.write(f"Std ‖s1‖: {stats['std_s1_norm']:.6f} m\\n\\n")
        f.write(f"Mean ‖error‖: {stats['mean_error']:.6f} m\\n")
        f.write(f"Std ‖error‖: {stats['std_error']:.6f} m\\n")
        f.write(f"Median ‖error‖: {stats['median_error']:.6f} m\\n")
        f.write(f"Max ‖error‖: {stats['max_error']:.6f} m\\n")
        f.write(f"Min ‖error‖: {stats['min_error']:.6f} m\\n")
        f.write(f"95th percentile: {stats['percentile_95']:.6f} m\\n")
        f.write(f"99th percentile: {stats['percentile_99']:.6f} m\\n\\n")
        f.write(f"Error as % of state norm: {stats['error_as_pct']:.2f}%\\n\\n")
        f.write("Component errors:\\n")
        f.write(f"  Position: {stats['error_pos_mean']:.6f} m\\n")
        f.write(f"  Orientation: {stats['error_ori_mean']:.6f} rad\\n")
        f.write(f"  Gripper: {stats['error_grip_mean']:.6f}\\n")
    
    print(f"✓ Saved statistics to: {stats_file}")
    
    # Error histogram
    hist_file = output_dir / f"{prefix}_eef8_diff_hist.png"
    plt.figure(figsize=(10, 6))
    plt.hist(results['error'], bins=20, edgecolor='black', alpha=0.7)
    plt.xlabel('L2 Error (m)', fontsize=12)
    plt.ylabel('Frequency', fontsize=12)
    plt.title('Simulator-Based Inverse Action Error Distribution', fontsize=14)
    plt.axvline(np.mean(results['error']), color='red', linestyle='--',
                label=f'Mean: {np.mean(results["error"]):.6f}', linewidth=2)
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.savefig(hist_file, dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"✓ Saved histogram to: {hist_file}")


# ============================================================================
# VISUALIZATION
# ============================================================================

def plot_trajectory_recovery(results: Dict, output_dir: Path) -> None:
    """Plot trajectory recovery visualization"""
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    # Position component
    s1_pos = np.array([s[:3] for s in results['s1']])
    s1_recon_pos = np.array([s[:3] for s in results['s1_reconstructed']])
    
    axes[0, 0].plot(s1_pos[:, 0], label='s1 (original)', marker='o', linewidth=2)
    axes[0, 0].plot(s1_recon_pos[:, 0], label='s1 (reconstructed)', marker='x',
                    alpha=0.7, linewidth=2, linestyle='--')
    axes[0, 0].set_ylabel('X position (m)', fontsize=11)
    axes[0, 0].set_title('Position X Recovery', fontsize=12, fontweight='bold')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)
    
    axes[0, 1].plot(s1_pos[:, 1], label='s1 (original)', marker='o', linewidth=2)
    axes[0, 1].plot(s1_recon_pos[:, 1], label='s1 (reconstructed)', marker='x',
                    alpha=0.7, linewidth=2, linestyle='--')
    axes[0, 1].set_ylabel('Y position (m)', fontsize=11)
    axes[0, 1].set_title('Position Y Recovery', fontsize=12, fontweight='bold')
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)
    
    # Error over time
    axes[1, 0].plot(results['error_pos'], label='Position error', marker='o',
                    linewidth=2, markersize=4)
    axes[1, 0].plot(results['error_ori'], label='Orientation error', marker='s',
                    linewidth=2, markersize=4)
    axes[1, 0].plot(results['error_grip'], label='Gripper error', marker='^',
                    linewidth=2, markersize=4)
    axes[1, 0].set_xlabel('Step', fontsize=11)
    axes[1, 0].set_ylabel('Error (m / rad)', fontsize=11)
    axes[1, 0].set_title('Component Errors Over Time', fontsize=12, fontweight='bold')
    axes[1, 0].legend()
    axes[1, 0].grid(True, alpha=0.3)
    
    # Total error
    axes[1, 1].plot(results['error'], label='Total L2 error', marker='o',
                    color='red', linewidth=2, markersize=4)
    axes[1, 1].axhline(np.mean(results['error']), color='red', linestyle='--',
                       label=f'Mean: {np.mean(results["error"]):.6f}', linewidth=2)
    axes[1, 1].set_xlabel('Step', fontsize=11)
    axes[1, 1].set_ylabel('L2 Error (m)', fontsize=11)
    axes[1, 1].set_title('Total Reversibility Error', fontsize=12, fontweight='bold')
    axes[1, 1].legend()
    axes[1, 1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    output_file = OUTPUT_DIR / "libero_inverse_action_sim_recovery.png"
    plt.savefig(output_file, dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"✓ Saved recovery plot to: {output_file}")


# ============================================================================
# MAIN
# ============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Test inverse action reversibility using MuJoCo simulator"
    )
    parser.add_argument("--demo_idx", type=int, default=0,
                        help="Demo index to test (default: 0)")
    parser.add_argument("--num_steps", type=int, default=50,
                        help="Number of steps to test (default: 50)")
    args = parser.parse_args()
    
    print("\\n" + "=" * 80)
    print("LIBERO INVERSE ACTION TEST - SIMULATOR BASED")
    print("=" * 80)
    
    # Load demo
    print(f"\\nLoading demo {args.demo_idx}...")
    trajectory_data = load_hdf5_trajectory(str(DEMO_PATH), demo_idx=args.demo_idx)
    print(f"  States shape: {trajectory_data['states_110d'].shape}")
    print(f"  Actions shape: {trajectory_data['actions'].shape}")
    
    # Create environment
    print("\\nCreating robosuite environment...")
    env = create_environment(str(DEMO_PATH))
    print(f"  Environment: {env}")
    
    # Run test
    results = test_inverse_action_simulator(env, trajectory_data, args.demo_idx,
                                             max_steps=args.num_steps)
    
    # Compute statistics
    stats = compute_statistics(results)
    
    # Print summary
    print("\\n" + "=" * 80)
    print("RESULTS SUMMARY")
    print("=" * 80)
    print(f"\\nNumber of steps tested: {stats['num_steps']}")
    print(f"Mean state norm ‖s1‖: {stats['mean_s1_norm']:.4f} m")
    print(f"Mean error ‖s1 - s1_reconstructed‖: {stats['mean_error']:.6f} m")
    print(f"Std dev error: {stats['std_error']:.6f} m")
    print(f"Error as % of state norm: {stats['error_as_pct']:.2f}%")
    print(f"\\nComponent errors:")
    print(f"  Position: {stats['error_pos_mean']:.6f} m")
    print(f"  Orientation: {stats['error_ori_mean']:.6f} rad")
    print(f"  Gripper: {stats['error_grip_mean']:.6f}")
    
    # Save results
    print("\\nSaving results...")
    save_results_to_file(results, stats, OUTPUT_DIR,
                        prefix="libero_inverse_action_sim")
    plot_trajectory_recovery(results, OUTPUT_DIR)
    
    print("\\n" + "=" * 80)
    print("✓ TEST COMPLETE")
    print("=" * 80)


if __name__ == "__main__":
    main()
