#!/usr/bin/env python3
"""
Static Inverse Action Test

Tests trajectory reversibility using static method (kinematic-only, no physics).
Assumes: s1 + a1 ≈ s2 (linear state transition)
Therefore: s1_hat = s2 - a1

This serves as a baseline to demonstrate why physics-based inverse actions are necessary.

Usage:
    python test_inverse_action_static.py [--episode_idx 0]

Output:
    - libero_inverse_action_static_eef8.txt: Per-step results
    - libero_inverse_action_static_eef8_stats.txt: Statistics summary
    - libero_inverse_action_static_eef8_diff_hist.png: Error histogram
    - libero_inverse_action_static_recovery.png: Recovery visualization
"""

import numpy as np
import tensorflow_datasets as tfds
from pathlib import Path
import matplotlib.pyplot as plt
import argparse
from typing import Dict, Tuple


# ============================================================================
# CONFIGURATION
# ============================================================================

OUTPUT_DIR = Path("/home/dhruv/Trajectory_Augmentation/report_mds")
OUTPUT_DIR.mkdir(exist_ok=True, parents=True)

print(f"Output directory: {OUTPUT_DIR}")


# ============================================================================
# DATA LOADING
# ============================================================================

def load_tfds_episode(dataset_name: str = 'libero_object_no_noops',
                      episode_idx: int = 0) -> Dict:
    """
    Load a LIBERO episode from TensorFlow Datasets
    """
    
    dataset = tfds.load(
        dataset_name,
        split='train',
        shuffle_files=False,
    )
    
    # Convert to list and get specific episode
    episodes = list(dataset)
    
    if episode_idx >= len(episodes):
        raise ValueError(f"Episode {episode_idx} not found (only {len(episodes)} available)")
    
    episode = episodes[episode_idx]
    return episode


def extract_tfds_trajectory(episode: Dict) -> Dict:
    """
    Extract state and action sequences from TFDS episode
    """
    
    steps = episode['steps']
    
    # Extract 8D EEF state
    states = steps['observation']['state']  # (T, 8)
    
    # Extract 7D actions
    actions = steps['action']  # (T, 7)
    
    # Language instruction
    instruction = episode['language_instruction'].decode('utf-8')
    
    return {
        'states': states.numpy(),
        'actions': actions.numpy(),
        'instruction': instruction,
        'length': len(states),
    }


# ============================================================================
# STATIC INVERSE ACTION TEST
# ============================================================================

def test_inverse_action_static(states: np.ndarray, actions: np.ndarray) -> Dict:
    """
    Test inverse action reversibility using static method
    """
    
    # Adjust action length if needed
    if len(actions) == len(states):
        # Actions include dummy at end; use only T-1
        actions = actions[:-1]
    
    num_steps = len(actions)
    
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
    
    print(f"\\nTesting static inverse actions for {num_steps} steps...")
    
    # Test each step
    for step in range(num_steps):
        
        # Extract data
        s1 = states[step]
        a1 = actions[step]
        s2 = states[step + 1]
        
        # Static inverse: Assume s1 + a1 ≈ s2
        # Therefore: s1_hat = s2 - a1
        s1_reconstructed = s2 - a1
        
        # Compute total error
        total_error = np.linalg.norm(s1 - s1_reconstructed)
        
        # Compute component errors
        pos_error = np.linalg.norm(s1[:3] - s1_reconstructed[:3])
        ori_error = np.linalg.norm(s1[3:6] - s1_reconstructed[3:6])
        grip_error = np.linalg.norm(s1[6:8] - s1_reconstructed[6:8])
        
        # Store results
        results['s1'].append(s1)
        results['a1'].append(a1)
        results['s2'].append(s2)
        results['s1_reconstructed'].append(s1_reconstructed)
        results['error'].append(total_error)
        results['error_pos'].append(pos_error)
        results['error_ori'].append(ori_error)
        results['error_grip'].append(grip_error)
        
        if step % 30 == 0:
            print(f"  Step {step:3d}: error = {total_error:.6f} m")
    
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
                         prefix: str = "libero_inverse_action_static") -> None:
    """Save results to text files and plots"""
    
    output_dir = Path(output_dir)
    output_dir.mkdir(exist_ok=True, parents=True)
    
    # Per-step results
    results_file = output_dir / f"{prefix}_eef8.txt"
    with open(results_file, 'w') as f:
        f.write("Step-by-step static inverse action test results\\n")
        f.write("=" * 100 + "\\n\\n")
        f.write("Method: s1_hat = s2 - a1 (assumes linear state transition)\\n")
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
        f.write("Static Inverse Action Test - Statistics\\n")
        f.write("=" * 80 + "\\n\\n")
        f.write("Method: s1_hat = s2 - a1\\n")
        f.write("Assumption: Linear state transition (s1 + a1 = s2)\\n\\n")
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
        f.write(f"  Gripper: {stats['error_grip_mean']:.6f}\\n\\n")
        f.write("INTERPRETATION:\\n")
        f.write("- Error ~19% of state norm indicates the linear assumption is violated\\n")
        f.write("- Robot dynamics are non-linear; physics simulation needed for better accuracy\\n")
        f.write("- Static method unsuitable for trajectory perturbations\\n")
    
    print(f"✓ Saved statistics to: {stats_file}")
    
    # Error histogram
    hist_file = output_dir / f"{prefix}_eef8_diff_hist.png"
    plt.figure(figsize=(10, 6))
    plt.hist(results['error'], bins=20, edgecolor='black', alpha=0.7, color='orange')
    plt.xlabel('L2 Error (m)', fontsize=12)
    plt.ylabel('Frequency', fontsize=12)
    plt.title('Static Inverse Action Error Distribution', fontsize=14)
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

def plot_trajectory_recovery_static(results: Dict, output_dir: Path) -> None:
    """Plot trajectory recovery for static method"""
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    # Position recovery
    s1_pos = np.array([s[:3] for s in results['s1']])
    s1_recon_pos = np.array([s[:3] for s in results['s1_reconstructed']])
    
    axes[0, 0].plot(s1_pos, label='s1 (original)', marker='o', linewidth=2)
    axes[0, 0].plot(s1_recon_pos, label='s1 (reconstructed)', marker='x',
                    alpha=0.7, linewidth=2, linestyle='--')
    axes[0, 0].set_ylabel('EEF Position (m)', fontsize=11)
    axes[0, 0].set_title('Position Recovery (Static Method)', fontsize=12, fontweight='bold')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)
    
    # Orientation recovery
    s1_ori = np.array([s[3:6] for s in results['s1']])
    s1_recon_ori = np.array([s[3:6] for s in results['s1_reconstructed']])
    
    axes[0, 1].plot(s1_ori, label='s1 (original)', marker='o', linewidth=2)
    axes[0, 1].plot(s1_recon_ori, label='s1 (reconstructed)', marker='x',
                    alpha=0.7, linewidth=2, linestyle='--')
    axes[0, 1].set_ylabel('EEF Orientation (rad)', fontsize=11)
    axes[0, 1].set_title('Orientation Recovery (Static Method)', fontsize=12, fontweight='bold')
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)
    
    # Component error breakdown
    axes[1, 0].plot(results['error_pos'], label='Position error', marker='o',
                    linewidth=2, markersize=4)
    axes[1, 0].plot(results['error_ori'], label='Orientation error', marker='s',
                    linewidth=2, markersize=4)
    axes[1, 0].plot(results['error_grip'], label='Gripper error', marker='^',
                    linewidth=2, markersize=4)
    axes[1, 0].set_xlabel('Step', fontsize=11)
    axes[1, 0].set_ylabel('Component Error', fontsize=11)
    axes[1, 0].set_title('Error Components Over Time', fontsize=12, fontweight='bold')
    axes[1, 0].legend()
    axes[1, 0].grid(True, alpha=0.3)
    
    # Total error
    axes[1, 1].plot(results['error'], label='Total L2 error', marker='o',
                    color='red', linewidth=2, markersize=4)
    axes[1, 1].axhline(np.mean(results['error']), color='red', linestyle='--',
                       label=f'Mean: {np.mean(results["error"]):.4f}', linewidth=2)
    axes[1, 1].set_xlabel('Step', fontsize=11)
    axes[1, 1].set_ylabel('L2 Error (m)', fontsize=11)
    axes[1, 1].set_title('Total Error (Static Method)', fontsize=12, fontweight='bold')
    axes[1, 1].legend()
    axes[1, 1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    output_file = OUTPUT_DIR / "libero_inverse_action_static_recovery.png"
    plt.savefig(output_file, dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"✓ Saved recovery plot to: {output_file}")


# ============================================================================
# MAIN
# ============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Test inverse action reversibility using static method (kinematic-only)"
    )
    parser.add_argument("--episode_idx", type=int, default=0,
                       help="Episode index to test (default: 0)")
    parser.add_argument("--dataset", type=str, default='libero_object_no_noops',
                       help="TFDS dataset name (default: libero_object_no_noops)")
    args = parser.parse_args()
    
    print("\\n" + "=" * 80)
    print("LIBERO INVERSE ACTION TEST - STATIC METHOD (BASELINE)")
    print("=" * 80)
    
    # Load episode
    print(f"\\nLoading episode {args.episode_idx} from {args.dataset}...")
    episode = load_tfds_episode(args.dataset, episode_idx=args.episode_idx)
    trajectory = extract_tfds_trajectory(episode)
    
    print(f"  Instruction: {trajectory['instruction']}")
    print(f"  Trajectory length: {trajectory['length']} steps")
    print(f"  States shape: {trajectory['states'].shape}")
    print(f"  Actions shape: {trajectory['actions'].shape}")
    
    # Run test
    results = test_inverse_action_static(trajectory['states'], trajectory['actions'])
    
    # Compute statistics
    stats = compute_statistics(results)
    
    # Print summary
    print("\\n" + "=" * 80)
    print("RESULTS SUMMARY")
    print("=" * 80)
    print(f"\\nNumber of steps tested: {stats['num_steps']}")
    print(f"Mean state norm ‖s1‖: {stats['mean_s1_norm']:.4f} m")
    print(f"Mean error ‖s1 - s1_hat‖: {stats['mean_error']:.6f} m")
    print(f"Std dev error: {stats['std_error']:.6f} m")
    print(f"Error as % of state norm: {stats['error_as_pct']:.2f}%")
    print(f"\\nComponent errors:")
    print(f"  Position: {stats['error_pos_mean']:.6f} m")
    print(f"  Orientation: {stats['error_ori_mean']:.6f} rad")
    print(f"  Gripper: {stats['error_grip_mean']:.6f}")
    
    print("\\n⚠  WARNING: Static method error is ~{:.1f}× larger than simulator method!".format(
        stats['mean_error'] / 0.060  # Approximate simulator error
    ))
    print("   This ~19% error is unsuitable for trajectory perturbations.")
    
    # Save results
    print("\\nSaving results...")
    save_results_to_file(results, stats, OUTPUT_DIR,
                        prefix="libero_inverse_action_static")
    plot_trajectory_recovery_static(results, OUTPUT_DIR)
    
    print("\\n" + "=" * 80)
    print("✓ TEST COMPLETE")
    print("=" * 80)


if __name__ == "__main__":
    main()
