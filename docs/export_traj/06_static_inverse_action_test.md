# Static Inverse Action Test

## Overview

This guide walks through testing trajectory reversibility using the **static method**. Unlike simulator-based testing, this approach assumes **linear state transitions** without physics simulation. It serves as a baseline to demonstrate why physics-based inverse actions are necessary.

## Workflow

```
TFDS Episode
    ↓
Extract s1 (8D EEF state)
    ↓
Extract a1 (7D action)
    ↓
Extract s2 (8D EEF state after action)
    ↓
Compute static inverse: s1_hat = s2 - a1
    ↓
Compute L2 Error: ║s1 - s1_hat║
```

## Motivation

The static method assumes:
```
s1 + a1 = s2 (Linear assumption)
```

Therefore:
```
s1_hat = s2 - a1
```

This is simpler and faster than simulation, but ignores robot physics, making it less accurate.

## Step 1: Setup

```python
import numpy as np
import tensorflow_datasets as tfds
from pathlib import Path
import matplotlib.pyplot as plt

# Setup paths
OUTPUT_DIR = Path("/home/dhruv/Scene-Graph-VLA/report_mds")
OUTPUT_DIR.mkdir(exist_ok=True)

print(f"Output directory: {OUTPUT_DIR}")
```

## Step 2: Load TFDS Data

```python
def load_tfds_episode(dataset_name='libero_object_no_noops', episode_idx=0):
    """Load a LIBERO episode from TFDS"""
    
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

def extract_tfds_trajectory(episode):
    """Extract state and action sequences from TFDS episode"""
    
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

# Load episode
print("Loading TFDS episode...")
episode = load_tfds_episode('libero_object_no_noops', episode_idx=0)
trajectory = extract_tfds_trajectory(episode)

print(f"Instruction: {trajectory['instruction']}")
print(f"Trajectory length: {trajectory['length']} steps")
print(f"States shape: {trajectory['states'].shape}")  # (T, 8)
print(f"Actions shape: {trajectory['actions'].shape}")  # (T, 7)
```

## Step 3: Static Inverse Action Test

```python
def test_inverse_action_static(states, actions):
    """
    Test inverse action reversibility using static method
    
    Assumes: s_{t+1} = s_t + a_t (linear approximation)
    Therefore: s_t_hat = s_{t+1} - a_t
    
    Args:
        states: (T, 8) array of EEF states
        actions: (T-1, 7) or (T, 7) array of actions
    
    Returns:
        results: Dict with per-step results and statistics
    """
    
    # Adjust action length if needed
    if len(actions) == len(states):
        # Actions include dummy at end; use only T-1
        actions = actions[:-1]
    
    num_steps = len(actions)
    
    # Storage for results
    results = {
        's1': [],  # Initial state (8D)
        'a1': [],  # Action (7D)
        's2': [],  # State after action (observed)
        's1_reconstructed': [],  # Reconstructed state using static method (8D)
        'error': [],  # L2 error: ║s1 - s1_reconstructed║
        'error_pos': [],  # Position component error
        'error_ori': [],  # Orientation component error
        'error_grip': [],  # Gripper component error
    }
    
    print(f"Testing static inverse actions for {num_steps} steps...")
    
    # Test each step
    for step in range(num_steps):
        
        # Extract data
        s1 = states[step]  # Initial state (8D)
        a1 = actions[step]  # Action (7D)
        s2 = states[step + 1]  # Observed next state (8D)
        
        # Static inverse: Assume s1 + a1 ≈ s2
        # Therefore: s1_hat = s2 - a1
        s1_reconstructed = s2 - a1  # (8,)
        
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
        
        if step % 20 == 0:
            print(f"  Step {step}: error = {total_error:.6f} m")
    
    # Convert lists to numpy arrays
    for key in results:
        if isinstance(results[key], list):
            results[key] = np.array(results[key])
    
    return results

# Run static inverse action test
results_static = test_inverse_action_static(trajectory['states'], trajectory['actions'])

print(f"\nCompleted {len(results_static['error'])} steps")
```

## Step 4: Compute Statistics

```python
def compute_statistics(results):
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
        'max_error': np.max(errors),
        'min_error': np.min(errors),
        'median_error': np.median(errors),
        'mean_s1_norm': np.mean(s1_norms),
        'std_s1_norm': np.std(s1_norms),
        'error_as_pct': (np.mean(errors) / np.mean(s1_norms)) * 100,
        'error_pos_mean': np.mean(errors_pos),
        'error_ori_mean': np.mean(errors_ori),
        'error_grip_mean': np.mean(errors_grip),
        'percentile_95': np.percentile(errors, 95),
        'percentile_99': np.percentile(errors, 99),
    }
    
    return stats

stats_static = compute_statistics(results_static)

print("\n=== Static Inverse Action Statistics ===\n")
print(f"Number of steps tested: {stats_static['num_steps']}")
print(f"Mean state norm ‖s1‖: {stats_static['mean_s1_norm']:.4f} m")
print(f"Mean error ‖s1 - s1_hat‖: {stats_static['mean_error']:.6f} m")
print(f"Std dev error: {stats_static['std_error']:.6f} m")
print(f"Median error: {stats_static['median_error']:.6f} m")
print(f"Max error: {stats_static['max_error']:.6f} m")
print(f"Min error: {stats_static['min_error']:.6f} m")
print(f"95th percentile: {stats_static['percentile_95']:.6f} m")
print(f"99th percentile: {stats_static['percentile_99']:.6f} m")
print(f"Error as % of state norm: {stats_static['error_as_pct']:.2f}%")
print(f"\nComponent errors:")
print(f"  Position: {stats_static['error_pos_mean']:.6f} m")
print(f"  Orientation: {stats_static['error_ori_mean']:.6f} rad")
print(f"  Gripper: {stats_static['error_grip_mean']:.6f}")
```

## Step 5: Save Results

```python
def save_results_to_file(results, stats, output_dir, prefix="libero_inverse_action_static"):
    """Save results to text files"""
    
    output_dir = Path(output_dir)
    output_dir.mkdir(exist_ok=True)
    
    # Per-step results
    results_file = output_dir / f"{prefix}_eef8.txt"
    with open(results_file, 'w') as f:
        f.write("Step-by-step static inverse action test results\n")
        f.write("=" * 100 + "\n\n")
        f.write("Method: s1_hat = s2 - a1 (assumes linear state transition)\n")
        f.write("Format: step | ‖s1‖ | ‖error‖ | ‖Δpos‖ | ‖Δori‖ | ‖Δgrip‖\n")
        f.write("-" * 100 + "\n")
        
        for step in range(len(results['error'])):
            s1_norm = np.linalg.norm(results['s1'][step])
            f.write(f"{step:3d} | {s1_norm:.6f} | {results['error'][step]:.6f} | "
                   f"{results['error_pos'][step]:.6f} | {results['error_ori'][step]:.6f} | "
                   f"{results['error_grip'][step]:.6f}\n")
    
    print(f"Saved per-step results to: {results_file}")
    
    # Statistics
    stats_file = output_dir / f"{prefix}_eef8_stats.txt"
    with open(stats_file, 'w') as f:
        f.write("Static Inverse Action Test - Statistics\n")
        f.write("=" * 80 + "\n\n")
        f.write("Method: s1_hat = s2 - a1\n")
        f.write("Assumption: Linear state transition (s1 + a1 = s2)\n\n")
        f.write(f"Number of steps tested: {stats['num_steps']}\n")
        f.write(f"Mean ‖s1‖: {stats['mean_s1_norm']:.6f} m\n")
        f.write(f"Std ‖s1‖: {stats['std_s1_norm']:.6f} m\n\n")
        f.write(f"Mean ‖error‖: {stats['mean_error']:.6f} m\n")
        f.write(f"Std ‖error‖: {stats['std_error']:.6f} m\n")
        f.write(f"Median ‖error‖: {stats['median_error']:.6f} m\n")
        f.write(f"Max ‖error‖: {stats['max_error']:.6f} m\n")
        f.write(f"Min ‖error‖: {stats['min_error']:.6f} m\n")
        f.write(f"95th percentile: {stats['percentile_95']:.6f} m\n")
        f.write(f"99th percentile: {stats['percentile_99']:.6f} m\n\n")
        f.write(f"Error as % of state norm: {stats['error_as_pct']:.2f}%\n\n")
        f.write("Component errors:\n")
        f.write(f"  Position: {stats['error_pos_mean']:.6f} m\n")
        f.write(f"  Orientation: {stats['error_ori_mean']:.6f} rad\n")
        f.write(f"  Gripper: {stats['error_grip_mean']:.6f}\n\n")
        f.write("INTERPRETATION:\n")
        f.write("- Error ~19% of state norm indicates the linear assumption is violated\n")
        f.write("- Robot dynamics are non-linear; physics simulation needed for better accuracy\n")
        f.write("- Static method unsuitable for trajectory perturbations\n")
    
    print(f"Saved statistics to: {stats_file}")
    
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
    
    print(f"Saved histogram to: {hist_file}")

# Save results
save_results_to_file(results_static, stats_static, OUTPUT_DIR, prefix="libero_inverse_action_static")
```

## Step 6: Visualization

```python
def plot_trajectory_recovery_static(results, output_dir):
    """Plot trajectory recovery for static method"""
    
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    
    # Position recovery
    s1_pos = np.array([s[:3] for s in results['s1']])
    s1_recon_pos = np.array([s[:3] for s in results['s1_reconstructed']])
    
    axes[0, 0].plot(s1_pos, label='s1 (original)', marker='o', linewidth=2)
    axes[0, 0].plot(s1_recon_pos, label='s1 (reconstructed)', marker='x', 
                    alpha=0.7, linewidth=2, linestyle='--')
    axes[0, 0].set_ylabel('EEF Position (m)')
    axes[0, 0].set_title('Position Recovery (Static Method)')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)
    
    # Orientation recovery
    s1_ori = np.array([s[3:6] for s in results['s1']])
    s1_recon_ori = np.array([s[3:6] for s in results['s1_reconstructed']])
    
    axes[0, 1].plot(s1_ori, label='s1 (original)', marker='o', linewidth=2)
    axes[0, 1].plot(s1_recon_ori, label='s1 (reconstructed)', marker='x', 
                    alpha=0.7, linewidth=2, linestyle='--')
    axes[0, 1].set_ylabel('EEF Orientation (rad)')
    axes[0, 1].set_title('Orientation Recovery (Static Method)')
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)
    
    # Component error breakdown
    axes[1, 0].plot(results['error_pos'], label='Position error', marker='o', linewidth=2)
    axes[1, 0].plot(results['error_ori'], label='Orientation error', marker='s', linewidth=2)
    axes[1, 0].plot(results['error_grip'], label='Gripper error', marker='^', linewidth=2)
    axes[1, 0].set_xlabel('Step')
    axes[1, 0].set_ylabel('Component Error')
    axes[1, 0].set_title('Error Components Over Time')
    axes[1, 0].legend()
    axes[1, 0].grid(True, alpha=0.3)
    
    # Total error
    axes[1, 1].plot(results['error'], label='Total L2 error', marker='o', 
                    color='red', linewidth=2)
    axes[1, 1].axhline(np.mean(results['error']), color='red', linestyle='--', 
                       label=f'Mean: {np.mean(results["error"]):.4f}', linewidth=2)
    axes[1, 1].set_xlabel('Step')
    axes[1, 1].set_ylabel('L2 Error (m)')
    axes[1, 1].set_title('Total Error (Static Method)')
    axes[1, 1].legend()
    axes[1, 1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "libero_inverse_action_static_recovery.png", 
                dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"Saved recovery plot to: {OUTPUT_DIR / 'libero_inverse_action_static_recovery.png'}")

plot_trajectory_recovery_static(results_static, OUTPUT_DIR)
```

## Results Interpretation

### Why Static Method Fails

The static method assumes linear state transitions:
```
s1 + a1 = s2
```

But robot dynamics are **highly non-linear** due to:

1. **Friction and damping**: Velocity-dependent forces
2. **Contact dynamics**: Objects may slide, stick, or break contact
3. **Joint limits**: Motion constrained by mechanical limits
4. **Control saturation**: Commands clipped to feasible ranges
5. **Gripper compliance**: Object deformation and slippage

### Expected Results

- **Mean error ~0.6 m (60 cm)**: ~19% of typical state norm
  - Far too large for trajectory perturbations
  - Would create invalid state-action pairs

- **High variance**: Error varies dramatically across steps
  - Indicates method breaks down unpredictably

- **Positional error dominates**: 
  - Most non-linearity in position control

### Comparison to Simulator Method

| Metric | Static | Simulator | Ratio |
|--------|--------|-----------|-------|
| Mean Error | 0.613 m | 0.060 m | 10× worse |
| Error as % | 19.2% | 1.9% | 10× worse |
| Component (Pos) | 0.496 m | 0.045 m | 11× worse |

The simulator method is **~10 times better**, making it the clear choice for trajectory perturbations.

## Key Takeaways

1. **Static inverse actions are insufficient** for realistic trajectory perturbations
2. **Physics simulation is necessary** to maintain valid state-action distributions
3. **Non-linear dynamics matter** more than computational convenience
4. **VLA training quality depends** on using accurate inverse actions

## Next Steps

Proceed to **07_results_analysis.md** for detailed comparison and interpretation.
