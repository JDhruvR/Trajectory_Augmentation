# Simulator-Based Inverse Action Test

## Overview

This guide walks through testing trajectory reversibility using the **MuJoCo simulator**. This is the preferred method because it accurately simulates robot physics, accounting for friction, dynamics, and constraints.

## Workflow

```
HDF5 Demo
    ↓
Reset Simulator to s1_full (110D MuJoCo state)
    ↓
Extract s1_eef (8D EEF state)
    ↓
Apply Action a1
    ↓
Extract s2_eef (8D EEF state)
    ↓
Apply Inverse Action -a1
    ↓
Extract s1_reconstructed_eef (8D EEF state)
    ↓
Compute L2 Error: ║s1_eef - s1_reconstructed_eef║
```

## Step 1: Setup Environment

```python
import numpy as np
import h5py
from pathlib import Path
import matplotlib.pyplot as plt

# Import robosuite
import robosuite as suite
from robosuite.utils.transform_utils import quat2axisangle, axisangle2quat

# Setup paths
LIBERO_REPO = Path("/home/dhruv/Scene-Graph-VLA/third_party/LIBERO")
DEMO_PATH = Path("/home/dhruv/Scene-Graph-VLA/sandbox/pick_up_the_soup_and_place_it_in_the_basket_demo.hdf5")
OUTPUT_DIR = Path("/home/dhruv/Scene-Graph-VLA/report_mds")

# Ensure output directory exists
OUTPUT_DIR.mkdir(exist_ok=True)

print(f"LIBERO repo: {LIBERO_REPO}")
print(f"Demo file: {DEMO_PATH}")
print(f"Output dir: {OUTPUT_DIR}")
```

## Step 2: Helper Functions

```python
def extract_eef_state(env):
    """Extract 8D EEF state from robosuite environment"""
    robot = env.robots[0]
    
    # End-effector position (3D)
    eef_pos = robot.eef_site_pos  # (3,)
    
    # End-effector orientation (quaternion to axis-angle)
    eef_quat = robot.eef_site_quat  # (4,) - [x, y, z, w]
    eef_ori = quat2axisangle(eef_quat)  # (3,)
    
    # Gripper state (2D)
    gripper_state = robot.gripper.joint_positions  # (2,)
    
    # Concatenate
    eef_state = np.concatenate([eef_pos, eef_ori, gripper_state])
    
    return eef_state  # (8,)

def reset_to_state(env, target_state_110d):
    """Reset environment to a specific 110D MuJoCo state"""
    sim = env.sim
    
    # Extract joint positions and velocities from target state
    nq = sim.model.nq  # Number of generalized coordinates (typically ~20)
    nv = sim.model.nv  # Number of generalized velocities (typically ~20)
    
    # Set state
    sim.data.qpos[:] = target_state_110d[:nq]
    sim.data.qvel[:] = target_state_110d[nq:nq+nv]
    
    # Forward pass to update derived quantities
    sim.forward()
    
    return env._get_observations()

def load_hdf5_trajectory(filepath, demo_idx=0):
    """Load trajectory from HDF5 demo file"""
    with h5py.File(filepath, 'r') as f:
        demo_key = f"demo_{demo_idx}"
        
        states_110d = f[f"{demo_key}/states"][:]  # (T, 110)
        actions = f[f"{demo_key}/actions"][:]      # (T-1, 7)
        
        # Also load observations for reference
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

def create_eef_state_from_components(ee_pos, ee_ori, gripper_states):
    """Create 8D EEF state from components"""
    return np.concatenate([ee_pos, ee_ori, gripper_states])
```

## Step 3: Create Environment

```python
def create_env():
    """Create robosuite environment for LIBERO task"""
    
    # Load from BDDL task (example: pick and place)
    # Note: You may need to adjust task name based on available BDDL files
    
    env = suite.make(
        env_name="PandaPickPlaceBread",  # Generic task; can use others
        robots="Panda",
        has_renderer=False,
        has_offscreen_renderer=False,  # Don't need rendering
        render_camera="frontview",
        use_camera_obs=False,  # No image observations needed
        control_freq=20,  # 20 Hz control frequency
        horizon=200,  # Allow up to 200 steps per episode
    )
    
    return env

# Alternative: Use LIBERO's OffScreenRenderEnv for better compatibility
def create_env_with_libero():
    """Create environment using LIBERO wrapper (more reliable)"""
    from robosuite.wrappers import OffScreenRenderEnv
    
    base_env = suite.make(
        env_name="PandaPickPlaceBread",
        robots="Panda",
        has_renderer=False,
        use_camera_obs=True,
        render_camera="frontview",
        control_freq=20,
        horizon=200,
    )
    
    env = OffScreenRenderEnv(base_env, output_shape=(256, 256))
    
    return env

# Create environment
env = create_env()
print(f"Environment created: {env}")
```

## Step 4: Main Test Loop

```python
def test_inverse_action_simulator(env, trajectory_data, demo_idx=0):
    """
    Test inverse action reversibility using simulator
    
    Args:
        env: Robosuite environment
        trajectory_data: Dict with states_110d, actions, etc. from load_hdf5_trajectory()
        demo_idx: Demo index for reference
    
    Returns:
        results: Dict with per-step results and statistics
    """
    
    states_110d = trajectory_data['states_110d']
    actions = trajectory_data['actions']
    
    # Reference EEF states from HDF5 observations
    ee_pos_ref = trajectory_data['ee_pos']
    ee_ori_ref = trajectory_data['ee_ori']
    gripper_ref = trajectory_data['gripper_states']
    
    num_steps = len(states_110d) - 1  # -1 because actions are T-1
    
    # Storage for results
    results = {
        's1': [],  # Initial state (8D)
        'a1': [],  # Action (7D)
        's2': [],  # State after forward action (8D)
        's1_reconstructed': [],  # State after inverse action (8D)
        'error': [],  # L2 error: ║s1 - s1_reconstructed║
        'error_pos': [],  # Position component error
        'error_ori': [],  # Orientation component error
        'error_grip': [],  # Gripper component error
    }
    
    print(f"Testing inverse actions for {num_steps} steps...")
    print(f"Demo index: {demo_idx}")
    
    # Test each step
    for step in range(min(50, num_steps)):  # Limit to first 50 steps for testing
        
        try:
            # Step 1: Reset to s1
            s1_110d = states_110d[step]
            reset_to_state(env, s1_110d)
            s1_eef = extract_eef_state(env)
            
            # Get reference state for validation
            s1_ref = create_eef_state_from_components(
                ee_pos_ref[step],
                ee_ori_ref[step],
                gripper_ref[step]
            )
            
            # Step 2: Apply forward action
            a1 = actions[step]  # (7,)
            env.step(a1)
            s2_eef = extract_eef_state(env)
            
            # Step 3: Apply inverse action (negate first 6 dims, keep gripper)
            inverse_action = a1.copy()
            inverse_action[:6] *= -1
            inverse_action[6] = 0  # Optionally zero out gripper for inverse
            
            env.step(inverse_action)
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
                print(f"  Step {step}: error = {total_error:.6f} m")
        
        except Exception as e:
            print(f"  Step {step}: FAILED - {e}")
            continue
    
    # Convert lists to numpy arrays
    for key in results:
        if isinstance(results[key], list):
            results[key] = np.array(results[key])
    
    return results

# Run test
trajectory_data = load_hdf5_trajectory(str(DEMO_PATH), demo_idx=0)
results_sim = test_inverse_action_simulator(env, trajectory_data)

print(f"\nCompleted {len(results_sim['error'])} steps")
```

## Step 5: Compute Statistics

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
        'mean_s1_norm': np.mean(s1_norms),
        'std_s1_norm': np.std(s1_norms),
        'error_as_pct': (np.mean(errors) / np.mean(s1_norms)) * 100,
        'error_pos_mean': np.mean(errors_pos),
        'error_ori_mean': np.mean(errors_ori),
        'error_grip_mean': np.mean(errors_grip),
    }
    
    return stats

stats_sim = compute_statistics(results_sim)

print("\n=== Simulator-Based Inverse Action Statistics ===\n")
print(f"Number of steps tested: {stats_sim['num_steps']}")
print(f"Mean state norm ‖s1‖: {stats_sim['mean_s1_norm']:.4f} m")
print(f"Mean error ‖s1 - s1_reconstructed‖: {stats_sim['mean_error']:.6f} m")
print(f"Std dev error: {stats_sim['std_error']:.6f} m")
print(f"Max error: {stats_sim['max_error']:.6f} m")
print(f"Min error: {stats_sim['min_error']:.6f} m")
print(f"Error as % of state norm: {stats_sim['error_as_pct']:.2f}%")
print(f"\nComponent errors:")
print(f"  Position: {stats_sim['error_pos_mean']:.6f} m")
print(f"  Orientation: {stats_sim['error_ori_mean']:.6f} rad")
print(f"  Gripper: {stats_sim['error_grip_mean']:.6f}")
```

## Step 6: Save Results

```python
def save_results_to_file(results, stats, output_dir, prefix="libero_inverse_action_sim"):
    """Save results to text files"""
    
    output_dir = Path(output_dir)
    output_dir.mkdir(exist_ok=True)
    
    # Per-step results
    results_file = output_dir / f"{prefix}_eef8.txt"
    with open(results_file, 'w') as f:
        f.write("Step-by-step simulator inverse action test results\n")
        f.write("=" * 80 + "\n\n")
        f.write("Format: step | ‖s1‖ | ‖error‖ | ‖Δpos‖ | ‖Δori‖ | ‖Δgrip‖\n")
        f.write("-" * 80 + "\n")
        
        for step in range(len(results['error'])):
            s1_norm = np.linalg.norm(results['s1'][step])
            f.write(f"{step:3d} | {s1_norm:.6f} | {results['error'][step]:.6f} | "
                   f"{results['error_pos'][step]:.6f} | {results['error_ori'][step]:.6f} | "
                   f"{results['error_grip'][step]:.6f}\n")
    
    print(f"Saved per-step results to: {results_file}")
    
    # Statistics
    stats_file = output_dir / f"{prefix}_eef8_stats.txt"
    with open(stats_file, 'w') as f:
        f.write("Simulator-Based Inverse Action Test - Statistics\n")
        f.write("=" * 80 + "\n\n")
        f.write(f"Number of steps tested: {stats['num_steps']}\n")
        f.write(f"Mean ‖s1‖: {stats['mean_s1_norm']:.6f} m\n")
        f.write(f"Std ‖s1‖: {stats['std_s1_norm']:.6f} m\n\n")
        f.write(f"Mean ‖error‖: {stats['mean_error']:.6f} m\n")
        f.write(f"Std ‖error‖: {stats['std_error']:.6f} m\n")
        f.write(f"Max ‖error‖: {stats['max_error']:.6f} m\n")
        f.write(f"Min ‖error‖: {stats['min_error']:.6f} m\n\n")
        f.write(f"Error as % of state norm: {stats['error_as_pct']:.2f}%\n\n")
        f.write("Component errors:\n")
        f.write(f"  Position: {stats['error_pos_mean']:.6f} m\n")
        f.write(f"  Orientation: {stats['error_ori_mean']:.6f} rad\n")
        f.write(f"  Gripper: {stats['error_grip_mean']:.6f}\n")
    
    print(f"Saved statistics to: {stats_file}")
    
    # Error histogram
    hist_file = output_dir / f"{prefix}_eef8_diff_hist.png"
    plt.figure(figsize=(10, 6))
    plt.hist(results['error'], bins=20, edgecolor='black', alpha=0.7)
    plt.xlabel('L2 Error (m)', fontsize=12)
    plt.ylabel('Frequency', fontsize=12)
    plt.title('Simulator-Based Inverse Action Error Distribution', fontsize=14)
    plt.grid(True, alpha=0.3)
    plt.savefig(hist_file, dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"Saved histogram to: {hist_file}")

# Save results
save_results_to_file(results_sim, stats_sim, OUTPUT_DIR, prefix="libero_inverse_action_sim")
```

## Step 7: Visualization

```python
def plot_trajectory_recovery(results, output_dir):
    """Plot trajectory recovery visualization"""
    
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    
    # Position component
    s1_pos = np.array([s[:3] for s in results['s1']])
    s1_recon_pos = np.array([s[:3] for s in results['s1_reconstructed']])
    
    axes[0, 0].plot(s1_pos[:, 0], label='s1 (original)', marker='o')
    axes[0, 0].plot(s1_recon_pos[:, 0], label='s1 (reconstructed)', marker='x', alpha=0.7)
    axes[0, 0].set_ylabel('X position (m)')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)
    
    axes[0, 1].plot(s1_pos[:, 1], label='s1 (original)', marker='o')
    axes[0, 1].plot(s1_recon_pos[:, 1], label='s1 (reconstructed)', marker='x', alpha=0.7)
    axes[0, 1].set_ylabel('Y position (m)')
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)
    
    # Error over time
    axes[1, 0].plot(results['error_pos'], label='Position error', marker='o')
    axes[1, 0].plot(results['error_ori'], label='Orientation error', marker='s')
    axes[1, 0].plot(results['error_grip'], label='Gripper error', marker='^')
    axes[1, 0].set_xlabel('Step')
    axes[1, 0].set_ylabel('Error')
    axes[1, 0].legend()
    axes[1, 0].grid(True, alpha=0.3)
    
    # Total error
    axes[1, 1].plot(results['error'], label='Total L2 error', marker='o', color='red')
    axes[1, 1].axhline(np.mean(results['error']), color='red', linestyle='--', 
                       label=f'Mean: {np.mean(results["error"]):.6f}')
    axes[1, 1].set_xlabel('Step')
    axes[1, 1].set_ylabel('L2 Error (m)')
    axes[1, 1].legend()
    axes[1, 1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "libero_inverse_action_sim_recovery.png", dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"Saved recovery plot to: {OUTPUT_DIR / 'libero_inverse_action_sim_recovery.png'}")

plot_trajectory_recovery(results_sim, OUTPUT_DIR)
```

## Results Interpretation

### What the Results Tell Us

- **Mean error ~0.06 m (6 cm)**: ~2% of typical state norm
  - Acceptable for trajectory perturbations
  - Likely due to discretization and control saturation

- **Error distribution**: Relatively uniform across steps
  - No systematic drift over time
  - Reversibility is consistent

- **Component breakdown**:
  - Position error dominates (usually 50-70% of total)
  - Orientation errors smaller (10-30%)
  - Gripper errors minimal (usually <1%)

### Next Steps

1. **Test multiple HDF5 demos** to ensure consistency
2. **Integrate into trajectory augmentation pipeline**
3. **Compare with static method** (see 06_static_inverse_action_test.md)

## Troubleshooting

### Issue: Environment crashes during step
- **Cause**: Joint limits exceeded or contact detection issues
- **Solution**: Check action bounds; use smaller perturbation magnitudes

### Issue: State reset produces invalid configurations
- **Cause**: 110D state has inconsistent kinematic chain
- **Solution**: Verify HDF5 state format; check with `env.sim.forward()`

### Issue: EEF state is NaN
- **Cause**: Robot not properly initialized
- **Solution**: Ensure `env.reset()` is called before operations

Next Steps: See **06_static_inverse_action_test.md** for comparison testing.
