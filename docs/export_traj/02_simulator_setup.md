# Simulator Setup: Robosuite Configuration for LIBERO

## Overview

Robosuite is a lightweight robot manipulation simulator built on MuJoCo. We use it to test inverse actions by:
1. Resetting to an intermediate state from a recorded trajectory
2. Applying forward actions
3. Applying inverse actions
4. Measuring state recovery

## Why Robosuite?

- **Physics-Based**: Uses MuJoCo for accurate dynamics simulation
- **Modular**: Supports multiple robot arms, end-effectors, and tasks
- **Extensible**: Works with BDDL task definitions from LIBERO
- **Action Control**: Supports operational space control (OSC) which matches LIBERO's action space

## Robosuite Architecture

### Key Components

1. **Environment** (`robosuite.make()`)
   - Task-specific environment with physics simulation
   - Handles state resets, action application, observation retrieval

2. **Robot Arm** (e.g., `Panda`, `UR5e`)
   - The manipulator that performs actions
   - State includes joint angles, velocities, and end-effector pose

3. **Controller** (e.g., `OSC_POSE`)
   - Converts high-level action commands to low-level torques
   - Controls gripper, arm position/orientation

4. **Task** (BDDL-defined)
   - Goal specification and reward function
   - Object placement, manipulation constraints

## Step 1: Create a Robosuite Environment

### Basic Environment Setup

```python
import robosuite as suite
from robosuite.utils.transform_utils import quat2axisangle, axisangle2quat

# Load environment with BDDL task
env = suite.make(
    env_name="PandaPickPlaceBread",  # Or other LIBERO task
    robots="Panda",
    has_renderer=False,  # No visualization needed for headless testing
    has_offscreen_renderer=True,  # For image observation (if needed)
    render_camera="frontview",
    use_camera_obs=False,  # We're only interested in state, not images
    control_freq=20,  # 20 Hz control frequency
    horizon=150,  # Episode length
)

print(f"Environment created: {env}")
print(f"Action space: {env.action_spec}")
print(f"Observation keys: {env.observation_spec.keys()}")
```

### Using OffScreenRenderEnv (LIBERO Method)

LIBERO provides a wrapper called `OffScreenRenderEnv` for better integration:

```python
from robosuite.wrappers import OffScreenRenderEnv

# Create base environment
env = suite.make(
    env_name="PandaPickPlaceBread",
    robots="Panda",
    has_renderer=False,
    use_camera_obs=True,
    render_camera="frontview",
)

# Wrap with off-screen renderer
env = OffScreenRenderEnv(env, output_shape=(256, 256))
```

## Step 2: Understanding the State and Action Space

### State Representation

The full MuJoCo state is 110-dimensional:
```python
obs = env.reset()
print(f"Full state shape: {env.state_dict()['states'].shape}")  # (110,)
```

For our experiments, we extract the 8D **end-effector (EEF) state**:

```python
import numpy as np

def extract_eef_state(env):
    """Extract 8D EEF state from environment"""
    # Get robot
    robot = env.robots[0]
    
    # End-effector position (3D)
    eef_pos = robot.eef_site_pos
    
    # End-effector orientation as quaternion (4D)
    eef_quat = robot.eef_site_quat  # (x, y, z, w)
    
    # Convert quaternion to axis-angle (3D)
    eef_ori = quat2axisangle(eef_quat)
    
    # Gripper state (2D - left and right finger positions)
    gripper_state = robot.gripper.joint_positions
    
    # Concatenate to 8D state
    eef_state = np.concatenate([eef_pos, eef_ori, gripper_state])
    
    return eef_state  # Shape: (8,)

# Example usage
state = extract_eef_state(env)
print(f"EEF state shape: {state.shape}")  # (8,)
print(f"EEF state: {state}")
```

### Action Representation

Actions in LIBERO are 7-dimensional:
- **Position delta** (3D): Change in EEF x, y, z
- **Orientation delta** (3D): Change in EEF orientation as axis-angle
- **Gripper command** (1D): Open/close signal

```python
# Example action
action = np.array([0.01, 0.0, 0.0, 0.0, 0.0, 0.0, -1.0])
#                  ^^^^^^  pos delta  ^^^^^^  ori delta  gripper

# Apply action
obs = env.step(action)
```

## Step 3: Working with Controller Parameters

### OSC_POSE Controller

The default controller for LIBERO tasks is **Operational Space Control (OSC)** with **POSE** targets.

```python
# In environment configuration
env = suite.make(
    ...,
    control_freq=20,
)

# Controller characteristics
# - Input type: delta (relative changes)
# - Reference frame: base (world frame)
# - Output limits:
#   - Position: [0.05, 0.05, 0.05] meters per step (5 cm max)
#   - Orientation: [0.5, 0.5, 0.5] radians per step
#   - Gripper: [-1, 1] (binary or continuous)
```

### Applying Actions

```python
def apply_action(env, action):
    """Apply action and return new state"""
    obs = env.step(action)
    return obs

# Forward step
obs_2 = apply_action(env, action_1)

# Inverse step (negate first 6 dims, keep gripper)
inverse_action = action_1.copy()
inverse_action[:6] *= -1  # Negate position/orientation deltas
obs_1_reconstructed = apply_action(env, inverse_action)
```

## Step 4: State Reset and Recovery

### Resetting to Arbitrary State

The key to testing inverse actions is the ability to reset to any recorded state:

```python
def reset_to_state(env, target_state_110d):
    """Reset environment to a specific 110D state"""
    # Get MuJoCo model and data
    sim = env.sim
    
    # Set joint positions and velocities
    # (target_state_110d contains full MuJoCo state)
    sim.data.qpos[:] = target_state_110d[:sim.model.nq]  # Joint positions
    sim.data.qvel[:] = target_state_110d[sim.model.nq:sim.model.nq + sim.model.nv]  # Joint velocities
    
    # Synchronize
    sim.forward()
    
    # Get observation
    obs = env._get_observations()
    return obs

# Usage
obs = reset_to_state(env, state_110d_from_hdf5)
```

## Step 5: Working with HDF5 Demo Data

### Loading HDF5 Trajectory

```python
import h5py

def load_hdf5_demo(filepath, demo_idx=0):
    """Load a demo from HDF5 file"""
    with h5py.File(filepath, 'r') as f:
        demo_key = f"demo_{demo_idx}"
        
        # Extract trajectory
        states = f[f"{demo_key}/states"][:]  # (T, 110) - full MuJoCo states
        actions = f[f"{demo_key}/actions"][:]  # (T-1, 7) - actions taken
        obs = {
            'ee_pos': f[f"{demo_key}/obs/ee_pos"][:],  # (T, 3)
            'ee_ori': f[f"{demo_key}/obs/ee_ori"][:],  # (T, 3) - axis-angle
            'gripper_states': f[f"{demo_key}/obs/gripper_states"][:],  # (T, 2)
        }
        
    return states, actions, obs

# Usage
states, actions, obs = load_hdf5_demo(
    "/home/dhruv/Scene-Graph-VLA/sandbox/pick_up_the_soup_and_place_it_in_the_basket_demo.hdf5"
)
print(f"Trajectory length: {len(states)}")
print(f"Actions shape: {actions.shape}")
```

## Step 6: Complete Inverse Action Test

```python
import numpy as np

def test_inverse_action_sim(env, s1_full, a1, steps=1):
    """
    Test inverse action reversibility in simulation
    
    Args:
        env: Robosuite environment
        s1_full: Initial 110D state
        a1: Action (7D)
        steps: Number of steps to forward/inverse
    
    Returns:
        dict: s1, s2, s1_reconstructed (as 8D EEF states), L2 error
    """
    # Reset to initial state
    reset_to_state(env, s1_full)
    s1_eef = extract_eef_state(env)
    
    # Apply action forward
    env.step(a1)
    s2_eef = extract_eef_state(env)
    
    # Apply inverse action
    inverse_action = a1.copy()
    inverse_action[:6] *= -1
    env.step(inverse_action)
    s1_reconstructed_eef = extract_eef_state(env)
    
    # Calculate error
    error = np.linalg.norm(s1_eef - s1_reconstructed_eef)
    
    return {
        's1': s1_eef,
        's2': s2_eef,
        's1_reconstructed': s1_reconstructed_eef,
        'error': error,
    }

# Usage
result = test_inverse_action_sim(env, states[1], actions[0])
print(f"Error: {result['error']:.6f}")
```

## Step 7: Visualizing Environment State

### Rendering with Visualization

```python
def render_frame(env, save_path=None):
    """Render current environment state"""
    frame = env.render(mode='rgb_array')  # RGB array
    
    if save_path:
        from PIL import Image
        Image.fromarray(frame).save(save_path)
    
    return frame

# Example
frame = render_frame(env, "debug_frame.png")
```

### Printing State Information

```python
def print_environment_state(env):
    """Print detailed state information"""
    robot = env.robots[0]
    
    print(f"EEF position: {robot.eef_site_pos}")
    print(f"EEF orientation (quat): {robot.eef_site_quat}")
    print(f"Gripper position: {robot.gripper.joint_positions}")
    print(f"Joint positions: {robot.joint_positions}")
    print(f"Joint velocities: {robot.joint_velocities}")
```

## Troubleshooting

### Issue: `MuJoCo Error: unknown body 'robot0_l_gripper_finger_contact'`
- **Cause**: Robot/gripper not properly initialized
- **Solution**: Ensure robot is created with correct model in `suite.make()`

### Issue: Action not being applied correctly
- **Cause**: Action space mismatch
- **Solution**: Verify action shape (should be 7D) and check controller configuration

### Issue: State reset fails
- **Cause**: Incompatible state dimensions
- **Solution**: Ensure state is 110D full MuJoCo state, not 8D EEF state

### Issue: Gripper state not changing
- **Cause**: Gripper command range mismatch
- **Solution**: Check if gripper uses [-1, 1] or [0, 1] convention

## Next Steps

Proceed to **03_inverse_action_concepts.md** for detailed mathematical formulation.
