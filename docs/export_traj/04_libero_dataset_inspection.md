# LIBERO Dataset Inspection and Structure

## Overview

LIBERO provides two complementary data formats:
1. **HDF5 demos**: Full MuJoCo simulator state + observations (used for sim-based inverse action testing)
2. **TensorFlow Datasets (TFDS)**: Processed data with images and compact state (used for static inverse action testing)

This document explains how to explore and work with both formats.

## HDF5 Demo Files

### Location

```
~/Scene-Graph-VLA/sandbox/pick_up_the_soup_and_place_it_in_the_basket_demo.hdf5
```

### File Structure

```
pick_up_the_soup_and_place_it_in_the_basket_demo.hdf5
├── demo_0/
│   ├── states                    (148, 110)    [Full MuJoCo state]
│   ├── actions                   (147, 7)      [Actions taken]
│   ├── obs/
│   │   ├── ee_pos                (148, 3)      [EEF position]
│   │   ├── ee_ori                (148, 3)      [EEF orientation, axis-angle]
│   │   ├── gripper_states        (148, 2)      [Gripper finger positions]
│   │   ├── joint_states          (148, 7)      [Panda joint angles]
│   │   └── images/
│   │       └── frontview         (148, 256, 256, 3)  [RGB observations]
│   └── (optional other fields)
```

### Loading HDF5 Data

```python
import h5py
import numpy as np

filepath = "/home/dhruv/Scene-Graph-VLA/sandbox/pick_up_the_soup_and_place_it_in_the_basket_demo.hdf5"

def inspect_hdf5_demo(filepath, demo_idx=0):
    """Inspect HDF5 demo structure and contents"""
    
    with h5py.File(filepath, 'r') as f:
        demo_key = f"demo_{demo_idx}"
        
        print(f"=== HDF5 Demo: {demo_key} ===\n")
        
        # States
        states = f[f"{demo_key}/states"][:]
        print(f"States shape: {states.shape}")  # (T, 110)
        print(f"  - Full MuJoCo state for exact reset")
        print(f"  - Includes: joint positions, velocities, and other internal state\n")
        
        # Actions
        actions = f[f"{demo_key}/actions"][:]
        print(f"Actions shape: {actions.shape}")  # (T-1, 7)
        print(f"  - 7D: [Δx, Δy, Δz, ωx, ωy, ωz, gripper]")
        print(f"  - Sample action: {actions[0]}\n")
        
        # Observations
        print("Observations:")
        
        # EEF position
        ee_pos = f[f"{demo_key}/obs/ee_pos"][:]
        print(f"  - EEF position shape: {ee_pos.shape}")  # (T, 3)
        print(f"    Sample: {ee_pos[0]}\n")
        
        # EEF orientation
        ee_ori = f[f"{demo_key}/obs/ee_ori"][:]
        print(f"  - EEF orientation shape: {ee_ori.shape}")  # (T, 3)
        print(f"    Format: axis-angle (ωx, ωy, ωz)")
        print(f"    Sample: {ee_ori[0]}\n")
        
        # Gripper state
        gripper_states = f[f"{demo_key}/obs/gripper_states"][:]
        print(f"  - Gripper states shape: {gripper_states.shape}")  # (T, 2)
        print(f"    Range: [0, 1] for open/close")
        print(f"    Sample: {gripper_states[0]}\n")
        
        # Joint states
        joint_states = f[f"{demo_key}/obs/joint_states"][:]
        print(f"  - Joint states shape: {joint_states.shape}")  # (T, 7)
        print(f"    Panda arm has 7 DOF")
        print(f"    Sample: {joint_states[0]}\n")
        
        # Images
        images = f[f"{demo_key}/obs/images/frontview"][:]
        print(f"  - Images shape: {images.shape}")  # (T, 256, 256, 3)
        print(f"    RGB format, uint8\n")
        
        # Trajectory length
        print(f"Trajectory length: {len(states)} steps")
        print(f"Trajectory duration: {len(states) * 0.05:.2f} seconds (at 20 Hz)")

# Usage
inspect_hdf5_demo(filepath)
```

### Extracting 8D EEF State

```python
def extract_eef_state_from_hdf5(filepath, demo_idx=0, step_idx=0):
    """Extract 8D EEF state from HDF5 demo"""
    
    with h5py.File(filepath, 'r') as f:
        demo_key = f"demo_{demo_idx}"
        
        # Get individual components
        ee_pos = f[f"{demo_key}/obs/ee_pos"][step_idx]  # (3,)
        ee_ori = f[f"{demo_key}/obs/ee_ori"][step_idx]  # (3,) - axis-angle
        gripper_states = f[f"{demo_key}/obs/gripper_states"][step_idx]  # (2,)
        
        # Concatenate to 8D state
        eef_state = np.concatenate([ee_pos, ee_ori, gripper_states])
    
    return eef_state  # Shape: (8,)

# Usage
s_0 = extract_eef_state_from_hdf5(filepath, demo_idx=0, step_idx=0)
print(f"8D EEF state at step 0: {s_0}")
print(f"Shape: {s_0.shape}")
```

### Loading Full State for Reset

```python
def get_full_state_for_reset(filepath, demo_idx=0, step_idx=0):
    """Get 110D full state for simulator reset"""
    
    with h5py.File(filepath, 'r') as f:
        demo_key = f"demo_{demo_idx}"
        state_110d = f[f"{demo_key}/states"][step_idx]  # (110,)
    
    return state_110d

# Usage
state_full = get_full_state_for_reset(filepath, demo_idx=0, step_idx=1)
print(f"Full state shape: {state_full.shape}")
```

## TensorFlow Datasets (TFDS)

### Available Splits

LIBERO provides TFDS for easier integration with TensorFlow pipelines:

```python
import tensorflow_datasets as tfds

# List available datasets
for dataset_name in ['libero_object', 'libero_object_no_noops', 'libero_goal', 'libero_100']:
    print(f"\n{dataset_name}:")
    info = tfds.builder(dataset_name).info
    print(f"  - Splits: {list(info.splits.keys())}")
    print(f"  - Features: {info.features}")
```

### Loading TFDS Data

```python
def load_tfds_episode(dataset_name='libero_object_no_noops', episode_idx=0):
    """Load a LIBERO episode from TFDS"""
    
    dataset = tfds.load(
        dataset_name,
        split='train',
        shuffle_files=False,
    )
    
    # Convert to list
    episodes = list(dataset)
    
    if episode_idx >= len(episodes):
        raise ValueError(f"Episode {episode_idx} not found (only {len(episodes)} available)")
    
    episode = episodes[episode_idx]
    return episode

# Usage
episode = load_tfds_episode('libero_object_no_noops', episode_idx=0)
print(f"Episode keys: {episode.keys()}")
```

### Episode Structure

```python
def inspect_tfds_episode(episode):
    """Inspect TFDS episode structure"""
    
    print("=== TFDS Episode ===\n")
    
    # Trajectory data
    for key in episode.keys():
        value = episode[key]
        if isinstance(value, dict):
            print(f"{key}:")
            for subkey, subvalue in value.items():
                if hasattr(subvalue, 'shape'):
                    print(f"  - {subkey}: {subvalue.shape} {subvalue.dtype}")
                else:
                    print(f"  - {subkey}: {type(subvalue)}")
        elif hasattr(value, 'shape'):
            print(f"{key}: {value.shape} {value.dtype}")
        else:
            print(f"{key}: {type(value)}")

# Example output:
# Episode 0 (libero_object_no_noops):
# 
# language_instruction: b'pick up the soup and place it in the basket'
# 
# steps:
#   - observation: (143, ...)
#     - state: (143, 8)          [8D EEF state]
#     - joint_state: (143, 7)    [Joint angles]
#     - images: (143, 256, 256, 3)  [RGB images]
#   - action: (143, 7)           [7D actions]
#   - is_terminal: (143,)        [Trajectory end marker]
```

### Extracting State and Action

```python
def extract_tfds_trajectory(episode):
    """Extract state and action sequences from TFDS episode"""
    
    steps = episode['steps']
    
    # Extract compact state (8D EEF)
    states = steps['observation']['state']  # (T, 8)
    
    # Extract actions
    actions = steps['action']  # (T, 7)
    
    # Extract language instruction
    instruction = episode['language_instruction'].decode('utf-8')
    
    return {
        'states': states.numpy(),
        'actions': actions.numpy(),
        'instruction': instruction,
        'length': len(states),
    }

# Usage
episode = load_tfds_episode('libero_object_no_noops', episode_idx=0)
trajectory = extract_tfds_trajectory(episode)

print(f"Instruction: {trajectory['instruction']}")
print(f"Trajectory length: {trajectory['length']} steps")
print(f"States shape: {trajectory['states'].shape}")
print(f"Actions shape: {trajectory['actions'].shape}")
```

## Comparison: HDF5 vs. TFDS

| Feature | HDF5 | TFDS |
|---------|------|------|
| **Full state** | ✓ (110D) | ✗ (8D EEF only) |
| **Observations** | ✓ (EEF, joint, images) | ✓ (state, joint, images) |
| **Simulator reset** | ✓ Exact | ✗ Not possible |
| **File size** | Large (~500MB per demo) | Smaller (~100MB per episode) |
| **Access pattern** | Random (good for testing) | Sequential (good for training) |
| **Use case** | Sim-based inverse actions | Static inverse actions, VLA training |

## Data Statistics

### HDF5 Demo Statistics

```python
def compute_hdf5_statistics(filepath, demo_idx=0):
    """Compute statistics for HDF5 demo"""
    
    with h5py.File(filepath, 'r') as f:
        demo_key = f"demo_{demo_idx}"
        
        # Load trajectory
        states = f[f"{demo_key}/obs/ee_pos"][:]  # (T, 3)
        actions = f[f"{demo_key}/actions"][:]    # (T-1, 7)
        
        # Compute statistics
        print(f"=== Statistics for {demo_key} ===\n")
        
        print("EEF Position:")
        print(f"  - Mean: {states.mean(axis=0)}")
        print(f"  - Std: {states.std(axis=0)}")
        print(f"  - Range: [{states.min(axis=0)}, {states.max(axis=0)}]\n")
        
        print("Actions:")
        print(f"  - Mean: {actions.mean(axis=0)}")
        print(f"  - Std: {actions.std(axis=0)}")
        print(f"  - Range: [{actions.min(axis=0)}, {actions.max(axis=0)}]\n")

# Usage
compute_hdf5_statistics(filepath)
```

### TFDS Episode Statistics

```python
def compute_tfds_statistics(dataset_name='libero_object_no_noops', num_episodes=10):
    """Compute statistics across TFDS episodes"""
    
    dataset = tfds.load(dataset_name, split='train')
    episodes = list(dataset.take(num_episodes))
    
    all_states = []
    all_actions = []
    
    for episode in episodes:
        trajectory = extract_tfds_trajectory(episode)
        all_states.append(trajectory['states'])
        all_actions.append(trajectory['actions'])
    
    states = np.concatenate(all_states, axis=0)
    actions = np.concatenate(all_actions, axis=0)
    
    print(f"=== TFDS Statistics ({num_episodes} episodes) ===\n")
    
    print("States (8D EEF):")
    print(f"  - Mean: {states.mean(axis=0)}")
    print(f"  - Std: {states.std(axis=0)}\n")
    
    print("Actions (7D):")
    print(f"  - Mean: {actions.mean(axis=0)}")
    print(f"  - Std: {actions.std(axis=0)}\n")

# Usage
compute_tfds_statistics(num_episodes=10)
```

## Practical Notes

### When to use HDF5
- Testing inverse actions with exact simulator reset
- Debugging physics interactions
- Computing ground-truth measurements
- Small-scale experiments

### When to use TFDS
- Training VLA models (native TF format)
- Large-scale data loading (efficient streaming)
- Multi-demo aggregation
- Production pipelines

### Memory Considerations

```python
# HDF5: Load entire demo into memory
with h5py.File(filepath) as f:
    states = f['demo_0/states'][:]  # Loads all 148 states to RAM

# TFDS: Stream data efficiently
dataset = tfds.load('libero_object_no_noops', split='train')
for episode in dataset:  # One episode at a time
    ...
```

## Next Steps

Proceed to **05_sim_inverse_action_test.md** for simulator-based inverse action testing.
