# Next Steps: Trajectory Augmentation and Integration

## Overview

Now that we've validated simulator-based inverse actions as accurate and reliable (~2% error), we can integrate this into a trajectory augmentation pipeline for VLA training. This document outlines:

1. Implementation roadmap
2. Trajectory augmentation strategies
3. Integration with VLA training
4. Validation and testing procedures
5. Performance monitoring

## Phase 1: Multi-Demo Validation

Before integrating into training, validate reversibility across different LIBERO tasks.

### Objective

Ensure inverse action reversibility is consistent across:
- Different task types (pick, place, sort, etc.)
- Different object properties (shape, size, weight)
- Different trajectory lengths

### Implementation

```python
import h5py
from pathlib import Path

def validate_inverse_action_across_demos(demo_dir, num_demos=5):
    """Test inverse actions on multiple HDF5 demos"""
    
    results_summary = {}
    
    demo_files = list(Path(demo_dir).glob("*.hdf5"))[:num_demos]
    
    for demo_file in demo_files:
        print(f"\nTesting {demo_file.name}...")
        
        # Load trajectory
        with h5py.File(demo_file, 'r') as f:
            demo_key = 'demo_0'
            states = f[f"{demo_key}/states"][:]
            actions = f[f"{demo_key}/actions"][:]
        
        # Run inverse action test (use first 30 steps)
        results = test_inverse_action_simulator(env, {
            'states_110d': states[:30],
            'actions': actions[:30],
            'ee_pos': ...,  # Load from HDF5
            'ee_ori': ...,
            'gripper_states': ...,
        })
        
        stats = compute_statistics(results)
        results_summary[demo_file.name] = stats
        
        print(f"  Mean error: {stats['mean_error']:.6f} m ({stats['error_as_pct']:.2f}%)")
    
    return results_summary

# Validation
validation_results = validate_inverse_action_across_demos(
    '/home/dhruv/Scene-Graph-VLA/sandbox',
    num_demos=5
)

# Print summary
print("\n=== Validation Summary ===")
for demo, stats in validation_results.items():
    print(f"{demo}: {stats['mean_error']:.6f} m ({stats['error_as_pct']:.2f}%)")
```

### Success Criteria

- All demos: error < 0.1 m (2-3% of state norm)
- Consistent across tasks
- No systematic drift over trajectory

## Phase 2: Trajectory Augmentation Pipeline

### Architecture

```
Original HDF5 Demo
    ↓
Sample random step t in trajectory
    ↓
Sample random perturbation δa
    ↓
Re-execute trajectory from step t with perturbed action
    ↓
Use inverse action to "return" to original trajectory
    ↓
Record augmented trajectory
```

### Implementation

```python
import numpy as np
import h5py

class TrajectoryAugmenter:
    """Augment trajectories using simulator-based inverse actions"""
    
    def __init__(self, env, num_augmentations=3, perturbation_std=0.005):
        """
        Args:
            env: Robosuite environment
            num_augmentations: Number of augmented trajectories per demo
            perturbation_std: Std dev of Gaussian perturbation (meters/radians)
        """
        self.env = env
        self.num_augmentations = num_augmentations
        self.perturbation_std = perturbation_std
    
    def sample_perturbation(self, action_template):
        """Sample small random perturbation"""
        perturbation = np.random.normal(0, self.perturbation_std, size=7)
        
        # Clamp to reasonable ranges
        perturbation[:3] = np.clip(perturbation[:3], -0.01, 0.01)  # ±1cm
        perturbation[3:6] = np.clip(perturbation[3:6], -0.05, 0.05)  # ±0.05 rad
        perturbation[6] = 0  # Don't perturb gripper
        
        return perturbation
    
    def augment_trajectory(self, states_110d, actions, obs_ee_pos, obs_ee_ori, 
                           obs_gripper, language_instruction):
        """
        Augment a single trajectory
        
        Args:
            states_110d: (T, 110) full MuJoCo states
            actions: (T-1, 7) actions
            obs_ee_pos: (T, 3) EEF positions
            obs_ee_ori: (T, 3) EEF orientations
            obs_gripper: (T, 2) gripper states
            language_instruction: Task instruction text
        
        Returns:
            augmented_trajectories: List of new trajectories
        """
        
        augmented = []
        T = len(states_110d)
        
        for aug_idx in range(self.num_augmentations):
            # Sample insertion point (avoid first/last few steps)
            insert_step = np.random.randint(5, max(6, T - 10))
            
            # Initialize augmented trajectory as copy
            aug_states = states_110d.copy()
            aug_actions = actions.copy()
            aug_obs_ee_pos = obs_ee_pos.copy()
            aug_obs_ee_ori = obs_ee_ori.copy()
            aug_obs_gripper = obs_gripper.copy()
            
            # Sample perturbation
            delta_a = self.sample_perturbation(actions[insert_step])
            
            # Reset to insertion point
            reset_to_state(self.env, states_110d[insert_step])
            
            # Apply perturbed action
            perturbed_action = actions[insert_step] + delta_a
            perturbed_action = np.clip(perturbed_action, -1, 1)  # Clamp to action bounds
            
            self.env.step(perturbed_action)
            s_perturbed = extract_eef_state(self.env)
            
            # Store perturbed state
            aug_obs_ee_pos[insert_step + 1] = s_perturbed[:3]
            aug_obs_ee_ori[insert_step + 1] = s_perturbed[3:6]
            aug_obs_gripper[insert_step + 1] = s_perturbed[6:8]
            
            # Apply inverse action to "return"
            inverse_action = delta_a.copy()
            inverse_action[:6] *= -1
            
            self.env.step(inverse_action)
            s_recovered = extract_eef_state(self.env)
            
            # Optionally apply small additional correction
            correction_error = np.linalg.norm(
                aug_obs_ee_pos[insert_step] - s_recovered[:3]
            )
            
            if correction_error > 0.02:  # If recovery error > 2cm
                # Apply small corrective action
                correction = (aug_obs_ee_pos[insert_step] - s_recovered[:3]) * 0.1
                correction_action = np.zeros(7)
                correction_action[:3] = correction
                self.env.step(correction_action)
                s_recovered = extract_eef_state(self.env)
            
            # Continue forward from recovered state
            for t in range(insert_step + 1, T - 1):
                self.env.step(actions[t])
                s_t = extract_eef_state(self.env)
                
                # Update observation
                aug_obs_ee_pos[t] = s_t[:3]
                aug_obs_ee_ori[t] = s_t[3:6]
                aug_obs_gripper[t] = s_t[6:8]
            
            augmented.append({
                'states_110d': aug_states,
                'actions': aug_actions,
                'obs_ee_pos': aug_obs_ee_pos,
                'obs_ee_ori': aug_obs_ee_ori,
                'obs_gripper': aug_obs_gripper,
                'language_instruction': language_instruction,
                'augmentation_type': 'inverse_action_perturbation',
                'perturbation_step': insert_step,
                'perturbation': delta_a,
            })
        
        return augmented
    
    def augment_dataset(self, input_dir, output_dir, num_demos=None):
        """Augment all HDF5 demos in directory"""
        
        Path(output_dir).mkdir(exist_ok=True, parents=True)
        
        demo_files = sorted(Path(input_dir).glob("*.hdf5"))
        if num_demos:
            demo_files = demo_files[:num_demos]
        
        for demo_idx, demo_file in enumerate(demo_files):
            print(f"\nAugmenting {demo_file.name} ({demo_idx+1}/{len(demo_files)})...")
            
            # Load original trajectory
            with h5py.File(demo_file, 'r') as f:
                demo_key = 'demo_0'
                states = f[f"{demo_key}/states"][:]
                actions = f[f"{demo_key}/actions"][:]
                obs_ee_pos = f[f"{demo_key}/obs/ee_pos"][:]
                obs_ee_ori = f[f"{demo_key}/obs/ee_ori"][:]
                obs_gripper = f[f"{demo_key}/obs/gripper_states"][:]
            
            # Augment
            augmented = self.augment_trajectory(
                states, actions, obs_ee_pos, obs_ee_ori, obs_gripper,
                language_instruction="Original trajectory"
            )
            
            # Save augmented trajectories
            output_file = Path(output_dir) / f"{demo_file.stem}_augmented.hdf5"
            with h5py.File(output_file, 'w') as f:
                for aug_idx, aug_traj in enumerate(augmented):
                    aug_key = f"demo_{aug_idx}"
                    
                    f.create_dataset(f"{aug_key}/states", data=aug_traj['states_110d'])
                    f.create_dataset(f"{aug_key}/actions", data=aug_traj['actions'])
                    f.create_group(f"{aug_key}/obs")
                    f.create_dataset(f"{aug_key}/obs/ee_pos", data=aug_traj['obs_ee_pos'])
                    f.create_dataset(f"{aug_key}/obs/ee_ori", data=aug_traj['obs_ee_ori'])
                    f.create_dataset(f"{aug_key}/obs/gripper_states", 
                                    data=aug_traj['obs_gripper'])
            
            print(f"  Saved {len(augmented)} augmented trajectories to {output_file}")

# Usage
augmenter = TrajectoryAugmenter(env, num_augmentations=3, perturbation_std=0.005)
augmenter.augment_dataset(
    input_dir='/home/dhruv/Scene-Graph-VLA/sandbox',
    output_dir='/home/dhruv/Scene-Graph-VLA/sandbox_augmented',
    num_demos=10
)
```

## Phase 3: VLA Training Integration

### Dataset Configuration

Update your VLA dataset loader to include both original and augmented trajectories:

```python
class LiberoAugmentedDataset:
    """LIBERO dataset with trajectory augmentation"""
    
    def __init__(self, original_dir, augmented_dir, split='train', 
                 augmentation_ratio=0.5):
        """
        Args:
            original_dir: Path to original HDF5 demos
            augmented_dir: Path to augmented HDF5 demos
            augmentation_ratio: Fraction of batch from augmented data
        """
        self.original_dir = Path(original_dir)
        self.augmented_dir = Path(augmented_dir)
        self.augmentation_ratio = augmentation_ratio
        
        # Load file lists
        self.original_files = sorted(
            self.original_dir.glob("*.hdf5")
        )
        self.augmented_files = sorted(
            self.augmented_dir.glob("*.hdf5")
        )
    
    def __len__(self):
        # Original + augmented data
        num_original = len(self.original_files)
        num_augmented = len(self.augmented_files)
        return num_original + num_augmented
    
    def __getitem__(self, idx):
        # Sample from original or augmented based on ratio
        if np.random.rand() < self.augmentation_ratio:
            # Sample from augmented
            file_idx = np.random.randint(len(self.augmented_files))
            return self._load_from_hdf5(self.augmented_files[file_idx])
        else:
            # Sample from original
            file_idx = np.random.randint(len(self.original_files))
            return self._load_from_hdf5(self.original_files[file_idx])
    
    def _load_from_hdf5(self, filepath):
        """Load single trajectory from HDF5"""
        with h5py.File(filepath, 'r') as f:
            demo_key = f"demo_{np.random.randint(len(f))}"
            
            states = f[f"{demo_key}/obs/ee_pos"][:]
            actions = f[f"{demo_key}/actions"][:]
            images = f[f"{demo_key}/obs/images/frontview"][:]  # If available
            
            return {
                'state': states,
                'action': actions,
                'images': images,
            }
```

### Training Loop

```python
def train_vla_with_augmentation(model, device, num_epochs=10, 
                                augmentation_ratio=0.5):
    """Train VLA with augmented trajectories"""
    
    # Create dataset
    dataset = LiberoAugmentedDataset(
        original_dir='/home/dhruv/Scene-Graph-VLA/sandbox',
        augmented_dir='/home/dhruv/Scene-Graph-VLA/sandbox_augmented',
        augmentation_ratio=augmentation_ratio,
    )
    
    loader = DataLoader(dataset, batch_size=32, shuffle=True)
    
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
    
    for epoch in range(num_epochs):
        print(f"Epoch {epoch + 1}/{num_epochs}")
        
        total_loss = 0
        num_batches = 0
        
        for batch_idx, batch in enumerate(loader):
            # Forward pass
            states = batch['state'].to(device)
            actions = batch['action'].to(device)
            images = batch['images'].to(device)
            
            # Predict actions
            predicted_actions = model(images, states)
            
            # Compute loss
            loss = F.mse_loss(predicted_actions, actions)
            
            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            num_batches += 1
            
            if batch_idx % 100 == 0:
                print(f"  Batch {batch_idx}: Loss = {loss.item():.6f}")
        
        avg_loss = total_loss / num_batches
        print(f"  Epoch average loss: {avg_loss:.6f}")
```

## Phase 4: Validation and Evaluation

### Metrics to Track

```python
def evaluate_augmentation_effects(model_original, model_augmented, 
                                  test_dataset):
    """Compare models trained with/without augmentation"""
    
    metrics = {
        'action_mse': [],
        'trajectory_error': [],
        'robustness_score': [],
    }
    
    for trajectory in test_dataset:
        states = trajectory['state']
        actions = trajectory['action']
        
        # Predictions from both models
        pred_original = model_original(states)
        pred_augmented = model_augmented(states)
        
        # Action prediction error
        mse_original = np.mean((actions - pred_original) ** 2)
        mse_augmented = np.mean((actions - pred_augmented) ** 2)
        
        metrics['action_mse'].append({
            'original': mse_original,
            'augmented': mse_augmented,
        })
    
    return metrics
```

### A/B Testing Framework

```python
def compare_model_performance(model_with_aug, model_without_aug, 
                             test_env, num_trials=10):
    """Compare model robustness in simulation"""
    
    results = {
        'success_rate_with_aug': 0,
        'success_rate_without_aug': 0,
        'final_error_with_aug': [],
        'final_error_without_aug': [],
    }
    
    for trial in range(num_trials):
        # Execute tasks with both models
        success_with, error_with = execute_task(model_with_aug, test_env)
        success_without, error_without = execute_task(model_without_aug, test_env)
        
        if success_with:
            results['success_rate_with_aug'] += 1
        if success_without:
            results['success_rate_without_aug'] += 1
        
        results['final_error_with_aug'].append(error_with)
        results['final_error_without_aug'].append(error_without)
    
    # Compute statistics
    results['success_rate_with_aug'] /= num_trials
    results['success_rate_without_aug'] /= num_trials
    
    print(f"Success rate with augmentation: {results['success_rate_with_aug']:.2%}")
    print(f"Success rate without augmentation: {results['success_rate_without_aug']:.2%}")
    
    return results
```

## Phase 5: Production Integration

### Checklist

- [ ] Multi-demo validation shows <0.1 m error across tasks
- [ ] Augmentation pipeline produces realistic trajectories
- [ ] VLA training integrates augmented data smoothly
- [ ] A/B testing shows improvement (>5% success rate increase)
- [ ] Inference speed acceptable (<50ms per prediction)
- [ ] Augmented data follows original distribution
- [ ] Error analysis shows no systematic bias

### Deployment

```bash
# Generate augmented dataset
python scripts/augment_trajectories.py \
  --input_dir sandbox \
  --output_dir sandbox_augmented \
  --num_augmentations 3 \
  --perturbation_std 0.005

# Train VLA with augmentation
python scripts/train_vla.py \
  --dataset libero_object_no_noops \
  --augmentation_ratio 0.5 \
  --num_epochs 50 \
  --batch_size 32

# Evaluate performance
python scripts/evaluate_vla.py \
  --model_path checkpoints/vla_with_aug.pth \
  --test_dataset libero_object_no_noops/test \
  --num_trials 20
```

## Conclusion

With simulator-based inverse actions validated, we can now:

1. **Augment trajectories** safely with ~2% perturbation error
2. **Improve VLA training** through data diversity
3. **Increase robustness** to real-world variations
4. **Measure improvements** systematically

The 10× accuracy improvement over static methods justifies the computational cost of simulation-based augmentation for high-quality VLA model training.

## References

- Implementation scripts: `/home/dhruv/Scene-Graph-VLA/scripts/`
- Augmented data: `/home/dhruv/Scene-Graph-VLA/sandbox_augmented/`
- Training configs: `/home/dhruv/Scene-Graph-VLA/configs/`
