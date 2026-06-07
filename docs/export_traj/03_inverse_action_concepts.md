# Inverse Action Concepts: Theory and Mathematics

## Overview

This document explains the theoretical foundation of inverse actions, how they work in simulation vs. static methods, and their use for trajectory augmentation.

## What is an Action?

An action is a command to the robot to change its state. In LIBERO, actions are **7-dimensional delta commands**:

```
a = [Δx, Δy, Δz, ωx, ωy, ωz, g]
```

Where:
- **Δx, Δy, Δz**: Changes in end-effector position (meters)
- **ωx, ωy, ωz**: Changes in end-effector orientation (radians, axis-angle format)
- **g**: Gripper command (±1, open/close)

## Forward Model

Given an initial state and an action, the robot transitions to a new state:

```
s_{t+1} = f(s_t, a_t)
```

Where `f` is the forward dynamics model (implemented in MuJoCo simulator).

### State Representation

We use an 8-dimensional end-effector (EEF) state:

```
s = [x, y, z, ωx, ωy, ωz, g_left, g_right]
```

- **Position** (3D): x, y, z coordinates of EEF
- **Orientation** (3D): ωx, ωy, ωz (axis-angle representation)
- **Gripper** (2D): Left and right finger positions

## The Inverse Action Problem

Given a state `s2` that resulted from applying action `a` to state `s1`, can we recover `s1` by applying `-a`?

```
Forward:  s1 + a → s2
Inverse:  s2 + (-a) → s1_reconstructed
Goal:     ║s1 - s1_reconstructed║ ≈ 0
```

### Why is this useful?

For **trajectory augmentation**, we can:

1. Pick a random point in a demonstration trajectory at time `t`
2. Apply a random perturbation `δa`
3. Use inverse action `-δa` to "undo" the perturbation
4. Create a slightly modified but realistic trajectory

This increases training data diversity without manually collecting new demonstrations.

## Method 1: Simulator-Based Inverse Actions (Preferred)

### How it works

1. **Load HDF5 demo** to get full MuJoCo state `s1_full` (110D)
2. **Reset simulator** to `s1_full`
3. **Apply forward action** `a`: `s1 → s2`
4. **Apply inverse action** `-a`: `s2 → s1_reconstructed`
5. **Extract 8D EEF state** from both `s1` and `s1_reconstructed`
6. **Measure L2 error**: `‖s1 - s1_reconstructed‖`

### Why it's accurate

- Uses **full physics simulation** (MuJoCo)
- Accounts for **non-linear dynamics** (friction, contact forces, joint limits)
- Respects **robot kinematics** and **constraints**
- Gripper state properly modeled in physics

### Mathematical formulation

Let the forward dynamics be:
```
s2 = s1 ⊕ a
```

Where `⊕` represents the composition operator including physics simulation.

The inverse is:
```
s1_reconstructed = s2 ⊕ (-a)
```

The **reversibility error** is:
```
error = ║s1 - s1_reconstructed║
      = ║s1 ⊕ a ⊕ (-a) - s1║
```

This error captures the **compounding effect** of:
- Discretization error (action applied in 50ms steps)
- Friction and damping
- Joint limits and control saturation
- Gripper slippage

### Results

From our experiments (HDF5 demo with 148 steps):

```
Mean ‖s1‖:           3.137 m
Mean ‖error‖:        0.0601 m
Std dev ‖error‖:     0.0133 m
Error as % of state: 1.9%
```

This ~2% error is acceptable for trajectory perturbations because:
- Small perturbations are absorbed by training
- Robot has natural correction behaviors
- EEF tolerances are typically ±1-2 cm

## Method 2: Static Inverse Actions (Baseline)

### How it works

1. **Load TFDS episode** (no full MuJoCo state available)
2. **Assume linear state transition**: `s2 = s1 + a`
3. **Compute inverse statically**: `s1_hat = s2 - a`
4. **Measure L2 error**: `‖s1 - s1_hat‖`

### Mathematical formulation

Assume linear dynamics:
```
s1 + a = s2  (assumed)
```

Then:
```
s1_hat = s2 - a
```

The **static error** is:
```
error = ‖s1 - s1_hat‖
      = ‖s1 - (s2 - a)‖
      = ‖s1 - s2 + a‖
```

### Why it fails

- **Ignores physics**: No friction, damping, or contacts
- **Assumes linearity**: Robot dynamics are highly non-linear
- **Missing coupling**: Position-orientation coupling not modeled
- **Gripper dynamics**: Handled incorrectly (object may deform, slip)

### Results

From our experiments (TFDS episode with 143 steps):

```
Mean ‖s1‖:           3.198 m
Mean ‖error‖:        0.613 m
Std dev ‖error‖:     0.239 m
Error as % of state: 19.2%
```

This ~19% error is **unacceptable** for trajectory perturbations because:
- Perturbations push states far from valid regions
- VLA training learns invalid state-action pairs
- Model becomes confused about valid trajectories

## Comparison: Sim vs. Static

| Aspect | Sim-Based | Static |
|--------|-----------|--------|
| **Physics** | Full MuJoCo dynamics | No physics |
| **Non-linearity** | Properly handled | Ignored (linear assumption) |
| **Error rate** | ~2% | ~19% |
| **Computational cost** | High (50+ sims per demo) | Low (arithmetic) |
| **Accuracy** | High | Low |
| **Recommended** | ✓ YES | Baseline only |

## Advanced Considerations

### Orientation Representation

End-effector orientation can be represented as:
- **Quaternion** (4D): Used in simulator (x, y, z, w)
- **Axis-angle** (3D): Used in actions and state (ωx, ωy, ωz)
- **Euler angles** (3D): Alternative (roll, pitch, yaw)

Conversion:
```python
# Quaternion → Axis-angle
axis_angle = quat2axisangle(quat)  # (3,)

# Axis-angle → Quaternion
quat = axisangle2quat(axis_angle)  # (4,)
```

### Gripper State Handling

The gripper has two components:
- **Command**: What we send (±1)
- **State**: What we observe (finger positions 0-1)

For inverse actions, we keep **command unchanged** because:
- Gripper position depends on object compliance
- Inverting a grasp/release can cause slip
- Conservative approach: don't modify gripper in inverse

```python
# Inverse action construction
inverse_action = action.copy()
inverse_action[:6] *= -1  # Negate position/orientation
# Leave gripper unchanged: inverse_action[6] unchanged
```

### Action Saturation

Robosuite OSC controller has output limits:
```
Position:    [0.05, 0.05, 0.05] m per step
Orientation: [0.5, 0.5, 0.5] rad per step
```

If action exceeds limits, it's clipped:
```
a_clipped = clip(a, a_min, a_max)
```

This **breaks linearity** and introduces systematic error in static inverse.

### Trajectory Perturbation Strategy

Once inverse actions are validated, we can augment trajectories:

1. **Sample perturbation** `δa` from small Gaussian distribution
2. **Insert perturbation** at random time step `t` in trajectory
3. **Re-execute** trajectory from `t` with perturbed action
4. **Record new trajectory** with modified segment

Example:
```
Original:  [s0, a0, s1, a1, s2, a2, s3]
Perturbed: [s0, a0, s1, a1+δa, s2+δs, a2-δa, s3]
           where δs ≈ s2 perturbed by (a1+δa)
           and we use -δa to "return" to original trajectory
```

This creates realistic augmented trajectories for VLA training.

## Practical Considerations

### When to use Sim-Based Inverse

- **Trajectory augmentation** for VLA training (primary use case)
- **Policy imitation learning** from demonstrations
- **Data efficiency improvement** in low-data regimes
- **Robustness testing** for perturbation handling

### When NOT to use Inverse Actions

- **Real robot execution**: Too risky; only use recorded data
- **Long trajectories**: Error accumulates over time
- **Complex contact dynamics**: Non-reversible events (e.g., breaking contact)

### Limitations

1. **Error accumulation**: Multiple inverse steps compound error
2. **Stochasticity**: Sim has randomness (contact, friction parameters)
3. **Gripper limitations**: Can't reliably inverse grasp/release
4. **Task-specific**: Different tasks may have different reversibility

## Next Steps

Proceed to **04_libero_dataset_inspection.md** to explore the datasets.
