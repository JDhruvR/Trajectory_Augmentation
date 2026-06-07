# Trajectory Inverse Actions: Project Overview

## Goal

Understand LIBERO dataset state/action semantics, test trajectory reversibility via inverse actions (using both simulator and static methods), and validate perturbation feasibility for trajectory augmentation in vision-language action (VLA) learning.

## What is an Inverse Action?

An **inverse action** is the negation of the original action applied to a state to attempt to reverse/undo the trajectory. In the context of this project:

- Given: initial state `s1`, action `a1`, resulting state `s2`
- Forward: `s1 + a1 → s2`
- Inverse: `s2 + (-a1) → s1_reconstructed`

The goal is to measure how close `s1_reconstructed` is to the original `s1` using L2 distance.

### Why is this important?

For **trajectory augmentation**, we want to perturb trajectories by:
1. Applying a random perturbation action to a state in the middle of a trajectory
2. Using inverse actions to "undo" the perturbation and return to the original trajectory
3. This creates slightly modified trajectories for data augmentation and improved VLA model robustness

## Project Scope

This project investigates two methods for trajectory reversibility:

1. **Simulator-based Method (Preferred)**
   - Use MuJoCo physics simulation (via robosuite)
   - Reset to intermediate state and apply inverse action through dynamics
   - More accurate because it follows actual robot physics
   - Result: Mean L2 error ~0.06 (6% of typical state norm)

2. **Static Method (Baseline)**
   - Use kinematic equation: `s1_hat = s2 - a1`
   - Does not account for physics; assumes linear state transitions
   - Simpler but less accurate
   - Result: Mean L2 error ~0.61 (19% of typical state norm)

## Key Components

- **LIBERO Dataset**: Benchmark for vision-guided robot manipulation with language instructions
- **Robosuite 1.4.1**: MuJoCo-based robot simulator with OSC_POSE controller
- **State Representation**: 8D EEF (end-effector) state = [pos_x, pos_y, pos_z, ori_x, ori_y, ori_z, gripper_cmd_1, gripper_cmd_2]
- **Action Representation**: 7D = [delta_pos_x, delta_pos_y, delta_pos_z, delta_ori_x, delta_ori_y, delta_ori_z, gripper_cmd]

## Comparison of Inverse Action Methods

| Metric | Sim-Based | Static |
|--------|-----------|--------|
| Mean ‖s1‖ | 3.137 | 3.198 |
| Mean ‖error‖ | 0.0601 | 0.613 |
| Std ‖error‖ | 0.0133 | 0.239 |
| Error as % of state | ~1.9% | ~19.2% |
| Physics accurate? | Yes | No |
| Computational cost | Higher | Lower |
| Recommendation | ✓ Preferred | Baseline only |

## Quick Facts

- **Simulator**: MuJoCo via robosuite 1.4.1 (not 1.5.2, which removed SingleArmEnv)
- **Conda Environment**: `qwen-vla` (all work done here)
- **Demo Dataset**: HDF5 file with 148 steps from "pick_up_the_soup_and_place_it_in_the_basket" task
- **EEF Orientation**: Represented as axis-angle in actions, converted from/to quaternion in simulator
- **OSC Controller**: Operational space control with delta input reference frame, max outputs [0.05, 0.05, 0.05, 0.5, 0.5, 0.5]

## Documentation Structure

1. **01_setup.md** - Environment and dependencies setup
2. **02_simulator_setup.md** - Robosuite installation and configuration
3. **03_inverse_action_concepts.md** - Detailed theory and math
4. **04_libero_dataset_inspection.md** - Dataset exploration
5. **05_sim_inverse_action_test.md** - Simulator-based testing procedure
6. **06_static_inverse_action_test.md** - Static method testing
7. **07_results_analysis.md** - Results and comparisons
8. **08_next_steps.md** - Future work (trajectory augmentation, multi-demo testing)

## Outputs Generated

All outputs are in `report_mds/`:

- `libero_inverse_action_sim_eef8.txt` - Per-step sim results
- `libero_inverse_action_sim_eef8_stats.txt` - Sim statistics
- `libero_inverse_action_sim_eef8_diff_hist.png` - Sim error histogram
- `libero_inverse_action_static_eef8.txt` - Per-step static results
- `libero_inverse_action_static_eef8_stats.txt` - Static statistics
- `libero_inverse_action_static_eef8_diff_hist.png` - Static error histogram
