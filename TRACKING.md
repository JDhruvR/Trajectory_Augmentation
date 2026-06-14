# Trajectory Augmentation Tracking & Setup Guide

This document tracks the current state of the repository, environment setup instructions, and the correct commands for validating inverse actions.

## 1. Environment Setup

The repository relies on the `qwen-vla` conda environment and the LIBERO benchmark dependencies.

### Conda Environment
Activate the required conda environment:
```bash
conda activate qwen-vla
```

### Installing LIBERO
The `third_party/LIBERO` package must be installed in editable mode:
```bash
cd third_party/LIBERO
pip install -e .
```

## 2. Running Inverse Action Validation

We validate the inverse actions by applying the reverse action physically in a MuJoCo simulation to ensure we arrive back at the original state (`s1`). 

**Crucial Configuration:** MuJoCo headless rendering often hangs over SSH. You *must* run these commands with hardware-accelerated EGL explicitly enabled.

### Run Single Demonstration Test
```bash
MUJOCO_GL=egl EGL_DEVICE_ID=0 conda run -n qwen-vla python scripts/test_inverse_action_simulator.py
```

### Run Batch Validation (across multiple demos)
```bash
MUJOCO_GL=egl EGL_DEVICE_ID=0 conda run -n qwen-vla python scripts/test_inverse_action_batch.py
```

## 3. Dataset Compatibility Notes & Learnings

The repository was migrated from an older project (`SGVLA-Export`) that operated on the `libero_object` dataset. The current codebase operates on the `libero_goal` dataset.

### Key Technical Differences:
1. **State Dimensionality:**
   - `libero_object` (old): 110-dimensional state (`PickPlaceBread` environment).
   - `libero_goal` (current): 79-dimensional state (`Libero_Tabletop_Manipulation` environment).
2. **Environment Initialization:**
   - Previous versions hardcoded `PickPlaceBread`, which caused massive tracking failures (34% error) when applied to the 79-dimensional `libero_goal` dataset because of dimension layout mismatches.
   - The current `scripts/` now dynamically parse the exact `.bddl` file from the demonstration's HDF5 metadata and dynamically instantiate the correct `Libero_Tabletop_Manipulation` environment.
3. **State Recovery Mechanism:**
   - Previous scripts manually sliced positions and velocities `[:nq]` and `[nq:nq+nv]`. In a 79-dimensional layout, the first index `[0]` represents `time`. Manual slicing shifted all positions by 1 index, causing catastrophic physics failures.
   - We now strictly use `sim.set_state_from_flattened(target_state)`, which is perfectly robust to any underlying dimension layout changes.

### Current Status
**Validation is PASSING**. The dynamic loading methodology achieves a reconstruction error of exactly **~1.87%**, perfectly aligning with the "properly right" baseline from the original `SGVLA-Export` reference.

## 4. Repository Structure

- `scripts/`: Contains the correct, dynamic scripts for `libero_goal`.
- `SGVLA-Export/`: The untouched reference implementation from the previous project.
- `report_mds/`: Output directory where metrics, statistics, and visual histograms are saved after running the simulator tests.
- `third_party/`: Contains the forked LIBERO source code.
