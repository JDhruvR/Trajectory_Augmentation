# Trajectory Inverse Actions Documentation

Complete guide to understanding, implementing, and deploying simulator-based trajectory reversibility for VLA training.

## Quick Navigation

### Getting Started
1. **[00_overview.md](00_overview.md)** - Project summary and key findings
   - What is an inverse action?
   - Why it matters for trajectory augmentation
   - Comparison of methods

2. **[01_setup.md](01_setup.md)** - Environment setup
   - Conda environment creation
   - Dependency installation
   - Verification procedures

### Theory and Background
3. **[02_simulator_setup.md](02_simulator_setup.md)** - Robosuite configuration
   - Environment creation
   - State/action spaces
   - Controller parameters
   - HDF5 integration

4. **[03_inverse_action_concepts.md](03_inverse_action_concepts.md)** - Mathematical foundations
   - Forward model definitions
   - Inverse action theory
   - Simulator vs. static methods
   - Advanced considerations

### Data Exploration
5. **[04_libero_dataset_inspection.md](04_libero_dataset_inspection.md)** - Dataset structure
   - HDF5 format overview
   - TFDS integration
   - Data loading utilities
   - Statistics computation

### Implementation and Testing
6. **[05_sim_inverse_action_test.md](05_sim_inverse_action_test.md)** - Simulator-based testing
   - Complete implementation guide
   - Helper functions
   - Results saving procedures
   - Visualization

7. **[06_static_inverse_action_test.md](06_static_inverse_action_test.md)** - Static method baseline
   - Kinematic-only approach
   - Why it fails
   - Expected errors
   - Comparison framework

### Analysis and Results
8. **[07_results_analysis.md](07_results_analysis.md)** - Results comparison
   - Simulator vs. static statistics
   - Component breakdown
   - Error sources
   - Practical implications

### Future Work
9. **[08_next_steps.md](08_next_steps.md)** - Implementation roadmap
   - Multi-demo validation
   - Trajectory augmentation pipeline
   - VLA training integration
   - Evaluation framework

## Key Results

| Method | Mean Error | Error % | Suitable? |
|--------|-----------|---------|-----------|
| **Simulator** | 0.060 m | 1.9% | ✓ YES |
| **Static** | 0.613 m | 19.2% | ✗ NO |

## Quick Start

```bash
# Activate environment
conda activate qwen-vla

# Run simulator test
python -c "
import robosuite as suite
from export_traj import test_inverse_action_simulator
# [See 05_sim_inverse_action_test.md for complete code]
"

# Run static test
python -c "
import tensorflow_datasets as tfds
from export_traj import test_inverse_action_static
# [See 06_static_inverse_action_test.md for complete code]
"
```

## Project Structure

```
export_traj/
├── README.md (this file)
├── 00_overview.md
├── 01_setup.md
├── 02_simulator_setup.md
├── 03_inverse_action_concepts.md
├── 04_libero_dataset_inspection.md
├── 05_sim_inverse_action_test.md
├── 06_static_inverse_action_test.md
├── 07_results_analysis.md
└── 08_next_steps.md
```

## Related Files

- **Implementation**: `~/Scene-Graph-VLA/scripts/` (to be created)
- **HDF5 Demos**: `~/Scene-Graph-VLA/sandbox/`
- **Results**: `~/Scene-Graph-VLA/report_mds/`
- **Augmented Data**: `~/Scene-Graph-VLA/sandbox_augmented/` (to be created)

## Key Concepts

### Inverse Action
An action that negates the effect of a previous action. Used for:
- Trajectory perturbations
- Data augmentation
- Trajectory reversal testing

### End-Effector (EEF) State
8-dimensional representation:
- Position (3D): x, y, z
- Orientation (3D): ωx, ωy, ωz (axis-angle)
- Gripper (2D): left and right finger positions

### State Transition
- **Forward**: `s_{t+1} = f(s_t, a_t)`
- **Inverse**: `s_t ≈ f(s_{t+1}, -a_t)`

## Important Notes

1. **Always use qwen-vla environment** for all work
2. **Use robosuite 1.4.1** (not 1.5.2+ which removed SingleArmEnv)
3. **Inverse gripper command** - keep gripper unchanged in inverse actions
4. **Limit trajectory length** - errors accumulate over long sequences
5. **Test on multiple tasks** - validate generalization

## Troubleshooting

**Q: ModuleNotFoundError: No module named 'robosuite'**
A: Make sure conda environment is activated: `conda activate qwen-vla`

**Q: MuJoCo Error: unknown body**
A: Check robot initialization in `suite.make()` - ensure correct robot name

**Q: State reset fails**
A: Verify state is 110D full MuJoCo state, not 8D EEF state

**Q: Gripper state not updating**
A: Check gripper command range; some environments use [-1,1], others [0,1]

## References

- Robosuite: https://robosuite.ai/
- LIBERO: https://libero.readthedocs.io/
- MuJoCo: https://mujoco.org/
- Scene-Graph-VLA Project: `~/Scene-Graph-VLA/`

## Contact and Updates

For questions or updates, refer to individual markdown files for detailed sections.
