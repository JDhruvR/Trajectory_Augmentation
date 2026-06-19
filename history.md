# Repository & Conversation History

This document serves as a succinct handoff summary for future agents or developers regarding the state of this repository.

## Goal
The primary objective of this session was to convert custom augmented LIBERO HDF5 datasets (`libero_goal`, `libero_object`, `libero_spatial`) into the highly efficient **LeRobot v3.0** dataset format (Parquet tabular chunks and SVT-AV1 video chunks) and verify that they are fully compatible with downstream `lerobot_train` scripts.

## Key Accomplishments & Pipeline Decisions

1. **Pivoted from Manual Patching to `datatrove`**:
   - Initial attempts to manually hack the LeRobot schema using custom scripts (`upgrade_schema_v3.py`) caused repeated failures due to strict metadata constraints and indexing issues.
   - We transitioned to a community-supported converter pipeline (`libero_to_lerobot/`) that utilizes `datatrove` for reliable multi-core processing and `SVT-AV1` for efficient temporal video compression.

2. **Native Resolution Adjustments**:
   - The default converter expected source images at `256x256x3`. We modified `libero_to_lerobot/libero2lerobot/libero_h5.py` to natively accept our dataset's `128x128x3` dimensions, skipping unnecessary upscaling and saving massive disk space.

3. **Complete Dataset Conversion**:
   - We successfully converted all 3 augmented suites (approx. 4,500 trajectories).
   - Our massive 45GB uncompressed numpy array dataset was flawlessly compressed down to **~1.44GB** using the AV1/Parquet v3.0 format.
   - Outputs are safely located at `/home/dhruv/Trajectory_Augmentation/data/lerobot_format/`.

4. **Training Dataloader Verification**:
   - Ran `test_dataset_load.py` to verify schema integrity.
   - Successfully executed a 20-step `lerobot_train` offline finetuning smoketest for an ACT policy directly on the converted data, proving complete API compatibility.

5. **Repository Cleanup**:
   - Grouped `generic_converter` and `libero2lerobot` into the clean `libero_to_lerobot/` pipeline directory.
   - Purged obsolete V2-to-V3 conversion scripts, redundant logs, and unused legacy clones.
   - Ignored large dynamic directories (`outputs/`, `data/`, `VLA-Export/`, `third_party/lerobot/`) via `.gitignore` to prevent git hangs.
   - Committed and pushed the finalized, clean state to the `main` branch.

## Directory Structure

*   `data/lerobot_format/`: The official, fully functional LeRobot v3.0 datasets.
*   `libero_to_lerobot/`: The `datatrove` multiprocessing pipeline scripts used for the dataset migration.
*   `scripts/`: Contains the correct, dynamic scripts for running inverse validation on the augmented data and the custom orchestrators (`benchmark_libero_pro.py`).
*   `third_party/`: Contains the forked LIBERO source code and local `lerobot` dependencies.

## Recent Context & Benchmarking (Phase 2)
After the data was converted and models were trained, we pivoted to **Robustness Evaluation** using the `LIBERO-PRO` suite to test OOD generalizability.
- We completely isolated the `qwen-vla` conda environment, pinning `lerobot v0.4.4` natively to avoid Python 3.12 syntax crashes in later versions.
- We generated and locked over 500 static `_env` BDDL states simulating Camera, Light, and Noise perturbations.
- A long-running evaluation script (`benchmark_libero_pro.py`) is currently orchestrating a multi-hour benchmark across the Baseline model and the 20k checkpoint inside a persistent `tmux` session named `benchmark`.
- The orchestrator specifically executes single-task loop batching to avoid crippling GPU OOM crashes caused by `lerobot_eval`'s default parallel vector instantiation.

## Next Steps for Future Agents
The repository is perfectly clean, emojis have been purged from stdout code, and the data is formally structured. The `benchmark` tmux session is currently crunching through the LIBERO-PRO physics scenarios. The user will be analyzing the resultant Markdown comparison tables upon completion!
