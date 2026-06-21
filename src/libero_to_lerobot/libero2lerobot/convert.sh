#!/bin/bash
export SVT_LOG=1
export HF_DATASETS_DISABLE_PROGRESS_BARS=TRUE
export HDF5_USE_FILE_LOCKING=FALSE

SUITES=("libero_goal" "libero_spatial" "libero_object")

for suite in "${SUITES[@]}"; do
    echo "Converting suite: $suite"
    python libero_h5.py \
        --src-paths /home/dhruv/Trajectory_Augmentation/data/LIBERO-datasets-augmented/$suite \
        --output-path /home/dhruv/Trajectory_Augmentation/data/lerobot_format/${suite}_augmented \
        --executor local \
        --tasks-per-job 3 \
        --workers 10
done