export SVT_LOG=1
export HF_DATASETS_DISABLE_PROGRESS_BARS=TRUE
export HDF5_USE_FILE_LOCKING=FALSE

python libero_h5.py \
    --src-paths /home/dhruv/Trajectory_Augmentation/data/LIBERO-datasets-augmented/libero_goal \
    --output-path /home/dhruv/Trajectory_Augmentation/data/lerobot_format/libero_goal_augmented \
    --executor local \
    --tasks-per-job 3 \
    --workers 10