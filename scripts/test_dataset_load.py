import logging
import os
import sys

logging.basicConfig(level=logging.INFO)

print("Importing LeRobotDataset...")
try:
    from lerobot.common.datasets.lerobot_dataset import LeRobotDataset
except ImportError:
    try:
        from lerobot.datasets.lerobot_dataset import LeRobotDataset
    except ImportError as e:
        print(f"Failed to import LeRobotDataset: {e}")
        sys.exit(1)

print("Import successful!")
dataset_path = "/home/dhruv/Trajectory_Augmentation/data/lerobot_format/libero_goal_augmented"

try:
    print(f"Loading dataset from: {dataset_path}")
    # Using the standard constructor
    dataset = LeRobotDataset(repo_id=None, root=dataset_path)
    print("Dataset successfully loaded!")
    print(f"Number of episodes: {dataset.num_episodes}")
    print(f"Number of frames: {dataset.num_frames}")
    print(f"Features: {dataset.features.keys()}")
    
    print("Loading first frame...")
    frame = dataset[0]
    print("Successfully loaded frame 0.")
except Exception as e:
    import traceback
    print(f"Failed to load dataset: {e}")
    traceback.print_exc()
