import os
import sys

# Configure logging to see issues
import logging
logging.basicConfig(level=logging.INFO)

try:
    from lerobot.datasets.lerobot_dataset import LeRobotDataset
except ImportError as e:
    try:
        from lerobot.common.datasets.lerobot_dataset import LeRobotDataset
    except ImportError as e:
        print(f"Failed to import lerobot: {e}")
        sys.exit(1)

def verify_dataset(dataset_path):
    print(f"Verifying dataset at: {dataset_path}")
    if not os.path.exists(dataset_path):
        print(f"Directory {dataset_path} does not exist!")
        return False
        
    try:
        # We need to initialize the dataset from a local folder
        dataset = LeRobotDataset.from_preloaded(root=dataset_path)
        print(f"Dataset successfully loaded!")
        print(f"Number of episodes: {dataset.num_episodes}")
        print(f"Number of frames: {dataset.num_frames}")
        print(f"Features: {dataset.features.keys()}")
        print(f"First frame: {dataset[0]}")
        return True
    except Exception as e:
        print(f"Failed to load dataset: {e}")
        return False

if __name__ == "__main__":
    original_path = "/home/dhruv/Trajectory_Augmentation/data/lerobot_format/libero_goal"
    verify_dataset(original_path)
