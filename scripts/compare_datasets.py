import torch
from lerobot.datasets.lerobot_dataset import LeRobotDataset

def print_dataset_info(name, path):
    print(f"=== {name} ===")
    try:
        ds = LeRobotDataset(repo_id="local", root=path)
        print(f"Total episodes: {ds.num_episodes}")
        print(f"Total frames: {ds.num_frames}")
        print(f"FPS: {ds.fps}")
        
        # Get first frame
        item = ds[0]
        print("\nFeature Shapes:")
        for k, v in item.items():
            if isinstance(v, torch.Tensor):
                print(f"  {k}: {v.shape} ({v.dtype})")
            else:
                print(f"  {k}: {type(v)} = {v}")
                
        # Get a full episode
        episode_dict = ds.hf_dataset.filter(lambda x: x["episode_index"] == 0)
        print(f"\nEpisode 0 length: {len(episode_dict)}")
    except Exception as e:
        print(f"Error loading {name}: {e}")
    print("\n")

print_dataset_info("NVIDIA Libero Goal", "/home/dhruv/Trajectory_Augmentation/data/nvidia_libero/libero_goal")
print_dataset_info("Our Augmented Libero Goal", "/home/dhruv/Trajectory_Augmentation/data/lerobot_format/libero_goal_augmented")
