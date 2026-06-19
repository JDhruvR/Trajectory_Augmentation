import sys

try:
    from lerobot.datasets.lerobot_dataset import LeRobotDataset
    
    print("Initializing LeRobotDataset pointing to Sylvest/libero_plus_lerobot...")
    print("This will download metadata from HuggingFace to verify the schema.")
    
    # We load it without downloading all videos just to verify it parses correctly.
    ds = LeRobotDataset("Sylvest/libero_plus_lerobot")
    
    print("\n✅ SMOKE TEST PASSED!")
    print(f"Dataset successfully loaded from HuggingFace.")
    print(f"Total episodes: {ds.num_episodes}")
    print(f"Total frames: {ds.num_frames}")
    print(f"Available features: {list(ds.features.keys())}")
    
except Exception as e:
    print(f"\n❌ SMOKE TEST FAILED: {e}")
    sys.exit(1)
