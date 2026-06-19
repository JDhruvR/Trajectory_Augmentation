import os
import sys
import h5py
import json
import numpy as np
import imageio
from pathlib import Path

LIBERO_REPO = Path("/home/dhruv/Trajectory_Augmentation/third_party/LIBERO")
if str(LIBERO_REPO) not in sys.path:
    sys.path.insert(0, str(LIBERO_REPO))

import libero.libero.envs
from robosuite import make

suite = "libero_spatial"
task_name = "pick_up_the_black_bowl_on_the_ramekin_and_place_it_on_the_plate"
orig_path = f"/home/dhruv/Trajectory_Augmentation/data/LIBERO-datasets/{suite}/{task_name}_demo.hdf5"

with h5py.File(orig_path, 'r') as f:
    env_args = json.loads(f['data'].attrs['env_args'])
    initial_state = f["data/demo_0/states"][0]
    actions = f["data/demo_0/actions"][:]
    
bddl_path = str(LIBERO_REPO / "libero" / "libero" / "bddl_files" / suite / f"{task_name}.bddl")

env = make(
    env_name=env_args["env_name"],
    bddl_file_name=bddl_path,
    robots=env_args["env_kwargs"].get("robots", ["Panda"]),
    has_renderer=False,
    has_offscreen_renderer=True,
    use_camera_obs=True,
    camera_names="agentview",
    camera_heights=512,
    camera_widths=512,
    control_freq=env_args["env_kwargs"].get("control_freq", 20),
    horizon=1000,
)

env.reset()
env.sim.set_state_from_flattened(initial_state)
env.sim.forward()

diffs = []
first_obs = env._get_observations()["agentview_image"]

for t in range(10):
    # Try passing 7D action directly
    try:
        env.step(actions[t])
    except Exception as e:
        print(f"Error stepping: {e}")
        break
        
    obs = env._get_observations()["agentview_image"]
    diff = np.sum(np.abs(obs - first_obs))
    diffs.append(diff)
    
print(f"Image differences from first frame: {diffs}")
