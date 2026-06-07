import robosuite as suite
import matplotlib.pyplot as plt
import numpy as np

print("Testing robots=['Panda']")
env1 = suite.make(
    env_name="PickPlaceBread",
    robots=["Panda"],
    has_renderer=False,
    has_offscreen_renderer=True,
    use_camera_obs=True,
    camera_names="agentview",
    camera_heights=128,
    camera_widths=128,
)
obs1 = env1.reset()
plt.imsave("test_list.png", np.flipud(obs1["agentview_image"]))

print("Testing robots='Panda'")
env2 = suite.make(
    env_name="PickPlaceBread",
    robots="Panda",
    has_renderer=False,
    has_offscreen_renderer=True,
    use_camera_obs=True,
    camera_names="agentview",
    camera_heights=128,
    camera_widths=128,
)
obs2 = env2.reset()
plt.imsave("test_str.png", np.flipud(obs2["agentview_image"]))
print("Done")
