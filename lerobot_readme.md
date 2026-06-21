---
license: apache-2.0
tags:
- lerobot
- robotics
- libero
- augmentation
- dataset
---

# Libero Trajectory Augmented Dataset

This repository contains the officially formatted LeRobot (`v3.0+`) datasets for the **LIBERO** benchmarks, augmented via physical state perturbation. 

## Overview & Motivation

Training robust offline reinforcement learning (RL) or Vision-Language-Action (VLA) models often suffers from the narrow distribution of expert demonstrations. This dataset aims to artificially inflate the robustness of offline datasets by providing **out-of-distribution (OOD) recovery trajectories** without requiring human teleoperation.

By learning from these augmented trajectories, policies are exposed to higher variability in trajectory lengths, altered spatial configurations, and must learn explicit recovery behaviors when driven slightly off the "golden path".

## Augmentation Details

The trajectories in this dataset were dynamically generated using the Robosuite MuJoCo engine. Instead of fully re-simulating episodes from scratch, we employed an **Inverse Action Paradigm**:

1. We load the original expert demonstration.
2. We roll out the environment to a critical state (e.g., moments before a grasp or interaction).
3. We inject structured, bounded 6D coordinate noise into the robot's end-effector state.
4. The expert policy is then forced to recover from this noisy, out-of-distribution state and successfully complete the task.

### Dataset Specifications

- **Augmentation Scale:** `2` augmented trajectories per `1` original expert demonstration. This effectively triples the size of the original dataset.
- **Variability:** The noise injection creates variability in trajectory lengths and forces the model to learn recovery actions, improving OOD resilience.
- **Orientation:** Perfect Native OpenCV (Right-side up, unmirrored). The underlying Robosuite OpenGL rendering artifacts (inverted X/Y axes) were dynamically corrected during generation to match the 8D NVIDIA schema natively.
- **Format:** Native Hugging Face LeRobot `v3.0` Datasets.
  - Highly compressed Parquet files for tabular state/action data.
  - High-efficiency `SVT-AV1` chunked MP4 videos for visual observations.

## Suites Included

This repository contains the fully processed sub-directories for the following LIBERO suites:

- `libero_goal`
- `libero_spatial`
- `libero_object`

## Usage

You can load these datasets natively using the Hugging Face `LeRobotDataset` API by pointing directly to the specific suite subdirectory.

```python
from lerobot.common.datasets.lerobot_dataset import LeRobotDataset

# Load the augmented libero_goal dataset
dataset = LeRobotDataset("JDhruvR/libero_trajectory_augmented/libero_goal")
```

## Citation

If you use this dataset or the LIBERO benchmark, please cite the original LIBERO paper:

```bibtex
@inproceedings{liu2023libero,
  title={LIBERO: Benchmarking Knowledge Transfer for Lifelong Robot Learning},
  author={Liu, Bo and Zhu, Yifeng and Gao, Chongkai and Feng, Yihao and Liu, Qiang and Zhu, Yuke and Stone, Peter},
  booktitle={Advances in Neural Information Processing Systems},
  year={2023}
}
```

Please also cite or reference LeRobot if you use their tooling or dataset loaders.

## References

- **LIBERO Project Page:** [https://libero-project.github.io/](https://libero-project.github.io/)
- **LIBERO arXiv Paper:** [https://arxiv.org/abs/2306.03310](https://arxiv.org/abs/2306.03310)
- **LIBERO GitHub Repository:** [https://github.com/Lifelong-Robot-Learning/LIBERO](https://github.com/Lifelong-Robot-Learning/LIBERO)
- **Hugging Face LeRobot LIBERO Documentation:** [https://huggingface.co/docs/lerobot/en/libero](https://huggingface.co/docs/lerobot/en/libero)
- **LeRobotDataset v3.0 Documentation:** [https://huggingface.co/docs/lerobot/lerobot-dataset-v3](https://huggingface.co/docs/lerobot/lerobot-dataset-v3)
- **Porting large datasets to LeRobot v3.0:** [https://huggingface.co/docs/lerobot/main/porting_datasets_v3](https://huggingface.co/docs/lerobot/main/porting_datasets_v3)
