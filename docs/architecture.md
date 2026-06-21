# Technical Architecture & Implementation Details

## Technology Stack & Core OSS Libraries
This project relies heavily on a specialized ecosystem of open-source robotics and deep learning libraries.

- **[PyTorch](https://pytorch.org/)**: Core deep learning framework underpinning the SmolVLA (Vision-Language-Action) policy architecture.
- **[Robosuite](https://robosuite.ai/)**: A simulation framework powered by MuJoCo for robotic learning. We use it to load physics states and generate new observations.
- **[LIBERO](https://libero-project.github.io/)**: A benchmark suite for lifelong robot learning. Provides the core datasets, BDDL environments, and language-conditioned tasks we augment.
- **[LeRobot](https://github.com/huggingface/lerobot)**: Hugging Face's end-to-end robotics library. We strictly adhere to the `v3.0` dataset format, utilizing their conversion utilities to output optimized SVT-AV1 parquet chunks.
- **[Datatrove](https://github.com/huggingface/datatrove)**: Data processing library utilized internally by LeRobot's `libero2lerobot` pipeline for high-throughput, multiprocessed dataset conversion.
- **[FFmpeg](https://ffmpeg.org/)**: Used for high-efficiency video manipulation and analysis during offline validations.
- **[H5py](https://www.h5py.org/)**: The interface to the HDF5 binary data format used by the raw LIBERO datasets.

## Technical Architecture

The Trajectory Augmentation pipeline is designed to artificially inflate the robustness of an offline dataset without requiring computationally expensive physics re-simulation for entire trajectories. 

### 1. Data Ingestion & State Extraction
We interface directly with the raw LIBERO `.hdf5` files. Rather than treating trajectories as rigid sequences, we parse the `env_args` and `bddl_file_name` to dynamically reconstruct the exact MuJoCo/Robosuite environment state at runtime.

### 2. State Injection & The "Inverse Action" Paradigm
Because hardcoding index-based slice offsets for multi-object tabletop environments is highly fragile, we utilize `env.sim.set_state_from_flattened(target_state)`. This allows us to jump to *any* timestamp in an expert trajectory instantly.
We identify a "critical state" (e.g., $t_{grasp} - 5$), jump to it, and inject a bounded random 6D noise vector into the end-effector. We capture the resultant simulated visual observations. 
We rely on the *Reversibility Assumption* (or *Inverse Action* paradigm)—rather than strictly computing an inverse physics path, we linearly stitch the noisy state back into the original expert trajectory, creating a highly local deviation that the VLA policy must learn to correct.

### 3. Rendering Alignments
The native Robosuite OpenGL renderer outputs arrays inverted on both the X and Y axes. A critical architectural feature of this pipeline is the application of `np.flip(img, axis=(0, 1))` to **every single frame**. This 180-degree physical rotation guarantees the final dataset is completely "natural" (human-readable) and natively matches the `nvidia/LIBERO_LeRobot_v3` standard.

### 4. LeRobot Conversion
The newly augmented trajectories are grouped by suite and passed into the `libero_h5.py` wrapper pipeline. The schema mapping strictly enforces an 8D `observation.state` vector to align 1:1 with NVIDIA. Datatrove workers parallel-encode the visual observations into AV1 MP4 chunks and the states into highly compressed parquet files.

## Salient Features
- **OOD Resilience:** By adding localized noise near critical path junctions, the resulting VLA policies show significantly higher macro-spatial robustness when evaluated on Out-Of-Distribution task setups (like `libero_pro_temp_x05`).
- **NVIDIA V3.0 Compliance:** The final dataset structure (`data/`, `meta/`, `videos/`) is perfectly nested and schema-aligned with Hugging Face's official repositories, allowing native `LeRobotDataset` initialization.
- **Highly Parallel:** Uses Python `multiprocessing.Pool` during HDF5 augmentation and `datatrove` worker execution during MP4/Parquet conversion.
