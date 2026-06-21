# Installation and User Guide

## Salient Features
The Trajectory Augmentation project introduces several key features over standard imitation learning pipelines:
- **Zero-Shot Robustness**: Automatically synthesizes complex out-of-distribution state recoveries without requiring human teleoperation for corrections.
- **LeRobot v3.0 Native**: Full end-to-end integration with Hugging Face's LeRobot format, outputting extremely dense `SVT-AV1` video chunks and parquets.
- **Orientation Auto-Correction**: Completely fixes the underlying upside-down/mirrored OpenGL simulation artifacts dynamically during dataset generation, ensuring human-natural validation data.
- **Multi-Suite Unification**: Custom `libero_h5.py` generic converters capable of safely normalizing disjointed schemas (`libero_goal`, `libero_object`, `libero_spatial`) into a monolithic, scalable hub repository.

## Installation Instructions

1. **Clone Repository & Setup Environment**
   It is highly recommended to use the generated `requirements.txt` inside a fresh Conda environment (Python 3.10 is required for Qwen-VLA compatibility).
   ```bash
   conda create -n qwen-vla python=3.10
   conda activate qwen-vla
   ```

2. **Install Dependencies**
   Install the exact packages utilized during generation:
   ```bash
   pip install -r requirements.txt
   ```

3. **Install LIBERO & Robosuite**
   Ensure that the MuJoCo physics engine and the custom Robosuite/LIBERO environments are placed in your `third_party` directory and added to your `PYTHONPATH`.
   ```bash
   export PYTHONPATH=$PYTHONPATH:/path/to/src/third_party/LIBERO
   ```

## User Guide

### 1. Generating Augmented Trajectories (HDF5)
To generate the raw, augmented trajectories in `.hdf5` format with native 180-degree rotation correction, utilize the main runner script. You can run this in the background using `tmux`.

```bash
python src/generate_augmented_dataset.py \
    --target_dir data/LIBERO-datasets/libero_goal \
    --output_dir data/LIBERO-datasets-augmented/libero_goal \
    --num_augmentations 5
```
*Note: Due to the intensive nature of physics simulation, this script spawns multiple CPU workers. It is expected to take 15-30 minutes per suite depending on core count.*

### 2. LeRobot Format Conversion
Once the HDF5s are generated, convert them into the highly-compressed Hugging Face dataset format.
```bash
export SVT_LOG=1
export HF_DATASETS_DISABLE_PROGRESS_BARS=TRUE
export HDF5_USE_FILE_LOCKING=FALSE

python src/libero_to_lerobot/libero2lerobot/libero_h5.py \
    --src-paths data/LIBERO-datasets-augmented/libero_goal \
    --output-path data/lerobot_format/hf_upload_repo/libero_goal \
    --executor local \
    --tasks-per-job 3 \
    --workers 10
```

### 3. Pushing to Hugging Face
Once the dataset structure is built locally in `hf_upload_repo`, you can push the dataset sequentially to Hugging Face using the CLI:
```bash
huggingface-cli upload your-username/libero_trajectory_augmented data/lerobot_format/hf_upload_repo . --repo-type dataset
```

