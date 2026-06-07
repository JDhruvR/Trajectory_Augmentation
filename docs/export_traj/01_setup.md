# Setup: Environment and Dependencies

## Prerequisites

- **OS**: Linux (tested on Ubuntu)
- **Python**: 3.8+ (preferably 3.9 or 3.10)
- **Conda**: Anaconda or Miniconda
- **Git**: For cloning repositories

## Step 1: Create and Activate Conda Environment

```bash
# Create a new conda environment named qwen-vla
conda create -n qwen-vla python=3.10 -y

# Activate the environment
conda activate qwen-vla
```

## Step 2: Install Core Dependencies

```bash
# Update pip
pip install --upgrade pip

# Install PyTorch (for CUDA 11.8 - adjust if you have different GPU support)
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

# Install core scientific packages
pip install numpy scipy matplotlib pandas scikit-learn

# Install HDF5 support
pip install h5py

# Install TensorFlow and TFDS (for LIBERO dataset)
pip install tensorflow tensorflow-datasets

# Install additional utilities
pip install easydict cloudpickle
```

## Step 3: Clone LIBERO Repository

```bash
# Navigate to project root
cd /home/dhruv/Scene-Graph-VLA

# Clone LIBERO into third_party
mkdir -p third_party
cd third_party
git clone https://github.com/Lifelong-Robot-Learning/LIBERO.git LIBERO
cd LIBERO

# Install LIBERO in development mode
pip install -e .
```

**Note**: LIBERO provides the dataset classes, task definitions, and BDDL files needed for environment setup.

## Step 4: Install Robosuite (Simulator)

```bash
# Install specific robosuite version (1.4.1)
pip install robosuite==1.4.1
```

**Why 1.4.1?** Version 1.5.2+ removed `SingleArmEnv` which is needed for LIBERO compatibility. Use 1.4.1.

## Step 5: Install BDDL and Additional Requirements

```bash
# Install BDDL (Behavior Description Definition Language) for task definitions
pip install bddl

# Install gym (older version for compatibility)
pip install gym==0.26.2

# Any other missing dependencies from LIBERO
pip install pillow pyyaml
```

## Step 6: Verify Installation

```bash
# Test imports
python -c "import robosuite; print(f'robosuite version: {robosuite.__version__}')"
python -c "import LIBERO; print('LIBERO imported successfully')"
python -c "import tensorflow_datasets; print('tensorflow_datasets imported successfully')"
python -c "import h5py; print(f'h5py version: {h5py.__version__}')"
```

Expected output:
```
robosuite version: 1.4.1
LIBERO imported successfully
tensorflow_datasets imported successfully
h5py version: X.X.X
```

## Step 7: Environment Variables (Optional but Recommended)

```bash
# Add to ~/.bashrc or ~/.zshrc for persistent environment variables
export LIBERO_REPO=/home/dhruv/Scene-Graph-VLA/third_party/LIBERO
export SCENE_GRAPH_VLA=/home/dhruv/Scene-Graph-VLA
export PYTHONPATH=$LIBERO_REPO:$SCENE_GRAPH_VLA:$PYTHONPATH
```

Then reload:
```bash
source ~/.bashrc  # or source ~/.zshrc
```

## Step 8: Verify Data Availability

```bash
# Check if HDF5 demo file exists
ls -lh ~/Scene-Graph-VLA/sandbox/pick_up_the_soup_and_place_it_in_the_basket_demo.hdf5

# Check LIBERO TFDS dataset
python -c "
import tensorflow_datasets as tfds
dataset = tfds.load('libero_object_no_noops', split='train', download_and_prepare=False)
print('LIBERO dataset found')
"
```

## Step 9: Directory Structure

Verify that your project structure looks like this:

```
~/Scene-Graph-VLA/
├── third_party/
│   └── LIBERO/                    # LIBERO repository
│       ├── libero/
│       ├── bddl_files/
│       └── ...
├── sandbox/
│   └── pick_up_the_soup_and_place_it_in_the_basket_demo.hdf5
├── data/
│   └── libero_graph_dataset.py    # Your dataset loader
├── report_mds/                     # Output directory
├── export_traj/                    # This documentation
└── ... (other project files)
```

## Troubleshooting

### Issue: `ModuleNotFoundError: No module named 'robosuite'`
- **Solution**: Make sure you're in the `qwen-vla` conda environment: `conda activate qwen-vla`

### Issue: `robosuite version conflicts`
- **Solution**: Uninstall and reinstall: `pip uninstall robosuite -y && pip install robosuite==1.4.1`

### Issue: `LIBERO not found`
- **Solution**: Make sure you cloned LIBERO into `third_party/` and installed it with `pip install -e .`

### Issue: HDF5 file not found
- **Solution**: Verify the path: `ls -lh ~/Scene-Graph-VLA/sandbox/`

### Issue: TFDS dataset download fails
- **Solution**: Check internet connection and disk space. LIBERO datasets can be large (~50GB).

## Quick Test Script

Create a test script `test_setup.py`:

```python
#!/usr/bin/env python3
import sys

def test_imports():
    """Test all critical imports"""
    print("Testing imports...")
    
    try:
        import robosuite
        print(f"✓ robosuite {robosuite.__version__}")
    except ImportError as e:
        print(f"✗ robosuite: {e}")
        return False
    
    try:
        import LIBERO
        print("✓ LIBERO")
    except ImportError as e:
        print(f"✗ LIBERO: {e}")
        return False
    
    try:
        import tensorflow_datasets as tfds
        print("✓ tensorflow_datasets")
    except ImportError as e:
        print(f"✗ tensorflow_datasets: {e}")
        return False
    
    try:
        import h5py
        print(f"✓ h5py {h5py.__version__}")
    except ImportError as e:
        print(f"✗ h5py: {e}")
        return False
    
    try:
        import numpy as np
        print(f"✓ numpy {np.__version__}")
    except ImportError as e:
        print(f"✗ numpy: {e}")
        return False
    
    return True

if __name__ == "__main__":
    success = test_imports()
    sys.exit(0 if success else 1)
```

Run it:
```bash
python test_setup.py
```

## Next Steps

Once setup is complete, proceed to:
- **02_simulator_setup.md** - Configure robosuite for LIBERO tasks
