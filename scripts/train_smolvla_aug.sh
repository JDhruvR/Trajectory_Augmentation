#!/bin/bash

# Ensure environments are correctly set
export USE_TF=0 
export USE_JAX=0
export WANDB_PROJECT="smolvla_finetuning"

echo "Starting SmolVLA Finetuning on augmented dataset..."

# We use the local path for the dataset
DATASET_PATH="/home/dhruv/Trajectory_Augmentation/data/lerobot_format/libero_all_augmented"

/home/dhruv/miniforge3/envs/qwen-vla/bin/python -m lerobot.scripts.lerobot_train \
  --dataset.repo_id=local \
  --dataset.root=$DATASET_PATH \
  --policy.type=smolvla \
  --policy.pretrained_path=HuggingFaceVLA/smolvla_libero \
  --output_dir=outputs/train/smolvla_libero_all_aug \
  --job_name=smolvla_libero_all_finetune \
  --policy.device=cuda \
  --policy.expert_width_multiplier=0.5 \
  --policy.repo_id=dhruv/smolvla_libero_all_aug \
  --batch_size=64 \
  --wandb.enable=true

echo "Training run completed or terminated."
