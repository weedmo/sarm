#!/bin/bash
# SARM Pretrain Script
#
# Usage: ./pretrain_sarm.sh
#
# Trains the SARM (Stage-Aware Reward Model) on the bimanual dataset.
# SARM predicts task progress and stage classification from image sequences.
#
# Annotation modes:
#   single_stage  - No annotations needed, uses task description as one stage (default)
#   dense_only    - Uses dense subtask boundaries (from 'd' key during recording)
#   dual          - Full dual-head with both sparse and dense annotations

# Required for torchcodec with PyTorch nightly (libnppicc.so.12)
export LD_LIBRARY_PATH="/home/weed/miniconda3/envs/lerobot/lib/python3.10/site-packages/nvidia/npp/lib:$LD_LIBRARY_PATH"

DATASET="weedmo/bimanual_merged"

lerobot-train \
    --policy.type=sarm \
    --policy.annotation_mode=dense_only \
    --policy.image_keys='["observation.images.left_wrist", "observation.images.right_wrist", "observation.images.right_head"]' \
    --policy.push_to_hub=false \
    --dataset.repo_id="$DATASET" \
    --batch_size=64 \
    --steps=50000 \
    --save_freq=5000 \
    --log_freq=200 \
    --wandb.enable=true \
    --wandb.project=sarm_pretrain
