#!/bin/bash
# SARM Single-Camera Inference Script
#
# Usage: ./compute_sarm_single.sh
#
# Computes SARM progress values from weedmo/sarm_single_1st (head cam only).
# Outputs sarm_progress.parquet which is used for SmolVLA RA-BC post-training.
#
# After this, run ./post_train_rabc_single.sh to post-train SmolVLA.

# Required for torchcodec with PyTorch nightly (libnppicc.so.12)
export LD_LIBRARY_PATH="/home/weed/miniconda3/envs/lerobot/lib/python3.10/site-packages/nvidia/npp/lib:$LD_LIBRARY_PATH"

SARM_MODEL="weedmo/sarm_single_1st"
DATASET="weedmo/bimanual_merged"

python src/lerobot/policies/sarm/compute_rabc_weights.py \
    --dataset-repo-id "$DATASET" \
    --reward-model-path "$SARM_MODEL" \
    --head-mode sparse \
    --device cuda \
    --stride 3 \
    --num-visualizations 5 \
    --output-dir ./sarm_viz_single \
    --push-to-hub
