#!/bin/bash
# SARM Subtask Annotation Script
#
# Usage: ./annotate_sarm.sh
#
# Uses Qwen3-VL to automatically annotate dense subtask boundaries in the dataset.
# Generates temporal_proportions_sparse.json and temporal_proportions_dense.json
# which are required for SARM dense_only training mode.
#
# After annotation, run ./pretrain_sarm.sh with annotation_mode=dense_only
#
# Requirements: GPU with 16GB+ VRAM for Qwen3-VL-30B-A3B

# Required for torchcodec with PyTorch nightly (libnppicc.so.12)
export LD_LIBRARY_PATH="/home/weed/miniconda3/envs/lerobot/lib/python3.10/site-packages/nvidia/npp/lib:$LD_LIBRARY_PATH"

DATASET="weedmo/bimanual_merged"
VIDEO_KEY="observation.images.right_head"

# Dense subtask descriptions (comma-separated)
# Adjust these to match your task's actual subtask stages
DENSE_SUBTASKS="Open the drawstring bag by pulling strings apart,Pick up first caramel and place inside bag,Pick up second caramel and place inside bag,Pull both ends of string to close bag"

python -m lerobot.data_processing.sarm_annotations.subtask_annotation \
    --repo-id "$DATASET" \
    --dense-only \
    --dense-subtasks "$DENSE_SUBTASKS" \
    --video-key "$VIDEO_KEY" \
    --model Qwen/Qwen2.5-VL-3B-Instruct \
    --device cuda \
    --dtype bfloat16 \
    --skip-existing \
    --push-to-hub
