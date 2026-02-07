#!/bin/bash
# SmolVLA RA-BC Post-Training Script
# Usage: ./post_train_rabc.sh

# Required for torchcodec with PyTorch nightly (libnppicc.so.12)
export LD_LIBRARY_PATH="/home/weed/miniconda3/envs/lerobot/lib/python3.10/site-packages/nvidia/npp/lib:$LD_LIBRARY_PATH"

python -m lerobot.policies.smolvla.post_train_smolvla \
    --smolvla-path lerobot/smolvla_base \
    --dataset-repo-id weedmo/bimanual_merged \
    --rabc-progress-path /home/weed/.cache/huggingface/lerobot/weedmo/bimanual_merged/sarm_progress.parquet \
    --rename-map '{"observation.images.left_wrist":"observation.images.camera1","observation.images.right_head":"observation.images.camera2","observation.images.right_wrist":"observation.images.camera3"}' \
    --steps 7000 \
    --batch-size 8 \
    --save-freq 2000 \
    --log-freq 100 \
    --warmup-steps 300 \
    --wandb-project smolvla_rabc
