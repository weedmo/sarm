#!/bin/bash
# SmolVLA RA-BC Post-Training Script (Single Head Camera SARM)
#
# Usage: ./post_train_rabc_single.sh
#
# Post-trains SmolVLA using SARM progress values from weedmo/sarm_single_1st.
# Run ./compute_sarm_single.sh first to generate sarm_progress.parquet.
#
# The SARM model uses only the head cam (observation.images.right_head),
# but SmolVLA still uses all 3 cameras for policy inference.

# Required for torchcodec with PyTorch nightly (libnppicc.so.12)
export LD_LIBRARY_PATH="/home/weed/miniconda3/envs/lerobot/lib/python3.10/site-packages/nvidia/npp/lib:$LD_LIBRARY_PATH"

SMOLVLA_PATH="lerobot/smolvla_base"
DATASET="weedmo/bimanual_merged"
SARM_PROGRESS="/home/weed/.cache/huggingface/lerobot/weedmo/bimanual_merged/sarm_progress.parquet"

python -m lerobot.policies.smolvla.post_train_smolvla \
    --smolvla-path "$SMOLVLA_PATH" \
    --dataset-repo-id "$DATASET" \
    --rabc-progress-path "$SARM_PROGRESS" \
    --rename-map '{"observation.images.left_wrist":"observation.images.camera1","observation.images.right_head":"observation.images.camera2","observation.images.right_wrist":"observation.images.camera3"}' \
    --steps 6000 \
    --batch-size 8 \
    --save-freq 2000 \
    --log-freq 100 \
    --warmup-steps 300 \
    --wandb-project smolvla_rabc_single
