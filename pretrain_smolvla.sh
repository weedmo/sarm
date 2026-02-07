#!/bin/bash
# SmolVLA Pretrain (Fine-tuning) Script
# Usage:
#   ./pretrain_smolvla.sh          # 새로 학습
#   ./pretrain_smolvla.sh resume   # 이어서 학습

# Required for torchcodec with PyTorch nightly (libnppicc.so.12)
export LD_LIBRARY_PATH="/home/weed/miniconda3/envs/lerobot/lib/python3.10/site-packages/nvidia/npp/lib:$LD_LIBRARY_PATH"

if [ "$1" = "resume" ]; then
    CHECKPOINT_DIR="outputs/train/2026-02-08/03-07-15_smolvla/checkpoints"
    echo "Resuming from ${CHECKPOINT_DIR}/last ..."
    lerobot-train \
        --resume=true \
        --config_path="${CHECKPOINT_DIR}/last/pretrained_model/train_config.json" \
        --steps=30000
else
    lerobot-train \
        --policy.type=smolvla \
        --policy.pretrained_path=lerobot/smolvla_base \
        --policy.push_to_hub=false \
        --dataset.repo_id=weedmo/bimanual_merged \
        --batch_size=8 \
        --steps=30000 \
        --save_freq=5000 \
        --log_freq=200 \
        --wandb.enable=true \
        --wandb.project=smolvla_pretrain
fi
