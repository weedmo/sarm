#!/bin/bash
# LeRobot 데이터셋 Hugging Face Hub 업로드 스크립트

HF_USER="weedmo"
DATASET_NAME="bimanual_merged"
DATA_ROOT="/home/weed/.cache/huggingface/lerobot/${HF_USER}/${DATASET_NAME}"

python -c "
from lerobot.datasets.lerobot_dataset import LeRobotDataset
dataset = LeRobotDataset(repo_id='${HF_USER}/${DATASET_NAME}', root='${DATA_ROOT}')
dataset.push_to_hub()
"
