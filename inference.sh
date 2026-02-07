#!/bin/bash
# SmolVLA Inference with Bimanual SO100
#
# Usage: ./inference.sh
#
# Runs the fine-tuned SmolVLA model (weedmo/smol_1st) on the bimanual robot.
# Teleop arms are connected so you can intervene if needed.
#
# Keyboard controls:
#   Right arrow (→)  - Stop current episode and save
#   Left arrow  (←)  - Discard current episode and re-record
#   Escape      (Esc) - Stop recording entirely

# Required for torchcodec with PyTorch nightly (libnppicc.so.12)
export LD_LIBRARY_PATH="/home/weed/miniconda3/envs/lerobot/lib/python3.10/site-packages/nvidia/npp/lib:$LD_LIBRARY_PATH"


POLICY_PATH="weedmo/smol_2nd"
REPO_ID="weedmo/eval_smol_2nd"
TASK="Perform a multi-stage manipulation: first open the drawstring bag by grasping and pulling its strings apart; then identify the two caramels aligned with the red line on the right, pick them up one by one, place them carefully inside the opened bag, and finally pull both ends of the string to close the bag tightly. Ensure each action is smooth and visually verified — the bag opening should be wide during placement and fully closed when finished."
NUM_EPISODES=10
FPS=30
EPISODE_TIME=300
RESET_TIME=60

lerobot-record \
    --robot.type=bi_so_follower \
    --robot.left_arm_config.port=/dev/ttyACM3 \
    --robot.right_arm_config.port=/dev/ttyACM1 \
    --robot.id=bimanual_follower \
    --teleop.type=bi_so_leader \
    --teleop.left_arm_config.port=/dev/ttyACM2 \
    --teleop.right_arm_config.port=/dev/ttyACM0 \
    --teleop.id=bimanual_leader \
    --robot.left_arm_config.cameras='{
        wrist: {"type": "opencv", "index_or_path": 8, "width": 1280, "height": 720, "fps": 30, "fourcc": "MJPG"}
    }' \
    --robot.right_arm_config.cameras='{
        wrist: {"type": "opencv", "index_or_path": 0, "width": 1280, "height": 720, "fps": 30, "fourcc": "MJPG"},
        head: {"type": "opencv", "index_or_path": 7, "width": 640, "height": 480, "fps": 30}
    }' \
    --policy.path="$POLICY_PATH" \
    --dataset.single_task="$TASK" \
    --dataset.repo_id="$REPO_ID" \
    --dataset.num_episodes=$NUM_EPISODES \
    --dataset.fps=$FPS \
    --dataset.episode_time_s=$EPISODE_TIME \
    --dataset.reset_time_s=$RESET_TIME \
    --dataset.rename_map='{"observation.images.left_wrist": "observation.images.camera1", "observation.images.right_wrist": "observation.images.camera2", "observation.images.right_head": "observation.images.camera3"}' \
    --display_data=true
