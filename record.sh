#!/bin/bash
# Bimanual SO100 Data Recording
#
# Usage: ./record.sh
#
# Keyboard controls during recording:
#   Right arrow (→)  - Stop current episode and save with task instruction
#   Space       ( )  - Stop current episode and save WITHOUT task instruction (empty string)
#   Left arrow  (←)  - Discard current episode and re-record
#   Escape      (Esc) - Stop recording entirely
#
# Task instruction examples:
#   "Pick up the red cup and place it on the plate"
#   "Open the drawer, grab the towel, and close the drawer"
#   "Stack the blue block on top of the green block"
#   Tips:
#     - Be specific about objects (color, shape, position)
#     - Describe the full action sequence in order
#     - Use consistent naming across episodes of the same task

REPO_ID="weedmo/bimanual_so100_dataset_1"
TASK="Perform a multi-stage manipulation: first open the drawstring bag by grasping and pulling its strings apart; then identify the two caramels aligned with the red line on the right, pick them up one by one, place them carefully inside the opened bag, and finally pull both ends of the string to close the bag tightly. Ensure each action is smooth and visually verified — the bag opening should be wide during placement and fully closed when finished."
NUM_EPISODES=100
FPS=30
EPISODE_TIME=300    # max seconds per episode (press → to end early)
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
    --dataset.single_task="$TASK"  \
    --dataset.repo_id="$REPO_ID" \
    --resume=true \
    --dataset.num_episodes=$NUM_EPISODES \
    --dataset.fps=$FPS \
    --dataset.episode_time_s=$EPISODE_TIME \
    --dataset.reset_time_s=$RESET_TIME \
    --display_data=true
