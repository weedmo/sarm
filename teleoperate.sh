#!/bin/bash
# Bimanual SO100 Teleoperation

lerobot-teleoperate \
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
        head: {"type": "opencv", "index_or_path": 6, "width": 640, "height": 480, "fps": 30}
    }' \
    --display_data=true
