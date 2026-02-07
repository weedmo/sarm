#!/bin/bash
# 모터 오프셋 진단 스크립트
# 사용법: ./check_motors.sh [left_port] [right_port]

LEFT_PORT="${1:-/dev/ttyACM2}"
RIGHT_PORT="${2:-/dev/ttyACM0}"

python3 << PYEOF
from lerobot.motors.feetech import FeetechMotorsBus
from lerobot.motors.motors_bus import Motor, MotorNormMode

MOTORS = {
    "shoulder_pan": Motor(1, "sts3215", MotorNormMode.RANGE_M100_100),
    "shoulder_lift": Motor(2, "sts3215", MotorNormMode.RANGE_M100_100),
    "elbow_flex": Motor(3, "sts3215", MotorNormMode.RANGE_M100_100),
    "wrist_flex": Motor(4, "sts3215", MotorNormMode.RANGE_M100_100),
    "wrist_roll": Motor(5, "sts3215", MotorNormMode.RANGE_M100_100),
    "gripper": Motor(6, "sts3215", MotorNormMode.RANGE_0_100),
}

MAX_OFFSET = 2047  # sign-magnitude 11bit limit

def check_arm(name, port):
    print(f"\n{'='*50}")
    print(f" {name} (port: {port})")
    print(f"{'='*50}")
    try:
        bus = FeetechMotorsBus(port=port, motors=dict(MOTORS))
        bus.connect()
        bus.reset_calibration()
        positions = bus.sync_read("Present_Position", normalize=False)
        print(f"{'motor':<16} {'position':>8} {'offset':>8} {'status'}")
        print("-" * 50)
        for motor, pos in positions.items():
            offset = pos - 2047
            ok = abs(offset) <= MAX_OFFSET
            status = "OK" if ok else f"OVERFLOW! (|{abs(offset)}| > {MAX_OFFSET})"
            print(f"{motor:<16} {pos:>8} {offset:>8} {status}")
        bus.disconnect()
    except Exception as e:
        print(f"  ERROR: {e}")

check_arm("LEFT ARM", "${LEFT_PORT}")
check_arm("RIGHT ARM", "${RIGHT_PORT}")
PYEOF
