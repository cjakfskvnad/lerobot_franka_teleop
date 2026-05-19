import yaml
from pathlib import Path
from typing import Dict, Any
from lerobot_robot_franka import FrankaConfig, Franka
import logging

logging.basicConfig(level=logging.INFO, format="%(message)s")

def main():
    parent_path = Path(__file__).resolve().parent
    cfg_path = parent_path.parent / "config" / "record_cfg.yaml"
    with open(cfg_path, 'r') as f:
        cfg = yaml.safe_load(f)

    # 创建机器人配置
    robot_cfg = cfg["record"]["robot"]
    robot_config = FrankaConfig(
        robot_ip=robot_cfg.get("robot_sn", robot_cfg["ip"]),
        network_interface=robot_cfg.get("network_interface"),
        gripper_name=robot_cfg.get("gripper_name"),
        gripper_init=robot_cfg.get("gripper_init", False),
        gripper_init_wait_sec=robot_cfg.get("gripper_init_wait_sec", 5.0),
        home_plan=robot_cfg.get("home_plan", "PLAN-Home"),
        home_joints=robot_cfg.get("home_joints"),
        command_frequency=robot_cfg.get("command_frequency", 50),
        use_gripper=robot_cfg["use_gripper"],
        close_threshold=robot_cfg["close_threshold"],
        gripper_bin_threshold=robot_cfg["gripper_bin_threshold"],
        gripper_reverse=robot_cfg["gripper_reverse"],
        gripper_max_open=robot_cfg["gripper_max_open"],
        debug=False
    )
    
    # 创建机器人实例并连接
    robot = Franka(robot_config)
    robot.connect()
    
    # 重置机器人到初始位置
    logging.info("Resetting robot to home position...")
    robot.reset()
    
    # 断开连接
    robot.disconnect()
    logging.info("Robot reset completed successfully.")

if __name__ == "__main__":
    main()
