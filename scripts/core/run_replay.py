import time
import yaml
import logging
logging.basicConfig(level=logging.WARNING, format="%(message)s")
from pathlib import Path
from typing import Dict, Any
from lerobot_robot_franka import FrankaConfig, Franka
from lerobot.datasets.lerobot_dataset import LeRobotDataset
from lerobot.utils.robot_utils import busy_wait
from lerobot.utils.utils import log_say

class ReplayConfig:
    def __init__(self, cfg: Dict[str, Any]):
        robot = cfg["robot"]

        # global config
        self.dataset_name: str = cfg["dataset_name"]
        self.episode_idx: str = cfg.get("episode_idx", 0)

        # robot config
        self.robot_ip: str = robot.get("robot_sn", robot["ip"])
        self.network_interface: str | None = robot.get("network_interface")
        self.gripper_name: str | None = robot.get("gripper_name")
        self.home_plan: str = robot.get("home_plan", "PLAN-Home")
        self.home_joints: list[float] | None = robot.get("home_joints")
        self.command_frequency: int = robot.get("command_frequency", 50)
        self.control_mode: str = cfg["control_mode"]

def run_replay(replay_cfg: ReplayConfig):
    episode_idx = replay_cfg.episode_idx

    robot_config = FrankaConfig(
        robot_ip=replay_cfg.robot_ip,
        network_interface=replay_cfg.network_interface,
        gripper_name=replay_cfg.gripper_name,
        home_plan=replay_cfg.home_plan,
        home_joints=replay_cfg.home_joints,
        command_frequency=replay_cfg.command_frequency,
        debug = False,
        gripper_reverse = False,
        control_mode = replay_cfg.control_mode
    )
    
    robot = robot = Franka(robot_config)
    robot.connect()
    dataset = LeRobotDataset(replay_cfg.dataset_name, episodes=[episode_idx])
    actions = dataset.hf_dataset.select_columns("action")
    log_say(f"Replaying episode {episode_idx}")
    for idx in range(dataset.num_frames):
        t0 = time.perf_counter()
        action = {
            name: float(actions[idx]["action"][i]) for i, name in enumerate(dataset.features["action"]["names"])
        }
        # print(f"action: {action}")
        robot.send_action(action)

        busy_wait(1.0 / dataset.fps - (time.perf_counter() - t0))

    robot.disconnect()

def main():
    parent_path = Path(__file__).resolve().parent
    cfg_path = parent_path.parent / "config" / "record_cfg.yaml"
    with open(cfg_path, 'r') as f:
        cfg = yaml.safe_load(f)

    replay_cfg = ReplayConfig(cfg["replay"])

    run_replay(replay_cfg)
