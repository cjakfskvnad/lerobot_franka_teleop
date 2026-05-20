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
    def __init__(self, cfg: Dict[str, Any], robot_defaults: Dict[str, Any] | None = None):
        robot = {**(robot_defaults or {}), **cfg.get("robot", {})}

        # global config
        self.dataset_name: str = cfg["dataset_name"]
        self.episode_idx: int = cfg.get("episode_idx", 0)

        # Flexiv robot config. The imported class names are kept for compatibility:
        # lerobot_robot_franka wraps Flexiv RDK behind the legacy Franka API.
        if "robot_sn" not in robot and "ip" not in robot:
            raise ValueError("replay.robot must define robot_sn or ip, or inherit one from record.robot")
        self.robot_ip: str = robot.get("robot_sn", robot["ip"])
        self.network_interface: str | None = robot.get("network_interface")
        self.gripper_name: str | None = robot.get("gripper_name")
        self.gripper_init: bool = robot.get("gripper_init", False)
        self.gripper_init_wait_sec: float = robot.get("gripper_init_wait_sec", 5.0)
        self.home_plan: str = robot.get("home_plan", "PLAN-Home")
        self.home_joints: list[float] | None = robot.get("home_joints")
        self.command_frequency: int = robot.get("command_frequency", 50)
        self.use_gripper: bool = robot.get("use_gripper", True)
        self.close_threshold: float = robot.get("close_threshold", 0.7)
        self.gripper_reverse: bool = robot.get("gripper_reverse", False)
        self.gripper_bin_threshold: float = robot.get("gripper_bin_threshold", 0.98)
        self.gripper_max_open: float = robot.get("gripper_max_open", 0.08)
        self.control_mode: str = cfg.get("control_mode", robot.get("control_mode", "isoteleop"))
        self.execute_mode: str = robot.get("execute_mode", "ee_pose")

def run_replay(replay_cfg: ReplayConfig):
    episode_idx = replay_cfg.episode_idx

    robot_config = FrankaConfig(
        robot_ip=replay_cfg.robot_ip,
        network_interface=replay_cfg.network_interface,
        gripper_name=replay_cfg.gripper_name,
        gripper_init=replay_cfg.gripper_init,
        gripper_init_wait_sec=replay_cfg.gripper_init_wait_sec,
        home_plan=replay_cfg.home_plan,
        home_joints=replay_cfg.home_joints,
        command_frequency=replay_cfg.command_frequency,
        debug=False,
        use_gripper=replay_cfg.use_gripper,
        close_threshold=replay_cfg.close_threshold,
        gripper_reverse=replay_cfg.gripper_reverse,
        gripper_bin_threshold=replay_cfg.gripper_bin_threshold,
        gripper_max_open=replay_cfg.gripper_max_open,
        control_mode=replay_cfg.control_mode,
        execute_mode=replay_cfg.execute_mode,
    )

    dataset = LeRobotDataset(replay_cfg.dataset_name)
    actions = dataset.hf_dataset.select_columns("action")
    episode_frame_indices = [
        idx for idx, ep_idx in enumerate(dataset.hf_dataset["episode_index"]) if int(ep_idx) == episode_idx
    ]
    if not episode_frame_indices:
        raise ValueError(f"Episode {episode_idx} not found in dataset {replay_cfg.dataset_name}")

    robot = Franka(robot_config)
    try:
        robot.connect()
        log_say(f"Replaying episode {episode_idx} ({len(episode_frame_indices)} frames)")
        for idx in episode_frame_indices:
            t0 = time.perf_counter()
            action = {
                name: float(actions[idx]["action"][i]) for i, name in enumerate(dataset.features["action"]["names"])
            }
            robot.send_action(action)

            busy_wait(1.0 / dataset.fps - (time.perf_counter() - t0))
    finally:
        robot.disconnect()

def main():
    parent_path = Path(__file__).resolve().parent
    cfg_path = parent_path.parent / "config" / "record_cfg.yaml"
    with open(cfg_path, 'r') as f:
        cfg = yaml.safe_load(f)

    replay_cfg = ReplayConfig(cfg["replay"], robot_defaults=cfg.get("record", {}).get("robot", {}))

    run_replay(replay_cfg)

if __name__ == "__main__":
    main()
