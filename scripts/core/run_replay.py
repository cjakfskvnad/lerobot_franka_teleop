import time
import yaml
import logging
logging.basicConfig(level=logging.WARNING, format="%(message)s")
import json
import math
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
        self.gripper_force: float = robot.get("gripper_force", 40.0)
        self.gripper_always_grasp: bool = robot.get("gripper_always_grasp", False)
        self.control_mode: str = cfg.get("control_mode", robot.get("control_mode", "isoteleop"))
        self.execute_mode: str = robot.get("execute_mode", "ee_pose")
        self.policy_io_schema: str = robot.get("policy_io_schema", "default")
        self.pre_replay_movej: bool = cfg.get("pre_replay_movej", True)
        self.pre_replay_joint_pose_file: str = cfg.get(
            "pre_replay_joint_pose_file",
            str(Path(__file__).resolve().parents[3] / "example_py" / "saved_joint_pose.json"),
        )
        self.pre_replay_joint_positions: list[float] | None = cfg.get("pre_replay_joint_positions")
        self.pre_replay_joint_unit: str = cfg.get("pre_replay_joint_unit", "rad")
        self.pre_replay_movej_velocity: float = cfg.get("pre_replay_movej_velocity", 0.2)
        self.pre_replay_movej_timeout: float = cfg.get("pre_replay_movej_timeout", 120.0)


def _load_pre_replay_joints_deg(replay_cfg: ReplayConfig) -> list[float]:
    if replay_cfg.pre_replay_joint_positions is not None:
        joints = [float(v) for v in replay_cfg.pre_replay_joint_positions]
        unit = replay_cfg.pre_replay_joint_unit
    else:
        pose_path = Path(replay_cfg.pre_replay_joint_pose_file).expanduser()
        data = json.loads(pose_path.read_text(encoding="utf-8"))
        if "q_deg" in data:
            joints = [float(v) for v in data["q_deg"]]
            unit = "deg"
        elif "q_rad" in data:
            joints = [float(v) for v in data["q_rad"]]
            unit = "rad"
        else:
            raise ValueError(f"{pose_path} must contain q_deg or q_rad")

    if len(joints) != 7:
        raise ValueError(f"Pre-replay MoveJ target must contain 7 joints, got {len(joints)}")

    if unit == "deg":
        return joints
    if unit == "rad":
        return [math.degrees(v) for v in joints]
    raise ValueError(f"Unsupported pre_replay_joint_unit: {unit}")

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
        gripper_force=replay_cfg.gripper_force,
        gripper_always_grasp=replay_cfg.gripper_always_grasp,
        control_mode=replay_cfg.control_mode,
        execute_mode=replay_cfg.execute_mode,
        policy_io_schema=replay_cfg.policy_io_schema,
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
        if replay_cfg.pre_replay_movej:
            target_deg = _load_pre_replay_joints_deg(replay_cfg)
            logging.warning(
                "Moving to pre-replay fixed joint pose with MoveJ: %s",
                [round(v, 3) for v in target_deg],
            )
            robot.movej_to_joint_positions_deg(
                target_deg,
                velocity=replay_cfg.pre_replay_movej_velocity,
                timeout=replay_cfg.pre_replay_movej_timeout,
            )
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
