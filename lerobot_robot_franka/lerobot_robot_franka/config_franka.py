from dataclasses import dataclass, field

from lerobot.cameras import CameraConfig

from lerobot.robots.config import RobotConfig

@RobotConfig.register_subclass("franka_robot")
@dataclass
class FrankaConfig(RobotConfig):
    use_gripper: bool = True
    gripper_reverse: bool = True
    # Backward-compatible field name. For Flexiv this should be the robot serial
    # number, for example "Rizon4s-123456".
    robot_ip: str = "Rizon4s-123456"
    network_interface: str | None = None
    gripper_name: str | None = None
    gripper_init: bool = False
    gripper_init_wait_sec: float = 5.0
    home_plan: str = "PLAN-Home"
    home_joints: list[float] | None = None
    command_frequency: int = 50
    gripper_bin_threshold: float = 0.98
    gripper_max_open: float = 0.0801  # gripper max open width in meters
    gripper_force: float = 40.0
    gripper_always_grasp: bool = False
    debug: bool = True
    close_threshold: float = 0.7
    cameras: dict[str, CameraConfig] = field(default_factory=dict)
    control_mode: str = "isoteleop"
    # Execute mode for oculus: "ee_pose" (cartesian impedance) or "joint" (joint impedance via IK)
    execute_mode: str = "ee_pose"
    # "tdk_pose_joint" matches Flexiv TDK datasets:
    # [tcp pose qwxyz + 7 follower joint positions].
    policy_io_schema: str = "default"
    include_force_observation: bool = False
    force_observation_frame: str = "tcp"
