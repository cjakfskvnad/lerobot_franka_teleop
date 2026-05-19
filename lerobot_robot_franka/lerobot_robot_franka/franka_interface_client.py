"""Flexiv RDK adapter with the legacy FrankaInterfaceClient API.

The rest of this teleoperation project was written against a small Franka/Polymetis
client.  Keeping the same method names lets the LeRobot integration use Flexiv
without rewriting the recording, replay, and teleoperator paths.
"""

from __future__ import annotations

import logging
import time
from typing import Iterable

import numpy as np
from scipy.spatial.transform import Rotation as R

log = logging.getLogger(__name__)


class FrankaInterfaceClient:
    """Compatibility wrapper around ``flexivrdk.Robot`` and ``flexivrdk.Gripper``.

    ``ip`` is kept for backward compatibility, but for Flexiv it should contain the
    robot serial number, for example ``Rizon4s-123456``.  Optionally pass
    ``network_interface`` as the local network interface IP if discovery needs to be
    restricted.
    """

    def __init__(
        self,
        ip: str = "Rizon4s-123456",
        port: int = 4242,
        *,
        robot_sn: str | None = None,
        network_interface: str | None = None,
        gripper_name: str | None = None,
        command_frequency: int = 50,
        home_plan: str = "PLAN-Home",
    ):
        del port  # Kept only to preserve the legacy constructor signature.
        try:
            import flexivrdk
        except ImportError as exc:
            raise ImportError(
                "flexivrdk is required for Flexiv control. Install the Flexiv RDK Python "
                "package in the active environment."
            ) from exc

        self._flexivrdk = flexivrdk
        self._mode = flexivrdk.Mode
        self._robot_sn = robot_sn or ip
        self._network_interface = network_interface
        self._gripper_name = gripper_name
        self._home_plan = home_plan
        self._period = 1.0 / command_frequency
        self._gripper = None

        whitelist = [network_interface] if network_interface else []
        self.robot = flexivrdk.Robot(self._robot_sn, whitelist)
        self._prepare_robot()
        log.info("Connected to Flexiv robot %s", self._robot_sn)

    def _prepare_robot(self) -> None:
        if self.robot.fault():
            log.warning("Fault occurred on Flexiv robot, trying to clear ...")
            if not self.robot.ClearFault():
                raise RuntimeError("Flexiv robot fault cannot be cleared")

        if not self.robot.operational():
            log.info("Enabling Flexiv robot ...")
            self.robot.Enable()
            while not self.robot.operational():
                time.sleep(0.5)

    @staticmethod
    def _as_list(values: Iterable[float]) -> list[float]:
        return [float(v) for v in values]

    @staticmethod
    def _tcp_pose_to_rotvec_pose(tcp_pose: Iterable[float]) -> np.ndarray:
        pose = np.asarray(tcp_pose, dtype=float)
        quat_xyzw = [pose[4], pose[5], pose[6], pose[3]]
        rotvec = R.from_quat(quat_xyzw).as_rotvec()
        return np.concatenate([pose[:3], rotvec])

    @staticmethod
    def _rotvec_pose_to_tcp_pose(pose: Iterable[float]) -> list[float]:
        pose = np.asarray(pose, dtype=float)
        quat_xyzw = R.from_rotvec(pose[3:6]).as_quat()
        return [
            float(pose[0]),
            float(pose[1]),
            float(pose[2]),
            float(quat_xyzw[3]),
            float(quat_xyzw[0]),
            float(quat_xyzw[1]),
            float(quat_xyzw[2]),
        ]

    def _switch_mode(self, mode) -> None:
        if self.robot.mode() != mode:
            self.robot.SwitchMode(mode)

    def gripper_initialize(self):
        if not self._gripper_name:
            log.warning("No Flexiv gripper_name configured; gripper control is disabled")
            return
        self._gripper = self._flexivrdk.Gripper(self.robot)
        self._gripper.Enable(self._gripper_name)
        log.info("Connected to Flexiv gripper %s", self._gripper_name)

    def gripper_goto(
        self,
        width: float,
        speed: float,
        force: float,
        epsilon_inner: float = -1.0,
        epsilon_outer: float = -1.0,
        blocking: bool = True,
    ):
        del epsilon_inner, epsilon_outer
        if self._gripper is None:
            return
        self._gripper.Move(float(width), float(speed), float(force))
        if blocking:
            while self._gripper.states().is_moving:
                time.sleep(0.02)

    def gripper_grasp(
        self,
        speed: float,
        force: float,
        grasp_width: float = 0.0,
        epsilon_inner: float = -1.0,
        epsilon_outer: float = -1.0,
        blocking: bool = True,
    ):
        del speed, grasp_width, epsilon_inner, epsilon_outer, blocking
        if self._gripper is not None:
            self._gripper.Grasp(float(force))

    def gripper_get_state(self) -> dict:
        if self._gripper is None:
            return {"width": 0.0, "force": 0.0, "is_moving": False}
        state = self._gripper.states()
        return {
            "width": float(state.width),
            "force": float(state.force),
            "is_moving": bool(state.is_moving),
        }

    def robot_get_joint_positions(self):
        return np.asarray(self.robot.states().q[:7], dtype=float)

    def robot_get_joint_velocities(self):
        return np.asarray(self.robot.states().dq[:7], dtype=float)

    def robot_get_ee_pose(self):
        return self._tcp_pose_to_rotvec_pose(self.robot.states().tcp_pose)

    def robot_move_to_joint_positions(
        self,
        positions: np.ndarray,
        time_to_go: float | None = None,
        delta: bool = False,
        Kq: np.ndarray | None = None,
        Kqd: np.ndarray | None = None,
    ):
        del Kq, Kqd
        self.robot_start_joint_impedance_control()
        current = self.robot_get_joint_positions()
        target = np.asarray(positions, dtype=float)
        if delta:
            target = current + target

        duration = float(time_to_go or 0.0)
        steps = max(1, int(duration / self._period))
        for pos in np.linspace(current, target, steps):
            self.robot_update_desired_joint_positions(pos)
            time.sleep(self._period)

    def robot_go_home(self):
        self._switch_mode(self._mode.NRT_PLAN_EXECUTION)
        self.robot.ExecutePlan(self._home_plan)
        while self.robot.busy():
            time.sleep(0.2)

    def robot_move_to_ee_pose(
        self,
        pose: np.ndarray,
        time_to_go: float | None = None,
        delta: bool = False,
        Kx: np.ndarray | None = None,
        Kxd: np.ndarray | None = None,
        op_space_interp: bool = True,
    ):
        del Kx, Kxd, op_space_interp
        self.robot_start_cartesian_impedance_control(None, None)
        target = np.asarray(pose, dtype=float)
        current = self.robot_get_ee_pose()
        if delta:
            target = current + target

        duration = float(time_to_go or 0.0)
        steps = max(1, int(duration / self._period))
        for interp_pose in np.linspace(current, target, steps):
            self.robot_update_desired_ee_pose(interp_pose)
            time.sleep(self._period)

    def robot_start_joint_impedance_control(
        self,
        Kq: np.ndarray | None = None,
        Kqd: np.ndarray | None = None,
        adaptive: bool = True,
    ):
        del Kq, Kqd, adaptive
        self._switch_mode(self._mode.NRT_JOINT_POSITION)
        log.info("[ROBOT] Flexiv joint position control started")

    def robot_start_cartesian_impedance_control(self, Kx: np.ndarray | None, Kxd: np.ndarray | None):
        del Kx, Kxd
        self._switch_mode(self._mode.NRT_CARTESIAN_MOTION_FORCE)
        self.robot.SetForceControlAxis([False, False, False, False, False, False])
        log.info("[ROBOT] Flexiv cartesian motion control started")

    def robot_update_desired_joint_positions(self, positions: np.ndarray):
        dof = len(self.robot.info().q_min) or 7
        target_pos = self._as_list(positions)[:dof]
        target_vel = [0.0] * len(target_pos)
        max_vel = [2.0] * len(target_pos)
        max_acc = [3.0] * len(target_pos)
        self.robot.SendJointPosition(target_pos, target_vel, max_vel, max_acc)

    def robot_update_desired_ee_pose(self, pose: np.ndarray):
        self.robot.SendCartesianMotionForce(self._rotvec_pose_to_tcp_pose(pose))

    def robot_terminate_current_policy(self):
        self.robot.Stop()

    def close(self):
        if self._gripper is not None:
            try:
                self._gripper.Stop()
            except Exception:
                log.exception("Failed to stop Flexiv gripper")
        self.robot.Stop()


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    flexiv = FrankaInterfaceClient(ip="Rizon4s-123456")
    flexiv.robot_go_home()
    print(f"Current joint positions: {flexiv.robot_get_joint_positions()}")
