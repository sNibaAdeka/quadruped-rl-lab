"""Small real-time MuJoCo world used by the motor test mode."""
from __future__ import annotations

from math import asin, atan2
from typing import Any

import mujoco
import numpy as np

from config import load_all
from robot.motors import MotorBank
from simulation.robot import JOINT_KINDS, QuadrupedModel, build_quadruped


def _euler_from_wxyz(quaternion: np.ndarray) -> tuple[float, float, float]:
    w, x, y, z = map(float, quaternion)
    roll = atan2(2 * (w * x + y * z), 1 - 2 * (x * x + y * y))
    pitch = asin(float(np.clip(2 * (w * y - z * x), -1.0, 1.0)))
    yaw = atan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z))
    return roll, pitch, yaw


class PhysicsWorld:
    """A live world.  It provides no policy and no animation controller."""

    def __init__(self, configs: dict[str, dict[str, Any]] | None = None) -> None:
        self.config = configs or load_all()
        self.robot: QuadrupedModel = build_quadruped(self.config["robot"], self.config["motors"], self.config["physics"])
        self.model = self.robot.model
        self.data = mujoco.MjData(self.model)
        self.control_dt = 1 / self.config["motors"]["control_frequency_hz"]
        self.substeps = int(self.config["motors"]["physics_substeps"])
        self.motor = MotorBank(self.robot, self.config["motors"], self.control_dt)
        self.last_torques = np.zeros(len(self.robot.joint_names))
        self.paused = False
        self.mode = "MOTOR TEST"
        self.reset("motor_test")

    def neutral_pose(self) -> np.ndarray:
        defaults = self.config["robot"]["joint_defaults_rad"]
        return np.array([defaults[next(kind for kind in JOINT_KINDS if name.endswith(kind))] for name in self.robot.joint_names], dtype=float)

    def reset(self, mode: str = "motor_test") -> None:
        mujoco.mj_resetData(self.model, self.data)
        self.data.qpos[0:7] = [0, 0, self.config["robot"]["body"]["start_height_m"], 1, 0, 0, 0]
        initial = self.neutral_pose()
        for name, value in zip(self.robot.joint_names, initial):
            self.data.qpos[int(self.model.jnt_qposadr[self.robot.joint_id(name)])] = value
        self.motor.reset(initial)
        self.last_torques.fill(0.0)
        self.mode = "FREE DROP" if mode == "drop_test" else "MOTOR TEST"
        mujoco.mj_forward(self.model, self.data)

    def set_target(self, joint: str, target_rad: float) -> float:
        if joint not in self.robot.joint_names:
            raise KeyError(joint)
        self.mode = "MOTOR TEST"
        return self.motor.set_target(joint, target_rad)

    def restore_neutral(self) -> None:
        for name, target in zip(self.robot.joint_names, self.neutral_pose()):
            self.motor.set_target(name, float(target))
        self.mode = "MOTOR TEST"

    def step(self, control_frames: int = 1) -> None:
        for _ in range(control_frames):
            if self.mode == "FREE DROP":
                self.last_torques.fill(0.0)
                self.data.ctrl[:] = 0.0
            else:
                self.last_torques = self.motor.apply(self.data)
                self.data.ctrl[:] = self.last_torques
            mujoco.mj_step(self.model, self.data, nstep=self.substeps)
            if not np.isfinite(self.data.qpos).all() or not np.isfinite(self.data.qvel).all():
                self.reset("motor_test")
                raise FloatingPointError("MuJoCo produced a non-finite state; world was reset.")

    def snapshot(self) -> dict[str, Any]:
        position = self.data.qpos[0:3].copy()
        roll, pitch, yaw = _euler_from_wxyz(self.data.qpos[3:7])
        contacts = {name: False for name in ("front_left", "front_right", "rear_left", "rear_right")}
        for index in range(self.data.ncon):
            contact = self.data.contact[index]
            first = mujoco.mj_id2name(self.model, mujoco.mjtObj.mjOBJ_GEOM, contact.geom1) or ""
            second = mujoco.mj_id2name(self.model, mujoco.mjtObj.mjOBJ_GEOM, contact.geom2) or ""
            for leg in contacts:
                if f"{leg}_foot_geom" in (first, second):
                    contacts[leg] = True
        telemetry = self.motor.telemetry(self.data, self.last_torques)
        joints = []
        for name, row in zip(self.robot.joint_names, telemetry):
            lower, upper = self.robot.joint_limits[name]
            joints.append({"name": name, "kind": next(kind for kind in JOINT_KINDS if name.endswith(kind)), "limit": [lower, upper], **row.__dict__})
        return {
            "phase": "PHASE 3 · MANUAL MOTOR TEST",
            "mode": self.mode,
            "paused": self.paused,
            "sim_time_s": round(float(self.data.time), 3),
            "base": {
                "position_m": [round(float(value), 4) for value in position],
                "quaternion_wxyz": [round(float(value), 5) for value in self.data.qpos[3:7]],
                "euler_rad": [round(roll, 4), round(pitch, 4), round(yaw, 4)],
                "linear_velocity_m_s": [round(float(value), 4) for value in self.data.qvel[0:3]],
            },
            "joints": joints,
            "contacts": contacts,
            "energy_j": round(float(np.sum(self.motor.energy)), 4),
            "motor_mode": self.config["motors"]["mode"],
            "control_frequency_hz": self.config["motors"]["control_frequency_hz"],
            "physics_timestep_s": self.config["physics"]["timestep_s"],
        }
