"""Non-ideal PD motor bank used by the phase-3 manual test mode."""
from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from typing import Iterable

import numpy as np

from simulation.robot import QuadrupedModel


@dataclass
class MotorTelemetry:
    target_rad: float
    position_rad: float
    velocity_rad_s: float
    torque_nm: float
    temperature_c: float
    energy_j: float


class MotorBank:
    """Turns desired joint angles into bounded, delayed and noisy torques.

    The UI never sets a joint pose directly. It only changes a target; MuJoCo
    advances the resulting torque and contact dynamics.
    """

    def __init__(self, robot: QuadrupedModel, config: dict, timestep: float, seed: int = 7) -> None:
        self.robot = robot
        self.config = config
        self.dt = timestep
        self.names = robot.joint_names
        self.rng = np.random.default_rng(seed)
        limits = config["limits"]
        self.max_torque = float(limits["max_torque_nm"])
        self.max_speed = float(limits["max_angular_velocity_rad_s"])
        self.max_target_speed = float(limits["max_target_velocity_rad_s"])
        self.max_target_accel = float(limits["max_target_acceleration_rad_s2"])
        self.delay_steps = max(1, round(float(limits["response_delay_s"]) / timestep))
        thermal = config["thermal_proxy"]
        self.ambient = float(thermal["ambient_c"])
        self.heat_per_joule = float(thermal["heating_per_joule"])
        self.cooling = float(thermal["cooling_per_second"])
        self.shutdown = float(thermal["shutdown_c"])
        self.desired = np.zeros(len(self.names))
        self.applied = np.zeros(len(self.names))
        self.target_velocity = np.zeros(len(self.names))
        self.temperature = np.full(len(self.names), float(thermal["initial_c"]))
        self.energy = np.zeros(len(self.names))
        self.buffers = [deque([0.0] * self.delay_steps, maxlen=self.delay_steps) for _ in self.names]

    def reset(self, targets: Iterable[float]) -> None:
        values = np.asarray(list(targets), dtype=float)
        self.desired[:] = values
        self.applied[:] = values
        self.target_velocity.fill(0.0)
        self.energy.fill(0.0)
        self.temperature.fill(float(self.config["thermal_proxy"]["initial_c"]))
        self.buffers = [deque([target] * self.delay_steps, maxlen=self.delay_steps) for target in values]

    def set_target(self, joint: str, radians: float) -> float:
        index = self.names.index(joint)
        lower, upper = self.robot.joint_limits[joint]
        value = float(np.clip(radians, lower, upper))
        self.desired[index] = value
        return value

    def targets(self) -> dict[str, float]:
        return dict(zip(self.names, self.desired.tolist()))

    def apply(self, data) -> np.ndarray:
        """Return a physical torque command for each MuJoCo motor actuator."""
        kp = float(self.config["pd"]["kp"])
        kd = float(self.config["pd"]["kd"])
        noise = float(self.config["limits"]["torque_noise_std_nm"])
        torques = np.zeros(len(self.names))
        for index, name in enumerate(self.names):
            self.buffers[index].append(float(self.desired[index]))
            delayed = self.buffers[index][0]
            requested_velocity = np.clip((delayed - self.applied[index]) * 16.0, -self.max_target_speed, self.max_target_speed)
            self.target_velocity[index] += np.clip(requested_velocity - self.target_velocity[index], -self.max_target_accel * self.dt, self.max_target_accel * self.dt)
            self.applied[index] += self.target_velocity[index] * self.dt
            joint_id = self.robot.joint_id(name)
            qpos = int(self.robot.model.jnt_qposadr[joint_id])
            dof = int(self.robot.model.jnt_dofadr[joint_id])
            position, velocity = float(data.qpos[qpos]), float(data.qvel[dof])
            speed_factor = max(0.0, 1.0 - abs(velocity) / self.max_speed)
            thermal_factor = max(0.0, 1.0 - max(0.0, self.temperature[index] - 70.0) / 40.0)
            torque = kp * (self.applied[index] - position) - kd * velocity
            torque = float(np.clip(torque, -self.max_torque * speed_factor * thermal_factor, self.max_torque * speed_factor * thermal_factor))
            if self.temperature[index] >= self.shutdown:
                torque = 0.0
            torque += float(self.rng.normal(0.0, noise))
            power = abs(torque * velocity)
            self.energy[index] += power * self.dt
            self.temperature[index] += self.heat_per_joule * power * self.dt - self.cooling * (self.temperature[index] - self.ambient) * self.dt
            torques[index] = torque
        return torques

    def telemetry(self, data, applied_torques: np.ndarray) -> list[MotorTelemetry]:
        rows: list[MotorTelemetry] = []
        for index, name in enumerate(self.names):
            joint_id = self.robot.joint_id(name)
            rows.append(MotorTelemetry(
                target_rad=round(float(self.desired[index]), 4),
                position_rad=round(float(data.qpos[int(self.robot.model.jnt_qposadr[joint_id])]), 4),
                velocity_rad_s=round(float(data.qvel[int(self.robot.model.jnt_dofadr[joint_id])]), 4),
                torque_nm=round(float(applied_torques[index]), 4),
                temperature_c=round(float(self.temperature[index]), 2),
                energy_j=round(float(self.energy[index]), 4),
            ))
        return rows
