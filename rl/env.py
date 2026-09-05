"""A real Gymnasium task: a 12-DOF robot tracks a forward velocity in MuJoCo.

There is intentionally no gait generator, animation, IK controller or
hand-authored foot trajectory in this file.  PPO receives observations and
produces only 12 joint-position targets; the non-ideal motor model and MuJoCo
contacts decide the resulting movement.
"""
from __future__ import annotations

from typing import Any

import gymnasium as gym
from gymnasium import spaces
import mujoco
import numpy as np

from simulation.world import PhysicsWorld


class QuadrupedWalkEnv(gym.Env):
    metadata = {"render_modes": []}

    def __init__(self, command_vx: float = 0.32, max_steps: int = 900) -> None:
        super().__init__()
        self.world = PhysicsWorld()
        self.command = np.array([command_vx, 0.0, 0.0], dtype=np.float32)  # vx, vy, yaw rate
        self.max_steps = max_steps
        self.step_count = 0
        self.previous_action = np.zeros(12, dtype=np.float32)
        self.last_energy = 0.0
        self.episode_reward = 0.0
        self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(12,), dtype=np.float32)
        # orientation, angular velocity, linear velocity, 12 positions, 12 velocities,
        # 4 contacts, 3 commanded velocities, and previous 12 actions
        self.observation_space = spaces.Box(low=-1.0, high=1.0, shape=(52,), dtype=np.float32)

    def _joint_values(self) -> tuple[np.ndarray, np.ndarray]:
        positions, velocities = [], []
        for name in self.world.robot.joint_names:
            joint_id = self.world.robot.joint_id(name)
            positions.append(self.world.data.qpos[int(self.world.model.jnt_qposadr[joint_id])])
            velocities.append(self.world.data.qvel[int(self.world.model.jnt_dofadr[joint_id])])
        return np.asarray(positions, dtype=np.float32), np.asarray(velocities, dtype=np.float32)

    def observation(self) -> np.ndarray:
        snapshot = self.world.snapshot()
        base = snapshot["base"]
        position, velocity = self._joint_values()
        ranges = np.asarray([self.world.robot.joint_limits[name] for name in self.world.robot.joint_names], dtype=np.float32)
        center = ranges.mean(axis=1)
        half_range = np.maximum(.01, (ranges[:, 1] - ranges[:, 0]) / 2)
        normalized_position = np.clip((position - center) / half_range, -1, 1)
        contacts = np.asarray([float(snapshot["contacts"][leg]) for leg in ("front_left", "front_right", "rear_left", "rear_right")], dtype=np.float32)
        values = np.concatenate((
            np.clip(np.asarray(base["euler_rad"], dtype=np.float32) / np.pi, -1, 1),
            np.clip(self.world.data.qvel[3:6].astype(np.float32) / 12.0, -1, 1),
            np.clip(self.world.data.qvel[0:3].astype(np.float32) / 4.0, -1, 1),
            normalized_position,
            np.clip(velocity / 18.0, -1, 1),
            contacts,
            self.command / np.asarray([1.0, 1.0, 2.0], dtype=np.float32),
            self.previous_action,
        )).astype(np.float32)
        return values

    def _targets_from_action(self, action: np.ndarray) -> np.ndarray:
        ranges = np.asarray([self.world.robot.joint_limits[name] for name in self.world.robot.joint_names], dtype=np.float32)
        neutral = self.world.neutral_pose().astype(np.float32)
        half_range = (ranges[:, 1] - ranges[:, 0]) * .5
        # Relative targets preserve a useful standing region while still leaving enough
        # motion to discover a gait. The policy is not given a gait or phase signal.
        return np.clip(neutral + np.clip(action, -1, 1) * half_range * .58, ranges[:, 0], ranges[:, 1])

    def apply_action(self, action: np.ndarray) -> None:
        targets = self._targets_from_action(np.asarray(action, dtype=np.float32))
        for name, target in zip(self.world.robot.joint_names, targets):
            self.world.motor.set_target(name, float(target))
        self.world.mode = "RL POLICY"

    def _fallen(self) -> bool:
        roll, pitch, _ = self.world.snapshot()["base"]["euler_rad"]
        return self.world.data.qpos[2] < .23 or abs(roll) > .82 or abs(pitch) > .82

    def reward(self, action: np.ndarray, terminated: bool) -> tuple[float, dict[str, float]]:
        snap = self.world.snapshot()
        roll, pitch, _ = snap["base"]["euler_rad"]
        height = float(self.world.data.qpos[2])
        vx, vy = map(float, self.world.data.qvel[0:2])
        upright = float(np.exp(-5.0 * (roll * roll + pitch * pitch)))
        height_reward = float(np.exp(-110.0 * (height - .37) ** 2))
        velocity_tracking = float(np.exp(-12.0 * ((vx - self.command[0]) ** 2 + (vy - self.command[1]) ** 2)))
        smoothness = float(np.mean((action - self.previous_action) ** 2))
        energy = max(0.0, snap["energy_j"] - self.last_energy)
        speed_shortfall = max(0.0, float(self.command[0] - vx))
        # This is velocity tracking, not a reward for absolute X position.  Remaining
        # motionless is explicitly worse than matching the commanded speed.
        total = .75 * upright + .65 * height_reward + 2.5 * velocity_tracking + .9 * np.clip(vx, -.25, .60) - 2.0 * speed_shortfall - .018 * energy - .030 * smoothness
        if terminated:
            total -= 8.0
        return float(total), {"upright": upright, "height": height_reward, "velocity_tracking": velocity_tracking, "energy": energy, "smoothness": smoothness}

    def reset(self, *, seed: int | None = None, options: dict[str, Any] | None = None):
        super().reset(seed=seed)
        self.world.reset("motor_test")
        # Small pose noise makes the policy learn balance rather than memorising one reset state.
        for name in self.world.robot.joint_names:
            joint_id = self.world.robot.joint_id(name)
            qpos = int(self.world.model.jnt_qposadr[joint_id])
            lower, upper = self.world.robot.joint_limits[name]
            self.world.data.qpos[qpos] = np.clip(self.world.data.qpos[qpos] + self.np_random.uniform(-.035, .035), lower, upper)
        self.world.data.qpos[0:2] = self.np_random.uniform(-.03, .03, size=2)
        mujoco.mj_forward(self.world.model, self.world.data)
        self.step_count = 0
        self.previous_action.fill(0.0)
        self.last_energy = self.world.snapshot()["energy_j"]
        self.episode_reward = 0.0
        return self.observation(), {"command_vx": float(self.command[0])}

    def step(self, action: np.ndarray):
        action = np.asarray(action, dtype=np.float32)
        self.apply_action(action)
        self.world.step()
        self.step_count += 1
        terminated = self._fallen()
        truncated = self.step_count >= self.max_steps
        reward, parts = self.reward(action, terminated)
        self.episode_reward += reward
        self.previous_action = action.copy()
        self.last_energy = self.world.snapshot()["energy_j"]
        info = {"reward_parts": parts, "base_height": float(self.world.data.qpos[2]), "forward_velocity": float(self.world.data.qvel[0])}
        return self.observation(), reward, terminated, truncated, info
