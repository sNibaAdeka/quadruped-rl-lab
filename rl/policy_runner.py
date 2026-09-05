"""Runs whichever real PPO checkpoint training most recently produced."""
from __future__ import annotations

import json
from pathlib import Path

from stable_baselines3 import PPO

from rl.env import QuadrupedWalkEnv


class LivePolicyRunner:
    def __init__(self, root: Path) -> None:
        self.root = root
        self.policy_path = root / "models" / "live_policy.zip"
        self.state_path = root / "runs" / "training_state.json"
        self.env = QuadrupedWalkEnv()
        self.observation, _ = self.env.reset(seed=101)
        self.model: PPO | None = None
        self.loaded_mtime = -1.0
        self.visual_episodes = 0
        self.visual_reward = 0.0

    def training_state(self) -> dict:
        try:
            return json.loads(self.state_path.read_text(encoding="utf-8"))
        except (FileNotFoundError, json.JSONDecodeError):
            return {"status": "starting", "step": 0, "total_timesteps": 900_000, "episodes": 0, "mean_episode_reward": 0.0}

    def _reload_if_needed(self) -> None:
        try:
            mtime = self.policy_path.stat().st_mtime
        except FileNotFoundError:
            return
        if mtime <= self.loaded_mtime:
            return
        try:
            self.model = PPO.load(str(self.policy_path), device="cpu")
            self.loaded_mtime = mtime
        except (OSError, ValueError):
            # A checkpoint can be observed during its atomic replacement; keep the last valid policy.
            return

    def step(self) -> None:
        self._reload_if_needed()
        if self.model is None:
            return
        train_state = self.training_state()
        # During early learning the PPO distribution is sampled, not scripted.  Later the visible
        # robot uses the learned deterministic mean so improvement is observable.
        deterministic = int(train_state.get("step", 0)) >= 24_000
        action, _ = self.model.predict(self.observation, deterministic=deterministic)
        self.observation, reward, terminated, truncated, _ = self.env.step(action)
        self.visual_reward += float(reward)
        if terminated or truncated:
            self.visual_episodes += 1
            self.observation, _ = self.env.reset()
            self.visual_reward = 0.0

    def snapshot(self) -> dict:
        state = self.env.world.snapshot()
        train = self.training_state()
        state["training"] = {
            "status": train.get("status", "starting"),
            "algorithm": train.get("algorithm", "PPO"),
            "step": int(train.get("step", 0)),
            "total_timesteps": int(train.get("total_timesteps", 900_000)),
            "episodes": int(train.get("episodes", 0)),
            "mean_episode_reward": float(train.get("mean_episode_reward", 0.0)),
            "visual_episodes": self.visual_episodes,
            "policy_mode": "LEARNED POLICY" if int(train.get("step", 0)) >= 24_000 else "PPO EXPLORATION",
        }
        return state
