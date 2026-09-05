"""Headless PPO training with atomic live checkpoints for the visualiser."""
from __future__ import annotations

import argparse
import json
import os
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.monitor import Monitor

from rl.env import QuadrupedWalkEnv


ROOT = Path(__file__).resolve().parents[1]
MODELS = ROOT / "models"
RUNS = ROOT / "runs"
STATE_PATH = RUNS / "training_state.json"


def write_state(**updates) -> None:
    RUNS.mkdir(parents=True, exist_ok=True)
    previous = {}
    if STATE_PATH.exists():
        try:
            previous = json.loads(STATE_PATH.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            pass
    previous.update(updates)
    STATE_PATH.write_text(json.dumps(previous, ensure_ascii=False), encoding="utf-8")


class LiveTrainingCallback(BaseCallback):
    def __init__(self, total_timesteps: int, save_every: int = 2_048) -> None:
        super().__init__()
        self.total_timesteps = total_timesteps
        self.save_every = save_every
        self.episodes = 0
        self.recent_rewards: list[float] = []

    def _save_live_policy(self) -> None:
        MODELS.mkdir(parents=True, exist_ok=True)
        temporary = MODELS / "live_policy_tmp.zip"
        final = MODELS / "live_policy.zip"
        self.model.save(str(temporary))
        os.replace(temporary, final)

    def _on_training_start(self) -> None:
        self._save_live_policy()  # An untrained PPO policy is still a real policy for early visual episodes.

    def _on_step(self) -> bool:
        for info in self.locals.get("infos", []):
            episode = info.get("episode")
            if episode:
                self.episodes += 1
                self.recent_rewards.append(float(episode["r"]))
                self.recent_rewards = self.recent_rewards[-80:]
        if self.num_timesteps % 256 == 0:
            mean_reward = float(np.mean(self.recent_rewards)) if self.recent_rewards else 0.0
            write_state(status="running", algorithm="PPO", total_timesteps=self.total_timesteps, step=self.num_timesteps, episodes=self.episodes, mean_episode_reward=round(mean_reward, 3), policy_path="models/live_policy.zip", updated_at=datetime.now(timezone.utc).isoformat())
        if self.num_timesteps % self.save_every == 0:
            self._save_live_policy()
        return True


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--timesteps", type=int, default=900_000)
    args = parser.parse_args()
    MODELS.mkdir(exist_ok=True); RUNS.mkdir(exist_ok=True)
    write_state(status="starting", algorithm="PPO", objective="forward velocity tracking at 0.32 m/s", total_timesteps=args.timesteps, step=0, episodes=0, mean_episode_reward=0.0, started_at=datetime.now(timezone.utc).isoformat())
    env = Monitor(QuadrupedWalkEnv())
    model = PPO("MlpPolicy", env, verbose=0, n_steps=1_024, batch_size=128, learning_rate=3e-4, gamma=.99, gae_lambda=.95, ent_coef=.003, device="cpu", tensorboard_log=str(RUNS / "tensorboard"), seed=42)
    callback = LiveTrainingCallback(args.timesteps)
    try:
        model.learn(total_timesteps=args.timesteps, callback=callback, progress_bar=False)
        callback._save_live_policy()
        write_state(status="complete", step=args.timesteps, completed_at=datetime.now(timezone.utc).isoformat())
    except Exception as error:
        write_state(status="failed", error=f"{type(error).__name__}: {error}", updated_at=datetime.now(timezone.utc).isoformat())
        raise
    finally:
        env.close()


if __name__ == "__main__":
    main()
