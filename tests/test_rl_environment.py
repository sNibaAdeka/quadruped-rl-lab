from __future__ import annotations

import unittest

import numpy as np

from rl.env import QuadrupedWalkEnv


class WalkEnvironmentTests(unittest.TestCase):
    def test_observation_and_action_shapes_are_consistent(self) -> None:
        env = QuadrupedWalkEnv(max_steps=5)
        observation, _ = env.reset(seed=4)
        self.assertEqual(observation.shape, (52,))
        self.assertTrue(env.observation_space.contains(observation))
        next_observation, reward, terminated, truncated, _ = env.step(np.zeros(12, dtype=np.float32))
        self.assertEqual(next_observation.shape, (52,))
        self.assertTrue(np.isfinite(reward))
        self.assertIsInstance(terminated, bool)
        self.assertIsInstance(truncated, bool)
        env.close()


if __name__ == "__main__":
    unittest.main()
