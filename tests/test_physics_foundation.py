from __future__ import annotations

import unittest

import numpy as np

from simulation.world import PhysicsWorld


class PhysicsFoundationTests(unittest.TestCase):
    def setUp(self) -> None:
        self.world = PhysicsWorld()

    def test_has_exactly_twelve_controllable_joints(self) -> None:
        self.assertEqual(len(self.world.robot.joint_names), 12)
        self.assertEqual(len(self.world.robot.actuator_names), 12)

    def test_manual_target_is_clamped_to_joint_limit(self) -> None:
        name = "front_left_knee"
        lower, upper = self.world.robot.joint_limits[name]
        self.assertEqual(self.world.set_target(name, 99.0), upper)
        self.assertEqual(self.world.set_target(name, -99.0), lower)

    def test_motor_torque_respects_configured_limit(self) -> None:
        self.world.set_target("front_left_hip_flexion", -1.2)
        self.world.step(5)
        maximum = self.world.config["motors"]["limits"]["max_torque_nm"]
        self.assertLessEqual(float(np.max(np.abs(self.world.last_torques))), maximum + 0.05)

    def test_free_drop_moves_body_down_under_gravity(self) -> None:
        self.world.reset("drop_test")
        start_z = float(self.world.data.qpos[2])
        self.world.step(12)
        self.assertLess(float(self.world.data.qpos[2]), start_z)

    def test_state_is_finite_and_contains_all_joint_rows(self) -> None:
        self.world.step(4)
        state = self.world.snapshot()
        self.assertEqual(len(state["joints"]), 12)
        self.assertTrue(np.isfinite(np.asarray(self.world.data.qpos)).all())
        self.assertTrue(np.isfinite(np.asarray(self.world.data.qvel)).all())


if __name__ == "__main__":
    unittest.main()
