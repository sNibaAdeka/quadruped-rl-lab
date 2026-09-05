"""Construct the 12-DOF engineering quadruped as a MuJoCo model.

This module owns geometry, joints and collision shapes only.  It contains no
locomotion sequence, inverse kinematics or learned policy.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import mujoco


LEGS = (
    ("front_left", 0.20, 0.135),
    ("front_right", 0.20, -0.135),
    ("rear_left", -0.20, 0.135),
    ("rear_right", -0.20, -0.135),
)
JOINT_KINDS = ("hip_abduction", "hip_flexion", "knee")


@dataclass(frozen=True)
class QuadrupedModel:
    model: mujoco.MjModel
    joint_names: tuple[str, ...]
    actuator_names: tuple[str, ...]
    joint_limits: dict[str, tuple[float, float]]

    def joint_id(self, name: str) -> int:
        return mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, name)

    def actuator_id(self, name: str) -> int:
        return mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_ACTUATOR, name)


def _range(values: list[float]) -> str:
    return f"{values[0]} {values[1]}"


def _leg_xml(name: str, x: float, y: float, robot: dict[str, Any]) -> str:
    upper = robot["leg"]["upper_length_m"]
    lower = robot["leg"]["lower_length_m"]
    radius = robot["leg"]["foot_radius_m"]
    j = robot["joints"]
    return f'''
      <body name="{name}_hip_mount" pos="{x} {y} 0">
        <joint name="{name}_hip_abduction" type="hinge" axis="1 0 0" range="{_range(j['hip_abduction']['range_rad'])}" damping="{j['hip_abduction']['damping']}" frictionloss="{j['hip_abduction']['friction_loss']}"/>
        <geom name="{name}_hip_housing" type="sphere" size="0.045" mass="0.045" material="joint"/>
        <body name="{name}_upper" pos="0 0 0">
          <joint name="{name}_hip_flexion" type="hinge" axis="0 1 0" range="{_range(j['hip_flexion']['range_rad'])}" damping="{j['hip_flexion']['damping']}" frictionloss="{j['hip_flexion']['friction_loss']}"/>
          <geom name="{name}_upper_link" type="capsule" fromto="0 0 0 0 0 -{upper}" size="0.030" mass="{robot['leg']['upper_mass_kg']}" material="link"/>
          <body name="{name}_knee_mount" pos="0 0 -{upper}">
            <joint name="{name}_knee" type="hinge" axis="0 1 0" range="{_range(j['knee']['range_rad'])}" damping="{j['knee']['damping']}" frictionloss="{j['knee']['friction_loss']}"/>
            <geom name="{name}_knee_housing" type="sphere" size="0.037" mass="0.035" material="joint"/>
            <geom name="{name}_lower_link" type="capsule" fromto="0 0 0 0 0 -{lower}" size="0.027" mass="{robot['leg']['lower_mass_kg']}" material="link"/>
            <body name="{name}_foot" pos="0 0 -{lower}">
              <geom name="{name}_foot_geom" type="sphere" size="{radius}" mass="0.055" friction="1.2 0.02 0.002" material="foot"/>
            </body>
          </body>
        </body>
      </body>'''


def build_quadruped(robot: dict[str, Any], motors: dict[str, Any], physics: dict[str, Any]) -> QuadrupedModel:
    """Build a free-floating chassis with four 3-DOF legs and torque motors."""
    body = robot["body"]
    dims = body["dimensions_m"]
    half = [value / 2 for value in dims]
    legs = "\n".join(_leg_xml(name, x, y, robot) for name, x, y in LEGS)
    joint_names = tuple(f"{leg}_{kind}" for leg, _, _ in LEGS for kind in JOINT_KINDS)
    torque = motors["limits"]["max_torque_nm"]
    actuators = "\n".join(
        f'<motor name="{name}_motor" joint="{name}" ctrllimited="true" ctrlrange="-{torque} {torque}" forcelimited="true" forcerange="-{torque} {torque}"/>'
        for name in joint_names
    )
    xml = f'''<mujoco model="qrl12_phase_1_to_3">
  <compiler angle="radian" coordinate="local"/>
  <option timestep="{physics['timestep_s']}" gravity="{' '.join(map(str, physics['gravity_m_s2']))}" integrator="{physics['solver']}"/>
  <size nconmax="256" njmax="1024"/>
  <default>
    <geom condim="4" friction="{physics['ground_friction']} 0.02 0.002" margin="{physics['contact_margin_m']}"/>
  </default>
  <asset>
    <material name="chassis" rgba="0.16 0.53 0.74 1"/>
    <material name="link" rgba="0.72 0.84 0.89 1"/>
    <material name="joint" rgba="0.06 0.12 0.17 1"/>
    <material name="foot" rgba="0.93 0.53 0.13 1"/>
    <material name="ground" rgba="0.15 0.19 0.23 1"/>
  </asset>
  <worldbody>
    <light name="key" pos="-2 -3 5" dir="2 3 -5" diffuse="0.9 0.94 1"/>
    <light name="fill" pos="3 2 3" dir="-3 -2 -3" diffuse="0.35 0.55 0.72"/>
    <geom name="ground" type="plane" size="{robot['world_size_m']} {robot['world_size_m']} 0.1" material="ground"/>
    <body name="robot_base" pos="0 0 {body['start_height_m']}">
      <freejoint name="base_free"/>
      <geom name="chassis" type="box" size="{half[0]} {half[1]} {half[2]}" mass="{body['mass_kg']}" material="chassis"/>
      <site name="center_of_mass" pos="{' '.join(map(str, body['center_of_mass_offset_m']))}" size="0.012" rgba="1 0.2 0.2 1"/>
      {legs}
    </body>
  </worldbody>
  <actuator>{actuators}</actuator>
</mujoco>'''
    model = mujoco.MjModel.from_xml_string(xml)
    limits = {name: tuple(map(float, robot["joints"][name.rsplit("_", 1)[-1] if name.endswith("knee") else "hip_abduction"].get("range_rad", [-1, 1]))) for name in ()}
    # Name parsing is deliberately explicit: it makes the 12 controllable DOF auditable.
    limits = {}
    for name in joint_names:
        kind = next(kind for kind in JOINT_KINDS if name.endswith(kind))
        limits[name] = tuple(float(value) for value in robot["joints"][kind]["range_rad"])
    return QuadrupedModel(model, joint_names, tuple(f"{name}_motor" for name in joint_names), limits)
