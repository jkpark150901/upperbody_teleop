"""mink QP-based alternative to RBY1UpperBodyRetargeter's closed-form arm
solve.

Calibration, --bvh-scale and target computation (_targets() /
_wrist_rotation_target()) are reused unchanged from the base retargeter --
only how the elbow/wrist targets get turned into joint angles changes:
from the closed-form ZXZ/ZYZ branch search (robot_hand/upper_body_retarget.py)
to mink's damped-least-squares QP over the whole 7-DOF arm chain at once,
same design mink itself uses in humanoid-retarget/GMR (see
bvh_upper_body_mapping.md). Needs `pip install mink`, which is NOT a hard
dependency of the rest of this project -- only imported here, so nothing
else breaks if it's absent.
"""
from __future__ import annotations

from typing import Dict

import mink
import mujoco
import numpy as np

from robot_hand.mujoco_urdf import mujoco_compatible_urdf
from robot_hand.upper_body_retarget import RBY1UpperBodyRetargeter, UpperBodyLandmarks

_ARM_SIDES = ("left", "right")
_ARM_JOINTS = tuple(f"{side}_arm_{i}" for side in _ARM_SIDES for i in range(7))


class MinkUpperBodyIK:
    def __init__(
        self,
        base: RBY1UpperBodyRetargeter,
        urdf_path,
        solver: str = "daqp",
        elbow_position_cost: float = 5.0,
        wrist_position_cost: float = 10.0,
        wrist_orientation_cost: float = 5.0,
        posture_cost: float = 1e-2,
        damping: float = 1e-2,
    ):
        self.base = base
        self.solver = solver
        self.damping = damping

        xml_path = mujoco_compatible_urdf(urdf_path)
        self.model = mujoco.MjModel.from_xml_path(str(xml_path))
        self.configuration = mink.Configuration(self.model)
        self._sync_non_arm_from_base()

        self.posture_task = mink.PostureTask(self.model, cost=posture_cost)
        self.tasks: Dict[str, mink.FrameTask] = {}
        for side in _ARM_SIDES:
            self.tasks[f"{side}_elbow"] = mink.FrameTask(
                frame_name=f"link_{side}_arm_3", frame_type="body",
                position_cost=elbow_position_cost, orientation_cost=0.0,
                lm_damping=damping,
            )
            self.tasks[f"{side}_wrist"] = mink.FrameTask(
                frame_name=f"link_{side}_arm_6", frame_type="body",
                position_cost=wrist_position_cost, orientation_cost=wrist_orientation_cost,
                lm_damping=damping,
            )

        # Only the 14 arm joints move -- torso/head/fingers are frozen via
        # zero velocity limit, same "torso pinned" assumption
        # RBY1UpperBodyRetargeter.update() makes for its own per-arm blocks.
        velocities = {
            self.model.joint(i).name: 0.0
            for i in range(self.model.njnt)
            if self.model.joint(i).name not in _ARM_JOINTS
        }
        velocities.update({name: 3.0 for name in _ARM_JOINTS})
        self.velocity_limit = mink.VelocityLimit(self.model, velocities)
        self.configuration_limit = mink.ConfigurationLimit(self.model)

    def _sync_non_arm_from_base(self) -> None:
        """Push the base retargeter's current q (torso/head/etc) into this
        mink model so frozen joints don't sit at MuJoCo's own keyframe
        default instead of the base retargeter's actual current pose."""
        for name, value in self.base._values(self.base.q).items():
            joint = self.model.joint(name)
            self.configuration.q[self.model.jnt_qposadr[joint.id]] = value
        self.configuration.update()

    def update(self, human: UpperBodyLandmarks, dt: float = 1.0 / 30.0, iterations: int = 20) -> Dict[str, float]:
        if self.base._human_to_robot is None:
            self.base.calibrate(human)
        target = self.base._targets(human)
        self._sync_non_arm_from_base()

        self.posture_task.set_target_from_configuration(self.configuration)
        for side in _ARM_SIDES:
            self.tasks[f"{side}_elbow"].set_target(
                mink.SE3.from_rotation_and_translation(mink.SO3.identity(), target[f"{side}_elbow"])
            )
            rot_target = self.base._wrist_rotation_target(side, human)
            if rot_target is None:
                rot_target = self.configuration.get_transform_frame_to_world(
                    f"link_{side}_arm_6", "body"
                ).rotation().as_matrix()
            self.tasks[f"{side}_wrist"].set_target(
                mink.SE3.from_rotation_and_translation(
                    mink.SO3.from_matrix(rot_target), target[f"{side}_wrist"]
                )
            )

        tasks = [self.posture_task, *self.tasks.values()]
        limits = [self.configuration_limit, self.velocity_limit]
        for _ in range(iterations):
            vel = mink.solve_ik(
                configuration=self.configuration, tasks=tasks, limits=limits,
                dt=dt, solver=self.solver, damping=self.damping,
            )
            self.configuration.integrate_inplace(vel, dt)

        values: Dict[str, float] = {}
        for name in _ARM_JOINTS:
            joint = self.model.joint(name)
            value = float(self.configuration.q[self.model.jnt_qposadr[joint.id]])
            values[name] = value
            # Keep the base retargeter's own q in sync (calibration state,
            # its smoothness/tie-break references elsewhere) so switching
            # back to the closed-form solver mid-run doesn't jump.
            self.base.q[self.base.names.index(name)] = value
        return values
