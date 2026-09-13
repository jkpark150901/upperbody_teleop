"""Custom mjvScene geom helpers for drawing wrist/chest poses as RGB axis
triads and hand joints as points — no URDF/robot bodies required yet
(plan section 21, steps [3]-[4]: "PN -> MuJoCo EE marker" before "-> robot
arm IK"). Geoms are pushed directly into an mjvScene each frame rather than
baked into the MJCF model, so this works against an otherwise-empty scene.
"""

from __future__ import annotations

import numpy as np

import mujoco

_AXIS_COLORS = (
    (1.0, 0.15, 0.15, 1.0),  # X - red
    (0.15, 1.0, 0.15, 1.0),  # Y - green
    (0.25, 0.45, 1.0, 1.0),  # Z - blue
)


def _has_room(scene: mujoco.MjvScene, n: int) -> bool:
    if scene.ngeom + n > scene.maxgeom:
        print(
            f"[markers] scene geom capacity exceeded "
            f"(ngeom={scene.ngeom}, +{n}, maxgeom={scene.maxgeom}); skipping"
        )
        return False
    return True


def add_line(scene: mujoco.MjvScene, p1, p2, radius: float, rgba) -> None:
    if not _has_room(scene, 1):
        return
    geom = scene.geoms[scene.ngeom]
    mujoco.mjv_initGeom(
        geom,
        type=mujoco.mjtGeom.mjGEOM_CAPSULE,
        size=np.zeros(3),
        pos=np.zeros(3),
        mat=np.eye(3).flatten(),
        rgba=np.array(rgba, dtype=np.float32),
    )
    mujoco.mjv_connector(geom, mujoco.mjtGeom.mjGEOM_CAPSULE, radius, np.asarray(p1), np.asarray(p2))
    scene.ngeom += 1


def add_point(scene: mujoco.MjvScene, pos, radius: float = 0.008, rgba=(1.0, 0.85, 0.0, 1.0)) -> None:
    if not _has_room(scene, 1):
        return
    geom = scene.geoms[scene.ngeom]
    mujoco.mjv_initGeom(
        geom,
        type=mujoco.mjtGeom.mjGEOM_SPHERE,
        size=np.array([radius, 0.0, 0.0]),
        pos=np.asarray(pos, dtype=float),
        mat=np.eye(3).flatten(),
        rgba=np.array(rgba, dtype=np.float32),
    )
    scene.ngeom += 1


def add_axis_triad(scene: mujoco.MjvScene, pose, length: float = 0.06, radius: float = 0.004) -> None:
    """Draw a pose as three RGB capsules along its local +X/+Y/+Z axes."""
    if not _has_room(scene, 3):
        return
    endpoints = pose.axis_endpoints(length)
    for axis in range(3):
        add_line(scene, pose.pos, endpoints[axis], radius, _AXIS_COLORS[axis])


def add_hand_points(
    scene: mujoco.MjvScene,
    world_joint_positions: np.ndarray,
    radius: float = 0.006,
    rgba=(1.0, 0.85, 0.0, 1.0),
) -> None:
    """world_joint_positions: (5, 4, 3) -> one point marker per joint."""
    flat = world_joint_positions.reshape(-1, 3)
    if not _has_room(scene, flat.shape[0]):
        return
    for p in flat:
        add_point(scene, p, radius=radius, rgba=rgba)
