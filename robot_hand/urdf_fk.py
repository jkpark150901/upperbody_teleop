"""Minimal URDF kinematic-tree parser + forward kinematics.

Only what's needed for the DG5F hand extracted by
scripts/extract_hand_urdf.py into robot_hand/urdf/dg5f_{left,right}.urdf:
<link>/<joint> with revolute/fixed types, <origin>, <axis>, <limit>, and a
single <collision><geometry><capsule radius="" length=""/></geometry>
</collision> per finger-segment link (see extract_hand_urdf.py -- this is a
MuJoCo URDF-import extension, not standard URDF, matching the convention
rby1_dg5f.urdf's own arm links already use). No <mimic>, no
continuous/prismatic/planar joints, no other collision primitives.

Each link's <visual><mesh filename=""/> and its <origin> are captured too
(robot_hand/mesh_loader.py resolves the filename and loads the actual
mesh -- see dg5f/ at the repo root -- for scripts/hand_retarget_vedo.py's
--render mesh mode).

Reuses teleop/geometry.py's Pose/quat helpers so hand link poses compose
the same way wrist retargeting does (teleop/calibration.py).
"""

from __future__ import annotations

import xml.etree.ElementTree as ET
from dataclasses import dataclass, field
from typing import Dict, Iterable, List, Optional, Tuple

import numpy as np

from teleop.geometry import Pose, euler_xyz_to_quat, quat_identity


def _parse_xyz(s: Optional[str]) -> np.ndarray:
    if not s:
        return np.zeros(3)
    return np.array([float(x) for x in s.split()])


def _parse_origin(elem: Optional[ET.Element]) -> Pose:
    if elem is None:
        return Pose.identity()
    pos = _parse_xyz(elem.get("xyz"))
    rpy = _parse_xyz(elem.get("rpy"))  # URDF roll/pitch/yaw, radians
    quat = euler_xyz_to_quat(np.degrees(rpy))
    return Pose(pos, quat)


def axis_angle_to_quat(axis: np.ndarray, angle: float) -> np.ndarray:
    axis = np.asarray(axis, dtype=float)
    norm = np.linalg.norm(axis)
    if norm < 1e-9 or angle == 0.0:
        return quat_identity()
    axis = axis / norm
    half = angle / 2.0
    return np.array([np.cos(half), *(axis * np.sin(half))])


@dataclass
class UrdfJoint:
    name: str
    type: str
    parent: str
    child: str
    origin: Pose
    axis: np.ndarray = field(default_factory=lambda: np.array([0.0, 0.0, 1.0]))
    lower: float = 0.0
    upper: float = 0.0


@dataclass
class UrdfLink:
    name: str
    visual_mesh: Optional[str] = None
    visual_origin: Pose = field(default_factory=Pose.identity)
    # Collision capsule, in this link's local frame (see module docstring).
    # None for links with no <collision> (mount/base/palm/tip in the
    # extracted hand URDFs -- robot_hand/collision.py's self-collision
    # check only needs the 4 driven finger-segment links per finger).
    collision_origin: Optional[Pose] = None
    collision_radius: float = 0.0
    collision_length: float = 0.0


@dataclass
class UrdfTree:
    root_link: str
    links: Dict[str, UrdfLink]
    joints: Dict[str, UrdfJoint]
    children: Dict[str, List[str]]  # parent link name -> [joint names]
    _parent_of: Dict[str, Tuple[str, str]] = field(default_factory=dict, repr=False, compare=False)

    def _joint_pose(self, joint: UrdfJoint, joint_values: Dict[str, float]) -> Pose:
        joint_pose = joint.origin
        if joint.type in ("revolute", "continuous"):
            angle = joint_values.get(joint.name, 0.0)
            if angle != 0.0:
                rot = Pose(np.zeros(3), axis_angle_to_quat(joint.axis, angle))
                joint_pose = joint_pose.compose(rot)
        return joint_pose

    def _parent_link_map(self) -> Dict[str, Tuple[str, str]]:
        """child link -> (joint name, parent link), built once and cached.

        Lets forward_kinematics(..., only=...) walk just the ancestor chain
        of the links a caller actually needs instead of the whole tree.
        """
        if not self._parent_of:
            for jname, joint in self.joints.items():
                self._parent_of[joint.child] = (jname, joint.parent)
        return self._parent_of

    def forward_kinematics(
        self,
        joint_values: Dict[str, float],
        root_pose: Optional[Pose] = None,
        only: Optional[Iterable[str]] = None,
    ) -> Dict[str, Pose]:
        """World-frame Pose for every link, given a value (radians) for each
        movable joint (missing/omitted joints default to 0).

        `only`, when given, restricts the work to the ancestor chains of
        those link names (memoized across shared prefixes) instead of
        walking the entire tree -- e.g. a numerical IK Jacobian that only
        cares about 8 upper-body landmark links has no reason to also
        recompute every one of a hand's ~30 finger joints on each column.
        """
        root_pose = root_pose if root_pose is not None else Pose.identity()
        poses = {self.root_link: root_pose}

        if only is None:
            def visit(link_name: str):
                for jname in self.children.get(link_name, []):
                    joint = self.joints[jname]
                    child_pose = poses[link_name].compose(self._joint_pose(joint, joint_values))
                    poses[joint.child] = child_pose
                    visit(joint.child)

            visit(self.root_link)
            return poses

        parent_of = self._parent_link_map()

        def resolve(link_name: str) -> Pose:
            cached = poses.get(link_name)
            if cached is not None:
                return cached
            jname, parent_link = parent_of[link_name]
            joint = self.joints[jname]
            pose = resolve(parent_link).compose(self._joint_pose(joint, joint_values))
            poses[link_name] = pose
            return pose

        for link_name in only:
            resolve(link_name)
        return poses


def load_urdf(path: str) -> UrdfTree:
    root = ET.parse(path).getroot()

    links: Dict[str, UrdfLink] = {}
    for link_elem in root.findall("link"):
        name = link_elem.get("name")
        visual_elem = link_elem.find("visual")
        mesh_elem = visual_elem.find("./geometry/mesh") if visual_elem is not None else None
        mesh = mesh_elem.get("filename") if mesh_elem is not None else None
        visual_origin = _parse_origin(visual_elem.find("origin")) if visual_elem is not None else Pose.identity()
        link = UrdfLink(name=name, visual_mesh=mesh, visual_origin=visual_origin)

        # A link may carry more than one <collision> (e.g. the source
        # mesh-based one alongside the capsule scripts/extract_hand_urdf.py
        # authors) -- match <origin> and <capsule> from the SAME <collision>
        # element rather than the first of each independently (a bare
        # ".//origin" / ".//capsule" findall pairing would silently mix
        # them up across sibling <collision> blocks).
        for collision_elem in link_elem.findall("collision"):
            capsule_elem = collision_elem.find("./geometry/capsule")
            if capsule_elem is not None:
                link.collision_origin = _parse_origin(collision_elem.find("origin"))
                link.collision_radius = float(capsule_elem.get("radius", 0.0))
                link.collision_length = float(capsule_elem.get("length", 0.0))
                break

        links[name] = link

    joints: Dict[str, UrdfJoint] = {}
    children: Dict[str, List[str]] = {}
    child_links = set()
    for joint_elem in root.findall("joint"):
        name = joint_elem.get("name")
        jtype = joint_elem.get("type")
        parent = joint_elem.find("parent").get("link")
        child = joint_elem.find("child").get("link")
        origin = _parse_origin(joint_elem.find("origin"))
        axis_elem = joint_elem.find("axis")
        axis = _parse_xyz(axis_elem.get("xyz")) if axis_elem is not None else np.array([0.0, 0.0, 1.0])
        limit_elem = joint_elem.find("limit")
        lower = float(limit_elem.get("lower", 0.0)) if limit_elem is not None else 0.0
        upper = float(limit_elem.get("upper", 0.0)) if limit_elem is not None else 0.0

        joints[name] = UrdfJoint(
            name=name, type=jtype, parent=parent, child=child,
            origin=origin, axis=axis, lower=lower, upper=upper,
        )
        children.setdefault(parent, []).append(name)
        child_links.add(child)

    root_candidates = [name for name in links if name not in child_links]
    if len(root_candidates) != 1:
        raise ValueError(f"expected exactly one root link in {path}, got {root_candidates}")

    return UrdfTree(root_link=root_candidates[0], links=links, joints=joints, children=children)
