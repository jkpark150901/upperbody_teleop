"""SenseGlove finger flexion -> DG5F robot hand joint angles, with
self-collision-aware motion limiting.

This is the plan's "Phase 1. 단순 finger mapping" (perception_neuron_
senseglove_mujoco_teleop_plan.md section 9), applied to the actual DG5F
hand geometry (robot_hand/urdf/dg5f_{left,right}.urdf) for the first time,
rather than SenseGlove's raw joint points as in scripts/vedo_local_preview.py.

Each DG5F finger is a 4-revolute-joint serial chain named
"<side>_hand_<r|l>j_dg_<finger 1-5>_<1-4>" (see scripts/extract_hand_urdf.py
/ robot_hand/urdf_fk.py), <finger>_tip fixed on the end:

    palm -> _1 -> _2 -> _3 -> _4 -> (fixed) _tip

_1 is each finger's base axis. For the index/middle/ring fingers this is a
small-range ab/adduction (spread) joint; for the thumb and pinky it instead
carries most of that finger's opposition range. SenseGlove's per-finger
flexion is a single open/closed scalar with no spread or opposition
information, so _1 is simply held at its rest angle (0 rad) for every
finger -- a real simplification, not a full retarget (fingertip-position
matching is plan section 9's Phase 2, not attempted here).

_2/_3/_4 are that finger's curl chain and are driven together from the same
per-finger scale value (0 = every joint's own rest angle, 1 = each joint's
own limit bound with the larger magnitude -- the DG5F's revolute limits are
authored so the "closing" direction is whichever bound is farther from
zero, true for every curl joint in rby1_dg5f.urdf's right_hand_rj_dg_*
_2/_3/_4 <limit> values).

Self-collision motion limiting: scripts/extract_hand_urdf.py authors a
<collision><capsule/></collision> per finger segment link (see that
script's and robot_hand/urdf_fk.py's docstrings). Before returning joint
angles, this module checks every ADJACENT finger pair (thumb-index,
index-middle, middle-ring, ring-pinky -- non-adjacent pairs, e.g.
thumb-pinky, aren't physically reachable into contact and are skipped for
speed) for capsule interpenetration and, if the raw flexion-driven pose
would collide, binary-searches the largest shared scale that keeps both
fingers just clear of each other -- i.e. it stops a closing hand right at
the point fingers would start pushing through each other, rather than
letting them visually overlap.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Dict, List, Tuple

import numpy as np

from devices.senseglove.sg_types import FINGER_NAMES
from robot_hand.collision import capsule_world_segment, segment_segment_distance
from robot_hand.urdf_fk import Pose, UrdfTree

_FINGER_INDEX = {name: i + 1 for i, name in enumerate(FINGER_NAMES)}  # thumb=1 .. pinky=5
_ADJACENT_PAIRS = list(zip(FINGER_NAMES[:-1], FINGER_NAMES[1:]))  # (thumb,index), (index,middle), ...

_COLLISION_MARGIN = 0.0005  # meters of clearance kept between capsules
_BISECTION_ITERS = 12


def _finger_chain_joints(tree: UrdfTree, joint_prefix: str, finger_idx: int) -> List[str]:
    """Revolute joints for one finger, in chain order (base -> tip-adjacent)."""
    pattern = re.compile(rf"^{re.escape(joint_prefix)}_{finger_idx}_(\d+)$")
    matches = []
    for jname, joint in tree.joints.items():
        if joint.type == "fixed":
            continue
        m = pattern.match(jname)
        if m:
            matches.append((int(m.group(1)), jname))
    matches.sort()
    return [name for _, name in matches]


@dataclass
class FingerClampInfo:
    """Diagnostics from the most recent retarget() call, for one finger --
    lets a viewer (e.g. scripts/hand_retarget_vedo.py) show when/how much
    self-collision avoidance is actually engaging."""

    requested: float  # flexion value asked for, 0..1
    applied: float  # scale actually used after collision clamping, 0..requested


class Dg5fHandRetargeter:
    """flexion (5,), ordered per devices.senseglove.sg_types.FINGER_NAMES,
    -> {joint_name: angle_rad} for every revolute joint in one DG5F hand."""

    def __init__(self, tree: UrdfTree, joint_prefix: str):
        """joint_prefix: "right_hand_rj_dg" or "left_hand_lj_dg" -- the
        common prefix of every revolute joint name in that hand's URDF,
        i.e. everything before "_<finger>_<1-4>"."""
        self._tree = tree
        self.chains: Dict[str, List[str]] = {
            finger: _finger_chain_joints(tree, joint_prefix, idx)
            for finger, idx in _FINGER_INDEX.items()
        }
        for finger, chain in self.chains.items():
            if len(chain) != 4:
                raise ValueError(
                    f"expected 4 revolute joints for finger '{finger}' under "
                    f"prefix '{joint_prefix}', found {len(chain)}: {chain}"
                )
        # The capsule-bearing links used for self-collision checking are
        # the child links of chain[1:] (_2.._4) -- NOT _1, even though
        # extract_hand_urdf.py authors a capsule for it too. _1 is each
        # finger's base ab/adduction-or-opposition joint, always held at
        # its rest angle regardless of `scale` (see _finger_angles below),
        # so its link's world pose never changes with flexion -- checking
        # it can only ever find whatever overlap the DG5F's own rest-pose
        # base geometry already has with its neighbor's (a wide mounting
        # block, not a slender bone -- a real per-mesh capsule radius
        # there routinely exceeds the rest-pose gap between fingers), which
        # would clamp every finger to 0 immediately rather than reacting to
        # anything the SenseGlove input actually did.
        # Public: scripts/hand_retarget_vedo.py uses this to find which
        # capsule actors belong to a finger for the collision-clamp color.
        self.capsule_links: Dict[str, List[str]] = {
            finger: [tree.joints[j].child for j in chain[1:]] for finger, chain in self.chains.items()
        }
        self.last_clamp: Dict[str, FingerClampInfo] = {}

    def _finger_angles(self, finger: str, scale: float) -> Dict[str, float]:
        chain = self.chains[finger]
        angles = {chain[0]: 0.0}  # base/ab-duction-or-opposition joint: held at rest
        for jname in chain[1:]:
            joint = self._tree.joints[jname]
            closed = joint.upper if abs(joint.upper) >= abs(joint.lower) else joint.lower
            angles[jname] = scale * closed
        return angles

    def _pair_collides(self, finger_a: str, scale_a: float, finger_b: str, scale_b: float) -> bool:
        angles = {**self._finger_angles(finger_a, scale_a), **self._finger_angles(finger_b, scale_b)}
        poses = self._tree.forward_kinematics(angles, Pose.identity())
        for link_a in self.capsule_links[finger_a]:
            cap_a = self._tree.links[link_a]
            pa0, pa1 = capsule_world_segment(poses[link_a], cap_a.collision_origin, cap_a.collision_length)
            for link_b in self.capsule_links[finger_b]:
                cap_b = self._tree.links[link_b]
                pb0, pb1 = capsule_world_segment(poses[link_b], cap_b.collision_origin, cap_b.collision_length)
                gap = segment_segment_distance(pa0, pa1, pb0, pb1) - (cap_a.collision_radius + cap_b.collision_radius)
                if gap < _COLLISION_MARGIN:
                    return True
        return False

    def _largest_safe_t(self, finger_a: str, scale_a: float, finger_b: str, scale_b: float) -> float:
        """Largest t in [0, 1] such that (t*scale_a, t*scale_b) doesn't
        collide, found by bisection (both scales are already <= their
        finger's own current value, so t=0 -- both fingers at their
        as-clamped-so-far pose scaled toward fully open -- is assumed safe:
        the neutral/open pose has no self-collision by construction)."""
        if not self._pair_collides(finger_a, scale_a, finger_b, scale_b):
            return 1.0
        lo, hi = 0.0, 1.0
        for _ in range(_BISECTION_ITERS):
            mid = (lo + hi) / 2.0
            if self._pair_collides(finger_a, mid * scale_a, finger_b, mid * scale_b):
                hi = mid
            else:
                lo = mid
        return lo

    def retarget(self, flexion: np.ndarray) -> Dict[str, float]:
        scale = {
            finger: float(np.clip(flexion[idx - 1], 0.0, 1.0)) for finger, idx in _FINGER_INDEX.items()
        }
        requested = dict(scale)

        for finger_a, finger_b in _ADJACENT_PAIRS:
            t = self._largest_safe_t(finger_a, scale[finger_a], finger_b, scale[finger_b])
            if t < 1.0:
                scale[finger_a] *= t
                scale[finger_b] *= t

        self.last_clamp = {
            finger: FingerClampInfo(requested=requested[finger], applied=scale[finger])
            for finger in _FINGER_INDEX
        }

        angles: Dict[str, float] = {}
        for finger, s in scale.items():
            angles.update(self._finger_angles(finger, s))
        return angles
