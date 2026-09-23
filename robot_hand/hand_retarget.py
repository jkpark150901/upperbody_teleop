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
small-range ab/adduction (spread) joint; for the thumb and pinky it carries
part of that finger's base/opposition range. Readers that only expose a
single open/closed scalar (SenseGlove flexion) leave _1 at rest (0 rad).
Readers that expose a base spread estimate (Quest hand tracking) can pass it
to retarget(..., spread=...) so _1 follows ab/adduction within the URDF limit.

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
from typing import Dict, List, Optional, Tuple

import numpy as np

from devices.senseglove.sg_types import FINGER_NAMES
from robot_hand.collision import capsule_world_segment, segment_segment_distance
from robot_hand.urdf_fk import Pose, UrdfTree

_FINGER_INDEX = {name: i + 1 for i, name in enumerate(FINGER_NAMES)}  # thumb=1 .. pinky=5
_ADJACENT_PAIRS = list(zip(FINGER_NAMES[:-1], FINGER_NAMES[1:]))  # (thumb,index), (index,middle), ...

_COLLISION_MARGIN = 0.0005  # meters of clearance kept between capsules
_BISECTION_ITERS = 12


def _unit(v: np.ndarray) -> np.ndarray:
    n = np.linalg.norm(v)
    if n < 1e-9:
        return np.zeros(3)
    return v / n


def _angle_between(a: np.ndarray, b: np.ndarray) -> float:
    ua, ub = _unit(a), _unit(b)
    if np.linalg.norm(ua) < 1e-9 or np.linalg.norm(ub) < 1e-9:
        return 0.0
    return float(np.arccos(np.clip(np.dot(ua, ub), -1.0, 1.0)))


def _base_spread_flex(direction: np.ndarray) -> Tuple[float, float]:
    """Wrist-local base direction -> (lateral spread, flexion) in radians.

    Quest wrist-local convention for the current stream: +y is up, +x is
    human-right, +z points back toward the body, so an open finger points
    roughly along -z.
    """
    v = _unit(direction)
    if np.linalg.norm(v) < 1e-9:
        return 0.0, 0.0
    forward = max(1e-6, -float(v[2]))
    spread = float(np.arctan2(v[0], forward))
    flex = float(np.arctan2(v[1], forward))
    return spread, flex


def _palm_frame(xr_pos: np.ndarray) -> np.ndarray:
    wrist = xr_pos[1]
    index_metacarpal = xr_pos[6]
    middle_metacarpal = xr_pos[11]
    pinky_metacarpal = xr_pos[21]

    forward = _unit(middle_metacarpal - wrist)
    lateral = _unit(index_metacarpal - pinky_metacarpal)
    forward = _unit(forward - np.dot(forward, lateral) * lateral)
    normal = _unit(np.cross(lateral, forward))
    if np.linalg.norm(forward) < 1e-9 or np.linalg.norm(lateral) < 1e-9 or np.linalg.norm(normal) < 1e-9:
        return np.eye(3)
    return np.column_stack([lateral, forward, normal])


def _joint_sign_from_limits(lower: float, upper: float) -> float:
    if upper <= 1e-9 and abs(lower) > abs(upper):
        return -1.0
    return 1.0


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
        self._neutral_spread: Optional[np.ndarray] = None
        self._neutral_joint_positions: Optional[np.ndarray] = None
        self._human_zero: Optional[Dict[str, float]] = None
        self.sign: Dict[str, float] = {
            jname: _joint_sign_from_limits(joint.lower, joint.upper)
            for jname, joint in tree.joints.items()
            if joint.type != "fixed"
        }
        self.gain: Dict[str, float] = {
            jname: 1.0 for jname, joint in tree.joints.items() if joint.type != "fixed"
        }

    def _finger_angles(self, finger: str, scale: float, spread: float = 0.0) -> Dict[str, float]:
        chain = self.chains[finger]
        base_joint = self._tree.joints[chain[0]]
        angles = {chain[0]: float(np.clip(spread, base_joint.lower, base_joint.upper))}
        for jname in chain[1:]:
            joint = self._tree.joints[jname]
            closed = joint.upper if abs(joint.upper) >= abs(joint.lower) else joint.lower
            angles[jname] = scale * closed
        return angles

    def _pair_collides(
        self,
        finger_a: str,
        scale_a: float,
        finger_b: str,
        scale_b: float,
        spread: Optional[Dict[str, float]] = None,
    ) -> bool:
        spread = spread or {}
        angles = {
            **self._finger_angles(finger_a, scale_a, spread.get(finger_a, 0.0)),
            **self._finger_angles(finger_b, scale_b, spread.get(finger_b, 0.0)),
        }
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

    def _largest_safe_t(
        self,
        finger_a: str,
        scale_a: float,
        finger_b: str,
        scale_b: float,
        spread: Optional[Dict[str, float]] = None,
    ) -> float:
        """Largest t in [0, 1] such that (t*scale_a, t*scale_b) doesn't
        collide, found by bisection (both scales are already <= their
        finger's own current value, so t=0 -- both fingers at their
        as-clamped-so-far pose scaled toward fully open -- is assumed safe:
        the neutral/open pose has no self-collision by construction; spread
        remains fixed while curl is backed off)."""
        if not self._pair_collides(finger_a, scale_a, finger_b, scale_b, spread):
            return 1.0
        lo, hi = 0.0, 1.0
        for _ in range(_BISECTION_ITERS):
            mid = (lo + hi) / 2.0
            if self._pair_collides(finger_a, mid * scale_a, finger_b, mid * scale_b, spread):
                hi = mid
            else:
                lo = mid
        return lo

    def retarget(self, flexion: np.ndarray, spread: Optional[np.ndarray] = None) -> Dict[str, float]:
        scale = {
            finger: float(np.clip(flexion[idx - 1], 0.0, 1.0)) for finger, idx in _FINGER_INDEX.items()
        }
        base_spread = {
            finger: float(spread[idx - 1]) if spread is not None else 0.0
            for finger, idx in _FINGER_INDEX.items()
        }
        requested = dict(scale)

        for finger_a, finger_b in _ADJACENT_PAIRS:
            t = self._largest_safe_t(
                finger_a, scale[finger_a], finger_b, scale[finger_b], base_spread
            )
            if t < 1.0:
                scale[finger_a] *= t
                scale[finger_b] *= t

        self.last_clamp = {
            finger: FingerClampInfo(requested=requested[finger], applied=scale[finger])
            for finger in _FINGER_INDEX
        }

        angles: Dict[str, float] = {}
        for finger, s in scale.items():
            angles.update(self._finger_angles(finger, s, base_spread[finger]))
        return angles

    def retarget_joint_positions(self, joint_positions: np.ndarray, hand: str) -> Dict[str, float]:
        """Quest skeleton joint positions -> DG5F joint angles.

        ``joint_positions`` is HandState.joint_positions: (5, 4, 3), in
        FINGER_NAMES order. Thumb rows are T0..T3. Other fingers are I1..I4,
        M1..M4, R1..R4, P1..P4 because OpenXR's metacarpal point is skipped
        in HandState for the four long fingers.

        Mapping requested for this Quest path:
          T1 -> dg_1_1, dg_1_2, dg_1_3 as a spherical base approximation
          T2 -> dg_1_4 (interpreting the repeated dg_1_3 note as the child joint)
          I/M/R/P1 -> dg_N_1, dg_N_2
          I/M/R/P2 -> dg_N_3
          I/M/R/P3 -> dg_N_4
        """
        pts = np.asarray(joint_positions, dtype=float)
        if pts.shape != (5, 4, 3):
            raise ValueError(f"expected Quest joint_positions shape (5,4,3), got {pts.shape}")

        raw_spread = np.zeros(5)
        for i in range(5):
            raw_spread[i], _ = _base_spread_flex(pts[i, 1] - pts[i, 0])
        if self._neutral_spread is None:
            self._neutral_spread = raw_spread.copy()
        if self._neutral_joint_positions is None:
            self._neutral_joint_positions = pts.copy()

        angles: Dict[str, float] = {}

        # Thumb: T1 spherical approximation plus T2 distal bend.
        thumb = self.chains["thumb"]
        t0, t1, t2, t3 = pts[0]
        n0, n1, n2, n3 = self._neutral_joint_positions[0]
        spread_t, flex_t = _base_spread_flex(t2 - t1)
        spread_n, flex_n = _base_spread_flex(n2 - n1)
        bend_t1 = _angle_between(t1 - t0, t2 - t1) - _angle_between(n1 - n0, n2 - n1)
        bend_t2 = _angle_between(t2 - t1, t3 - t2) - _angle_between(n2 - n1, n3 - n2)
        thumb_q2_sign = -1.0 if hand == "right" else 1.0
        thumb_values = {
            thumb[0]: flex_t - flex_n,
            thumb[1]: thumb_q2_sign * abs(spread_t - spread_n),
            thumb[2]: max(0.0, bend_t1),
            thumb[3]: bend_t2,
        }
        for jname, value in thumb_values.items():
            joint = self._tree.joints[jname]
            angles[jname] = float(np.clip(value, joint.lower, joint.upper))

        # Long fingers: P1/I1/... gives base spread + MCP flexion, then the
        # next two skeleton joints provide PIP/DIP bend.
        for finger_idx, finger in enumerate(FINGER_NAMES[1:], start=1):
            chain = self.chains[finger]
            p1, p2, p3, p4 = pts[finger_idx]
            spread_raw, flex = _base_spread_flex(p2 - p1)
            values = {
                chain[0]: spread_raw - float(self._neutral_spread[finger_idx]),
                chain[1]: flex,
                chain[2]: _angle_between(p2 - p1, p3 - p2),
                chain[3]: _angle_between(p3 - p2, p4 - p3),
            }
            for jname, value in values.items():
                joint = self._tree.joints[jname]
                angles[jname] = float(np.clip(value, joint.lower, joint.upper))

        self.last_clamp = {
            finger: FingerClampInfo(requested=1.0, applied=1.0)
            for finger in _FINGER_INDEX
        }
        return angles

    def _quest_human_angles(self, xr_joints: np.ndarray) -> Dict[str, float]:
        pos = np.asarray(xr_joints, dtype=float)[:, :3]
        if pos.shape != (26, 3):
            raise ValueError(f"expected Quest/OpenXR joints shape (26,7) or (26,3), got {np.asarray(xr_joints).shape}")
        r_palm = _palm_frame(pos)

        human: Dict[str, float] = {}

        # Thumb: base pitch/yaw plus MCP/IP flexion.
        t0 = _unit(pos[3] - pos[2])
        t1 = _unit(pos[4] - pos[3])
        t2 = _unit(pos[5] - pos[4])
        t0_local = r_palm.T @ t0
        human["thumb_base_pitch"] = float(np.arctan2(t0_local[2], np.hypot(t0_local[0], t0_local[1])))
        human["thumb_base_yaw"] = float(np.arctan2(t0_local[0], t0_local[1]))
        human["thumb_mcp"] = _angle_between(t0, t1)
        human["thumb_ip"] = _angle_between(t1, t2)

        long_fingers = {
            "index": (6, 7, 8, 9, 10),
            "middle": (11, 12, 13, 14, 15),
            "ring": (16, 17, 18, 19, 20),
            "pinky": (21, 22, 23, 24, 25),
        }
        for finger, chain in long_fingers.items():
            f0, f1, f2, f3, f4 = [pos[i] for i in chain]
            b1 = _unit(f2 - f1)
            b2 = _unit(f3 - f2)
            b3 = _unit(f4 - f3)
            b1_local = r_palm.T @ b1
            human[f"{finger}_abd"] = float(np.arctan2(b1_local[0], b1_local[1]))
            human[f"{finger}_mcp"] = float(np.arctan2(-b1_local[2], np.hypot(b1_local[0], b1_local[1])))
            human[f"{finger}_pip"] = _angle_between(b1, b2)
            human[f"{finger}_dip"] = _angle_between(b2, b3)

        return human

    def retarget_openxr_joints(self, xr_joints: np.ndarray) -> Dict[str, float]:
        """Quest/OpenXR 26-joint skeleton -> DG5F joint angles.

        This follows quest_openxr_dg5f_retargeting_rules.md: build a
        palm-local frame, extract human joint-angle features, subtract the
        first valid/open-hand neutral feature values, then apply per-joint
        sign/gain and URDF joint-limit clamps.
        """
        human = self._quest_human_angles(xr_joints)
        if self._human_zero is None:
            self._human_zero = dict(human)

        mapping = {
            "thumb": ("thumb_base_pitch", "thumb_base_yaw", "thumb_mcp", "thumb_ip"),
            "index": ("index_abd", "index_mcp", "index_pip", "index_dip"),
            "middle": ("middle_abd", "middle_mcp", "middle_pip", "middle_dip"),
            "ring": ("ring_abd", "ring_mcp", "ring_pip", "ring_dip"),
            "pinky": ("pinky_abd", "pinky_mcp", "pinky_pip", "pinky_dip"),
        }

        angles: Dict[str, float] = {}
        for finger in FINGER_NAMES:
            for jname, feature in zip(self.chains[finger], mapping[finger]):
                joint = self._tree.joints[jname]
                delta = human[feature] - self._human_zero[feature]
                value = self.sign.get(jname, 1.0) * self.gain.get(jname, 1.0) * delta
                angles[jname] = float(np.clip(value, joint.lower, joint.upper))

        self.last_clamp = {
            finger: FingerClampInfo(requested=1.0, applied=1.0)
            for finger in _FINGER_INDEX
        }
        return angles
