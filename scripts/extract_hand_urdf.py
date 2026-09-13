"""One-off/rerunnable extraction: pulls just the DG5F hand (right + left)
out of the full rby1_dg5f.urdf upper-body robot, as two standalone URDFs
whose root link is the hand's mount plate -- so they can be loaded and
posed on their own, without the arm/torso links the full URDF also
contains. See robot_hand/urdf_fk.py for the loader that reads these, and
scripts/hand_retarget_vedo.py for what uses them.

The DG5F hand's mesh files (../dg5f/meshes/... relative to the source
URDF) aren't present in this repo -- only the URDF geometry/kinematics
were available -- so downstream visualization draws primitive skeleton
geometry (spheres/lines) instead of loading these meshes. The
<mesh filename="..."> references are still copied through verbatim in
case the mesh files get added later (same relative path works, since
robot_hand/urdf/ and the original rby1_dg5f.urdf are both one level away
from a project-root-level `dg5f/` mesh folder).

This also *authors* new geometry that isn't in the source URDF at all: a
<collision><geometry><capsule radius="" length=""/></geometry></collision>
per finger-segment link (each finger's _1.._4, both hands -- not the
mount/base/palm/tip links), so robot_hand/hand_retarget.py's self-collision
avoidance has something to check distances between. `<capsule>` is a
MuJoCo URDF-import extension rather than standard URDF, matching the
convention rby1_dg5f.urdf's own arm links already use (e.g.
"link_torso_hp"'s <collision><geometry><capsule .../></geometry>) -- so
these two files stay loadable by the same tooling. Capsule placement
(direction + length) is derived from each link's own outgoing joint origin
(the vector to the next joint in its finger's chain -- see
robot_hand/collision.py's direction_rpy); radius is a fixed per-segment-
index guess (no real hand cross-section data was available), tapering
from the base segment (_1, thicker) to the one nearest the tip (_4).

Run again if rby1_dg5f.urdf changes:
    python scripts/extract_hand_urdf.py
"""

from __future__ import annotations

import pathlib
import re
import sys
import xml.etree.ElementTree as ET

import numpy as np

_ROOT = pathlib.Path(__file__).parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from robot_hand.collision import direction_rpy  # noqa: E402

_SOURCE = _ROOT / "rby1_dg5f.urdf"
_OUT_DIR = _ROOT / "robot_hand" / "urdf"

_SIDES = {
    "right": {"link_prefix": "right_hand_", "out": "dg5f_right.urdf", "robot_name": "dg5f_right_hand"},
    "left": {"link_prefix": "left_hand_", "out": "dg5f_left.urdf", "robot_name": "dg5f_left_hand"},
}

# Finger-segment link name -> (finger 1-5, segment 1-4), e.g.
# "right_hand_rl_dg_2_3" -> (2, 3). Matches both sides' naming.
_FINGER_SEGMENT_RE = re.compile(r"_dg_(\d)_(\d)$")

# Per-segment-index capsule radius (m): base segment thicker, tapering
# toward the segment nearest the fingertip. No real cross-section data was
# available for the DG5F hand -- these are plausible guesses for a small
# gripper hand, good enough to make finger-vs-finger self-collision
# checking meaningful, not a fidelity claim.
_SEGMENT_RADIUS = {1: 0.008, 2: 0.007, 3: 0.006, 4: 0.005}


def extract_side(source_root: ET.Element, link_prefix: str, robot_name: str) -> ET.Element:
    new_root = ET.Element("robot", {"name": robot_name})

    kept_links = {}  # name -> link Element
    for link in source_root.findall("link"):
        name = link.get("name")
        if name.startswith(link_prefix):
            new_root.append(link)
            kept_links[name] = link

    kept_joints = {}  # parent link name -> joint Element (finger chains never branch)
    for joint in source_root.findall("joint"):
        parent = joint.find("parent").get("link")
        child = joint.find("child").get("link")
        # Drops the single fixed joint that anchors the hand to the arm
        # (e.g. "right_hand_mount": link_right_arm_6 -> ..._mount), since
        # its parent link doesn't exist in a hand-only URDF. The mount
        # link itself becomes this tree's root.
        if parent in kept_links and child in kept_links:
            new_root.append(joint)
            kept_joints[parent] = joint

    add_finger_collision_capsules(kept_links, kept_joints)

    return new_root


def add_finger_collision_capsules(kept_links: dict, kept_joints: dict) -> None:
    for link_name, link_elem in kept_links.items():
        m = _FINGER_SEGMENT_RE.search(link_name)
        if m is None:
            continue  # mount/base/palm/tip -- no capsule authored for these
        segment = int(m.group(2))

        next_joint = kept_joints.get(link_name)
        if next_joint is None:
            continue  # shouldn't happen for _1.._4 (each has an outgoing joint)
        origin_elem = next_joint.find("origin")
        direction = np.array([float(x) for x in origin_elem.get("xyz").split()])
        length = float(np.linalg.norm(direction))
        if length < 1e-6:
            continue

        rpy = direction_rpy(direction)
        midpoint = direction / 2.0

        collision = ET.SubElement(link_elem, "collision")
        origin = ET.SubElement(collision, "origin")
        origin.set("xyz", f"{midpoint[0]:.6f} {midpoint[1]:.6f} {midpoint[2]:.6f}")
        origin.set("rpy", f"{rpy[0]:.6f} {rpy[1]:.6f} {rpy[2]:.6f}")
        geometry = ET.SubElement(collision, "geometry")
        capsule = ET.SubElement(geometry, "capsule")
        capsule.set("radius", str(_SEGMENT_RADIUS[segment]))
        capsule.set("length", f"{length:.6f}")


def main():
    source_root = ET.parse(_SOURCE).getroot()

    _OUT_DIR.mkdir(parents=True, exist_ok=True)
    for side, cfg in _SIDES.items():
        new_root = extract_side(source_root, cfg["link_prefix"], cfg["robot_name"])
        out_path = _OUT_DIR / cfg["out"]
        try:
            ET.indent(new_root, space="  ")
        except AttributeError:
            pass  # Python < 3.9: skip pretty-printing, content is unaffected
        ET.ElementTree(new_root).write(out_path, encoding="utf-8", xml_declaration=True)
        n_links = len(new_root.findall("link"))
        n_joints = len(new_root.findall("joint"))
        n_capsules = len(new_root.findall("./link/collision/geometry/capsule"))
        print(
            f"[extract_hand_urdf] {side}: {n_links} links, {n_joints} joints, "
            f"{n_capsules} collision capsules -> {out_path}"
        )


if __name__ == "__main__":
    main()
