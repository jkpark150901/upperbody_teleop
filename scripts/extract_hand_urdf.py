"""One-off/rerunnable extraction: pulls just the DG5F hand (right + left)
out of the full rby1_dg5f.urdf upper-body robot, as two standalone URDFs
whose root link is the hand's mount plate -- so they can be loaded and
posed on their own, without the arm/torso links the full URDF also
contains. See robot_hand/urdf_fk.py for the loader that reads these, and
scripts/hand_retarget_vedo.py for what uses them.

The DG5F hand's mesh files now live at the repo-root dg5f/ (see
robot_hand/mesh_loader.py, used by scripts/hand_retarget_vedo.py's
--render mesh mode). The <mesh filename="..."> references are copied
through verbatim from the source URDF into the extracted hand-only ones.

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
robot_hand/collision.py's direction_rpy); radius is the median (see
_RADIUS_PERCENTILE) of that same link's real <collision><mesh> vertices'
distance from that axis (see _segment_radius_from_mesh) -- a fixed
per-segment-index guess table was used here before the dg5f/ mesh files
were available, and undershot real cross-sections badly (e.g. ~7mm guessed
vs ~11.4mm actual median for a middle segment), which is why
robot_hand.hand_retarget's self-collision clamp essentially never engaged:
the real finger meshes could visibly overlap well before the undersized
capsules did.

A too-high percentile overshoots the other way: adjacent fingers (e.g.
index/middle) sit only ~24.5mm apart center-to-center at rest, but the
95th percentile of either one's own cross-section is ~14mm, so a
p95-radius capsule pair already "overlaps" by a few mm in the neutral
pose -- which breaks robot_hand.hand_retarget's core assumption that the
neutral pose is collision-free (see its self-collision-checking
docstring) and collapses that finger to fully open for any nonzero
flexion. The median keeps a small positive gap at rest while still
reflecting the real mesh, not a guess.

Run again if rby1_dg5f.urdf changes:
    python scripts/extract_hand_urdf.py
"""

from __future__ import annotations

import pathlib
import re
import sys
import xml.etree.ElementTree as ET
from typing import Optional

import numpy as np
import trimesh

_ROOT = pathlib.Path(__file__).parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from robot_hand.collision import direction_rpy  # noqa: E402
from robot_hand.mesh_loader import resolve_mesh_path  # noqa: E402

_SOURCE = _ROOT / "rby1_dg5f.urdf"
_OUT_DIR = _ROOT / "robot_hand" / "urdf"

_SIDES = {
    "right": {"link_prefix": "right_hand_", "out": "dg5f_right.urdf", "robot_name": "dg5f_right_hand"},
    "left": {"link_prefix": "left_hand_", "out": "dg5f_left.urdf", "robot_name": "dg5f_left_hand"},
}

# Finger-segment link name -> (finger 1-5, segment 1-4), e.g.
# "right_hand_rl_dg_2_3" -> (2, 3). Matches both sides' naming.
_FINGER_SEGMENT_RE = re.compile(r"_dg_(\d)_(\d)$")

# Fallback only, for a link whose <collision><mesh> is missing/unreadable
# (shouldn't happen -- every finger segment has one in rby1_dg5f.urdf).
_FALLBACK_SEGMENT_RADIUS = {1: 0.008, 2: 0.007, 3: 0.006, 4: 0.005}

_RADIUS_PERCENTILE = 50.0  # median: "typical" cross-section, not near-max bulge in every direction (see module docstring)


def _segment_radius_from_mesh(link_elem: ET.Element, direction: np.ndarray, length: float) -> Optional[float]:
    """Real cross-section radius for this link's finger-segment capsule:
    the given percentile of its own <collision><mesh> vertices' distance
    from the capsule's central axis (the same direction/length the capsule
    itself is placed along -- see add_finger_collision_capsules), so the
    authored capsule actually bounds the real geometry it's standing in
    for instead of an arbitrary guess."""
    mesh_elem = link_elem.find("./collision/geometry/mesh")
    if mesh_elem is None:
        return None
    try:
        path = resolve_mesh_path(mesh_elem.get("filename"))
        mesh = trimesh.load(str(path))
    except (ValueError, FileNotFoundError):
        return None
    verts = (
        mesh.vertices
        if not isinstance(mesh, trimesh.Scene)
        else np.vstack([g.vertices for g in mesh.geometry.values()])
    )
    dir_hat = direction / length
    rel = verts - (direction / 2.0)
    along = rel @ dir_hat
    radial = np.linalg.norm(rel - np.outer(along, dir_hat), axis=1)
    return float(np.percentile(radial, _RADIUS_PERCENTILE))


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

        radius = _segment_radius_from_mesh(link_elem, direction, length)
        if radius is None:
            radius = _FALLBACK_SEGMENT_RADIUS[segment]

        collision = ET.SubElement(link_elem, "collision")
        origin = ET.SubElement(collision, "origin")
        origin.set("xyz", f"{midpoint[0]:.6f} {midpoint[1]:.6f} {midpoint[2]:.6f}")
        origin.set("rpy", f"{rpy[0]:.6f} {rpy[1]:.6f} {rpy[2]:.6f}")
        geometry = ET.SubElement(collision, "geometry")
        capsule = ET.SubElement(geometry, "capsule")
        capsule.set("radius", f"{radius:.6f}")
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
