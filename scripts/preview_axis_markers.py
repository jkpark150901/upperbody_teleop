"""Offline smoke test: render one frame of the axis/point markers to a PNG
using mock PN + SenseGlove data, without needing two machines or a live
UDP link. Useful for sanity-checking mujoco_teleop/markers.py changes.

Usage:
    python scripts/preview_axis_markers.py --out preview.png
"""

from __future__ import annotations

import argparse
import pathlib
import sys

_ROOT = pathlib.Path(__file__).parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import mujoco  # noqa: E402
import numpy as np  # noqa: E402

from devices.perception_neuron.pn_reader import MockPNReader  # noqa: E402
from devices.senseglove.sg_reader import MockSenseGloveReader  # noqa: E402
from mujoco_teleop import markers  # noqa: E402
from teleop.calibration import SideCalibration, WristRetargeter, map_hand_points_to_world  # noqa: E402
from teleop.geometry import Pose, quat_identity  # noqa: E402

_SCENE_XML = _ROOT / "mujoco_teleop" / "scene_axis.xml"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", default=str(_ROOT / "scratch_preview.png"))
    parser.add_argument("--width", type=int, default=1280)
    parser.add_argument("--height", type=int, default=720)
    args = parser.parse_args()

    pn = MockPNReader()
    sg = MockSenseGloveReader()
    pn.connect()
    sg.connect()

    # advance the mock generators a bit so poses aren't at t=0
    import time
    time.sleep(0.7)

    pn_frame = pn.poll()
    left_hand, right_hand = sg.poll()

    left_cal = SideCalibration(
        marker_origin=Pose(np.array([0.25, 0.25, 1.0]), quat_identity()),
        scale=0.7,
        clamp_min=np.array([-0.4, -0.6, 0.4]),
        clamp_max=np.array([0.7, 0.6, 1.8]),
    )
    right_cal = SideCalibration(
        marker_origin=Pose(np.array([0.25, -0.25, 1.0]), quat_identity()),
        scale=0.7,
        clamp_min=np.array([-0.4, -0.6, 0.4]),
        clamp_max=np.array([0.7, 0.6, 1.8]),
    )
    retargeter = WristRetargeter(left_cal, right_cal)
    retargeter.recenter(pn_frame.chest, pn_frame.left_wrist, pn_frame.right_wrist)

    # move a bit further and recompute so recenter offset is visibly non-trivial
    time.sleep(0.4)
    pn_frame = pn.poll()
    left_hand, right_hand = sg.poll()

    left_target = retargeter.update("left", pn_frame.chest, pn_frame.left_wrist)
    right_target = retargeter.update("right", pn_frame.chest, pn_frame.right_wrist)

    model = mujoco.MjModel.from_xml_path(str(_SCENE_XML))
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)

    renderer = mujoco.Renderer(model, height=args.height, width=args.width)
    renderer.update_scene(data)
    # mujoco.Renderer allocates its own mjvScene sized for the model's static
    # geoms; grow headroom for our custom markers by reusing its scene object
    # directly (Renderer exposes `.scene`).
    scene = renderer.scene

    markers.add_axis_triad(scene, Pose(np.array([0.0, 0.0, 1.4]), pn_frame.chest.quat), length=0.1, radius=0.006)
    markers.add_axis_triad(scene, left_target, length=0.07)
    markers.add_axis_triad(scene, right_target, length=0.07)

    left_pts = map_hand_points_to_world(left_target, left_hand.joint_positions)
    right_pts = map_hand_points_to_world(right_target, right_hand.joint_positions)
    markers.add_hand_points(scene, left_pts, radius=0.008, rgba=(1.0, 0.85, 0.0, 1.0))
    markers.add_hand_points(scene, right_pts, radius=0.008, rgba=(0.2, 0.85, 1.0, 1.0))

    pixels = renderer.render()

    from PIL import Image
    Image.fromarray(pixels).save(args.out)
    print(f"wrote {args.out}  ({pixels.shape[1]}x{pixels.shape[0]})")
    print(f"left_target  pos={np.round(left_target.pos, 3)}")
    print(f"right_target pos={np.round(right_target.pos, 3)}")


if __name__ == "__main__":
    main()
