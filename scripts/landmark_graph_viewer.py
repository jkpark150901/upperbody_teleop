"""Standalone raw-human-landmark viewer: Open3D (freely rotatable 3D, with
per-joint axis triads where orientation is tracked) + matplotlib (fixed-
view node-edge graph) side by side, driven by UDP packets from
scripts/upper_body_retarget_mujoco.py's --show-landmarks.

Runs as a SEPARATE PROCESS from the mujoco driver script on purpose: an
earlier in-process vedo (VTK/OpenGL) window crashed alongside
mujoco.viewer's own GLFW-based passive viewer, and Open3D also owns a
GLFW/OpenGL context -- putting either in the same process as mujoco.viewer
risks the same crash. A UDP socket (loopback, tiny JSON payload) is the
whole IPC surface, so this process's GL context never shares a process
with mujoco's.

Usage (normally auto-spawned by --show-landmarks; run manually to attach
to an already-running sender, or to watch from a second machine):
  python -m scripts.landmark_graph_viewer --port 5099
"""
from __future__ import annotations

import argparse
import json
import socket
import time

import numpy as np

_POINTS = (
    "hips", "chest", "head",
    "left_shoulder", "left_elbow", "left_wrist",
    "right_shoulder", "right_elbow", "right_wrist",
)
_BONES = (
    ("chest", "left_shoulder"), ("left_shoulder", "left_elbow"), ("left_elbow", "left_wrist"),
    ("chest", "right_shoulder"), ("right_shoulder", "right_elbow"), ("right_elbow", "right_wrist"),
    ("chest", "head"), ("chest", "hips"),
)
_BONE_IDX = [(_POINTS.index(a), _POINTS.index(b)) for a, b in _BONES]

# Joints whose orientation is tracked (see UpperBodyLandmarks.left_wrist_rotation
# etc.), and which arrow(s) to draw for each: (joint, local-frame unit vector
# to rotate by that joint's tracked rotation matrix, RGB color, label).
# For the wrists, plain X/Y/Z-colored axes don't tell you which one is
# "fingers" at a glance -- confirmed on real hardware that MocapApi's
# LeftHand/RightHand local frame has dorsal (back of hand) along +Y on
# *both* sides, but fingers flips sign between sides (+X on LeftHand, -X on
# RightHand -- same mirrored-rigging convention robot_hand/upper_body_retarget.py's
# _BVH_HAND_FINGERS_LOCAL now uses), so draw exactly those two per side,
# explicitly labeled, instead of a generic RGB triad. Head has no such
# established semantic axes, so it keeps the generic X/Y/Z triad.
_AXIS_SPECS = (
    ("left_wrist", np.array([1.0, 0.0, 0.0]), (1.0, 0.0, 1.0), "L fingers"),
    ("left_wrist", np.array([0.0, -1.0, 0.0]), (1.0, 0.5, 0.0), "L dorsal"),
    ("right_wrist", np.array([-1.0, 0.0, 0.0]), (1.0, 0.0, 1.0), "R fingers"),
    ("right_wrist", np.array([0.0, -1.0, 0.0]), (1.0, 0.5, 0.0), "R dorsal"),
    ("head", np.array([1.0, 0.0, 0.0]), (1.0, 0.0, 0.0), "head X"),
    ("head", np.array([0.0, 1.0, 0.0]), (0.0, 1.0, 0.0), "head Y"),
    ("head", np.array([0.0, 0.0, 1.0]), (0.0, 0.0, 1.0), "head Z"),
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--port", type=int, default=5099)
    parser.add_argument("--hz", type=float, default=30.0, help="max UI refresh rate")
    parser.add_argument("--axis-length", type=float, default=8.0, help="joint axis triad length, same units as the landmark data")
    args = parser.parse_args()

    import open3d as o3d
    import matplotlib
    matplotlib.use("TkAgg")
    import matplotlib.pyplot as plt

    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    sock.bind(("127.0.0.1", args.port))
    sock.setblocking(False)

    # Open3D: freely rotatable 3D view.
    o3d_vis = o3d.visualization.Visualizer()
    o3d_vis.create_window(window_name="Raw human landmarks (Open3D, rotatable)", width=600, height=600)
    line_set = o3d.geometry.LineSet(
        points=o3d.utility.Vector3dVector(np.zeros((len(_POINTS), 3))),
        lines=o3d.utility.Vector2iVector(np.array(_BONE_IDX)),
    )
    line_set.colors = o3d.utility.Vector3dVector(np.tile([0.27, 0.51, 0.71], (len(_BONE_IDX), 1)))
    point_cloud = o3d.geometry.PointCloud(points=o3d.utility.Vector3dVector(np.zeros((len(_POINTS), 3))))
    point_cloud.paint_uniform_color([0.8, 0.1, 0.1])

    # Axis arrows: one LineSet, one segment per _AXIS_SPECS entry, built at
    # a fixed size from the start (same-size updates render correctly;
    # growing/shrinking a LineSet/PointCloud after creation does not --
    # confirmed the hard way with vedo's equivalent gotcha earlier in this
    # project). A joint with no rotation this tick collapses to a
    # zero-length segment (still present, just invisible) instead of being
    # added/removed.
    n_axis = len(_AXIS_SPECS)
    axis_lines = o3d.geometry.LineSet(
        points=o3d.utility.Vector3dVector(np.zeros((n_axis * 2, 3))),
        lines=o3d.utility.Vector2iVector(np.array([[2 * i, 2 * i + 1] for i in range(n_axis)])),
    )
    axis_lines.colors = o3d.utility.Vector3dVector(np.array([spec[2] for spec in _AXIS_SPECS]))

    o3d_vis.add_geometry(line_set)
    o3d_vis.add_geometry(point_cloud)
    o3d_vis.add_geometry(axis_lines)
    o3d_first_frame = True

    # matplotlib: fixed-view node-edge graph (labeled), the "그래프 형태" view.
    plt.ion()
    fig = plt.figure("Raw human landmarks (graph)", figsize=(5, 5))
    ax = fig.add_subplot(111, projection="3d")
    fig.show()

    print(f"[landmark_graph_viewer] listening on UDP 127.0.0.1:{args.port}")
    latest = None
    period = 1.0 / args.hz
    last_draw = 0.0
    try:
        while True:
            # Drain fully so the UI always shows the newest frame, not a
            # growing backlog (same pattern as this project's other UDP
            # readers -- see devices/quest_hand/quest_hand_reader.py).
            while True:
                try:
                    data, _ = sock.recvfrom(65536)
                    latest = json.loads(data.decode("utf-8"))
                except BlockingIOError:
                    break
            if not o3d_vis.poll_events():
                break
            o3d_vis.update_renderer()

            now = time.monotonic()
            if latest is not None and now - last_draw >= period:
                last_draw = now
                points = latest["points"]
                rotation = latest.get("rotation", {})
                pts = np.array([points[name] for name in _POINTS], dtype=float)

                line_set.points = o3d.utility.Vector3dVector(pts)
                point_cloud.points = o3d.utility.Vector3dVector(pts)

                axis_pts = np.zeros((n_axis * 2, 3))
                for i, (joint_name, local_vec, _color, _label) in enumerate(_AXIS_SPECS):
                    origin = np.array(points[joint_name], dtype=float)
                    r = rotation.get(joint_name)
                    axis_pts[2 * i] = origin
                    if r is not None:
                        direction = np.array(r, dtype=float) @ local_vec
                        axis_pts[2 * i + 1] = origin + args.axis_length * direction
                    else:
                        axis_pts[2 * i + 1] = origin  # no rotation this tick -- collapse to invisible
                axis_lines.points = o3d.utility.Vector3dVector(axis_pts)

                o3d_vis.update_geometry(line_set)
                o3d_vis.update_geometry(point_cloud)
                o3d_vis.update_geometry(axis_lines)
                if o3d_first_frame:
                    o3d_vis.reset_view_point(True)
                    o3d_first_frame = False

                ax.cla()
                ax.scatter(pts[:, 0], pts[:, 1], pts[:, 2], c="red", s=40, depthshade=False)
                for a, b in _BONES:
                    p0, p1 = np.array(points[a]), np.array(points[b])
                    ax.plot([p0[0], p1[0]], [p0[1], p1[1]], [p0[2], p1[2]], c="steelblue", lw=2)
                for joint_name, local_vec, color, label in _AXIS_SPECS:
                    r = rotation.get(joint_name)
                    if r is None:
                        continue
                    origin = np.array(points[joint_name], dtype=float)
                    tip = origin + args.axis_length * (np.array(r, dtype=float) @ local_vec)
                    ax.plot([origin[0], tip[0]], [origin[1], tip[1]], [origin[2], tip[2]], c=color, lw=2)
                    ax.text(tip[0], tip[1], tip[2], label, size=6, color=color)
                for name, p in zip(_POINTS, pts):
                    ax.text(p[0], p[1], p[2], name, size=7)
                ax.set_title("Raw human landmarks (graph)")
                center = pts.mean(axis=0)
                radius = max(1e-3, float(np.abs(pts - center).max()))
                ax.set_xlim(center[0] - radius, center[0] + radius)
                ax.set_ylim(center[1] - radius, center[1] + radius)
                ax.set_zlim(center[2] - radius, center[2] + radius)
                fig.canvas.draw_idle()
                fig.canvas.flush_events()
            else:
                time.sleep(0.001)
    except KeyboardInterrupt:
        pass
    finally:
        o3d_vis.destroy_window()
        plt.close(fig)
        sock.close()


if __name__ == "__main__":
    main()
