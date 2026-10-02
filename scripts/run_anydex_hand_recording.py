"""Replay an AnyDex MediaPipe PKL on an AnyDex-supported MuJoCo hand.

This is a lightweight alternative to third_party/AnyDexRetarget/example/
teleop_sim.py for environments without OpenCV. It only supports replay PKL
input, which is enough for checking a converted Quest recording on Inspire.

The robot hand is rendered with mujoco.viewer as usual (so physics -- real
mj_step with actuators/contacts, not just a kinematic qpos override --
still applies). The raw input hand skeleton (the recorded MediaPipe-order
landmarks that drove the retarget) is shown separately in its own Open3D
window, kept in sync with the same frame index. 'P' (in either window)
toggles pause/resume; while paused, both windows stay responsive but stop
advancing frames/stepping physics. A UDP listener on --control-port accepts
the same toggle, so a launcher UI (e.g. scripts/retarget_compare_ui.py) can
drive pause/resume with a button instead of needing window focus.
"""
from __future__ import annotations

import argparse
import importlib
import pathlib
import pickle
import socket
import sys
import threading
import time

import mujoco
import numpy as np
import open3d as o3d
import yaml

_ROOT = pathlib.Path(__file__).resolve().parent.parent
_ANYDEX_EXAMPLE = _ROOT / "third_party" / "AnyDexRetarget" / "example"
_ANYDEX_ROOT = _ROOT / "third_party" / "AnyDexRetarget"
for path in (str(_ANYDEX_ROOT), str(_ANYDEX_EXAMPLE)):
    if path not in sys.path:
        sys.path.insert(0, path)

from anydexretarget import Retargeter  # noqa: E402
from output.sim.mujoco_output import ROBOT_HAND_CONFIGS, retarget_to_mujoco_target  # noqa: E402

# Standard MediaPipe 21-point hand landmark connectivity (wrist=0, thumb
# 1-4, index 5-8, middle 9-12, ring 13-16, pinky 17-20, plus the palm arcs
# between finger bases) -- matches the index order openxr_to_mediapipe21()
# produces, so this draws correctly over our own Quest-derived skeletons too.
_HAND_CONNECTIONS = [
    (0, 1), (1, 2), (2, 3), (3, 4),
    (0, 5), (5, 6), (6, 7), (7, 8),
    (0, 9), (9, 10), (10, 11), (11, 12),
    (0, 13), (13, 14), (14, 15), (15, 16),
    (0, 17), (17, 18), (18, 19), (19, 20),
    (5, 9), (9, 13), (13, 17),
]


def _resolve_example_path(path: pathlib.Path) -> pathlib.Path:
    if path.is_absolute():
        return path
    return _ANYDEX_EXAMPLE / path


def _load_recording(path: pathlib.Path) -> list[dict]:
    with path.open("rb") as f:
        frames = pickle.load(f)
    if not frames:
        raise RuntimeError(f"{path}: empty replay")
    return frames


def _build_skeleton_geoms() -> tuple[o3d.geometry.PointCloud, o3d.geometry.LineSet]:
    points = np.zeros((21, 3), dtype=np.float64)
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)
    pcd.paint_uniform_color((1.0, 0.1, 0.1))

    lines = o3d.geometry.LineSet()
    lines.points = o3d.utility.Vector3dVector(points)
    lines.lines = o3d.utility.Vector2iVector(np.asarray(_HAND_CONNECTIONS, dtype=np.int32))
    lines.paint_uniform_color((1.0, 0.1, 0.1))
    return pcd, lines


def _update_skeleton_geoms(pcd: o3d.geometry.PointCloud, lines: o3d.geometry.LineSet, landmarks: np.ndarray) -> None:
    points = landmarks.astype(np.float64)
    pcd.points = o3d.utility.Vector3dVector(points)
    lines.points = o3d.utility.Vector3dVector(points)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=pathlib.Path, default=pathlib.Path("config/adaptive/quest3/quest3_inspire_hand.yaml"))
    parser.add_argument("--play", type=pathlib.Path, default=pathlib.Path("data/quest_hand_latest_anydex.pkl"))
    parser.add_argument("--hand", choices=("left", "right"), default="right")
    parser.add_argument("--speed", type=float, default=1.0)
    parser.add_argument("--loop", action="store_true")
    parser.add_argument("--headless", action="store_true")
    parser.add_argument("--max-frames", type=int, default=None)
    parser.add_argument("--control-port", type=int, default=0, help="UDP port to listen for 'toggle'/'pause'/'resume' pause commands (0 = disabled)")
    args = parser.parse_args()

    config_path = _resolve_example_path(args.config)
    play_path = _resolve_example_path(args.play)
    frames = _load_recording(play_path)

    with config_path.open("r", encoding="utf-8") as f:
        config = yaml.safe_load(f)
    robot_type = config.get("robot", {}).get("type", "inspire_hand")
    hand_cfg = ROBOT_HAND_CONFIGS[robot_type]
    model_path = pathlib.Path(hand_cfg["model_path"](args.hand))

    model = mujoco.MjModel.from_xml_path(str(model_path))
    data = mujoco.MjData(model)
    retargeter = Retargeter.from_yaml(str(config_path), args.hand)

    qpos_servo_alpha = hand_cfg.get("qpos_servo_alpha")
    direct_qpos_mode = bool(hand_cfg.get("direct_qpos", False))
    qpos_servo_mode = qpos_servo_alpha is not None and not direct_qpos_mode
    actuator_mode = model.nu > 0 and not direct_qpos_mode and not qpos_servo_mode
    if not actuator_mode and not qpos_servo_mode:
        direct_qpos_mode = True
    target_len = model.nu if actuator_mode else model.nq
    latest_target = np.zeros(target_len, dtype=np.float32)

    if actuator_mode:
        for i in range(model.nu):
            if model.actuator_ctrllimited[i]:
                lo, hi = model.actuator_ctrlrange[i]
                data.ctrl[i] = 0.5 * (lo + hi)
        for _ in range(100):
            mujoco.mj_step(model, data)
    else:
        mujoco.mj_forward(model, data)

    paused = [False]

    def _toggle_pause(*_args) -> bool:
        paused[0] = not paused[0]
        print(f"[run_anydex_hand_recording] {'paused' if paused[0] else 'resumed'}")
        return False

    def _key_callback(keycode: int) -> None:
        # Space (GLFW_KEY_SPACE=32) is already bound by mujoco's own viewer to
        # its internal pause -- use 'P' instead to avoid fighting that, and to
        # keep our own `paused` flag (which also holds the skeleton window and
        # the retarget step, not just mujoco's physics) as the single source
        # of truth.
        if keycode in (ord("P"), ord("p")):
            _toggle_pause()

    if args.control_port:
        def _control_listener() -> None:
            sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
            sock.bind(("127.0.0.1", args.control_port))
            while True:
                data_bytes, _ = sock.recvfrom(64)
                cmd = data_bytes.decode("utf-8", errors="ignore").strip().lower()
                if cmd == "pause":
                    paused[0] = True
                elif cmd == "resume":
                    paused[0] = False
                else:  # "toggle" or anything else
                    paused[0] = not paused[0]
                print(f"[run_anydex_hand_recording] {'paused' if paused[0] else 'resumed'} (control port)")

        threading.Thread(target=_control_listener, daemon=True).start()
        print(f"[run_anydex_hand_recording] control-port={args.control_port} (udp 'pause'/'resume'/'toggle')")

    viewer = None
    skel_vis = None
    skel_pcd = skel_lines = None
    if not args.headless:
        mujoco_viewer = importlib.import_module("mujoco.viewer")
        viewer = mujoco_viewer.launch_passive(model, data, key_callback=_key_callback)

        skel_vis = o3d.visualization.VisualizerWithKeyCallback()
        skel_vis.create_window(window_name=f"input skeleton ({args.hand}) -- P=pause", width=500, height=500)
        skel_vis.register_key_callback(ord("P"), _toggle_pause)
        skel_vis.get_render_option().point_size = 8.0
        skel_pcd, skel_lines = _build_skeleton_geoms()
        # Seed with the first frame that actually has TRACKED landmarks (not
        # just the first frame with the key present -- convert_quest_
        # recording_to_anydex_replay.py always writes a (21,3) array, filling
        # untracked frames with all-zeros rather than omitting the key, so
        # "key present" alone picks up an all-zero/origin frame just as
        # easily as a real one). An all-zero seed collapses the initial
        # bounding box to the origin, which is what made the skeleton appear
        # as "nothing but the axis" -- reset_view_point(True) below had
        # nothing real to frame.
        first_landmarks = None
        for fr in frames:
            raw = fr.get(f"{args.hand}_fingers")
            if raw is None:
                continue
            arr = np.asarray(raw)
            if np.any(arr):
                first_landmarks = arr
                break
        if first_landmarks is not None:
            _update_skeleton_geoms(skel_pcd, skel_lines, first_landmarks)
        skel_vis.add_geometry(skel_pcd)
        skel_vis.add_geometry(skel_lines)
        skel_vis.add_geometry(o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.05))
        skel_vis.reset_view_point(True)

    print(f"[run_anydex_hand_recording] robot={robot_type} hand={args.hand}")
    print(f"[run_anydex_hand_recording] config={config_path}")
    print(f"[run_anydex_hand_recording] replay={play_path} frames={len(frames)}")
    print(
        "[run_anydex_hand_recording] control="
        + ("actuator" if actuator_mode else "qpos_servo" if qpos_servo_mode else "direct_qpos")
    )
    print("[run_anydex_hand_recording] P (in either window) = pause/resume")

    rendered = 0
    idx = 0
    n_frames = len(frames)
    prev_t = float(frames[0].get("t", 0.0))
    try:
        while True:
            if viewer is not None and not viewer.is_running():
                return
            if skel_vis is not None and not skel_vis.poll_events():
                return

            if paused[0]:
                if viewer is not None:
                    viewer.sync()
                if skel_vis is not None:
                    skel_vis.update_renderer()
                time.sleep(0.03)
                continue

            frame = frames[idx]
            t = float(frame.get("t", prev_t))
            dt = max(0.0, (t - prev_t) / max(args.speed, 1e-6))
            prev_t = t
            if viewer is not None and dt > 0.0:
                time.sleep(dt)

            target = retarget_to_mujoco_target(
                fingers_data=frame,
                hand_side=args.hand,
                retargeter=retargeter,
                hand_cfg=hand_cfg,
                target_len=target_len,
            )
            if target is not None:
                latest_target[:] = target
                if direct_qpos_mode:
                    data.qpos[:target_len] = latest_target
                    data.qvel[:] = 0.0
                    mujoco.mj_forward(model, data)
                elif actuator_mode:
                    data.ctrl[:] = latest_target
                    for _ in range(10):
                        mujoco.mj_step(model, data)
                else:
                    data.qpos[:target_len] += float(qpos_servo_alpha) * (latest_target - data.qpos[:target_len])
                    data.qvel[:] = 0.0
                    mujoco.mj_forward(model, data)

            if viewer is not None:
                viewer.sync()
            if skel_vis is not None:
                landmarks = frame.get(f"{args.hand}_fingers")
                if landmarks is not None:
                    _update_skeleton_geoms(skel_pcd, skel_lines, np.asarray(landmarks))
                    skel_vis.update_geometry(skel_pcd)
                    skel_vis.update_geometry(skel_lines)
                skel_vis.update_renderer()

            rendered += 1
            if args.max_frames is not None and rendered >= args.max_frames:
                return

            idx += 1
            if idx >= n_frames:
                if not args.loop:
                    return
                idx = 0
                prev_t = float(frames[0].get("t", 0.0))
    except KeyboardInterrupt:
        pass
    finally:
        if viewer is not None:
            viewer.close()
        if skel_vis is not None:
            skel_vis.destroy_window()


if __name__ == "__main__":
    main()
