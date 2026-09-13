"""UDP receiver + MuJoCo passive-viewer visualizer for the teleop pipeline.

Runs on the PC that has MuJoCo (plan section 5: the Ubuntu/GR00T PC, though
this code is plain cross-platform Python and can run wherever the MuJoCo
window should appear). Receives HumanTeleopState packets sent by
scripts/teleop_sender.py -- which runs on the PC with Perception Neuron +
SenseGlove attached -- over UDP, applies recenter/retargeting
(teleop/calibration.py), and draws:

  - chest / left wrist target / right wrist target as RGB axis triads
  - SenseGlove hand joint points as small spheres, rigidly attached to the
    retargeted wrist pose

No robot/URDF yet (plan Phase 4: "MuJoCo EE Marker") -- axis + point
markers only, drawn as custom mjvScene geoms so no MJCF body is needed.

Controls (viewer window focused):
  r      recenter (capture current human pose as the zero reference)
  space  pause / resume marker updates
"""

from __future__ import annotations

import argparse
import pathlib
import socket
import sys
import threading
import time
from typing import Optional

import mujoco
import mujoco.viewer
import numpy as np
import yaml

_ROOT = pathlib.Path(__file__).parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from mujoco_teleop import markers  # noqa: E402
from teleop.calibration import SideCalibration, WristRetargeter, map_hand_points_to_world  # noqa: E402
from teleop.geometry import Pose, euler_xyz_to_quat, quat_mul  # noqa: E402
from teleop.udp_protocol import ReceivedTeleopPacket, decode_state  # noqa: E402

_SCENE_XML = pathlib.Path(__file__).parent / "scene_axis.xml"


class UDPReceiverThread(threading.Thread):
    """Background UDP receiver that always keeps only the newest packet.

    Dropping older packets on arrival of a newer one is intentional for
    real-time teleop (plan section 20, "Teleoperation latency": prefer the
    freshest sample over queueing stale ones).
    """

    def __init__(self, listen_ip: str, listen_port: int):
        super().__init__(daemon=True)
        self._sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self._sock.bind((listen_ip, listen_port))
        self._sock.settimeout(0.5)
        self._lock = threading.Lock()
        self._latest: Optional[ReceivedTeleopPacket] = None
        self._running = True
        self.n_received = 0
        self.n_decode_errors = 0

    def run(self) -> None:
        while self._running:
            try:
                data, _addr = self._sock.recvfrom(65536)
            except socket.timeout:
                continue
            except OSError:
                break
            try:
                packet = decode_state(data)
            except Exception as e:
                self.n_decode_errors += 1
                print(f"[udp_receiver] decode error: {e}")
                continue
            self.n_received += 1
            with self._lock:
                self._latest = packet

    def latest(self) -> Optional[ReceivedTeleopPacket]:
        with self._lock:
            return self._latest

    def stop(self) -> None:
        self._running = False
        self._sock.close()


def _build_side_calibration(cfg: dict, side: str, align_quat: np.ndarray) -> SideCalibration:
    side_cfg = cfg["sides"][side]
    return SideCalibration(
        marker_origin=Pose(np.array(side_cfg["marker_origin"]), align_quat.copy()),
        scale=cfg.get("workspace_scale", 0.7),
        align_quat=align_quat,
        offset_quat=euler_xyz_to_quat(np.array(side_cfg.get("offset_euler_deg", [0.0, 0.0, 0.0]))),
        clamp_min=np.array(cfg["clamp"]["xyz_min"]),
        clamp_max=np.array(cfg["clamp"]["xyz_max"]),
        max_linear_velocity=cfg.get("max_linear_velocity", 1.5),
    )


def main():
    parser = argparse.ArgumentParser(description="MuJoCo teleop axis/point viewer (UDP receiver)")
    parser.add_argument(
        "--config",
        default=str(pathlib.Path(__file__).parent.parent / "configs" / "teleop.yaml"),
    )
    parser.add_argument("--listen-ip", default=None)
    parser.add_argument("--listen-port", type=int, default=None)
    args = parser.parse_args()

    with open(args.config, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    listen_ip = args.listen_ip or cfg["network"]["listen_ip"]
    listen_port = args.listen_port or cfg["network"]["listen_port"]

    align_quat = euler_xyz_to_quat(np.array(cfg.get("align_euler_deg", [0.0, 0.0, 0.0])))
    retargeter = WristRetargeter(
        left_cfg=_build_side_calibration(cfg, "left", align_quat),
        right_cfg=_build_side_calibration(cfg, "right", align_quat),
    )

    receiver = UDPReceiverThread(listen_ip, listen_port)
    receiver.start()
    print(f"[teleop_receiver_viewer] listening on {listen_ip}:{listen_port}")

    model = mujoco.MjModel.from_xml_path(str(_SCENE_XML))
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)

    state = {"paused": False}

    def key_callback(keycode):
        key = chr(keycode) if 0 < keycode < 256 else ""
        if key.lower() == "r":
            packet = receiver.latest()
            if packet is not None:
                retargeter.recenter(packet.chest, packet.left_wrist, packet.right_wrist)
            else:
                print("[teleop_receiver_viewer] no packet received yet, cannot recenter")
        elif key == " ":
            state["paused"] = not state["paused"]
            print(f"[teleop_receiver_viewer] paused={state['paused']}")

    chest_view_origin = np.array(cfg["chest_view_origin"])
    last_stats_t = time.time()

    with mujoco.viewer.launch_passive(model, data, key_callback=key_callback) as viewer:
        while viewer.is_running():
            step_start = time.time()

            viewer.user_scn.ngeom = 0
            packet = receiver.latest()

            if packet is not None and not state["paused"]:
                age = time.time() - packet.t_sample

                chest_view = Pose(chest_view_origin, quat_mul(align_quat, packet.chest.quat))
                markers.add_axis_triad(viewer.user_scn, chest_view, length=0.08, radius=0.005)

                left_target = retargeter.update("left", packet.chest, packet.left_wrist)
                if left_target is not None:
                    markers.add_axis_triad(viewer.user_scn, left_target, length=0.06)
                    world_pts = map_hand_points_to_world(left_target, packet.left_hand.joint_positions)
                    markers.add_hand_points(viewer.user_scn, world_pts, rgba=(1.0, 0.85, 0.0, 1.0))

                right_target = retargeter.update("right", packet.chest, packet.right_wrist)
                if right_target is not None:
                    markers.add_axis_triad(viewer.user_scn, right_target, length=0.06)
                    world_pts = map_hand_points_to_world(right_target, packet.right_hand.joint_positions)
                    markers.add_hand_points(viewer.user_scn, world_pts, rgba=(0.2, 0.85, 1.0, 1.0))

                if not retargeter.calibrated:
                    print("[teleop_receiver_viewer] press 'r' in the viewer to recenter", end="\r")

                if age > 0.3:
                    print(f"[teleop_receiver_viewer] WARNING stale packet, age={age:.2f}s")

            mujoco.mj_forward(model, data)
            viewer.sync()

            if time.time() - last_stats_t > 5.0:
                print(
                    f"[teleop_receiver_viewer] received={receiver.n_received} "
                    f"decode_errors={receiver.n_decode_errors}"
                )
                last_stats_t = time.time()

            elapsed = time.time() - step_start
            time.sleep(max(0.0, 1.0 / 60.0 - elapsed))

    receiver.stop()


if __name__ == "__main__":
    main()
