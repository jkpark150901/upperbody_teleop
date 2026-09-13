"""Perception Neuron readers.

Two backends are provided behind a common interface (`PNReaderBase`):

- `MockPNReader`   : synthetic wrist/chest motion, no hardware needed. Used
                     to validate the rest of the pipeline (UDP link, MuJoCo
                     axis/point rendering) without Axis Studio running.
- `MocapApiPNReader`: real Perception Neuron integration via Noitom's
                     MocapApi (https://github.com/pnmocap/MocapApi), talking
                     to Axis Studio's BVH/Calc UDP/TCP stream.

`MocapApiPNReader` needs the official `MocapApi` Python module (SWIG wrapper
shipped in the MocapApi SDK, alongside the native MocapApi DLL) importable on
this machine. That SDK is not installed on the machine this code was written
on, so this adapter is written against the SDK's publicly documented shape
(MCPApplication / MCPSettings / MCPEvent / MCPAvatar / MCPJoint) but has not
been exercised against real hardware. Before relying on it:

  1. Install Axis Studio + MocapApi on the PN PC, confirm `import MocapApi`
     works from this environment (put MocapApi.py + the native DLL on
     PYTHONPATH / next to this file).
  2. Run `python -m devices.perception_neuron.pn_reader --backend mocapapi`
     and check the joint names printed at start-up against JOINT_NAME_CHEST /
     JOINT_NAME_LEFT_WRIST / JOINT_NAME_RIGHT_WRIST below — Axis Studio's
     avatar skeleton naming can vary by actor/template and may need
     adjusting.
"""

from __future__ import annotations

import abc
import argparse
import time
from typing import Optional

import numpy as np

from teleop.geometry import Pose, quat_from_xyzw, quat_identity
from devices.perception_neuron.pn_types import PNFrame

# Avatar joint names as commonly exported by Axis Studio's default actor
# template. Verify against `avatar.get_joints()` names on the real rig —
# see module docstring.
JOINT_NAME_CHEST = "Spine3"
JOINT_NAME_LEFT_WRIST = "LeftHand"
JOINT_NAME_RIGHT_WRIST = "RightHand"


class PNReaderBase(abc.ABC):
    def connect(self) -> None:
        """Open the connection. No-op for backends that don't need it."""

    @abc.abstractmethod
    def poll(self) -> Optional[PNFrame]:
        """Return the latest frame, or None if nothing new is available.

        Must be non-blocking (or bounded by a very small timeout) since the
        caller drives its own fixed-rate loop.
        """

    def close(self) -> None:
        pass


class MockPNReader(PNReaderBase):
    """Synthetic chest/wrist motion for pipeline testing without hardware."""

    def __init__(self, seed: int = 0):
        self._t0 = time.time()
        self._rng = np.random.default_rng(seed)

    def connect(self) -> None:
        self._t0 = time.time()

    def poll(self) -> Optional[PNFrame]:
        t = time.time() - self._t0

        chest = Pose(np.array([0.0, 0.0, 1.3]), quat_identity())

        lw_pos = np.array([
            0.25 + 0.10 * np.sin(0.6 * t),
            0.20 + 0.05 * np.sin(0.9 * t + 1.0),
            1.15 + 0.08 * np.sin(0.5 * t + 0.5),
        ])
        rw_pos = np.array([
            0.25 + 0.10 * np.sin(0.6 * t + 3.14159),
            -0.20 - 0.05 * np.sin(0.9 * t + 2.0),
            1.15 + 0.08 * np.sin(0.5 * t + 1.5),
        ])

        wobble = 0.15 * np.sin(0.4 * t)
        left_wrist = Pose(lw_pos, _small_yaw_quat(wobble))
        right_wrist = Pose(rw_pos, _small_yaw_quat(-wobble))

        return PNFrame(
            timestamp=time.time(),
            chest=chest,
            left_wrist=left_wrist,
            right_wrist=right_wrist,
        )


def _small_yaw_quat(angle_rad: float) -> np.ndarray:
    half = angle_rad / 2.0
    return np.array([np.cos(half), 0.0, 0.0, np.sin(half)])


class MocapApiPNReader(PNReaderBase):
    """Real Perception Neuron reader built on Noitom's MocapApi.

    See module docstring for the caveats around this backend — it is
    written against MocapApi's documented API surface but unverified against
    real hardware/SDK on this machine.
    """

    def __init__(
        self,
        udp_port: int = 7009,
        joint_chest: str = JOINT_NAME_CHEST,
        joint_left_wrist: str = JOINT_NAME_LEFT_WRIST,
        joint_right_wrist: str = JOINT_NAME_RIGHT_WRIST,
    ):
        try:
            import MocapApi  # type: ignore
        except ImportError as e:
            raise RuntimeError(
                "MocapApi module not importable. Install Axis Studio's "
                "MocapApi SDK and place MocapApi.py + native DLL on "
                "PYTHONPATH. See devices/perception_neuron/pn_reader.py "
                "module docstring."
            ) from e

        self._mcp = MocapApi
        self._udp_port = udp_port
        self._joint_chest = joint_chest
        self._joint_left_wrist = joint_left_wrist
        self._joint_right_wrist = joint_right_wrist

        self._app = None
        self._avatar_handle = None
        self._latest: Optional[PNFrame] = None

    def connect(self) -> None:
        mcp = self._mcp
        settings = mcp.MCPSettings()
        settings.SetSettingsUDP(self._udp_port)

        app = mcp.MCPApplication()
        app.SetSettings(settings)
        # BVH data over MocapApi is typically Y-up, right-handed; adjust if
        # Axis Studio is configured differently.
        app.SetBvhRotation(mcp.EMCPBvhRotation.BvhRotationYXZ if hasattr(
            mcp, "EMCPBvhRotation") else 0)
        app.Open()
        self._app = app

    def poll(self) -> Optional[PNFrame]:
        if self._app is None:
            raise RuntimeError("call connect() before poll()")

        mcp = self._mcp
        events = self._app.PollEvents()
        for evt in events:
            if evt.event_type == mcp.EMCPEventType.AvatarUpdated:
                self._avatar_handle = evt.event_data.avatar_handle

        if self._avatar_handle is None:
            return None

        avatar = mcp.MCPAvatar(self._avatar_handle)
        joints = {j.get_name(): j for j in avatar.get_joints()}

        try:
            chest_j = joints[self._joint_chest]
            lw_j = joints[self._joint_left_wrist]
            rw_j = joints[self._joint_right_wrist]
        except KeyError as e:
            raise RuntimeError(
                f"Joint name {e} not found in avatar skeleton. Available: "
                f"{sorted(joints.keys())}. Adjust JOINT_NAME_* in "
                f"pn_reader.py to match this Axis Studio actor template."
            ) from e

        frame = PNFrame(
            timestamp=time.time(),
            chest=_joint_to_pose(chest_j),
            left_wrist=_joint_to_pose(lw_j),
            right_wrist=_joint_to_pose(rw_j),
        )
        self._latest = frame
        return frame

    def close(self) -> None:
        if self._app is not None:
            self._app.Close()
            self._app = None


def _joint_to_pose(joint) -> Pose:
    p = joint.get_local_position()  # (x, y, z) in meters, world frame
    q = joint.get_local_rotation()  # (x, y, z, w)
    return Pose(np.array([p.x, p.y, p.z]), quat_from_xyzw(q.x, q.y, q.z, q.w))


def _main():
    parser = argparse.ArgumentParser(description="Perception Neuron reader smoke test")
    parser.add_argument("--backend", choices=["mock", "mocapapi"], default="mock")
    parser.add_argument("--hz", type=float, default=20.0)
    args = parser.parse_args()

    reader: PNReaderBase
    if args.backend == "mock":
        reader = MockPNReader()
    else:
        reader = MocapApiPNReader()

    reader.connect()
    period = 1.0 / args.hz
    try:
        while True:
            frame = reader.poll()
            if frame is not None:
                print(
                    f"t={frame.timestamp:.3f} "
                    f"chest={np.round(frame.chest.pos, 3)} "
                    f"lw={np.round(frame.left_wrist.pos, 3)} "
                    f"rw={np.round(frame.right_wrist.pos, 3)}"
                )
            time.sleep(period)
    except KeyboardInterrupt:
        pass
    finally:
        reader.close()


if __name__ == "__main__":
    _main()
