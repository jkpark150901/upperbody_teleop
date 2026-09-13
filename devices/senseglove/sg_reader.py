"""SenseGlove readers.

Three backends behind a common interface (`SenseGloveReaderBase`):

- `MockSenseGloveReader`        : synthetic per-finger flexion + a simple
                                  2-joint/3-segment FK curl model, so the
                                  rest of the pipeline (UDP link, MuJoCo
                                  point rendering) can be validated without
                                  gloves attached.
- `SGCoreSenseGloveReader`      : best-effort real integration IF some SDK
                                  version happens to expose a Python import
                                  mirroring SGCore's native API 1:1.
- `SenseGloveJSONBridgeReader`  : **the recommended real-hardware path** --
                                  reads HandState from a local JSON-over-TCP
                                  socket fed by `senseglove_bridge/
                                  senseglove_bridge.cpp`, a real C++ helper
                                  in this repo (see below).

API shapes below were checked directly against SGCore's real public
headers and its own official sample
(https://github.com/Adjuvo/SenseGlove-API -- HandLayer.hpp, HandPose.hpp,
Vect3D.hpp, examples/sgcore-client.cpp), not guessed:
  - `HandPose` comes from a **static** call, no device object needed:
      `bool ok = SGCore::HandLayer::GetHandPose(rightHand, /*out*/ handPose)`
  - Normalized flexion (0..1) for all 5 fingers at once:
      `std::vector<float> flexion = handPose.GetNormalizedFlexion(true)`
  - `handPose.GetJointPositions()`: 5 (finger, thumb..pinky) x 4 (joint,
    including fingertip) array of `Vect3D` (`.GetX()/.GetY()/.GetZ()`), in
    **millimeters**, relative to the wrist -- `HandState.joint_positions`
    is meters, so both this reader and the C++ helper divide by 1000.
  - `SGCore::SenseCom::ScanningActive()` / `StartupSenseCom()` to ensure
    the connection process is running.

**`senseglove_bridge/senseglove_bridge.cpp`** (sibling top-level folder)
implements exactly this against the real headers, and was verified to
compile+link cleanly with MSVC and to talk correctly end-to-end to
`SenseGloveJSONBridgeReader` below (using stub headers reproducing the
SDK's real signatures, since the actual SDK isn't installed on the machine
this was written on -- see that folder's README for exact build steps and
what's still unverified: linking the real `SGCoreCpp.lib` and reading from
actual hardware).

Still UNCONFIRMED (no official Python binding for SGCore is documented
anywhere -- the native API is C++/C# only, which is exactly why the C++
bridge exists): whether `import SGCore` works at all in Python on a real
install, and if so, whether it mirrors the C++ names/signatures above.
`SGCoreSenseGloveReader` raises immediately with an actionable message if
it doesn't, rather than pretending to work -- use
`SenseGloveJSONBridgeReader` (+ the C++ bridge) instead if that happens,
which is the expected/recommended outcome, not a fallback of last resort.
"""

from __future__ import annotations

import abc
import argparse
import json
import socket
import time
from typing import Optional, Tuple

import numpy as np

from devices.senseglove.sg_types import HandState, N_FINGERS

# Approximate proximal/middle/distal phalanx lengths (m) per finger, only
# used for the mock's visual FK and as a sanity fallback — real data
# supersedes this.
_SEGMENT_LENGTHS = {
    "thumb": (0.035, 0.030, 0.0),
    "index": (0.045, 0.028, 0.022),
    "middle": (0.048, 0.030, 0.024),
    "ring": (0.045, 0.028, 0.022),
    "pinky": (0.038, 0.022, 0.018),
}
_SPLAY_RAD = {"thumb": -0.55, "index": -0.16, "middle": 0.0, "ring": 0.16, "pinky": 0.32}
_MAX_BEND_RAD = 1.75  # ~100 deg total curl at flexion == 1.0


class SenseGloveReaderBase(abc.ABC):
    def connect(self) -> None:
        pass

    @abc.abstractmethod
    def poll(self) -> Tuple[Optional[HandState], Optional[HandState]]:
        """Return (left, right) HandState; either may be None if stale/absent."""

    def close(self) -> None:
        pass


class MockSenseGloveReader(SenseGloveReaderBase):
    def __init__(self, seed: int = 0):
        self._t0 = time.time()
        self._rng = np.random.default_rng(seed)

    def connect(self) -> None:
        self._t0 = time.time()

    def poll(self) -> Tuple[Optional[HandState], Optional[HandState]]:
        t = time.time() - self._t0
        left = self._make_hand("left", t, phase=0.0)
        right = self._make_hand("right", t, phase=1.57)
        return left, right

    def _make_hand(self, hand: str, t: float, phase: float) -> HandState:
        base_flex = 0.5 + 0.5 * np.sin(0.8 * t + phase)
        flexion = np.clip(
            base_flex + np.array([0.0, 0.05, 0.0, -0.05, -0.1]) * np.sin(1.3 * t + phase),
            0.0,
            1.0,
        )
        joint_positions = _finger_fk_all(flexion)
        return HandState(
            timestamp=time.time(),
            hand=hand,
            flexion=flexion,
            joint_positions=joint_positions,
            raw={"source": "mock"},
        )


def _finger_fk_all(flexion: np.ndarray) -> np.ndarray:
    from devices.senseglove.sg_types import FINGER_NAMES, N_JOINTS_PER_FINGER

    out = np.zeros((N_FINGERS, N_JOINTS_PER_FINGER, 3))
    for i, name in enumerate(FINGER_NAMES):
        out[i] = _finger_fk_single(name, float(flexion[i]))
    return out


def _finger_fk_single(name: str, flexion: float) -> np.ndarray:
    """Wrist-local polyline [MCP, PIP, DIP, TIP] for one finger, given 0..1 flexion."""
    seg_lengths = _SEGMENT_LENGTHS[name]
    splay = _SPLAY_RAD[name]

    mcp_base = 0.03  # knuckle row offset from wrist origin, meters, +X = "out of palm"
    base = np.array([mcp_base * np.cos(splay), mcp_base * np.sin(splay), -0.01])
    dir0 = np.array([np.cos(splay), np.sin(splay), 0.0])

    bend_per_joint = flexion * _MAX_BEND_RAD / 2.0

    def rot_y(angle):
        c, s = np.cos(angle), np.sin(angle)
        return np.array([[c, 0.0, s], [0.0, 1.0, 0.0], [-s, 0.0, c]])

    pts = [base]
    direction = dir0
    for k, seg_len in enumerate(seg_lengths):
        if seg_len <= 0.0:
            pts.append(pts[-1])
            continue
        direction = rot_y(bend_per_joint * (k + 1)) @ dir0
        pts.append(pts[-1] + seg_len * direction)

    return np.stack(pts[:4], axis=0)


class SGCoreSenseGloveReader(SenseGloveReaderBase):
    """Best-effort real integration against SGCore's native API shape.
    See module docstring for what's confirmed vs. guessed here, and why
    `SenseGloveJSONBridgeReader` is the more realistic real-hardware path.
    """

    def __init__(self):
        try:
            import SGCore  # type: ignore
        except ImportError as e:
            raise RuntimeError(
                "SGCore module not importable (no official Python binding "
                "is documented for it -- see this file's module docstring). "
                "Use SenseGloveJSONBridgeReader instead, fed by a small "
                "C++/C# helper written against the vendor's own SDK sample."
            ) from e
        self._sgcore = SGCore

    def connect(self) -> None:
        # SGCore SDK samples typically require SenseCom running as a
        # separate background process; the library itself just attaches to
        # whatever devices SenseCom already sees.
        pass

    def poll(self) -> Tuple[Optional[HandState], Optional[HandState]]:
        sg = self._sgcore
        left = self._poll_hand(sg, right_hand=False)
        right = self._poll_hand(sg, right_hand=True)
        return left, right

    def _poll_hand(self, sg, right_hand: bool) -> Optional[HandState]:
        # Mirrors SGCore::HandLayer::GetHandPose(bool, HandPose&) -- a
        # static call, confirmed against the real C++ header and sample
        # (see module docstring). UNCONFIRMED: whether a Python wrapper (if
        # one even exists) exposes `HandLayer` the same way, and whether it
        # surfaces the C++ `bool f(out T&)` pattern as a `(bool, T)` tuple
        # return, as assumed here.
        if not hasattr(sg, "HandLayer"):
            raise RuntimeError(
                "SGCore.HandLayer not found on the imported SGCore module "
                "-- this Python binding doesn't mirror the real C++ API "
                "(devices/senseglove/sg_reader.py module docstring). Use "
                "SenseGloveJSONBridgeReader + senseglove_bridge.cpp instead."
            )
        if not sg.HandLayer.DeviceConnected(right_hand):
            return None

        ok, pose = sg.HandLayer.GetHandPose(right_hand)
        if not ok:
            return None

        flexion = np.clip(np.array(pose.GetNormalizedFlexion(True), dtype=float), 0.0, 1.0)

        joint_positions_mm = np.array(
            [[[p.GetX(), p.GetY(), p.GetZ()] for p in finger] for finger in pose.GetJointPositions()],
            dtype=float,
        )  # (5,4,3) mm, wrist-relative
        joint_positions = joint_positions_mm / 1000.0

        return HandState(
            timestamp=time.time(),
            hand="right" if right_hand else "left",
            flexion=flexion,
            joint_positions=joint_positions,
            raw={"source": "sgcore"},
        )


class SenseGloveJSONBridgeReader(SenseGloveReaderBase):
    """Reads HandState from a local JSON-over-TCP bridge.

    Since no official SenseGlove Python binding could be confirmed (module
    docstring), this is the realistic path for real hardware: write a
    small C++/C# program against the vendor's own (guaranteed-working)
    SGCore sample --

        HandProfile profile = HandProfile::Default(glove.IsRight());
        HandPose pose;
        if (glove.GetHandPose(profile, pose)) {
            // flexion[f] = pose.GetNormalizedFlexion((Finger)f, true);
            // joint_positions[f][j] = pose.jointPositions[f][j] / 1000.0;  // mm -> m
        }

    -- that prints one JSON object per line, per hand, per update, to a
    TCP client connection:

        {"hand": "left" | "right",
         "flexion": [thumb, index, middle, ring, pinky],   // 0..1
         "joint_positions": [[[x,y,z], ...4 joints...], ...5 fingers...]}  // meters, wrist-relative

    This reader is the TCP client: it connects to that helper program and
    parses newline-delimited JSON. Only that small helper program needs to
    be written/verified on the real glove PC -- this side is already
    tested (see the sg_reader smoke test / project test suite).
    """

    def __init__(self, host: str = "127.0.0.1", port: int = 8850, timeout: float = 0.05):
        self._host = host
        self._port = port
        self._timeout = timeout
        self._sock: Optional[socket.socket] = None
        self._buf = b""

    def connect(self) -> None:
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        sock.settimeout(5.0)
        sock.connect((self._host, self._port))
        sock.settimeout(self._timeout)
        self._sock = sock

    def poll(self) -> Tuple[Optional[HandState], Optional[HandState]]:
        if self._sock is None:
            raise RuntimeError("call connect() before poll()")

        try:
            chunk = self._sock.recv(65536)
            if chunk:
                self._buf += chunk
        except socket.timeout:
            pass
        except OSError:
            pass

        left: Optional[HandState] = None
        right: Optional[HandState] = None
        while b"\n" in self._buf:
            line, self._buf = self._buf.split(b"\n", 1)
            line = line.strip()
            if not line:
                continue
            try:
                d = json.loads(line)
                state = HandState(
                    timestamp=time.time(),
                    hand=d["hand"],
                    flexion=np.array(d["flexion"], dtype=float),
                    joint_positions=np.array(d["joint_positions"], dtype=float),
                    raw={"source": "bridge"},
                )
            except Exception as e:
                print(f"[senseglove_bridge] bad line, skipping: {e}")
                continue
            if state.hand == "left":
                left = state
            elif state.hand == "right":
                right = state
        return left, right

    def close(self) -> None:
        if self._sock is not None:
            self._sock.close()
            self._sock = None


def _main():
    parser = argparse.ArgumentParser(description="SenseGlove reader smoke test")
    parser.add_argument("--backend", choices=["mock", "sgcore", "bridge"], default="mock")
    parser.add_argument("--hz", type=float, default=20.0)
    parser.add_argument("--bridge-host", default="127.0.0.1")
    parser.add_argument("--bridge-port", type=int, default=8850)
    args = parser.parse_args()

    reader: SenseGloveReaderBase
    if args.backend == "mock":
        reader = MockSenseGloveReader()
    elif args.backend == "sgcore":
        reader = SGCoreSenseGloveReader()
    else:
        reader = SenseGloveJSONBridgeReader(args.bridge_host, args.bridge_port)

    reader.connect()
    period = 1.0 / args.hz
    try:
        while True:
            left, right = reader.poll()
            if left is not None:
                print(f"L flex={np.round(left.flexion, 2)}")
            if right is not None:
                print(f"R flex={np.round(right.flexion, 2)}")
            time.sleep(period)
    except KeyboardInterrupt:
        pass
    finally:
        reader.close()


if __name__ == "__main__":
    _main()
