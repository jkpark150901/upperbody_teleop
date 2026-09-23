"""Quest 3 hand tracking (XR_EXT_hand_tracking, streamed by a modified
XrHandsFB APK) -> HandState.

Pipeline (see cloudxr_quest3_windows_setup.md's replacement, no CloudXR):

    Quest 3 (XrHandsFB APK, xrLocateHandJointsEXT) --UDP, Wi-Fi--> this reader

Unlike the SenseGlove bridge, there is no separate bridge *process*: the
headset itself is the sender, over a connectionless UDP socket the host
just binds and listens on. Built and adb-installed from
Meta-OpenXR-SDK/Samples/XrSamples/XrHandsFB (see that sample's main.cpp,
SendHandJointsUDP, for the sender side and the host IP/port to edit before
building).

Role in the teleop pipeline: mocap supplies wrist pose (arm IK); this
supplies finger articulation, in place of the SenseGlove. It produces the
same ``HandState`` the SenseGlove readers do, so the DG5F hand retargeter
and the main loop consume it unchanged (``--hand-backend quest``).

Wire format v1 (matches SendHandJointsUDP in XrHandsFB's main.cpp exactly --
little-endian, fields memcpy'd in order, no struct padding on either side):

    header (24 bytes):
        magic        u32   0x31485258 ("XRH1")
        version      u8    1
        hand         u8    0=left, 1=right
        active       u8    XrHandJointLocationsEXT.isActive
        reserved     u8    0
        seq          u32   per-hand send counter (UDP is unordered/lossy)
        t            f64   device XrTime-derived predicted-display time,
                            seconds -- device-local, NOT wall-clock and NOT
                            synchronized with the host's clock; diagnostic only
        tracked_mask u32   bit i set iff joint i has both
                            XR_SPACE_LOCATION_POSITION_VALID_BIT and
                            XR_SPACE_LOCATION_ORIENTATION_VALID_BIT
    joints (26 x 28 = 728 bytes):
        26 rows of [px, py, pz, qx, qy, qz, qw], float32, metres, in
        XR_EXT_hand_tracking's XrHandJointEXT order, all in the app's
        xrLocateHandJointsEXT base space (LocalSpace in the sample) --
        only relative (wrist-frame) geometry is used here, so the absolute
        reference space doesn't matter to this reader.

Total packet size: 752 bytes, well under a Wi-Fi MTU (no fragmentation).
"""

from __future__ import annotations

import socket
import struct
import time
from dataclasses import dataclass, field
from typing import Dict, Optional, Sequence, Tuple

import numpy as np

from devices.senseglove.sg_reader import SenseGloveReaderBase
from devices.senseglove.sg_types import FINGER_NAMES, HandState

# XR_EXT_hand_tracking XrHandJointEXT order.
XR_PALM, XR_WRIST = 0, 1
N_XR_JOINTS = 26

MAGIC = 0x31485258  # "XRH1", little-endian u32 -- see module docstring
_HEADER_FMT = "<IBBBBIdI"
HEADER_SIZE = struct.calcsize(_HEADER_FMT)  # 24
_JOINTS_SIZE = N_XR_JOINTS * 7 * 4  # 728
PACKET_SIZE = HEADER_SIZE + _JOINTS_SIZE  # 752

# Bone chains per finger (FINGER_NAMES order), base -> tip. The thumb has no
# intermediate phalanx, so its chain is 4 joints; the others are 5.
_CHAIN: Dict[str, Tuple[int, ...]] = {
    "thumb": (2, 3, 4, 5),
    "index": (6, 7, 8, 9, 10),
    "middle": (11, 12, 13, 14, 15),
    "ring": (16, 17, 18, 19, 20),
    "pinky": (21, 22, 23, 24, 25),
}

# HandState.joint_positions is (5, 4, 3): the last four joints of each chain
# (thumb: metacarpal/proximal/distal/tip; others: proximal/intermediate/
# distal/tip -- the metacarpal is skipped).
_STATE_JOINTS = {name: chain[-4:] for name, chain in _CHAIN.items()}

# Total curl (sum of inter-bone angles, rad) that maps to flexion 0 and 1.
# FIRST-PASS values from typical finger ROM (MCP ~90 + PIP ~100 + DIP ~60 deg
# for a full fist); tune against the real headset -- see curl_range below.
DEFAULT_CURL_RANGE: Dict[str, Tuple[float, float]] = {
    "thumb": (0.15, 1.6),
    "index": (0.2, 4.0),
    "middle": (0.2, 4.0),
    "ring": (0.2, 4.0),
    "pinky": (0.2, 4.0),
}


def _quat_xyzw_to_matrix(q: np.ndarray) -> np.ndarray:
    x, y, z, w = q / np.linalg.norm(q)
    return np.array([
        [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
        [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
        [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
    ])


def chain_curl(points: np.ndarray) -> float:
    """Sum of unsigned angles between consecutive bones of a joint chain."""
    bones = np.diff(points, axis=0)
    norms = np.linalg.norm(bones, axis=1, keepdims=True)
    if np.any(norms < 1e-6):
        raise ValueError("degenerate bone (coincident joints)")
    bones = bones / norms
    cos = np.clip(np.einsum("ij,ij->i", bones[:-1], bones[1:]), -1.0, 1.0)
    return float(np.arccos(cos).sum())


def chain_spread(local_points: np.ndarray) -> float:
    """Finger base direction in the wrist-local palm plane.

    OpenXR gives full joint positions, so unlike SenseGlove's single flexion
    scalar we can estimate ab/adduction from the first finger bone. In the
    wrist-local frame used by this reader, x is lateral across the palm and
    -z is roughly fingertip-forward for the synthetic self-check hand. The
    returned angle is positive toward +x.
    """
    if local_points.shape[0] < 2:
        raise ValueError("need at least two joints for spread")
    v = local_points[1] - local_points[0]
    lateral = float(v[0])
    forward = float(-v[2])
    if abs(lateral) < 1e-9 and abs(forward) < 1e-9:
        raise ValueError("degenerate base bone")
    return float(np.arctan2(lateral, forward))


def decode_packet(data: bytes) -> Optional[dict]:
    """Raw UDP payload -> header fields + (26, 7) joints array, or None if
    the packet doesn't match this wire format (wrong size/magic/version --
    e.g. a stray packet from something else on the port)."""
    if len(data) != PACKET_SIZE:
        return None
    magic, version, hand, active, _reserved, seq, t, tracked_mask = struct.unpack(
        _HEADER_FMT, data[:HEADER_SIZE]
    )
    if magic != MAGIC or version != 1:
        return None
    joints = np.frombuffer(data, dtype="<f4", count=N_XR_JOINTS * 7, offset=HEADER_SIZE)
    return {
        "hand": "left" if hand == 0 else "right",
        "active": bool(active),
        "seq": seq,
        "t": t,
        "tracked_mask": tracked_mask,
        "joints": joints.reshape(N_XR_JOINTS, 7),
    }


def xr_hand_to_state(
    hand: str,
    joints: Sequence[Sequence[float]],
    tracked: Optional[Sequence[bool]] = None,
    curl_range: Optional[Dict[str, Tuple[float, float]]] = None,
) -> Optional[HandState]:
    """26 XR joint poses -> HandState. None if the needed joints aren't tracked.

    flexion is computed from bone-direction angles, so it is independent of
    which reference space the headset located the joints in. joint_positions
    is expressed in the wrist joint's frame (metres). The full pose array
    (26, 7) is kept in ``raw["xr_joints"]`` for recording / a future
    joint-angle (rather than flexion-only) retargeter.
    """
    j = np.asarray(joints, dtype=float)
    if j.shape != (N_XR_JOINTS, 7):
        raise ValueError(f"expected ({N_XR_JOINTS}, 7) joints, got {j.shape}")
    if tracked is not None:
        ok = np.asarray(tracked, dtype=bool)
        needed = [XR_WRIST] + [i for c in _CHAIN.values() for i in c]
        if ok.shape != (N_XR_JOINTS,) or not ok[needed].all():
            return None

    ranges = {**DEFAULT_CURL_RANGE, **(curl_range or {})}
    pos = j[:, :3]
    flexion = np.zeros(len(FINGER_NAMES))
    for f, name in enumerate(FINGER_NAMES):
        lo, hi = ranges[name]
        flexion[f] = np.clip((chain_curl(pos[list(_CHAIN[name])]) - lo) / (hi - lo), 0.0, 1.0)

    r_wrist = _quat_xyzw_to_matrix(j[XR_WRIST, 3:])
    local = (pos - pos[XR_WRIST]) @ r_wrist  # rows: R^T (p - p_wrist)
    joint_positions = np.stack([local[list(_STATE_JOINTS[n])] for n in FINGER_NAMES])
    spread = np.array([chain_spread(local[list(_CHAIN[n])]) for n in FINGER_NAMES], dtype=float)

    return HandState(
        timestamp=time.time(),
        hand=hand,
        flexion=flexion,
        joint_positions=joint_positions,
        raw={"source": "quest_hand", "xr_joints": j, "spread": spread},
    )


@dataclass
class ReceiveStats:
    """Reception counters, exposed as ``QuestHandUDPReader.stats`` so both the
    periodic console log below and scripts/quest_hand_monitor.py's UI can
    answer "is the headset connected at all" independently of whether a
    given hand is currently tracked -- the two failure modes look identical
    from HandState alone (both just produce None), but need very different
    fixes (network/build/focus vs. "move your hand into view").
    """

    packets_total: int = 0  # every datagram received, valid or not
    packets_malformed: int = 0  # wrong size / bad magic / bad version
    # Well-formed packets per hand, regardless of active/tracked:
    packets_by_hand: Dict[str, int] = field(default_factory=lambda: {"left": 0, "right": 0})
    # Well-formed + active=1 (headset thinks it's tracking this hand):
    active_by_hand: Dict[str, int] = field(default_factory=lambda: {"left": 0, "right": 0})
    # active=1 but missing a joint xr_hand_to_state needs (partial occlusion):
    untracked_by_hand: Dict[str, int] = field(default_factory=lambda: {"left": 0, "right": 0})
    last_addr: Optional[Tuple[str, int]] = None
    last_packet_t: Optional[float] = None
    # Last well-formed packet for this hand, active or not -- "is the headset
    # streaming this hand at all" (independent of whether it's tracked):
    last_seen_t: Dict[str, Optional[float]] = field(
        default_factory=lambda: {"left": None, "right": None}
    )
    # Last well-formed + active=1 packet for this hand:
    last_active_t: Dict[str, Optional[float]] = field(
        default_factory=lambda: {"left": None, "right": None}
    )


class QuestHandUDPReader(SenseGloveReaderBase):
    """poll() -> (left, right) HandState from the Quest's UDP hand stream.

    UDP is connectionless, so unlike the SenseGlove TCP bridge there is
    nothing to "connect" to on the wire -- connect() just binds a local
    socket and waits for the headset to start sending. A hand whose
    tracking is lost on the headset (``active=0`` or a required joint not
    VALID) yields None for that side; the main loop then keeps the last
    flexion it saw, so the fingers hold their last pose rather than
    snapping open. Packets from the wrong sender/port are ignored, not
    just any datagram on the socket -- decode_packet() checks magic+size.

    Reception counters are kept in ``self.stats`` (see ReceiveStats) and a
    summary is logged to the console every few seconds -- "nothing is
    arriving at all" (network/firewall/build/focus) and "packets are
    arriving but this hand isn't tracked" (occlusion/lighting/out of view)
    are the two most common real-world failure modes and look identical
    from a bare HandState, so this is printed unconditionally, not behind a
    verbose flag. scripts/quest_hand_monitor.py is a tkinter UI on top of
    the same stats for a live, at-a-glance version of this.
    """

    def __init__(
        self,
        # Not "0.0.0.0": on this dev PC, binding the wildcard address did
        # not actually accept the headset's packets arriving on the
        # Ethernet adapter (confirmed by testing -- binding this adapter's
        # own address directly does). Multi-homed here (Ethernet + a
        # Hyper-V/WSL vEthernet adapter + Tailscale), so if this is run on a
        # different host, re-check with `ipconfig`/`ip addr` and pass the
        # right adapter's address explicitly rather than assuming 0.0.0.0
        # works.
        listen_host: str = "192.168.8.156",
        listen_port: int = 5005,
        curl_range: Optional[Dict[str, Tuple[float, float]]] = None,
        log_interval: float = 3.0,
    ):
        self._host = listen_host
        self._port = listen_port
        self._curl_range = curl_range
        self._sock: Optional[socket.socket] = None
        self.stats = ReceiveStats()
        self._log_interval = log_interval
        self._last_log_t = 0.0
        self._first_packet_logged = False

    def connect(self) -> None:
        sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        sock.bind((self._host, self._port))
        sock.setblocking(False)  # see poll(): never wait for a *future* packet
        self._sock = sock
        self.stats = ReceiveStats()
        self._last_log_t = time.time()
        self._first_packet_logged = False
        print(f"[quest_hand] listening on {self._host}:{self._port}")

    def poll(self) -> Tuple[Optional[HandState], Optional[HandState]]:
        if self._sock is None:
            raise RuntimeError("call connect() before poll()")

        now = time.time()
        latest: Dict[str, dict] = {}
        # Drain whatever is *already* queued (bounded so a fast sender can't
        # stall the caller's loop -- mirrors MocapLandmarkReader.poll()'s
        # event-drain pattern), then stop -- non-blocking, so this never
        # waits for the next packet to arrive. A blocking-with-timeout recv
        # would do that: as long as some new packet keeps landing within the
        # timeout window (guaranteed at any real tracking rate faster than
        # ~20 Hz), the loop never times out and instead blocks until it has
        # collected a full 64, i.e. up to ~64 packet intervals of latency
        # every tick -- with the headset well past 60 Hz per hand, that is
        # hundreds of ms, not the sub-ms this should cost. decode_packet()
        # is cheap (a magic/size check plus a zero-copy numpy view); only
        # the *last* packet per hand is worth the flexion/rotation math
        # below, so that's deferred out of this loop.
        for _ in range(64):
            try:
                data, addr = self._sock.recvfrom(65536)
            except BlockingIOError:
                break
            except OSError as e:
                print(f"[quest_hand] socket error: {e}")
                break
            self.stats.packets_total += 1
            self.stats.last_addr = addr
            self.stats.last_packet_t = now
            d = decode_packet(data)
            if d is None:
                self.stats.packets_malformed += 1
                continue
            self.stats.packets_by_hand[d["hand"]] += 1
            self.stats.last_seen_t[d["hand"]] = now
            if not d["active"]:
                continue
            self.stats.active_by_hand[d["hand"]] += 1
            self.stats.last_active_t[d["hand"]] = now
            latest[d["hand"]] = d

        left: Optional[HandState] = None
        right: Optional[HandState] = None
        for hand, d in latest.items():
            tracked = [(d["tracked_mask"] >> i) & 1 for i in range(N_XR_JOINTS)]
            try:
                state = xr_hand_to_state(d["hand"], d["joints"], tracked, self._curl_range)
            except ValueError as e:
                print(f"[quest_hand] bad packet, skipping: {e}")
                continue
            if state is None:
                self.stats.untracked_by_hand[hand] += 1
                continue
            state.raw["seq"] = d["seq"]
            state.raw["t_device"] = d["t"]
            if hand == "left":
                left = state
            else:
                right = state

        self._log_status(now)
        return left, right

    def _log_status(self, now: float) -> None:
        s = self.stats
        if s.packets_total > 0 and not self._first_packet_logged:
            self._first_packet_logged = True
            print(f"[quest_hand] first packet received, from {s.last_addr[0]}:{s.last_addr[1]}")
        if now - self._last_log_t < self._log_interval:
            return
        self._last_log_t = now
        if s.packets_total == 0:
            print(
                f"[quest_hand] no packets received yet on {self._host}:{self._port} -- check: "
                "headset awake & app in foreground focus (not paused), kHandUdpHostIp/Port in "
                "main.cpp match this host, same LAN, and this PC's firewall allows inbound UDP "
                f"{self._port}"
            )
            return
        addr = f"{s.last_addr[0]}:{s.last_addr[1]}" if s.last_addr else "?"
        print(
            f"[quest_hand] total={s.packets_total} malformed={s.packets_malformed} "
            f"last_sender={addr} last_packet={now - s.last_packet_t:.1f}s ago | "
            f"left: seen={s.packets_by_hand['left']} active={s.active_by_hand['left']} "
            f"untracked={s.untracked_by_hand['left']} | "
            f"right: seen={s.packets_by_hand['right']} active={s.active_by_hand['right']} "
            f"untracked={s.untracked_by_hand['right']}"
        )

    def close(self) -> None:
        if self._sock is not None:
            self._sock.close()
            self._sock = None
