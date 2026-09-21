"""Joint-space ("articulation") wire format + UDP/TCP sender for streaming the
retargeted robot state to a remote server.

Unlike teleop/udp_protocol.py (human wrist/hand poses, retargeting done on the
receiver), this carries the *result* of retargeting: one radian value per
revolute joint of rby1_dg5f.urdf (torso, both 7-DOF arms, head, and both DG5F
hands' 20 finger joints each), so the receiver just applies joint targets.

Two message types, both JSON:

  layout -- {"v":1,"type":"layout","layout":"<crc32 hex>","names":[...],"unit":"rad"}
            the ordered joint-name list. Frames carry only values in that
            order (a full named dict per frame would run ~2 KB; the ordered
            list is ~0.5 KB and fits one UDP datagram comfortably).
  frame  -- {"v":1,"type":"frame","seq":N,"t_send":unix,"t_sample":unix,
             "layout":"<same id>","q":[...],"hands":{"left":bool,"right":bool}}
            "hands" says whether that side's finger values are live glove
            data (true) or the rest pose held at 0 (false).

Transport:
  udp://host:port -- one datagram per message. UDP is lossy and a receiver
      may start after the sender, so the layout is re-sent every
      LAYOUT_INTERVAL_S; a receiver ignores frames until it has seen a
      layout whose id matches the frame's.
  tcp://host:port -- newline-delimited JSON (same framing as the SenseGlove
      bridge). The layout is sent first on every (re)connect. This side is
      the TCP *client*; the remote server listens.

Sending never blocks the caller (the retargeting/render loop): send() only
swaps a "latest packet" slot that a background thread drains, so a slow or
dead link drops stale frames instead of adding latency or stalling the UI.
"""

from __future__ import annotations

import json
import socket
import threading
import time
import zlib
from typing import Dict, List, Optional, Sequence, Tuple
from urllib.parse import urlparse

PROTOCOL_VERSION = 1
LAYOUT_INTERVAL_S = 2.0
_TCP_CONNECT_TIMEOUT_S = 2.0
_TCP_RETRY_S = 1.0


def layout_id(names: Sequence[str]) -> str:
    return format(zlib.crc32(",".join(names).encode("utf-8")) & 0xFFFFFFFF, "08x")


def encode_layout(names: Sequence[str]) -> bytes:
    payload = {
        "v": PROTOCOL_VERSION, "type": "layout", "layout": layout_id(names),
        "names": list(names), "unit": "rad",
    }
    return json.dumps(payload, separators=(",", ":")).encode("utf-8")


def encode_frame(
    seq: int, t_sample: float, layout: str, q: Sequence[float], hands: Dict[str, bool],
) -> bytes:
    payload = {
        "v": PROTOCOL_VERSION, "type": "frame", "seq": seq,
        "t_send": time.time(), "t_sample": t_sample, "layout": layout,
        "q": [round(float(v), 5) for v in q],
        "hands": {"left": bool(hands.get("left")), "right": bool(hands.get("right"))},
    }
    return json.dumps(payload, separators=(",", ":")).encode("utf-8")


def decode_message(data: bytes) -> dict:
    d = json.loads(data.decode("utf-8"))
    if d.get("v") != PROTOCOL_VERSION:
        raise ValueError(f"unsupported protocol version: {d.get('v')}")
    if d.get("type") not in ("layout", "frame"):
        raise ValueError(f"unknown message type: {d.get('type')!r}")
    return d


def parse_endpoint(url: str) -> Tuple[str, str, int]:
    """'udp://host:port' / 'tcp://host:port' -> (scheme, host, port)."""
    parsed = urlparse(url)
    if parsed.scheme not in ("udp", "tcp") or not parsed.hostname or not parsed.port:
        raise ValueError(f"expected udp://host:port or tcp://host:port, got {url!r}")
    return parsed.scheme, parsed.hostname, parsed.port


class ArticulationSender:
    def __init__(self, url: str, names: Sequence[str]):
        self.scheme, self.host, self.port = parse_endpoint(url)
        self.names: List[str] = list(names)
        self.layout = layout_id(self.names)
        self._layout_bytes = encode_layout(self.names)
        self._seq = 0
        self._cond = threading.Condition()
        self._pending: Optional[bytes] = None
        self._closed = False
        self.sent = 0
        self.dropped = 0  # frames overwritten in the slot before the link could send them
        self.connected = self.scheme == "udp"  # UDP has no connection state to report
        self._thread = threading.Thread(target=self._run, name="articulation-sender", daemon=True)
        self._thread.start()

    def send(self, q: Sequence[float], t_sample: float, hands: Dict[str, bool]) -> None:
        if len(q) != len(self.names):
            raise ValueError(f"expected {len(self.names)} joint values, got {len(q)}")
        packet = encode_frame(self._seq, t_sample, self.layout, q, hands)
        self._seq += 1
        with self._cond:
            if self._pending is not None:
                self.dropped += 1
            self._pending = packet
            self._cond.notify()

    def close(self) -> None:
        with self._cond:
            self._closed = True
            self._cond.notify()
        self._thread.join(timeout=2.0)

    def _take(self) -> Optional[bytes]:
        with self._cond:
            while self._pending is None and not self._closed:
                self._cond.wait(timeout=0.5)
            packet, self._pending = self._pending, None
            return packet

    def _run(self) -> None:
        if self.scheme == "udp":
            self._run_udp()
        else:
            self._run_tcp()

    def _run_udp(self) -> None:
        sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        dest = (self.host, self.port)
        next_layout = 0.0
        while not self._closed:
            packet = self._take()
            if packet is None:
                continue
            try:
                now = time.monotonic()
                if now >= next_layout:
                    sock.sendto(self._layout_bytes, dest)
                    next_layout = now + LAYOUT_INTERVAL_S
                sock.sendto(packet, dest)
                self.sent += 1
            except OSError:
                # Windows raises here when an earlier datagram drew an ICMP
                # port-unreachable (nothing listening yet) -- harmless for a
                # fire-and-forget stream, keep sending.
                pass
        sock.close()

    def _run_tcp(self) -> None:
        sock: Optional[socket.socket] = None
        while not self._closed:
            if sock is None:
                try:
                    sock = socket.create_connection((self.host, self.port), timeout=_TCP_CONNECT_TIMEOUT_S)
                    sock.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
                    sock.sendall(self._layout_bytes + b"\n")
                    self.connected = True
                    print(f"[articulation] connected tcp://{self.host}:{self.port}")
                except OSError:
                    sock = None
                    self.connected = False
                    with self._cond:  # nothing to send to yet: discard, don't queue stale frames
                        self._pending = None
                    time.sleep(_TCP_RETRY_S)
                    continue
            packet = self._take()
            if packet is None:
                continue
            try:
                sock.sendall(packet + b"\n")
                self.sent += 1
            except OSError:
                print(f"[articulation] tcp://{self.host}:{self.port} lost, reconnecting")
                self.connected = False
                sock.close()
                sock = None
        if sock is not None:
            sock.close()
