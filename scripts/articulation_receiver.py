"""Reference receiver for teleop/articulation_protocol.py -- run on the remote
server (or locally to test the link) to see what upper_body_retarget_vedo.py's
--send is streaming. Also a template for applying the joint targets.

    python scripts/articulation_receiver.py --proto udp --port 9100
    python scripts/articulation_receiver.py --proto tcp --port 9100 --show left_arm_0 right_arm_3

Prints every 2 s: frame rate, sequence gaps (lost/dropped frames), and
one-way latency (t_recv - t_send, only meaningful if both machines' clocks
are synced), plus the requested joints' latest values in degrees.
"""

from __future__ import annotations

import argparse
import math
import pathlib
import socket
import sys
import time
from typing import Dict, List, Optional

_ROOT = pathlib.Path(__file__).parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from teleop.articulation_protocol import decode_message  # noqa: E402


class Stats:
    def __init__(self, show: List[str]):
        self.show = show
        self.names: Optional[List[str]] = None
        self.layout: Optional[str] = None
        self.frames = 0
        self.lost = 0
        self.last_seq: Optional[int] = None
        self.latency_sum = 0.0
        self.last_q: Optional[List[float]] = None
        self.last_hands: Dict[str, bool] = {}
        self.t_report = time.monotonic()

    def on_message(self, data: bytes) -> None:
        try:
            msg = decode_message(data)
        except (ValueError, KeyError) as e:
            print(f"[receiver] bad message: {e}")
            return
        if msg["type"] == "layout":
            if msg["layout"] != self.layout:
                self.names, self.layout = msg["names"], msg["layout"]
                print(f"[receiver] layout {self.layout}: {len(self.names)} joints ({msg['unit']})")
            return
        if msg["layout"] != self.layout:
            return  # frame for a layout we haven't been told about yet
        if self.last_seq is not None and msg["seq"] > self.last_seq + 1:
            self.lost += msg["seq"] - self.last_seq - 1
        self.last_seq = msg["seq"]
        self.frames += 1
        self.latency_sum += time.time() - msg["t_send"]
        self.last_q, self.last_hands = msg["q"], msg["hands"]

    def maybe_report(self) -> None:
        now = time.monotonic()
        if now - self.t_report < 2.0:
            return
        dt = now - self.t_report
        if self.frames and self.names and self.last_q:
            values = dict(zip(self.names, self.last_q))
            shown = "  ".join(f"{n}={math.degrees(values[n]):.1f}deg" for n in self.show if n in values)
            print(
                f"[receiver] {self.frames / dt:.1f} Hz  lost={self.lost}  "
                f"latency={1000 * self.latency_sum / self.frames:.1f} ms  hands={self.last_hands}  {shown}"
            )
        else:
            print("[receiver] no frames yet")
        self.frames, self.latency_sum, self.t_report = 0, 0.0, now


def run_udp(port: int, host: str, stats: Stats) -> None:
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    sock.bind((host, port))
    sock.settimeout(0.5)
    print(f"[receiver] listening udp://{host}:{port}")
    while True:
        try:
            data, _ = sock.recvfrom(65535)
            stats.on_message(data)
        except socket.timeout:
            pass
        stats.maybe_report()


def run_tcp(port: int, host: str, stats: Stats) -> None:
    server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    server.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    server.bind((host, port))
    server.listen(1)
    server.settimeout(0.5)
    print(f"[receiver] listening tcp://{host}:{port}")
    while True:
        try:
            conn, addr = server.accept()
        except socket.timeout:
            stats.maybe_report()
            continue
        print(f"[receiver] client connected from {addr[0]}:{addr[1]}")
        conn.settimeout(0.5)
        buf = b""
        stats.last_seq = None
        while True:
            try:
                chunk = conn.recv(65536)
                if not chunk:
                    print("[receiver] client disconnected")
                    break
                buf += chunk
            except socket.timeout:
                pass
            except OSError:
                break
            while b"\n" in buf:
                line, buf = buf.split(b"\n", 1)
                if line.strip():
                    stats.on_message(line)
            stats.maybe_report()
        conn.close()


def main():
    parser = argparse.ArgumentParser(description="Articulation stream receiver / link test")
    parser.add_argument("--proto", choices=("udp", "tcp"), default="udp")
    parser.add_argument("--host", default="0.0.0.0", help="bind address")
    parser.add_argument("--port", type=int, default=9100)
    parser.add_argument("--show", nargs="*", default=["left_arm_0", "left_arm_3", "right_arm_0", "right_arm_3"])
    args = parser.parse_args()
    stats = Stats(args.show)
    try:
        (run_udp if args.proto == "udp" else run_tcp)(args.port, args.host, stats)
    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    main()
