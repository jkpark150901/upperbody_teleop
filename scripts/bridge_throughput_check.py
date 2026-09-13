"""Raw throughput/latency check for senseglove_bridge.exe, completely
independent of hand_retarget_vedo.py / senseglove_monitor.py's own
rendering cost -- connects to the bridge's TCP socket directly and just
timestamps each JSON line as it arrives.

Use this to answer "is the bridge itself slow, or is it something
downstream (retargeting, vedo rendering)?" -- if this script also shows a
low lines/sec rate or long gaps, the bottleneck is the bridge/SGCore/glove
link itself, not any Python-side visualization. Run alongside
senseglove_bridge.exe (must already be listening -- see
senseglove_bridge/SETUP.md) with SenseCom + the glove(s) active:

    python scripts/bridge_throughput_check.py --seconds 10
"""

from __future__ import annotations

import argparse
import json
import socket
import time
from collections import Counter


def main():
    parser = argparse.ArgumentParser(description="Raw senseglove_bridge.exe throughput/latency check")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8850)
    parser.add_argument("--seconds", type=float, default=10.0)
    args = parser.parse_args()

    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    sock.settimeout(5.0)
    print(f"[bridge_throughput_check] connecting to {args.host}:{args.port} ...")
    sock.connect((args.host, args.port))
    sock.settimeout(1.0)
    print(f"[bridge_throughput_check] connected -- sampling for {args.seconds:.0f}s ...")

    buf = b""
    per_hand = Counter()
    gaps = []
    last_line_t = None
    start = time.perf_counter()
    n_lines = 0
    n_recv_calls = 0
    n_timeouts = 0

    while time.perf_counter() - start < args.seconds:
        try:
            chunk = sock.recv(65536)
            n_recv_calls += 1
            if not chunk:
                print("[bridge_throughput_check] bridge closed the connection")
                break
            buf += chunk
        except socket.timeout:
            n_timeouts += 1
            continue

        while b"\n" in buf:
            line, buf = buf.split(b"\n", 1)
            if not line.strip():
                continue
            now = time.perf_counter()
            if last_line_t is not None:
                gaps.append(now - last_line_t)
            last_line_t = now
            n_lines += 1
            try:
                d = json.loads(line)
                per_hand[d.get("hand", "?")] += 1
            except Exception as e:
                print(f"[bridge_throughput_check] bad line: {e}")

    elapsed = time.perf_counter() - start
    print()
    print(f"[bridge_throughput_check] elapsed={elapsed:.1f}s  total_lines={n_lines}  "
          f"avg_rate={n_lines / elapsed:.1f} lines/s  recv_calls={n_recv_calls}  recv_timeouts={n_timeouts}")
    print(f"[bridge_throughput_check] per-hand line counts: {dict(per_hand)}  "
          f"(expected ~{args.seconds * 60:.0f} each at the bridge's 60 Hz update loop, "
          f"less for a hand that reports DeviceConnected()==false sometimes)")
    if gaps:
        gaps.sort()
        p50 = gaps[len(gaps) // 2]
        p99 = gaps[int(len(gaps) * 0.99)]
        print(f"[bridge_throughput_check] inter-line gap: min={min(gaps)*1000:.1f}ms "
              f"p50={p50*1000:.1f}ms p99={p99*1000:.1f}ms max={max(gaps)*1000:.1f}ms")
        print("[bridge_throughput_check] a large max/p99 gap here (well above ~17ms = 1/60Hz) "
              "means the bridge/SGCore/glove link itself is stalling -- not a downstream "
              "rendering issue, since nothing but this socket read is happening in this script.")
    else:
        print("[bridge_throughput_check] no lines received at all -- check senseglove_bridge.exe's "
              "own console for DeviceConnected()==false / SenseCom status")

    sock.close()


if __name__ == "__main__":
    main()
