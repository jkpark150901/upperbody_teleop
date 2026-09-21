"""Probe Axis Neuron UDP packets without MocapApi, GUI, or retargeting."""

from __future__ import annotations

import argparse
import socket
import time
import zlib


def main():
    parser = argparse.ArgumentParser(description="Raw UDP packet probe for Axis Neuron")
    parser.add_argument("--port", type=int, default=7002)
    parser.add_argument("--bind", default="0.0.0.0")
    args = parser.parse_args()

    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    sock.setsockopt(socket.SOL_SOCKET, socket.SO_RCVBUF, 4 * 1024 * 1024)
    sock.bind((args.bind, args.port))
    sock.settimeout(0.25)
    print(f"[udp_probe] listening on {args.bind}:{args.port} (Ctrl+C to stop)")

    report_at = time.perf_counter()
    first_packet_at = None
    packets = byte_count = changed = 0
    last_crc = None
    last_sender = None
    sizes = set()
    try:
        while True:
            try:
                payload, sender = sock.recvfrom(65535)
                now = time.perf_counter()
                if first_packet_at is None:
                    first_packet_at = now
                    print(f"[udp_probe] FIRST PACKET from {sender[0]}:{sender[1]}, {len(payload)} bytes")
                crc = zlib.crc32(payload)
                if last_crc is not None and crc != last_crc:
                    changed += 1
                last_crc, last_sender = crc, sender
                packets += 1
                byte_count += len(payload)
                sizes.add(len(payload))
            except socket.timeout:
                now = time.perf_counter()

            if now - report_at >= 1.0:
                dt = now - report_at
                if packets:
                    print(
                        f"[udp_probe] {packets / dt:6.1f} pkt/s | {byte_count / dt / 1024:7.1f} KiB/s | "
                        f"payload changed {changed}/{packets} | sizes={sorted(sizes)} | from={last_sender}"
                    )
                else:
                    print(f"[udp_probe] NO PACKETS on UDP {args.port}")
                report_at = now
                packets = byte_count = changed = 0
                sizes.clear()
    except KeyboardInterrupt:
        print("\n[udp_probe] stopped")
    finally:
        sock.close()


if __name__ == "__main__":
    main()
