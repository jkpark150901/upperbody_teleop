"""Device-PC entry point: PN + SenseGlove -> HumanTeleopState -> UDP.

Runs on the PC physically connected to Perception Neuron + SenseGlove (plan
section 5, the Windows PC). Streams raw human pose/hand data to the MuJoCo
PC at a fixed control rate over UDP; no calibration/retargeting happens
here (see mujoco_teleop/teleop_receiver_viewer.py + teleop/calibration.py --
recentering is a receiver-side, viewer-driven action).

PN and SenseGlove backends are selected independently (`--pn-backend`,
`--sg-backend`) since they'll likely need different real paths in
practice -- see devices/senseglove/sg_reader.py's module docstring for why
`bridge` (a local JSON-over-TCP helper written against the vendor's own
SGCore sample) is the realistic SenseGlove option, not `sgcore` directly.

Usage:
    python scripts/teleop_sender.py --pn-backend mock --sg-backend mock
    python scripts/teleop_sender.py --pn-backend mocapapi --sg-backend bridge --config configs/teleop.yaml
"""

from __future__ import annotations

import argparse
import pathlib
import socket
import sys
import time

_ROOT = pathlib.Path(__file__).parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import yaml  # noqa: E402

from devices.perception_neuron.pn_reader import MockPNReader, MocapApiPNReader  # noqa: E402
from devices.senseglove.sg_reader import (  # noqa: E402
    MockSenseGloveReader,
    SenseGloveJSONBridgeReader,
    SGCoreSenseGloveReader,
)
from teleop.teleop_state import TeleopAggregator  # noqa: E402
from teleop.udp_protocol import encode_state  # noqa: E402


def main():
    parser = argparse.ArgumentParser(description="PN + SenseGlove -> UDP teleop sender")
    parser.add_argument("--backend", choices=["mock", "real"], default=None,
                         help="shorthand for --pn-backend/--sg-backend mock, or mocapapi+bridge")
    parser.add_argument("--pn-backend", choices=["mock", "mocapapi"], default=None)
    parser.add_argument("--sg-backend", choices=["mock", "sgcore", "bridge"], default=None)
    parser.add_argument("--sg-bridge-host", default="127.0.0.1")
    parser.add_argument("--sg-bridge-port", type=int, default=8850)
    parser.add_argument("--config", default=str(_ROOT / "configs" / "teleop.yaml"))
    parser.add_argument("--target-ip", default=None, help="override configs/teleop.yaml network.target_ip")
    parser.add_argument("--target-port", type=int, default=None)
    parser.add_argument("--hz", type=float, default=None, help="override configs/teleop.yaml network.control_rate_hz")
    args = parser.parse_args()

    with open(args.config, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    target_ip = args.target_ip or cfg["network"]["target_ip"]
    target_port = args.target_port or cfg["network"]["target_port"]
    hz = args.hz or cfg.get("control_rate_hz", 90.0)

    pn_backend = args.pn_backend or ("mock" if args.backend != "real" else "mocapapi")
    sg_backend = args.sg_backend or ("mock" if args.backend != "real" else "bridge")

    pn_reader = MockPNReader() if pn_backend == "mock" else MocapApiPNReader(udp_port=cfg["pn"]["udp_port"])

    if sg_backend == "mock":
        sg_reader = MockSenseGloveReader()
    elif sg_backend == "sgcore":
        sg_reader = SGCoreSenseGloveReader()
    else:
        sg_reader = SenseGloveJSONBridgeReader(args.sg_bridge_host, args.sg_bridge_port)

    aggregator = TeleopAggregator(pn_reader, sg_reader)
    aggregator.connect()

    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    dest = (target_ip, target_port)
    print(
        f"[teleop_sender] pn={pn_backend} sg={sg_backend} -> "
        f"{target_ip}:{target_port} @ {hz:.0f} Hz"
    )

    period = 1.0 / hz
    seq = 0
    n_sent = 0
    last_stats_t = time.time()

    try:
        while True:
            loop_start = time.time()

            state = aggregator.poll()
            if state is not None:
                packet = encode_state(state, seq)
                sock.sendto(packet, dest)
                seq += 1
                n_sent += 1

            if time.time() - last_stats_t > 5.0:
                print(f"[teleop_sender] sent={n_sent} seq={seq}")
                last_stats_t = time.time()

            elapsed = time.time() - loop_start
            time.sleep(max(0.0, period - elapsed))
    except KeyboardInterrupt:
        pass
    finally:
        aggregator.close()
        sock.close()


if __name__ == "__main__":
    main()
