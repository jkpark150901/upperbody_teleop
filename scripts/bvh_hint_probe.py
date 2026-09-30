"""Live diagnostic for MocapLandmarkReader._bvh_joint_angle_hints().

Two modes:

--raw (recommended for re-deriving the axis mapping): prints the RightArm/
RightForeArm/RightHand (and Left*) bones' raw local Euler angles straight
from MocapApi, as a delta from the first frame seen, with NO index
reordering or sign flips applied. Do ONE isolated motion at a time (e.g.
right arm: pure abduction only, hold, release; then pure flexion only;
then pure internal/external rotation with elbow bent 90 as a lever) and
read off directly which of idx0/idx1/idx2 moves and in which direction for
each motion. That number is ground truth for fixing the index/sign mapping
in _bvh_joint_angle_hints() (scripts/upper_body_retarget_vedo.py) and
_BVH_DIRECT_JOINTS's consumer in scripts/upper_body_retarget_mujoco.py --
no more guessing from watching the mapped robot animation, which only
shows the *current* (possibly still-wrong) mapping's output.

--mapped (default): prints the resulting bvh_joint_angles hint dict
(current mapping applied) for a quick sanity check once the raw pass says
the mapping should be correct.

Usage:
  python -m scripts.bvh_hint_probe --port 7002 --raw
  python -m scripts.bvh_hint_probe --port 7002
"""
from __future__ import annotations

import argparse
import math
import time

from scripts.upper_body_retarget_vedo import MocapLandmarkReader, _BVH_ARM_ROTATION_NAMES


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--port", type=int, default=7002)
    parser.add_argument("--hz", type=float, default=2.0, help="print rate")
    parser.add_argument("--raw", action="store_true", help="print raw per-bone Euler deltas, unmapped")
    args = parser.parse_args()

    reader = MocapLandmarkReader(args.port)
    reader.connect()
    print(f"[bvh_hint_probe] connected on UDP {args.port}, waiting for frames...")

    baseline = {}  # bone name -> first-seen raw euler (radians), --raw only

    last_print = 0.0
    period = 1.0 / args.hz
    try:
        while True:
            now = time.monotonic()
            if now - last_print < period:
                time.sleep(0.005)
                continue

            if args.raw:
                # Reuse MocapLandmarkReader's own event drain + joint dict,
                # bypassing poll()'s UpperBodyLandmarks/hint construction so
                # a missing hand-tracking source etc. can't block this.
                newest = False
                for _ in range(64):
                    events = reader.app.poll_next_event()
                    if not events:
                        break
                    for event in events:
                        if event.event_type == reader.mcp.MCPEventType.AvatarUpdated:
                            reader.avatar_handle = event.event_data.avatar_handle
                            newest = True
                if not newest or reader.avatar_handle is None:
                    continue
                avatar = reader.mcp.MCPAvatar(reader.avatar_handle)
                joints = {j.get_name(): j for j in avatar.get_joints()}
                bone_names = [n for names in _BVH_ARM_ROTATION_NAMES.values() for n in names]
                if any(n not in joints for n in bone_names):
                    print(f"[bvh_hint_probe] missing bones; available={sorted(joints)}")
                    last_print = now
                    continue
                raw = {n: reader._local_euler_radians(joints[n]) for n in bone_names}
                if not baseline:
                    baseline.update({n: v.copy() for n, v in raw.items()})
                    print("[bvh_hint_probe] baseline captured from current pose -- hold pose, then do ONE isolated motion at a time")
                last_print = now
                parts = []
                for side in ("left", "right"):
                    upper_name, forearm_name, hand_name = _BVH_ARM_ROTATION_NAMES[side]
                    for label, name in (("upper", upper_name), ("forearm", forearm_name), ("hand", hand_name)):
                        delta = raw[name] - baseline[name]
                        deg = [math.degrees(v) for v in delta]
                        parts.append(f"{side}.{label}=[{deg[0]:+6.1f} {deg[1]:+6.1f} {deg[2]:+6.1f}]")
                print(" ".join(parts))
                continue

            landmarks = reader.poll()
            if landmarks is None:
                continue
            last_print = now
            hints = landmarks.bvh_joint_angles
            if not hints:
                print("[bvh_hint_probe] bvh_joint_angles is None -- LeftArm/LeftForeArm/"
                      "LeftHand/RightArm/RightForeArm/RightHand names not found in this stream")
                continue
            parts = []
            for side in ("left", "right"):
                vals = [hints[f"{side}_arm_{i}"] for i in range(7)]
                deg = [math.degrees(v) for v in vals]
                parts.append(
                    f"{side:>5}: " + " ".join(f"a{i}={d:+6.1f}" for i, d in enumerate(deg))
                )
            print(" | ".join(parts))
    except KeyboardInterrupt:
        pass
    finally:
        reader.close()


if __name__ == "__main__":
    main()
