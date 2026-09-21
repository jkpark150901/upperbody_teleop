# pn_stream_check.py

import math
import pathlib
import sys
import time
import ctypes

# Use the Python wrapper bundled with this repository.  Adding demo-py (rather
# than the package's inner directory) preserves the SDK's own import layout:
# ``from MocapApi.mocap_api import ...``.
_ROOT = pathlib.Path(__file__).resolve().parent
_MOCAP_PYTHON_SDK = _ROOT / "MocapApi" / "demo" / "demo-py"
if not _MOCAP_PYTHON_SDK.is_dir():
    raise RuntimeError(f"MocapApi Python SDK not found: {_MOCAP_PYTHON_SDK}")
if str(_MOCAP_PYTHON_SDK) not in sys.path:
    sys.path.insert(0, str(_MOCAP_PYTHON_SDK))

from MocapApi import mocap_api as MocapApi  # noqa: E402


# RBY1 teleop에 우선 필요한 joint
TARGET_JOINTS = {
    "Hips",
    "Spine",
    "Spine1",
    "Spine2",
    "Spine3",
    "Head",
    "LeftShoulder",
    "LeftArm",
    "LeftForeArm",
    "LeftHand",
    "RightShoulder",
    "RightArm",
    "RightForeArm",
    "RightHand",
}


def quat_to_rpy(x, y, z, w):
    """
    quaternion (x, y, z, w)
    -> roll, pitch, yaw [degree]
    """

    # roll (x)
    sinr_cosp = 2.0 * (w * x + y * z)
    cosr_cosp = 1.0 - 2.0 * (x * x + y * y)
    roll = math.atan2(sinr_cosp, cosr_cosp)

    # pitch (y)
    sinp = 2.0 * (w * y - z * x)
    if abs(sinp) >= 1:
        pitch = math.copysign(math.pi / 2, sinp)
    else:
        pitch = math.asin(sinp)

    # yaw (z)
    siny_cosp = 2.0 * (w * z + x * y)
    cosy_cosp = 1.0 - 2.0 * (y * y + z * z)
    yaw = math.atan2(siny_cosp, cosy_cosp)

    return (
        math.degrees(roll),
        math.degrees(pitch),
        math.degrees(yaw),
    )


def get_xyz(v):
    """MocapApi wrapper가 object 또는 tuple 어느 쪽이든 대응."""
    if hasattr(v, "x"):
        return float(v.x), float(v.y), float(v.z)

    return float(v[0]), float(v[1]), float(v[2])


def get_quat(q):
    """MocapApi quaternion -> x,y,z,w."""

    # wrapper가 q.x / q.y / q.z / q.w를 제공하는 경우
    if hasattr(q, "x"):
        return float(q.x), float(q.y), float(q.z), float(q.w)

    # 네 SDK에서 tuple 순서가 다르면 여기만 수정
    # 일단 (x, y, z, w)라고 가정
    # The bundled MocapApi Python wrapper returns tuples as (w, x, y, z).
    return float(q[1]), float(q[2]), float(q[3]), float(q[0])


def get_global_position(joint):
    """Read the animated world position; local BVH translations are constant bone offsets."""
    x, y, z = ctypes.c_float(), ctypes.c_float(), ctypes.c_float()
    err = joint.api.contents.GetJointGlobalPosition(
        ctypes.pointer(x), ctypes.pointer(y), ctypes.pointer(z), joint.handle
    )
    if err != MocapApi.MCPError.NoError:
        raise RuntimeError(f"GetJointGlobalPosition failed: {MocapApi.MCPError._fields[err]}")
    return x.value, y.value, z.value


def main():

    # --------------------------------------------------
    # MocapApi 설정
    # --------------------------------------------------

    settings = MocapApi.MCPSettings()

    # Axis Neuron Pro BVH UDP port와 동일하게
    settings.set_udp(7002)
    settings.set_bvh_data(MocapApi.MCPBvhData.Binary)
    settings.set_bvh_transformation(MocapApi.MCPBvhDisplacement.Enable)

    app = MocapApi.MCPApplication()
    app.set_settings(settings)

    # 네 기존 코드와 동일
    if hasattr(MocapApi, "MCPBvhRotation"):
        settings.set_bvh_rotation(MocapApi.MCPBvhRotation.YXZ)

    opened, message = app.open()
    if not opened:
        raise RuntimeError(f"Failed to open MocapApi on UDP 7002: {message}")
    app.disable_event_cache()

    print("Waiting for Axis Neuron Pro BVH stream on UDP 7002...")
    print("TARGET:", sorted(TARGET_JOINTS))
    print()

    avatar_handle = None
    last_wait_log = time.monotonic()

    try:
        while True:

            events = app.poll_next_event()

            for evt in events:

                if evt.event_type == MocapApi.MCPEventType.AvatarUpdated:
                    avatar_handle = evt.event_data.avatar_handle

            if avatar_handle is None:
                now = time.monotonic()
                if now - last_wait_log >= 1.0:
                    print("No AvatarUpdated event yet (still listening on UDP 7002)...")
                    last_wait_log = now
                time.sleep(0.01)
                continue

            avatar = MocapApi.MCPAvatar(avatar_handle)

            joints = avatar.get_joints()

            print("\033[2J\033[H", end="")  # 콘솔 화면 clear
            print("=== Perception Neuron Upper Body ===")
            print()

            for joint in joints:

                name = joint.get_name()

                if name not in TARGET_JOINTS:
                    continue

                pos = get_global_position(joint)
                rot = joint.get_local_rotation()

                px, py, pz = get_xyz(pos)

                qx, qy, qz, qw = get_quat(rot)

                roll, pitch, yaw = quat_to_rpy(
                    qx, qy, qz, qw
                )

                print(
                    f"{name:16s} "
                    f"XYZ = "
                    f"[{px:8.3f}, {py:8.3f}, {pz:8.3f}] "
                    f"RPY = "
                    f"[{roll:7.2f}, {pitch:7.2f}, {yaw:7.2f}] deg"
                )

            time.sleep(0.05)   # 약 20 Hz 출력

    except KeyboardInterrupt:
        print("\nStopping...")

    finally:
        app.close()


if __name__ == "__main__":
    main()
