"""RB-Y1(rby1ub, Standard Upper-Body)의 스톡 평행 그리퍼를 제거하고
양쪽 손목에 DG5F를 결합한다.

Talos와 같은 패턴(build_dg5f_talos.py): DG5F의 mount->palm 로컬축은 +Z이고,
RB-Y1 스톡 그리퍼(EE_BODY_*)는 arm_6 로컬 -Z으로 달려있어(자체 quat 없음
= identity, pos만 -Z 방향) Rx(180도)로 정렬한다. 마운트 위치는 스톡
그리퍼의 EE_BODY 위치(0,0,-0.1548)를 그대로 재사용한다.
"""
import math
import os.path as osp

import mujoco
import numpy as np

SCRIPT_DIR = osp.dirname(osp.realpath(__file__))
DG5F_DIR = osp.join(osp.dirname(SCRIPT_DIR), "dg5f")
SRC_XML = osp.join(SCRIPT_DIR, "rby1ub_src", "mujoco", "model_act.xml")

MOUNT_POS = [0, 0, -0.1548]
# DG5F +Z(mount->palm) 를 arm_6의 -Z(스톡 그리퍼가 뻗는 방향)에 맞추는 Rx(180도).
_c, _s = math.cos(math.pi / 2), math.sin(math.pi / 2)
MOUNT_QUAT = [_c, _s, 0, 0]

STOCK_EE_BODIES = {
    "right": ["FT_SENSOR_R", "EE_BODY_R", "ee_finger_r1", "ee_finger_r2"],
    "left": ["FT_SENSOR_L", "EE_BODY_L", "ee_finger_l1", "ee_finger_l2"],
}
STOCK_EQUALITY = ["right_finger", "left_finger"]
ARM_MOUNT_BODY = {"right": "link_right_arm_6", "left": "link_left_arm_6"}


def main():
    spec = mujoco.MjSpec.from_file(SRC_XML)

    # 손가락 equality(gripper_finger_r1/r2, l1/l2 커플링)를 먼저 지운다 --
    # 몸체보다 나중에 지우면 참조가 끊긴 채 남아 컴파일 에러가 난다
    # (Talos 작업에서 겪은 것과 같은 순서 문제).
    for name in STOCK_EQUALITY:
        eq = spec.equality(name)
        if eq is not None:
            spec.delete(eq)

    for side in ("right", "left"):
        for body_name in STOCK_EE_BODIES[side]:
            body = spec.body(body_name)
            if body is not None:
                spec.delete(body)

    mounts = {}
    for side in ("right", "left"):
        wrist = spec.body(ARM_MOUNT_BODY[side])
        mounts[side] = wrist.add_site(
            name=f"{side}_hand_mount", pos=MOUNT_POS, quat=MOUNT_QUAT
        )

    dg5f = {
        "right": mujoco.MjSpec.from_file(osp.join(DG5F_DIR, "dg5f_right.urdf")),
        "left": mujoco.MjSpec.from_file(osp.join(DG5F_DIR, "dg5f_left.urdf")),
    }
    # DG5F URDF에는 <transmission>이 전혀 없다 -- 힌지 관절마다 새 position
    # actuator를 만든다 (tip 관절은 고정이라 hinge가 아니라 제외됨).
    for side_spec in dg5f.values():
        for joint in side_spec.joints:
            if joint.type != mujoco.mjtJoint.mjJNT_HINGE:
                continue
            act = side_spec.add_actuator(name=joint.name, target=joint.name,
                                          trntype=mujoco.mjtTrn.mjTRN_JOINT)
            act.set_to_position(kp=5, kv=0.5)

    for side in ("right", "left"):
        spec.attach(dg5f[side], prefix=f"{side}_hand_", site=mounts[side])

    model = spec.compile()
    print(f"Compiled OK. nq={model.nq} nbody={model.nbody} nu={model.nu}")

    # 스톡 모델에 키프레임이 없어서(순수 팔+손, 다리/균형 문제 없음) 그냥
    # qpos=0, ctrl=0을 rest pose로 저장한다.
    key = spec.add_key(name="home")
    key.qpos = np.zeros(model.nq)
    key.ctrl = np.zeros(model.nu)
    spec.compile()

    out_xml = spec.to_xml()
    out_xml = out_xml.replace('file="meshes/dg5f', 'file="../dg5f/meshes/dg5f')
    out_xml = out_xml.replace('file="dg5f_right.urdf.', 'file="../dg5f/dg5f_right.urdf.')
    out_xml = out_xml.replace('file="dg5f_left.urdf.', 'file="../dg5f/dg5f_left.urdf.')
    out_path = osp.join(SCRIPT_DIR, "rby1_dg5f.xml")
    with open(out_path, "w") as f:
        f.write(out_xml)
    print(f"Saved {out_path}")


if __name__ == "__main__":
    main()
