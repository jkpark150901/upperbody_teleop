"""RB-Y1(rby1ub) 원본 URDF(rby1ub_src/urdf/model.urdf)에서 직접 스톡
평행 그리퍼를 떼어내고 DG5F 손 URDF를 붙인다.

MJCF를 거치지 않는다 -- rby1_dg5f.xml(MuJoCo)에서 URDF로 역변환한 이전
버전과 달리, Rainbow Robotics가 배포한 원본 URDF와 DG5F 원본 URDF를 XML
레벨에서 직접 합친다.

마운트: 원본의 tool_right/FT_Sensor_END_right 체인(arm_6 -> -0.1087 ->
-0.0461 = 총 -0.1548, identity rotation)과 정확히 같은 위치에 DG5F를
붙인다. DG5F의 mount->palm 로컬 축은 +Z, RB-Y1 손목의 그리퍼 부착 방향은
identity rotation의 -Z이므로 Rx(180도)로 정렬한다 (rby1_dg5f.xml 만들 때
렌더로 검증된 것과 동일).
"""
from __future__ import annotations

import os.path as osp
import xml.etree.ElementTree as ET

SCRIPT_DIR = osp.dirname(osp.realpath(__file__))
SRC_URDF = osp.join(SCRIPT_DIR, "rby1ub_src", "urdf", "model.urdf")
DG5F_DIR = osp.join(osp.dirname(SCRIPT_DIR), "dg5f")
OUT_URDF = osp.join(SCRIPT_DIR, "rby1_dg5f.urdf")

MOUNT_XYZ = "0.0 0.0 -0.1548"
MOUNT_RPY = "3.14159265358979 0.0 0.0"  # Rx(180deg)

# 스톡 그리퍼 체인 -- arm_6 이후로 전부 제거하고 그 자리에 DG5F를 붙인다.
STOCK_JOINTS = {
    "right": ["tool_right", "FT_Sensor_END_right", "gripper_finger_r1", "gripper_finger_r2"],
    "left": ["tool_left", "FT_Sensor_END_left", "gripper_finger_l1", "gripper_finger_l2"],
}
STOCK_LINKS = {
    "right": ["FT_sensor_R", "ee_right", "ee_finger_r1", "ee_finger_r2"],
    "left": ["FT_sensor_L", "ee_left", "ee_finger_l1", "ee_finger_l2"],
}
ARM_MOUNT_LINK = {"right": "link_right_arm_6", "left": "link_left_arm_6"}
DG5F_SRC = {"right": osp.join(DG5F_DIR, "dg5f_right.urdf"), "left": osp.join(DG5F_DIR, "dg5f_left.urdf")}
DG5F_ROOT_LINK = {"right": "rl_dg_mount", "left": "ll_dg_mount"}
DG5F_PREFIX = {"right": "right_hand_", "left": "left_hand_"}


def prefix_mesh_paths(elem, dg5f_own_dir_rel):
    for mesh in elem.iter("mesh"):
        filename = mesh.get("filename")
        if filename is not None:
            mesh.set("filename", osp.join(dg5f_own_dir_rel, filename))


def import_dg5f(robot, side):
    dg5f = ET.parse(DG5F_SRC[side]).getroot()
    prefix = DG5F_PREFIX[side]
    dg5f_dir_rel = osp.relpath(DG5F_DIR, SCRIPT_DIR)

    for link in dg5f.findall("link"):
        link.set("name", prefix + link.get("name"))
        prefix_mesh_paths(link, dg5f_dir_rel)
        robot.append(link)

    for joint in dg5f.findall("joint"):
        joint.set("name", prefix + joint.get("name"))
        for tag in ("parent", "child"):
            ref = joint.find(tag)
            ref.set("link", prefix + ref.get("link"))
        robot.append(joint)

    # arm_6 -> DG5F 루트(mount)로 잇는 새 fixed joint.
    mount_joint = ET.SubElement(robot, "joint", name=f"{side}_hand_mount", type="fixed")
    ET.SubElement(mount_joint, "parent", link=ARM_MOUNT_LINK[side])
    ET.SubElement(mount_joint, "child", link=prefix + DG5F_ROOT_LINK[side])
    ET.SubElement(mount_joint, "origin", xyz=MOUNT_XYZ, rpy=MOUNT_RPY)
    ET.SubElement(mount_joint, "axis", xyz="0 0 1")


def main():
    tree = ET.parse(SRC_URDF)
    robot = tree.getroot()

    # 원본 파일의 <mujoco><compiler meshdir="./meshes"/>가 이미 "./meshes/"로
    # 시작하는 mesh filename에 또 붙어서 MuJoCo 로더가 "meshes/meshes/..."를
    # 찾다가 실패한다 (원본 파일 자체에 있던 문제, Pinocchio는 <mujoco> 태그를
    # 무시해서 안 겪음). meshdir을 지워서 mesh filename을 그대로 쓰게 한다.
    compiler = robot.find("mujoco/compiler")
    if compiler is not None and "meshdir" in compiler.attrib:
        del compiler.attrib["meshdir"]

    # RB-Y1 자신의 mesh 경로("./meshes/...")는 원본 위치(rby1ub_src/urdf/)
    # 기준이라, 출력 파일 위치(이 폴더) 기준으로 다시 앵커링해야 한다.
    rby1_mesh_dir_rel = osp.relpath(osp.join(SCRIPT_DIR, "rby1ub_src", "urdf", "meshes"), SCRIPT_DIR)
    for mesh in robot.iter("mesh"):
        filename = mesh.get("filename")
        if filename is not None and filename.startswith("./meshes/"):
            mesh.set("filename", osp.join(rby1_mesh_dir_rel, filename[len("./meshes/"):]))

    for side in ("right", "left"):
        for joint_name in STOCK_JOINTS[side]:
            joint = next((j for j in robot.findall("joint") if j.get("name") == joint_name), None)
            if joint is not None:
                robot.remove(joint)
        for link_name in STOCK_LINKS[side]:
            link = next((l for l in robot.findall("link") if l.get("name") == link_name), None)
            if link is not None:
                robot.remove(link)
        import_dg5f(robot, side)

    ET.indent(robot, space="  ")
    tree.write(OUT_URDF, encoding="utf-8", xml_declaration=True)
    n_links = len(robot.findall("link"))
    n_joints = len(robot.findall("joint"))
    print(f"Saved {OUT_URDF}  (links={n_links}, joints={n_joints})")


if __name__ == "__main__":
    main()
