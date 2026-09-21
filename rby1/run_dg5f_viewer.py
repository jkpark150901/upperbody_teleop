"""RB-Y1(rby1ub) + 양손 DG5F 결합 모델을 인터랙티브 MuJoCo 뷰어로 연다.

순수 자세 확인용이라 물리 스텝은 안 돌린다 (mj_forward만) -- rby1ub_src의
torso_base가 지면에서 겨우 0.3m 위라 DG5F를 편 채로(qpos=0) 물리를 돌리면
손가락이 바닥에 ~10cm 파묻혀서 즉시 발산한다. 받침대에 올리는 건 실제
장면(테이블 등)을 만들 때 처리할 별도 작업.
"""
import os.path as osp
import time

import mujoco
import mujoco.viewer

SCRIPT_DIR = osp.dirname(osp.realpath(__file__))
MODEL_PATH = osp.join(SCRIPT_DIR, "rby1_dg5f.xml")


def main():
    model = mujoco.MjModel.from_xml_path(MODEL_PATH)
    data = mujoco.MjData(model)
    mujoco.mj_resetDataKeyframe(model, data, 0)
    mujoco.mj_forward(model, data)

    with mujoco.viewer.launch_passive(model, data) as viewer:
        while viewer.is_running():
            viewer.sync()
            time.sleep(0.03)


if __name__ == "__main__":
    main()
