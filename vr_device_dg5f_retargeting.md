# VR Device Based DG5F Retargeting

## 1. 실행 예시

아래 명령은 프로젝트 루트(`D:\upperbody_teleop`)에서 실행한다. 현재 검증 기준 Python은 `drt` conda 환경이다.

```powershell
C:\Users\admin\miniforge3\envs\drt\python.exe scripts\quest_hand_vedo.py --backend quest --quest-host 192.168.8.156 --quest-port 5005 --hand both --render capsule --retargeter anydex
```

가장 많이 쓰게 될 Quest hand 단독 확인 명령이다. Raw OpenXR skeleton과 DG5F URDF retarget 결과를 동시에 보여준다. `--retargeter anydex`는 AnyDexRetarget의 `KeyVectorOptimizer` 기반 DG5F config를 사용한다.

```powershell
C:\Users\admin\miniforge3\envs\drt\python.exe scripts\quest_hand_vedo.py --backend quest --quest-host 192.168.8.156 --quest-port 5005 --hand both --render capsule --retargeter analytic
```

기존 joint-angle 기반 rule retargeter와 비교할 때 사용한다.

```powershell
C:\Users\admin\miniforge3\envs\drt\python.exe scripts\quest_hand_vedo.py --backend quest --quest-host 192.168.8.156 --quest-port 5005 --hand both --render capsule --retargeter anydex --record recordings\quest_anydex_test.jsonl
```

리타게팅 튜닝용 녹화. 프레임별 OpenXR 26 joints, flexion/spread, 중간 human angle, 최종 DG5F joint angle을 JSONL로 저장한다.

```powershell
C:\Users\admin\miniforge3\envs\drt\python.exe scripts\quest_hand_vedo.py --backend mock --hand both --render capsule --retargeter anydex
```

Quest 없이 synthetic hand로 AnyDex backend를 확인한다.

```powershell
C:\Users\admin\miniforge3\envs\drt\python.exe scripts\quest_hand_monitor.py --host 192.168.8.156 --port 5005
```

URDF/retargeting 없이 UDP 수신 상태만 확인한다. 패킷 수신, malformed packet, 좌우 hand update 상태를 볼 때 사용한다.

```powershell
C:\Users\admin\miniforge3\envs\drt\python.exe scripts\quest_hand_selfcheck.py
```

하드웨어 없이 Quest UDP packet encode/decode, synthetic stream, reader end-to-end를 확인한다.

```powershell
C:\Users\admin\miniforge3\envs\drt\python.exe scripts\upper_body_retarget_vedo.py --backend mocapapi --port 7002 --hand-backend quest --hand-retargeter anydex --quest-host 192.168.8.156 --quest-port 5005
```

Perception Neuron 상체 + Quest hand + DG5F hand를 함께 구동한다. Quest hand sample에 대해서만 AnyDex retargeter를 사용한다.

```powershell
C:\Users\admin\miniforge3\envs\drt\python.exe scripts\upper_body_retarget_vedo.py --backend mocapapi --port 7002 --hand-backend quest --hand-retargeter analytic --quest-host 192.168.8.156 --quest-port 5005
```

상체 viewer에서 기존 analytic hand retargeter와 비교할 때 사용한다.

```powershell
C:\Users\admin\miniforge3\envs\drt\python.exe scripts\upper_body_retarget_vedo.py --backend mocapapi --port 7002 --hand-backend quest --hand-retargeter anydex --quest-host 192.168.8.156 --quest-port 5005 --send udp://192.168.0.20:9100
```

상체 + 손 리타게팅 결과를 원격 서버로 joint angle stream 전송한다.

## 2. 현재 파이프라인

```text
Quest / XrHandsFB APK
    -> UDP
    -> devices.quest_hand.quest_hand_reader.QuestHandUDPReader
    -> HandState(raw["xr_joints"] = OpenXR 26 joints)
    -> retarget backend
       - analytic: robot_hand.hand_retarget.Dg5fHandRetargeter
       - anydex:   robot_hand.anydex_retarget.AnyDexDg5fRetargeter
    -> DG5F URDF joint angles
    -> vedo UI / upper body viewer / articulation sender
```

UDP 수신 자체는 이미 구현되어 있고, 리타게팅 쪽은 backend를 선택하는 구조다.

## 3. Retarget Backend

### analytic

`robot_hand/hand_retarget.py`의 기존 rule 기반 리타게팅이다.

- OpenXR 26 joints에서 palm-local frame을 만든다.
- 사람 손 joint angle feature를 계산한다.
- 첫 valid frame을 neutral/open 기준으로 잡는다.
- DG5F joint limit에 맞춰 clamp한다.
- 엄지와 abduction/adduction 축은 sign/gain 튜닝이 필요할 수 있다.

### anydex

`robot_hand/anydex_retarget.py` wrapper를 통해 AnyDexRetarget 알고리즘만 사용한다.

- 입력 UDP, Quest reader, UI는 기존 것을 그대로 쓴다.
- OpenXR 26 joints를 MediaPipe-style 21 keypoints로 변환한다.
- AnyDexRetarget `KeyVectorOptimizer`를 호출한다.
- AnyDex qpos를 DG5F URDF joint name dict로 변환한다.

DG5F용 AnyDex config:

```text
configs/anydex/dg5f_left_vector_quest3.yaml
configs/anydex/dg5f_right_vector_quest3.yaml
```

AnyDexRetarget source:

```text
third_party/AnyDexRetarget
```

Pinocchio/urdfdom은 현재 DG5F URDF의 custom `<capsule>` collision을 이해하지 못한다. 그래서 `robot_hand/anydex_retarget.py`가 collision tag를 제거한 Pinocchio용 URDF를 `outputs/anydex_urdf/`에 자동 생성해서 사용한다.

## 4. OpenXR Joint Label

UI에서는 수신 skeleton에 아래 label을 붙인다.

```text
W, PALM
T0 T1 T2 T3
I0 I1 I2 I3 I4
M0 M1 M2 M3 M4
R0 R1 R2 R3 R4
P0 P1 P2 P3 P4
```

OpenXR index 기준:

```text
0  PALM
1  WRIST
2  THUMB_METACARPAL
3  THUMB_PROXIMAL
4  THUMB_DISTAL
5  THUMB_TIP
6  INDEX_METACARPAL
7  INDEX_PROXIMAL
8  INDEX_INTERMEDIATE
9  INDEX_DISTAL
10 INDEX_TIP
11 MIDDLE_METACARPAL
12 MIDDLE_PROXIMAL
13 MIDDLE_INTERMEDIATE
14 MIDDLE_DISTAL
15 MIDDLE_TIP
16 RING_METACARPAL
17 RING_PROXIMAL
18 RING_INTERMEDIATE
19 RING_DISTAL
20 RING_TIP
21 LITTLE_METACARPAL
22 LITTLE_PROXIMAL
23 LITTLE_INTERMEDIATE
24 LITTLE_DISTAL
25 LITTLE_TIP
```

## 5. OpenXR to AnyDex Keypoint Mapping

AnyDexRetarget는 21개 MediaPipe-style hand keypoint를 기대한다. 현재 wrapper는 OpenXR 26개를 아래처럼 재배열한다.

```text
MP 0  wrist  <- OpenXR 1
MP 1  thumb0 <- OpenXR 2
MP 2  thumb1 <- OpenXR 3
MP 3  thumb2 <- OpenXR 4
MP 4  thumb3 <- OpenXR 5
MP 5  index0 <- OpenXR 7
MP 6  index1 <- OpenXR 8
MP 7  index2 <- OpenXR 9
MP 8  index3 <- OpenXR 10
MP 9  middle0 <- OpenXR 12
MP 10 middle1 <- OpenXR 13
MP 11 middle2 <- OpenXR 14
MP 12 middle3 <- OpenXR 15
MP 13 ring0 <- OpenXR 17
MP 14 ring1 <- OpenXR 18
MP 15 ring2 <- OpenXR 19
MP 16 ring3 <- OpenXR 20
MP 17 pinky0 <- OpenXR 22
MP 18 pinky1 <- OpenXR 23
MP 19 pinky2 <- OpenXR 24
MP 20 pinky3 <- OpenXR 25
```

OpenXR의 metacarpal point 중 long finger metacarpal은 AnyDex 21-keypoint 입력에서는 제외한다.

## 6. DG5F Mapping Rule

현재 사람이 지정한 기본 규칙:

```text
T1 -> dg_1_1, dg_1_2, dg_1_3
T2 -> dg_1_4

I1 -> dg_2_1, dg_2_2
I2 -> dg_2_3
I3 -> dg_2_4

M1 -> dg_3_1, dg_3_2
M2 -> dg_3_3
M3 -> dg_3_4

R1 -> dg_4_1, dg_4_2
R2 -> dg_4_3
R3 -> dg_4_4

P1 -> dg_5_1, dg_5_2
P2 -> dg_5_3
P3 -> dg_5_4
```

AnyDex config에서는 이 규칙을 직접 각도식으로 쓰지 않고, DG5F link 위치와 human keypoint vector를 맞추는 방식으로 최적화한다. 따라서 실제 튜닝 포인트는 `key_vectors[].scale`, target link, `mediapipe_rotation`이다.

## 7. 녹화

Quest hand 단독 UI에서 녹화:

```powershell
C:\Users\admin\miniforge3\envs\drt\python.exe scripts\quest_hand_vedo.py --backend quest --quest-host 192.168.8.156 --quest-port 5005 --hand both --render capsule --retargeter anydex --record recordings\quest_anydex_test.jsonl
```

저장 내용:

- `xr_joints`: OpenXR 26 joints
- `joint_positions`: compact hand joint positions
- `flexion`
- `spread`
- `human_angles`
- `dg5f_angles`

이 파일은 AnyDex config scale/rotation 튜닝 시 기준 데이터로 쓴다.

## 8. 의존성

기본 viewer/Quest 경로는 기존 requirements를 사용한다.

```powershell
C:\Users\admin\miniforge3\envs\drt\python.exe -m pip install -r requirements.txt
```

AnyDex backend만 따로 설치하려면 아래 파일을 사용한다.

```powershell
C:\Users\admin\miniforge3\envs\drt\python.exe -m pip install -r requirements-anydex.txt
```

Windows에서는 Pinocchio/NLopt를 conda-forge로 설치하는 쪽이 더 안정적이다.

```powershell
C:\Users\admin\miniforge3\Scripts\conda.exe install -n drt -c conda-forge pinocchio nlopt -y
```

현재 `drt` 환경에는 `nlopt` 설치를 완료했고, `pinocchio`, `yaml`, `scipy` import도 확인했다.

확인 명령:

```powershell
C:\Users\admin\miniforge3\envs\drt\python.exe -c "import nlopt, pinocchio, scipy, yaml; print('anydex deps ok')"
```

## 9. 튜닝 메모

- 첫 단계는 `scripts/quest_hand_vedo.py --retargeter anydex`로 손 단독 UI에서 확인한다.
- synthetic hand 기준으로는 일부 DG5F 축이 limit에 붙는 경향이 있다. 실제 Quest 손 녹화로 config tuning이 필요하다.
- 우선 조정할 값은 `configs/anydex/dg5f_*_vector_quest3.yaml`의 `key_vectors[].scale`이다.
- 전체 손 방향이 틀어지면 `retarget.mediapipe_rotation`을 조정한다.
- 특정 finger segment가 이상하면 해당 `task` link를 `_2`, `_3`, `_tip` 중 어느 것으로 잡는지 확인한다.
- analytic backend는 비교 기준으로 유지한다.
