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

## 11. MuJoCo로 URDF 제어 (2026-09-29)

같은 리타게팅 파이프라인을 vedo 대신 MuJoCo에 꽂았다. 물리 시뮬레이션(중력/충돌/액추에이터)은 아직 안 쓴다 -- 매 틱마다 리타게팅 결과를 `qpos`에 직접 쓰고 `mj_forward()`(kinematics만)만 호출하는 kinematic puppet이다. `mj_step()` 기반 실제 액추에이터 제어는 이게 검증된 다음 단계다.

```powershell
python scripts\upper_body_retarget_mujoco.py --backend mock
python scripts\upper_body_retarget_mujoco.py --backend quest --hand-backend quest --quest-host 192.168.8.156 --quest-port 5005 --hand-retargeter anydex
python scripts\upper_body_retarget_mujoco.py --backend mocapapi --port 7002 --hand-backend quest --hand-retargeter anydex --quest-host 192.168.8.156 --quest-port 5005
```

### URDF 메시 변환

MuJoCo URDF 컴파일러는 Collada(`.dae`)를 못 읽는다 (`rby1_dg5f.urdf`를 바로 로드하면 "no decoder found for mesh file ...dae"). `robot_hand/mujoco_urdf.py`가 `robot_hand/anydex_retarget.py`의 `_pinocchio_compatible_urdf`와 같은 패턴으로, `.dae`는 trimesh로 `.obj` 변환하고 이미 `.STL`인 collision 메시는 그대로 두되 전부 절대경로로 재작성한 URDF 사본을 `outputs/mujoco_urdf/`에 생성한다 (소스 mtime 기준 캐시, 매번 재생성 안 함). 첫 로드 확인: `nq=58, njnt=58, nbody=59`.

### 헤드/카메라 고정

실제 로봇은 머리 자유도가 없고 카메라가 몸체에 고정된다. 그래서:
- `head_0`/`head_1`은 리타게팅 결과를 아예 MuJoCo에 쓰지 않는다(`MujocoJointWriter(skip=_HEAD_JOINTS)`) -- URDF의 qpos0 그대로 고정.
- `mujoco.viewer`의 카메라도 매 프레임 갱신하지 않고 시작할 때 한 번만 고정 시점으로 세팅한다(`viewer.cam.lookat/distance/azimuth/elevation`).
- 아직 안 한 것: 로봇 몸체에 고정 부착된 `<camera>`(1인칭/에고센트릭 렌더링용)는 URDF에 없다. 필요하면 `mujoco_urdf.py`가 생성하는 사본에 `<camera>` 태그를 추가하는 식으로 넣을 수 있다.
- MuJoCo 뷰어가 `<collision>`(단순 STL/캡슐)과 `<visual>`(디테일한 obj) geom을 동시에 렌더링해서 손이 "단순화"돼 보였던 문제: URDF 임포트가 둘을 다른 geom group(콜리전=0, 비주얼=1)으로 나눴는데 뷰어 기본값이 둘 다 켜져 있었다. `upper_body_retarget_mujoco.py`가 시작할 때 콜리전 전용 그룹(모든 geom이 `contype`/`conaffinity` != 0인 그룹)을 렌더링에서만 끄도록 고쳤다 -- 물리 충돌 계산엔 영향 없음.

## 12. 좌표축 정렬 버그 수정 (2026-09-29)

`--backend quest`의 손목 IK 델타 변환(`scripts/upper_body_retarget_vedo.py`, `scripts/upper_body_retarget_mujoco.py`)에서 **X/Y축이 뒤바뀐 버그**가 있었다.

로봇 축(검증됨, 추측 아님): `rby1_dg5f.urdf`의 `base`→`torso_hp`→`torso_5`→`head_base`→`head_0`→`head_1` 체인 전체가 `origin rpy="0 0 0"`이라 head 프레임이 base와 원점에서 완전히 같은 축을 공유한다. `right_arm_0`의 origin이 `y=-0.220`인 것도 확인 -- **X=정면, Y=왼쪽, Z=위**.

OpenXR LOCAL space는 X=오른쪽, Y=위, Z=뒤(정면=-Z). 올바른 매핑:

```text
robot.x (정면) = -quest.z
robot.y (왼쪽) = -quest.x
robot.z (위)   =  quest.y
```

수정 전엔 `[quest.x, -quest.z, quest.y]`로 X/Y가 사실상 뒤섞여 있었다. IK 수렴 재검증(warm-start incremental, `robot_hand/upper_body_retarget.py`의 `solve_wrist`):

```text
수정 전: 9.2cm 델타에도 잔차 71mm (거의 못 따라감)
수정 후: 26.2cm 델타(전방 25cm+상승 8cm)에서 잔차 18mm
```

이전에 "스케일 문제"로 봤던 떨림의 상당 부분이 실은 이 축 버그였을 가능성이 높다. `--wrist-scale`은 여전히 유효한 보조 수단으로 남겨뒀다.

## 13. 손가락 접촉(핀치) 미구현 원인 (2026-09-29)

`--retargeter anydex`가 쓰는 `KeyVectorOptimizer`(`configs/anydex/dg5f_*_vector_quest3.yaml`)는 **손가락 간 접촉을 전혀 모른다** -- 각 손가락의 (origin→task) 벡터를 사람 쪽 대응 벡터와 독립적으로만 맞추는 단순 vector-matching이라, 엄지와 검지가 실제로 맞닿아도 로봇 쪽에서 떨어져 있을 수 있다.

`third_party/AnyDexRetarget`의 다른 옵티마이저 `AdaptiveOptimizerAnalytical`에는 정확히 이 기능이 있다 (`analytical_optimizer.py`):
- `_compute_pinch_alpha()`: 사람 keypoint에서 엄지-각 손가락 tip 거리로 핀치 감지 (`pinch_thresholds.d1/d2`).
- 핀치 감지되면 엄지+활성 손가락의 목표 벡터를 "균일 스케일된 wrist→tip 벡터"로 블렌딩해서 opposition contact가 손가락별 독립 스케일링 때문에 깨지지 않게 함.
- `w_contact`: 엄지-활성 손가락 tip 간 거리를 직접 페널티로 넣는 contact loss.

`third_party/AnyDexRetarget/example/config/adaptive/quest3/quest3_gaia_hand20.yaml`에 이미 튜닝된 레퍼런스 값이 있어서(같은 gaia_hand20, 5절/9절과 같은 차용 방식), `configs/anydex/dg5f_{left,right}_adaptive_quest3.yaml`을 새로 만들었다. `robot_hand/anydex_retarget.py`는 코드 변경 없이 `--anydex-config-left/right`로 이미 임의 config를 받을 수 있어서 그대로 씀:

```powershell
python scripts\quest_hand_vedo.py --backend quest --quest-host 192.168.8.156 --quest-port 5005 --hand both --render capsule --retargeter anydex --anydex-config-left configs\anydex\dg5f_left_adaptive_quest3.yaml --anydex-config-right configs\anydex\dg5f_right_adaptive_quest3.yaml
```

drt 환경에서 로드 + `retarget_openxr_joints` 호출까지 확인함 (synthetic curl 0.0/0.5/1.0 세 단계에서 크래시 없이 손가락 관절각이 입력에 따라 달라지는 것 확인). 실제 핀치 동작 품질은 아직 실기 검증 안 됨 -- `mediapipe_rotation`(right-hand 기준, left는 미검증)과 `pinch_thresholds`/`w_contact`가 1차 튜닝 포인트.
- **2026-09-29**: `configs/anydex/dg5f_{left,right}_vector_quest3.yaml`의 scale/
  `mediapipe_rotation`을 처음부터 새로 튜닝하는 대신, 구조가 거의 동일한
  `third_party/AnyDexRetarget/example/config/vector/quest3/quest3_gaia_hand20.yaml`
  (5지, `_3`/`_4`/`tip` 세그먼트, task_kp 배선이 완전히 동일, DOF도 20개로 일치)의
  값을 그대로 가져와 시작점으로 삼았다. `right`는 gaia 쪽도 right 기준이라 그대로
  썼고, `left`는 **미검증**이다(같은 숫자를 그대로 넣었을 뿐, 좌우 반전 시 축
  부호가 뒤집혀야 할 수도 있음) -- 실제 왼손 녹화로 반드시 재확인할 것.

## 10. `configs/anydex/dg5f_*_vector_quest3.yaml` 작성 규칙

`optimizer.type: "KeyVectorOptimizer"`를 쓸 때의 스키마다. 출처:
`third_party/AnyDexRetarget/anydexretarget/optimizer/base_optimizer.py`
(`BaseOptimizer.__init__`, `from_config`)와
`third_party/AnyDexRetarget/anydexretarget/optimizer/key_vector_optimizer.py`
(`KeyVectorOptimizer.__init__`, `_compute_target_vectors`, `_loss_and_grad`).

### 최상위 구조

```yaml
optimizer:
  type: "KeyVectorOptimizer"      # 이거 아니면 AdaptiveOptimizerAnalytical (다른 스키마)

robot:
  type: "dg5f"                    # 임의 문자열 -- ROBOT_CONFIGS에 없으면 기본값(shadow_hand)으로
                                   # fallback하지만, 아래 필드를 전부 명시하면 그 fallback은 안 씀
  urdf_path: "robot_hand/urdf/dg5f_left.urdf"   # repo root 기준 상대경로
  origin_link: "..."              # 손목/팔목 기준 링크 (모든 key_vector의 origin 후보)
  tip_links: [5개]                # 손가락 tip 링크, MediaPipe TIP과 매칭될 후보
  link1_names: [5개]              # 손가락 root(첫 관절) 링크 -- finger_root_vectors 계산에 쓰임
  link3_names: [5개]              # 중간 링크 (Adaptive optimizer용이지만 Base에서 항상 파싱)
  link4_names: [5개]              # distal 링크
  num_fingers: 5                  # 4면 새끼손가락 제외(Allegro/Leap류)
  neutral_qpos: [0.0, ... x N]    # URDF의 전체 DOF 개수(mimic 포함)와 길이 일치해야 함.
                                   # 초기값이자 norm_delta 정규화의 기준 자세
  tip_offsets / link3_offsets / link4_offsets:   # 선택. (3,) 또는 (N,3), 링크 로컬좌표계 offset(m)

retarget:
  huber_delta: 2.0     # cm 단위 residual에 대한 Huber loss delta
  norm_delta: 0.04     # 이전 프레임 대비 관절속도 페널티 가중치 (튐 억제)
  lp_alpha: 0.35        # 출력 qpos에 적용하는 저역통과 필터 alpha (0=필터없음~1=매우느림)
  clamp_joint_lower:    # 선택. {"이름에 포함된 substring": 최소값} -- 로봇 URDF의
    dg_1_2: -0.1        #   기본 lower limit보다 더 빡빡하게 잡고 싶을 때만 씀
  mediapipe_rotation:   # 손 전체를 사람 좌표계에서 로봇 좌표계로 미리 돌릴 때 (extrinsic XYZ, deg)
    x: 0.0
    y: 0.0
    z: 0.0
    # 좌/우 다르게 주고 싶으면: {default: {...}, left: {...}, right: {...}} 형태도 가능
    # (retarget.py:_resolve_hand_side_config)

  key_vectors:          # KeyVectorOptimizer는 이게 없으면 즉시 ValueError
    - origin: <로봇 링크 이름>
      task: <로봇 링크 이름>
      origin_kp: <0~20 정수>   # MediaPipe-style keypoint index (아래 표)
      task_kp: <0~20 정수>
      scale: 1.0               # 사람 벡터에 곱하는 배율. 기본 1.0
      origin_offset: [x,y,z]   # 선택, origin 링크 로컬좌표계 offset(m). 기본 [0,0,0]
      task_offset: [x,y,z]     # 선택, task 링크 로컬좌표계 offset(m)
```

### 각 `key_vectors` 행이 뭘 하는지

한 줄 = "로봇의 (origin→task) 벡터가 사람의 (origin_kp→task_kp) 벡터를 `scale`배 한 것과
같아지도록" 만드는 제약 하나다. 전체 loss는:

```
L(q) = mean_i( Huber(‖ FK(task_i)-FK(origin_i) - scale_i*(mp[task_kp_i]-mp[origin_kp_i]) ‖) )
       + norm_delta * ‖q - q_prev‖²
```

`origin`은 보통 전부 palm(`0`,`origin_kp: 0`)으로 고정해두고, `task`/`task_kp`만 바꿔가며
각 손가락 세그먼트마다 한 줄씩 추가하는 식으로 쓴다 (현재 파일이 그 패턴).

### MediaPipe-style keypoint 인덱스 (`origin_kp`/`task_kp`)

`BaseOptimizer`가 정의한 고정 인덱스:

```text
0            wrist
1,5,9,13,17  각 손가락 MCP(=손가락 뿌리, 엄지는 CMC)   -- MP_MCP_INDICES
2,6,10,14,18 각 손가락 PIP(엄지는 MCP)               -- MP_PIP_INDICES
3,7,11,15,19 각 손가락 DIP                            -- MP_DIP_INDICES
4,8,12,16,20 각 손가락 TIP                            -- MP_TIP_INDICES
             (순서: 엄지, 검지, 중지, 약지, 소지)
```

우리 wrapper(`robot_hand/anydex_retarget.py:_OPENXR_TO_MEDIAPIPE_21`)가 OpenXR 26 joints를
이 21개 배열로 미리 재배열해서 넘기므로, YAML에서는 OpenXR 인덱스가 아니라 **항상 이
0~20 MediaPipe 인덱스**를 써야 한다.

### 흔한 함정

- **`dg5f`는 좌/우 자동 치환 리스트에 없다.** `KeyVectorOptimizer`는 `shadow_hand`
  (`rh_`→`lh_`), `unitree_dex5_hand`, `linker_l20`/`sharpa_hand`/`gaia_hand20`/
  `inspire_hand`(`right_`→`left_`)만 자동으로 좌/우 링크 이름을 바꿔준다. `dg5f`는
  해당 안 되므로 **`dg5f_left_*.yaml`/`dg5f_right_*.yaml`에 좌우 링크 이름을 각각
  완전한 형태로 직접 써야 한다** -- 한쪽만 고치고 다른 쪽을 안 고치면 그대로 어긋난다.
- `neutral_qpos` 길이는 URDF의 전체 DOF(`model.nq`, mimic joint 포함) 개수와 정확히
  같아야 한다. 안 맞으면 broadcasting 에러가 남.
- `origin`/`task`에 같은 링크 이름을 쓰더라도 `origin_offset`/`task_offset`이 다르면
  서로 다른 점으로 취급한다(중복 계산 안 됨) -- 한 링크의 관절원점과 실제 표면 두 점을
  동시에 쓰고 싶을 때 이 방식을 쓰면 된다.
- mimic 관절(URDF `<mimic>` 태그)은 최적화 대상에서 자동으로 빠지고, source 관절값에서
  `multiplier`/`offset`으로 계산된다 -- YAML에서 손댈 수 있는 부분이 아니라 URDF 쪽 문제다.

## 14. 3-프로세스 분리 실행 (2026-09-30)

리타게팅(손/모션캡쳐)과 MuJoCo 물리 시뮬레이션을 별도 프로세스로 완전히 분리.
`teleop/articulation_protocol.py`의 `ArticulationSender`/`ArticulationReceiver`(UDP, JSON)로
관절각만 주고받는다 -- MuJoCo 프로세스는 리타게팅 로직을 전혀 모른다.

```
# 한 번에 세 창 띄우기 (IP/포트는 파일 열어서 환경에 맞게 수정)
run_all.bat

# 또는 개별 실행 (conda activate upperbody_teleop 먼저)

# 1) Quest 손 트래킹 -> DG5F 손가락 리타게팅 + 그래프 창 + 녹화
python -m scripts.hand_process --quest-host 192.168.8.156 --quest-port 5005 --send udp://127.0.0.1:6100
python -m scripts.hand_process --quest-host 192.168.8.156 --quest-port 5005 --send udp://127.0.0.1:6100 --hand-retargeter anydex --anydex-config-left configs/anydex/dg5f_left_adaptive_quest3.yaml --anydex-config-right configs/anydex/dg5f_right_adaptive_quest3.yaml
python -m scripts.hand_process --quest-host 192.168.8.156 --quest-port 5005 --send udp://127.0.0.1:6100 --record recordings/hand_session1.jsonl

# 2) Axis Studio(MocapApi) -> 팔/몸통 IK 리타게팅 + 그래프 창 + 녹화
python -m scripts.mocap_process --port 7002 --send udp://127.0.0.1:6100
python -m scripts.mocap_process --port 7002 --send udp://127.0.0.1:6100 --arm-solver mink
python -m scripts.mocap_process --port 7002 --send udp://127.0.0.1:6100 --bvh-scale --record recordings/mocap_session1.jsonl

# 3) MuJoCo 물리 시뮬레이션만 (리타게팅 없음, 1)/2) 둘 다 여기로 UDP 전송)
python -m scripts.mujoco_physics_process --port 6100
```

- 1)과 2)는 서로 다른 관절 이름 집합(손가락 vs 팔/몸통)을 같은 포트(`--send`)로 보내도
  된다 -- `ArticulationReceiver`가 layout id로 구분해서 merge.
- 3)은 `robot_hand/mujoco_urdf.py`의 `build_actuated_model()`(액추에이터+중력/접촉 물리,
  같은 팔/손 안에서는 충돌 검사 끄고 팔 간/팔-몸통 간만 검사, 몸통은 사실상 용접)을 그대로
  쓴다. `--sim-mode`류 옵션은 없고 항상 physics 모드.
- 시각화는 둘 다 `scripts/skeleton_graph.py` 공용 (matplotlib, 노드+뼈대선+`x,y,z / rx,ry,rz`
  텍스트). MuJoCo 뷰어와 별 프로세스라 vedo/Open3D 창과 GLFW 충돌 걱정 없음.
- 녹화 포맷: 1)은 `hand_xr_v1`(원시 26-joint XR 배열 그대로, 나중에 다른 리타게터로 재생
  가능), 2)는 기존 `upper_body_combined_v1`(`CombinedRecorder` 재사용).
