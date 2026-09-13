# Perception Neuron + SenseGlove 기반 MuJoCo 의류 조작 Teleoperation 구현 계획

## 1. 목표

최종 목표는 사람이 **Perception Neuron + SenseGlove**를 착용하고 MuJoCo 상의 상반신 휴머노이드/양손 로봇을 직접 조작하여 옷 개기 시연 데이터를 생성하는 것이다.

초기 단계에서는 외부 인간 시연 데이터(VR-Folding, DynamicIntelligence 등)를 직접 리타게팅하지 않는다.  
대신 사람이 시뮬레이터를 보면서 직접 조작하여 아래 데이터를 동기화해 저장한다.

- Robot-mounted RGB / Depth
- Robot joint state
- Left / Right wrist target pose
- Left / Right robot hand state
- SenseGlove raw hand data
- Perception Neuron raw / processed wrist pose
- Cloth mesh vertex state
- Contact / grasp state
- Executed robot action

생성한 데이터는 이후 다음 순서로 사용한다.

```text
Teleoperation demonstration
        ↓
Behavior Cloning
        ↓
Policy rollout
        ↓
Human correction / DAgger
        ↓
RL fine-tuning
```

---

# 2. 전체 시스템 구조

```text
               Human Operator
                    │
        ┌───────────┴───────────┐
        │                       │
Perception Neuron           SenseGlove
        │                       │
 torso / wrist pose         finger pose
        │                       │
        ▼                       ▼
 task-space               hand retargeting
 retargeting                    │
        │                       │
        └───────────┬───────────┘
                    ▼
                MuJoCo Robot
        ┌───────────┴────────────┐
        │                        │
      Arm IK                 Robot Hand
        │                        │
        └───────────┬────────────┘
                    ▼
                 Garment
                    │
          ┌─────────┼─────────┐
          ▼         ▼         ▼
         RGB-D    Cloth GT   Contact
          │         │         │
          └─────────┴─────────┘
                    ▼
            Demonstration Dataset
```

---

# 3. 장비 역할 분담

## 3.1 Perception Neuron

역할:

- torso orientation / pose
- left wrist pose
- right wrist pose
- 필요 시 elbow / shoulder pose
- teleoperation용 손목 6DoF 입력

초기에는 전신 리타게팅을 하지 않는다.

사용할 핵심 값:

```text
Chest pose
Left wrist pose
Right wrist pose
```

추천 입력 표현:

```text
T_chest_to_left_wrist
T_chest_to_right_wrist
```

즉 global pose를 그대로 쓰지 않고 torso-relative pose를 사용한다.

```math
T_{C \rightarrow W}
=
T_C^{-1} T_W
```

이 방식의 장점:

- IMU global drift 영향 감소
- 사람 이동과 robot base 이동 분리
- 사람/로봇 좌표계 정렬이 간단함
- robot workspace scaling이 쉬움

---

## 3.2 SenseGlove

역할:

- finger flexion
- thumb opposition
- finger joint posture
- pinch / grasp 상태
- 필요 시 haptic feedback

초기 단계에서는 모든 hand DoF를 policy action에 바로 사용하지 않아도 된다.

우선 다음 둘 중 하나로 시작한다.

### Option A. 단순 finger mapping

```text
human finger flexion
        ↓ normalize
0 ~ 1
        ↓
robot joint range
```

### Option B. Hand synergy

```text
z1 = global grasp closure
z2 = thumb opposition
z3 = index pinch
z4 = middle/ring support
```

권장:

- 원본 SenseGlove 데이터는 항상 저장
- 실제 robot control에는 축약된 synergy를 사용해도 됨

---

# 4. 소프트웨어 구성

## 4.1 Perception Neuron

권장 구조:

```text
Perception Neuron sensors
        ↓
Axis Studio (Windows)
        ↓
BVH / Calc Streaming
        ↓ UDP/TCP
MocapApi receiver
        ↓
teleop bridge
        ↓
MuJoCo
```

설치 대상:

- Axis Studio
- Noitom MocapApi
- 필요 시 mocap_ros_cpp / mocap_ros_py

공식 MocapApi:

```text
https://github.com/pnmocap/MocapApi
```

테스트 목표:

```text
timestamp
chest position / quaternion
left wrist position / quaternion
right wrist position / quaternion
```

를 실시간 출력한다.

---

## 4.2 SenseGlove

권장 구조:

```text
SenseGlove
    ↓
SenseCom
    ↓
SGCore / SGConnect
    ↓
teleop bridge
    ↓
MuJoCo hand
```

설치 대상:

- SenseCom
- SenseGlove Native API
- SGCore / SGConnect

공식 Native API:

```text
https://github.com/Adjuvo/SenseGlove-API
```

첫 테스트:

```text
Left hand finger values
Right hand finger values
```

를 콘솔에 실시간 출력한다.

---

# 5. 개발 PC 구성 권장안

초기 구현은 다음처럼 분리하는 것이 편하다.

## Windows PC

```text
Axis Studio
SenseCom
Perception Neuron connection
SenseGlove connection
```

## Ubuntu PC

```text
MuJoCo
GR00T-WholeBodyControl
MocapApi receiver
SenseGlove receiver 또는 network relay
Teleoperation controller
Data recorder
```

Perception Neuron은 Axis Studio에서 UDP/TCP streaming을 보내고 Ubuntu에서 직접 수신한다.

SenseGlove는 초기에는 Windows에서 정상 연결을 먼저 확인한 뒤:

1. Ubuntu Native API 직접 연결
2. 또는 Windows → UDP relay

중 하나를 선택한다.

---

# 6. Teleoperation 데이터 구조

통합 입력 구조 예시:

```cpp
struct HumanTeleopState
{
    double timestamp;

    Pose chest;

    Pose left_wrist;
    Pose right_wrist;

    HandState left_hand;
    HandState right_hand;
};
```

MuJoCo에서 사용할 target:

```cpp
struct RobotTeleopTarget
{
    Pose left_ee_target;
    Pose right_ee_target;

    std::vector<double> left_hand_target;
    std::vector<double> right_hand_target;
};
```

---

# 7. 손목 Retargeting

사람과 로봇의 팔 길이가 다르므로 absolute position을 그대로 복사하지 않는다.

## 7.1 기준 자세 calibration

Teleoperation 시작 시:

```text
human chest pose      → 저장
human wrist pose      → 저장

robot torso pose      → 저장
robot wrist pose      → 저장
```

---

## 7.2 Position mapping

예시:

```math
p_{robot,target}
=
p_{robot,0}
+
S R_{align}
(p_{human}-p_{human,0})
```

여기서:

- `S`: workspace scale
- `R_align`: human → robot 좌표축 변환
- `p_human,0`: 시작 시 사람 손 위치
- `p_robot,0`: 시작 시 robot EE 위치

초기에는 scalar scale 하나로 시작:

```text
S = 0.5 ~ 1.0
```

이후 필요하면 XYZ별 scale 사용.

---

## 7.3 Orientation mapping

```math
R_{robot,target}
=
R_{robot,0}
R_{human,0}^{-1}
R_{human}
R_{offset}
```

`R_offset`은 사람 손 neutral orientation과 robot wrist neutral orientation의 차이를 보정한다.

---

## 7.4 권장 control 방식

초기에는 absolute mapping보다 아래가 안전하다.

```text
relative Δpose control
+
clutch / recenter
```

필수 기능:

- workspace clamp
- joint limit clamp
- singularity check
- max EE velocity limit
- max angular velocity limit
- emergency stop
- recenter button

---

# 8. Robot Arm Control

초기 구현은:

```text
Perception Neuron wrist target
        ↓
Desired EE pose
        ↓
IK
        ↓
Robot joint target
        ↓
MuJoCo actuator
```

권장 순서:

1. EE marker만 움직이기
2. IK 연결
3. 한 팔
4. 양팔
5. cloth 없이 test
6. cloth 추가

초기에는 torso를 고정해도 된다.

---

# 9. Robot Hand Retargeting

초기에는 full optimization보다 단순 mapping으로 시작한다.

## Phase 1

```text
SenseGlove flexion
        ↓
normalize
        ↓
robot finger joint range
```

## Phase 2

필요 시 fingertip-based retargeting:

```math
q^*
=
\arg\min_q
\sum_f
\|p_f^{robot}(q)-p_f^{human}\|^2
+
\lambda \|q-q_{prev}\|^2
```

## Phase 3

cloth task에 필요한 grasp primitive를 추출:

- pinch
- edge grasp
- power grasp
- release
- sliding grasp

---

# 10. Cloth Simulation

초기 목표는 현실 cloth를 완전히 재현하는 것이 아니다.

우선:

- garment 1종
- flat initial state
- 적당한 mesh resolution
- self-collision
- bending stiffness
- stretch stiffness
- friction
- mass

정도만 맞춘다.

초기 추천 task:

```text
1. towel folding
2. rectangular cloth folding
3. T-shirt folding
```

T-shirt부터 시작해도 되지만 디버깅 관점에서는 rectangular cloth가 훨씬 쉽다.

---

# 11. Camera 구성

Teleoperator가 보는 화면과 policy가 사용할 camera를 분리한다.

## Human operator

```text
MuJoCo viewer
```

또는 필요 시 HMD.

## Policy observation camera

Robot head / chest에:

```text
RGB camera
Depth camera
```

를 고정한다.

시연 생성 시 반드시 저장:

```text
RGB_t
Depth_t
```

최종 policy 목표:

```math
\pi(a_t | RGBD_t, q_t)
```

---

# 12. Demonstration Dataset Schema

한 timestep 예시:

```text
timestamp

observation/
    rgb
    depth
    robot_q
    robot_qdot
    robot_ee_left
    robot_ee_right

teleop_raw/
    pn_chest_pose
    pn_left_wrist_pose
    pn_right_wrist_pose

    senseglove_left_raw
    senseglove_right_raw

action/
    left_ee_target
    right_ee_target

    left_hand_target
    right_hand_target

privileged/
    cloth_vertices
    cloth_velocity
    contact_state
    grasp_state

metadata/
    garment_id
    episode_id
    success
```

중요:

**Raw teleoperation input을 반드시 별도로 저장한다.**

그래야 robot hand나 retargeting 알고리즘을 바꿔도 기존 시연을 다시 변환할 수 있다.

---

# 13. 초기상태 Randomization

처음에는 최소 randomization만 사용한다.

```text
cloth x/y translation
cloth yaw
small wrinkle perturbation
robot initial q noise
```

초기 성공 이후 점차 확대:

```text
cloth material
friction
mass
initial deformation
camera noise
lighting
```

---

# 14. 구현 단계

## Phase 0 — 장비 확인

- [ ] Perception Neuron 정확한 모델 확인
- [ ] Axis Studio 설치
- [ ] PN calibration 성공
- [ ] SenseGlove 정확한 모델 확인
- [ ] SenseCom 설치
- [ ] glove calibration 성공

산출물:

```text
장비 두 개 모두 vendor software에서 정상 동작
```

---

## Phase 1 — Perception Neuron Receiver

- [ ] Axis Studio streaming 활성화
- [ ] MocapApi 설치
- [ ] chest pose 출력
- [ ] left wrist pose 출력
- [ ] right wrist pose 출력
- [ ] timestamp 확인

산출물:

```text
pn_reader
```

출력:

```text
timestamp
chest SE(3)
left wrist SE(3)
right wrist SE(3)
```

---

## Phase 2 — SenseGlove Receiver

- [ ] SenseCom 실행
- [ ] SGCore 연결
- [ ] Left glove 인식
- [ ] Right glove 인식
- [ ] finger joint 값 출력
- [ ] raw sensor 값 저장 가능 확인

산출물:

```text
senseglove_reader
```

---

## Phase 3 — 통합 Teleop Input

- [ ] PN + SenseGlove timestamp 통합
- [ ] fixed-rate state publish
- [ ] dropped packet handling
- [ ] latency 기록

권장 rate:

```text
50 ~ 100 Hz control
```

---

## Phase 4 — MuJoCo EE Marker

아직 robot arm을 움직이지 않는다.

- [ ] human→robot coordinate transform
- [ ] scale 적용
- [ ] left target marker
- [ ] right target marker
- [ ] recenter
- [ ] workspace clamp

성공 기준:

```text
사람 손을 움직였을 때
MuJoCo marker가 자연스럽게 대응
```

---

## Phase 5 — Robot Arm IK

- [ ] left arm IK
- [ ] right arm IK
- [ ] joint limit
- [ ] velocity limit
- [ ] collision 확인
- [ ] torso fixed test

성공 기준:

```text
양손을 5~10분 움직여도
IK 발산/급격한 joint jump가 없음
```

---

## Phase 6 — SenseGlove → Robot Hand

- [ ] open hand
- [ ] close hand
- [ ] thumb opposition
- [ ] pinch
- [ ] release
- [ ] joint limit

성공 기준:

```text
주요 grasp가 사람이 의도한 대로 동작
```

---

## Phase 7 — PN + SenseGlove 통합 Teleoperation

- [ ] 양팔 + 양손 동시에 제어
- [ ] pause
- [ ] recenter
- [ ] emergency stop
- [ ] control latency 확인

목표:

```text
cloth 없이 물체 집기 가능
```

---

## Phase 8 — Cloth 추가

첫 test:

```text
grab
lift
move
release
```

다음:

```text
한 번 접기
```

그 다음:

```text
full folding sequence
```

---

## Phase 9 — Recorder

Teleoperation과 동시에 저장:

- [ ] RGB
- [ ] Depth
- [ ] robot state
- [ ] robot action
- [ ] raw PN
- [ ] raw SenseGlove
- [ ] cloth mesh
- [ ] contact
- [ ] success label

산출물:

```text
episode_000/
episode_001/
...
```

---

# 15. 첫 시연 데이터 목표

초기에는 데이터 양보다 pipeline 검증이 목적이다.

## Stage A

```text
5 successful demonstrations
```

목적:

- 저장 포맷 검증
- replay 검증
- timestamp sync 검증

## Stage B

```text
20 ~ 50 successful demonstrations
```

목적:

- Behavior Cloning 초기 실험

## Stage C

```text
100+ demonstrations
```

초기 garment pose randomization 포함.

---

# 16. Behavior Cloning

첫 BC는 단순하게 시작한다.

Observation:

```text
RGB-D
robot q
robot qdot
```

Action:

```text
Left wrist Δpose
Right wrist Δpose
Left hand synergy
Right hand synergy
```

초기에는 cloth mesh를 privileged observation으로 사용하는 별도 실험도 수행한다.

```text
cloth mesh embedding
+
robot state
        ↓
policy
```

이를 통해:

```text
vision 문제인지
control 문제인지
cloth 문제인지
```

를 분리해서 확인할 수 있다.

---

# 17. DAgger / Human Correction

BC policy rollout 중 실패하면:

```text
BC policy
    ↓
cloth state 이상
    ↓
human takeover
    ↓
Perception Neuron + SenseGlove
    ↓
recovery demonstration
    ↓
dataset 추가
```

이 방식으로 policy가 실제로 방문하는 state distribution을 보강한다.

---

# 18. RL Fine-tuning

BC가 기본 folding behavior를 만든 뒤 RL을 적용한다.

Reward 후보:

```text
semantic keypoint distance
fold-line alignment
cloth coverage
target region overlap
grasp stability
excessive stretching penalty
action smoothness
final folding score
```

초기 RL 목표는 인간 trajectory를 그대로 복사하는 것이 아니라:

```text
simulation dynamics에 맞게
grasp / lift / move / release를 수정
```

하는 것이다.

---

# 19. 초기 실험에서 하지 않을 것

다음은 초기 단계에서 제외한다.

- VR-Folding full-body retargeting
- human cloth mesh trajectory 복원
- real RGB → exact cloth mesh reconstruction
- SenseGlove 모든 DoF 직접 policy action
- full humanoid whole-body control
- 실제 garment material identification
- sim-to-real domain adaptation
- VLA fine-tuning

먼저:

```text
teleop → successful sim folding → dataset → BC
```

가 되는지만 확인한다.

---

# 20. 주요 리스크

## Perception Neuron drift

대응:

```text
global wrist pose 사용 최소화
torso-relative pose 사용
recenter 지원
```

## Human / robot workspace 차이

대응:

```text
relative mapping
scale
workspace clamp
IK feasibility check
```

## SenseGlove / robot hand morphology 차이

대응:

```text
normalized flexion
synergy
fingertip-based retargeting
```

## Cloth grasp가 어려움

대응:

```text
초기에는 단순 cloth
높은 friction
손가락 morphology 단순화
grasp primitive 사용
```

## Teleoperation latency

대응:

```text
timestamp 저장
control interpolation
low-pass filter
separate rendering / physics rates
```

---

# 21. 권장 구현 순서 요약

```text
[1]
Perception Neuron
→ wrist pose console 출력

[2]
SenseGlove
→ finger pose console 출력

[3]
PN
→ MuJoCo EE marker

[4]
EE marker
→ robot arm IK

[5]
SenseGlove
→ robot hand

[6]
PN + SenseGlove
→ full teleoperation

[7]
simple object grasp

[8]
cloth grasp / lift / release

[9]
single fold

[10]
RGB-D + mesh + action recorder

[11]
20~50 demos

[12]
Behavior Cloning

[13]
Human correction / DAgger

[14]
RL fine-tuning
```

---

# 22. 현재 가장 중요한 다음 작업

우선 아래 두 프로그램이 독립적으로 동작해야 한다.

```text
pn_reader
senseglove_reader
```

목표 출력:

```text
PN:
timestamp
left wrist xyz quaternion
right wrist xyz quaternion

SenseGlove:
timestamp
left finger state
right finger state
```

이 두 데이터가 안정적으로 나오면 이후 MuJoCo 쪽 통합 작업으로 넘어간다.

---

# 23. 초기 디렉터리 구조 제안

```text
humanoid_cloth_teleop/
│
├── devices/
│   ├── perception_neuron/
│   │   ├── pn_reader.cpp
│   │   └── pn_types.hpp
│   │
│   └── senseglove/
│       ├── sg_reader.cpp
│       └── sg_types.hpp
│
├── teleop/
│   ├── teleop_state.hpp
│   ├── wrist_retarget.cpp
│   ├── hand_retarget.cpp
│   └── calibration.yaml
│
├── mujoco/
│   ├── robot_controller.cpp
│   ├── arm_ik.cpp
│   ├── hand_controller.cpp
│   └── viewer.cpp
│
├── cloth/
│   ├── garment.xml
│   └── garment_config.yaml
│
├── recorder/
│   ├── recorder.cpp
│   └── dataset_schema.md
│
├── configs/
│   ├── teleop.yaml
│   ├── robot.yaml
│   └── camera.yaml
│
└── README.md
```

---

# 24. 최종 목표 구조

```text
Perception Neuron + SenseGlove
            ↓
      Human expert
            ↓
     MuJoCo Teleop
            ↓
 ┌──────────┼──────────┐
 RGB-D     Mesh GT     Action
 └──────────┼──────────┘
            ↓
   Demonstration Dataset
            ↓
     Behavior Cloning
            ↓
   Human Correction
            ↓
      RL Fine-tuning
            ↓
   Cloth Folding Policy
```

---

## 핵심 결정

현재 프로젝트에서는 외부 인간 시연 데이터를 MuJoCo에 직접 재현하는 것보다 다음 방식을 우선한다.

> **Perception Neuron으로 손목/팔 pose를 받고, SenseGlove로 손가락 pose를 받아 MuJoCo robot을 직접 teleoperation하여 시연 데이터를 생성한다.**

이 방식은 사람 시연의 cloth deformation을 별도로 복원할 필요가 없고, 시연 순간의 RGB-D / robot state / action / cloth mesh state를 동일한 simulation timeline에서 동시에 확보할 수 있다는 것이 가장 큰 장점이다.
