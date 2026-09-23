# Quest/OpenXR → DG5F Left Hand 리타게팅 규칙

## 1. 목적

Meta Quest 3의 OpenXR hand tracking으로 얻은 26개 hand joint pose를 이용해 사람 손의 관절각을 추정하고, 이를 DG5F left hand의 20개 revolute joint target으로 변환한다.

초기 구현은 **joint-angle 기반 rule-based retargeting**을 사용한다.

```text
Quest/OpenXR 26 joints
        ↓
palm-local 좌표계 변환
        ↓
사람 손 관절각 계산
        ↓
neutral offset / sign / gain
        ↓
DG5F joint limit clamp
        ↓
DG5F q_target (20 DoF)
```

목표는 손 크기 차이에 영향을 덜 받도록 **절대 위치보다 관절 사이 각도**를 사용하는 것이다.

---

## 2. OpenXR hand joint index

OpenXR 기본 hand skeleton:

| Index | Joint |
|---:|---|
| 0 | PALM |
| 1 | WRIST |
| 2 | THUMB_METACARPAL |
| 3 | THUMB_PROXIMAL |
| 4 | THUMB_DISTAL |
| 5 | THUMB_TIP |
| 6 | INDEX_METACARPAL |
| 7 | INDEX_PROXIMAL |
| 8 | INDEX_INTERMEDIATE |
| 9 | INDEX_DISTAL |
| 10 | INDEX_TIP |
| 11 | MIDDLE_METACARPAL |
| 12 | MIDDLE_PROXIMAL |
| 13 | MIDDLE_INTERMEDIATE |
| 14 | MIDDLE_DISTAL |
| 15 | MIDDLE_TIP |
| 16 | RING_METACARPAL |
| 17 | RING_PROXIMAL |
| 18 | RING_INTERMEDIATE |
| 19 | RING_DISTAL |
| 20 | RING_TIP |
| 21 | LITTLE_METACARPAL |
| 22 | LITTLE_PROXIMAL |
| 23 | LITTLE_INTERMEDIATE |
| 24 | LITTLE_DISTAL |
| 25 | LITTLE_TIP |

```python
OPENXR = {
    "palm": 0,
    "wrist": 1,
    "thumb":  [2, 3, 4, 5],
    "index":  [6, 7, 8, 9, 10],
    "middle": [11, 12, 13, 14, 15],
    "ring":   [16, 17, 18, 19, 20],
    "pinky":  [21, 22, 23, 24, 25],
}
```

---

## 3. DG5F left joint order

DG5F left URDF의 revolute joint는 총 20개이다.

```python
DG5F_ORDER = [
    "lj_dg_1_1", "lj_dg_1_2", "lj_dg_1_3", "lj_dg_1_4",
    "lj_dg_2_1", "lj_dg_2_2", "lj_dg_2_3", "lj_dg_2_4",
    "lj_dg_3_1", "lj_dg_3_2", "lj_dg_3_3", "lj_dg_3_4",
    "lj_dg_4_1", "lj_dg_4_2", "lj_dg_4_3", "lj_dg_4_4",
    "lj_dg_5_1", "lj_dg_5_2", "lj_dg_5_3", "lj_dg_5_4",
]
```

초기 finger mapping:

| DG5F finger | Human finger |
|---:|---|
| 1 | Thumb |
| 2 | Index |
| 3 | Middle |
| 4 | Ring |
| 5 | Pinky |

---

## 4. URDF joint limits

첨부된 `dg5f_left.urdf` 기준.

### Thumb

| Joint | Axis | Lower | Upper |
|---|---|---:|---:|
| `lj_dg_1_1` | X | -0.8901 | 0.3840 |
| `lj_dg_1_2` | Z | 0.0000 | 3.1416 |
| `lj_dg_1_3` | X | -1.5708 | 1.5708 |
| `lj_dg_1_4` | X | -1.5708 | 1.5708 |

### Index

| Joint | Axis | Lower | Upper |
|---|---|---:|---:|
| `lj_dg_2_1` | X | -0.6109 | 0.4189 |
| `lj_dg_2_2` | Y | 0.0000 | 2.0071 |
| `lj_dg_2_3` | Y | -1.5708 | 1.5708 |
| `lj_dg_2_4` | Y | -1.5708 | 1.5708 |

### Middle

| Joint | Axis | Lower | Upper |
|---|---|---:|---:|
| `lj_dg_3_1` | X | -0.6109 | 0.6109 |
| `lj_dg_3_2` | Y | 0.0000 | 1.9548 |
| `lj_dg_3_3` | Y | -1.5708 | 1.5708 |
| `lj_dg_3_4` | Y | -1.5708 | 1.5708 |

### Ring

| Joint | Axis | Lower | Upper |
|---|---|---:|---:|
| `lj_dg_4_1` | X | -0.4189 | 0.6109 |
| `lj_dg_4_2` | Y | 0.0000 | 1.9024 |
| `lj_dg_4_3` | Y | -1.5708 | 1.5708 |
| `lj_dg_4_4` | Y | -1.5708 | 1.5708 |

### Pinky

| Joint | Axis | Lower | Upper |
|---|---|---:|---:|
| `lj_dg_5_1` | Z | -1.0472 | 0.01745 |
| `lj_dg_5_2` | X | -0.6109 | 0.4189 |
| `lj_dg_5_3` | Y | -1.5708 | 1.5708 |
| `lj_dg_5_4` | Y | -1.5708 | 1.5708 |

> `lj_dg_5_1/5_2`는 다른 세 손가락과 축 구성이 다르므로, 초기 구현 후 시뮬레이터에서 의미와 부호를 반드시 검증한다.

---

## 5. Palm-local frame

월드/OpenXR local 좌표를 직접 쓰지 않고 매 프레임 palm-local frame을 만든다.

```python
W  = p[1]   # wrist
I0 = p[6]   # index metacarpal
M0 = p[11]  # middle metacarpal
P0 = p[21]  # little metacarpal

forward = normalize(M0 - W)
lateral = normalize(I0 - P0)

forward = normalize(
    forward - dot(forward, lateral) * lateral
)

normal = normalize(cross(lateral, forward))

R_palm = np.column_stack([lateral, forward, normal])
```

의미:

```text
forward : wrist → middle finger 방향
lateral : pinky → index 방향
normal  : palm plane에 수직
```

월드 벡터 `v`를 palm-local로 바꿀 때:

```python
v_local = R_palm.T @ v
```

왼손/오른손 또는 OpenXR 좌표계 정의에 따라 `normal` 부호가 뒤집힐 수 있으므로 calibration 시 확인한다.

---

## 6. Signed angle helper

```python
def signed_angle(a, b, axis):
    a = normalize(a)
    b = normalize(b)
    axis = normalize(axis)

    return np.arctan2(
        np.dot(axis, np.cross(a, b)),
        np.clip(np.dot(a, b), -1.0, 1.0)
    )
```

---

## 7. Index / Middle / Ring 규칙

Finger joint sequence:

```text
F0 = metacarpal
F1 = proximal
F2 = intermediate
F3 = distal
F4 = tip
```

bone vectors:

```python
b0 = normalize(F1 - F0)
b1 = normalize(F2 - F1)
b2 = normalize(F3 - F2)
b3 = normalize(F4 - F3)
```

### 7.1 MCP abduction/adduction

```python
b1_local = R_palm.T @ b1
mcp_abd = np.arctan2(b1_local[0], b1_local[1])
```

초기 매핑:

```text
Index  MCP abd/add → lj_dg_2_1
Middle MCP abd/add → lj_dg_3_1
Ring   MCP abd/add → lj_dg_4_1
```

### 7.2 MCP flexion

```python
mcp_flex = np.arctan2(
    -b1_local[2],
    np.sqrt(b1_local[0]**2 + b1_local[1]**2)
)
```

초기 매핑:

```text
Index  MCP flex → lj_dg_2_2
Middle MCP flex → lj_dg_3_2
Ring   MCP flex → lj_dg_4_2
```

### 7.3 PIP flexion

```python
pip_flex = np.arccos(
    np.clip(np.dot(b1, b2), -1.0, 1.0)
)
```

매핑:

```text
Index  PIP → lj_dg_2_3
Middle PIP → lj_dg_3_3
Ring   PIP → lj_dg_4_3
```

### 7.4 DIP flexion

```python
dip_flex = np.arccos(
    np.clip(np.dot(b2, b3), -1.0, 1.0)
)
```

매핑:

```text
Index  DIP → lj_dg_2_4
Middle DIP → lj_dg_3_4
Ring   DIP → lj_dg_4_4
```

---

## 8. Pinky 규칙

OpenXR:

```text
P0 = 21 metacarpal
P1 = 22 proximal
P2 = 23 intermediate
P3 = 24 distal
P4 = 25 tip
```

bone vectors:

```python
p0 = normalize(P1 - P0)
p1 = normalize(P2 - P1)
p2 = normalize(P3 - P2)
p3 = normalize(P4 - P3)
```

`lj_dg_5_3`, `lj_dg_5_4`는 우선 각각 PIP/DIP flexion에 대응한다.

```text
lj_dg_5_3 ← pinky PIP flex
lj_dg_5_4 ← pinky DIP flex
```

`lj_dg_5_1`, `lj_dg_5_2`는 URDF 축 구성이 다른 손가락과 다르므로 2-DoF base/MCP feature로 먼저 매핑한 뒤 시뮬레이터에서 swap/sign을 검증한다.

```python
p1_local = R_palm.T @ p1

pinky_abd = np.arctan2(
    p1_local[0],
    p1_local[1]
)

pinky_flex = np.arctan2(
    -p1_local[2],
    np.sqrt(p1_local[0]**2 + p1_local[1]**2)
)
```

초기안:

```text
lj_dg_5_1 ← pinky abduction/base spread
lj_dg_5_2 ← pinky MCP flexion
```

반드시 시뮬레이터에서 `5_1`, `5_2`를 각각 소각도로 움직여 실제 의미와 swap 여부를 확인한다.

---

## 9. Thumb 규칙

OpenXR thumb:

```text
T0 = 2  thumb metacarpal
T1 = 3  thumb proximal
T2 = 4  thumb distal
T3 = 5  thumb tip
```

bone vectors:

```python
t0 = normalize(T1 - T0)
t1 = normalize(T2 - T1)
t2 = normalize(T3 - T2)
```

초기 의미:

```text
DG5F 1_1, 1_2 : thumb base 2 DoF
DG5F 1_3      : thumb MCP flexion
DG5F 1_4      : thumb IP flexion
```

### 9.1 Thumb distal angles

```python
thumb_mcp = np.arccos(
    np.clip(np.dot(t0, t1), -1.0, 1.0)
)

thumb_ip = np.arccos(
    np.clip(np.dot(t1, t2), -1.0, 1.0)
)
```

초기 매핑:

```text
lj_dg_1_3 ← thumb_mcp
lj_dg_1_4 ← thumb_ip
```

### 9.2 Thumb base 2 DoF

```python
t0_local = R_palm.T @ t0

thumb_base_yaw = np.arctan2(
    t0_local[0],
    t0_local[1]
)

thumb_base_pitch = np.arctan2(
    t0_local[2],
    np.sqrt(t0_local[0]**2 + t0_local[1]**2)
)
```

초기 매핑:

```text
lj_dg_1_1 ← thumb_base_pitch
lj_dg_1_2 ← thumb_base_yaw
```

Thumb는 반드시 다음을 calibration한다.

```text
- 1_1 / 1_2 swap 여부
- 각 joint sign
- neutral offset
- gain
```

---

## 10. Neutral calibration

1. 사용자가 손을 자연스럽게 편다.
2. 1~2초 동안 human feature를 평균한다.
3. 이를 `human_zero`로 저장한다.
4. 해당 순간 DG5F 자세를 `robot_zero`로 정의한다.

```python
delta = q_human - q_human_zero
q_robot = q_robot_zero + gain * sign * delta
```

초기에는 `q_robot_zero = 0`으로 시작해도 된다.

---

## 11. Sign / Gain 설정

```python
SIGN = {
    "lj_dg_1_1": 1.0,
    "lj_dg_1_2": 1.0,
    # ...
}

GAIN = {
    "lj_dg_1_1": 1.0,
    "lj_dg_1_2": 1.0,
    # ...
}
```

최종 변환:

```python
q = q0_robot[joint] + SIGN[joint] * GAIN[joint] * (
    h - h0[joint]
)
```

권장 tuning 순서:

```text
1. sign
2. neutral offset
3. gain
4. joint-limit clamp
```

---

## 12. Joint limit clamp

```python
q = np.clip(q, q_min[joint], q_max[joint])
```

초기 테스트에서는 limit 전체를 쓰지 않고 90~95% 범위로 줄여도 된다.

---

## 13. Tracking validity 처리

필요한 OpenXR joint가 모두 valid일 때만 새 target을 계산한다.

권장 규칙:

```text
필요 joint valid
    → 새 target 계산

일부 joint invalid
    → 이전 valid target 유지

N frame 이상 invalid
    → neutral / safe pose로 천천히 복귀
```

tracking loss에서 target을 즉시 0으로 만들지 않는다.

---

## 14. Filtering

최종 robot joint target에 EMA를 적용한다.

```python
q_filtered = alpha * q_new + (1.0 - alpha) * q_prev
```

초기값:

```python
alpha = 0.25 ~ 0.5
```

---

## 15. Velocity limit

```python
dq = q_target - q_prev
dq = np.clip(dq, -dq_max * dt, dq_max * dt)
q_cmd = q_prev + dq
```

첨부 URDF의 revolute joint velocity limit은 `3.141592653589793 rad/s`이므로 초기 상한으로 활용할 수 있다.

---

## 16. 초기 joint mapping 요약

```text
Thumb
  base pitch/yaw → 1_1 / 1_2
  MCP flex       → 1_3
  IP flex        → 1_4

Index
  MCP abd/add → 2_1
  MCP flex    → 2_2
  PIP flex    → 2_3
  DIP flex    → 2_4

Middle
  MCP abd/add → 3_1
  MCP flex    → 3_2
  PIP flex    → 3_3
  DIP flex    → 3_4

Ring
  MCP abd/add → 4_1
  MCP flex    → 4_2
  PIP flex    → 4_3
  DIP flex    → 4_4

Pinky
  base/MCP abduction-like → 5_1
  MCP flexion-like        → 5_2
  PIP flex                → 5_3
  DIP flex                → 5_4
```

`Thumb 1_1/1_2`와 `Pinky 5_1/5_2`는 우선 시뮬레이터에서 실제 motion을 확인해 swap/sign을 확정한다.

---

## 17. Recommended data structure

```python
@dataclass
class HandRetargetState:
    human_angles: dict
    human_zero: dict

    robot_target: np.ndarray   # shape=(20,)
    robot_prev: np.ndarray     # shape=(20,)

    valid: bool
    timestamp: float
```

최종 출력은 named joint-space target:

```python
RobotJointTarget = {
    "names": DG5F_ORDER,
    "position": q_target,
}
```

---

## 18. Pseudocode

```python
def retarget(openxr_joint_positions, valid_flags, dt):

    if not required_joints_valid(valid_flags):
        return hold_or_safe_pose()

    R_palm = make_palm_frame(openxr_joint_positions)

    human = {}
    human.update(extract_thumb_angles(openxr_joint_positions, R_palm))
    human.update(extract_finger_angles("index", openxr_joint_positions, R_palm))
    human.update(extract_finger_angles("middle", openxr_joint_positions, R_palm))
    human.update(extract_finger_angles("ring", openxr_joint_positions, R_palm))
    human.update(extract_pinky_angles(openxr_joint_positions, R_palm))

    q = np.zeros(20)

    for i, joint_name in enumerate(DG5F_ORDER):
        h = human[HUMAN_FEATURE[joint_name]]

        q[i] = (
            ROBOT_ZERO[joint_name]
            + SIGN[joint_name]
            * GAIN[joint_name]
            * (h - HUMAN_ZERO[joint_name])
        )

        q[i] = np.clip(
            q[i],
            JOINT_LIMIT[joint_name][0],
            JOINT_LIMIT[joint_name][1]
        )

    q = low_pass(q, q_prev)
    q = velocity_limit(q, q_prev, dt)

    q_prev[:] = q
    return q
```

---

## 19. 초기 검증 순서

### Step 1 — robot joint 단독 테스트

각 DG5F joint를 하나씩 소각도로 움직여 실제 motion을 확인한다.

특히:

```text
thumb 1_1 / 1_2
pinky 5_1 / 5_2
```

### Step 2 — 사람 손 한 DoF씩 테스트

```text
손가락 벌리기/모으기
검지만 굽히기
PIP만 굽히기
DIP만 굽히기
엄지 벌리기
엄지 opposition
```

### Step 3 — sign 수정

반대로 움직이는 joint의 `SIGN`을 뒤집는다.

### Step 4 — neutral calibration

편 손 자세를 1~2초 평균한다.

### Step 5 — gain tuning

human ROM과 robot ROM을 비교해 gain을 조정한다.

### Step 6 — grasp 검증

```text
open hand
fist
pinch
tripod grasp
power grasp
```

---

## 20. V0 구현 원칙

초기 버전에서는 다음을 하지 않는다.

```text
- fingertip Cartesian IK
- nonlinear optimization
- learned retargeting
- hand mesh fitting
```

먼저:

```text
position-derived joint angles
→ offset/sign/gain
→ joint limits
```

만으로 안정적인 20-DoF target을 만든다.

개선 순서:

```text
V0 : 모든 joint rule-based
V1 : thumb fingertip/opposition correction
V2 : fingertip task-space optimization
V3 : learned retargeting (필요 시)
```

---

## 21. 구현 시 주의사항

1. OpenXR global/local position 자체를 robot joint에 직접 매핑하지 않는다.
2. 손 크기 차이를 줄이기 위해 bone length보다 angle을 사용한다.
3. palm-local frame을 매 프레임 갱신한다.
4. `acos()` 입력은 반드시 `[-1, 1]`로 clamp한다.
5. tracking loss에서 target을 즉시 0으로 만들지 않는다.
6. URDF joint limit을 마지막 단계에서 항상 적용한다.
7. sign / gain / zero는 YAML/JSON config로 분리한다.
8. left/right hand를 동시에 지원할 경우 palm-frame handedness를 명시적으로 처리한다.
9. Quest 입력 코드와 robot-specific mapping 코드를 분리한다.

추천 모듈 구조:

```text
quest_receiver.py
    ↓
openxr_hand.py
    ↓
human_hand_features.py
    ↓
dg5f_retargeter.py
    ↓
RobotJointTarget
```
