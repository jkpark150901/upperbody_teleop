# BVH → RBY1 상반신 매핑 규칙

참조: [YanjieZe/GMR](https://github.com/YanjieZe/GMR) (`general_motion_retargeting/ik_configs/bvh_xsens_to_g1.json` 등)

## GMR 방식 요약

GMR은 BVH 본의 로컬 오일러 각을 로봇 관절에 직접 대입하지 않는다. 대신 **BVH 관절의 글로벌
위치(+글로벌 회전)를 로봇 링크의 IK 타깃으로 이름 대 이름으로 매칭**하고, 각 쌍마다
position weight / orientation weight를 줘서 전신 IK를 한 번에 푼다
(`ik_match_table`: `robot_link -> [human_bone, pos_weight, rot_weight, pos_offset, rot_offset_quat]`).

이 프로젝트에서 최근 디버깅하던 `_bvh_joint_angle_hints` / `--mocap-drive bvh-joints`
(BVH 본의 **로컬** 오일러 채널을 인덱스별로 로봇 관절에 직접 꽂는 방식)는 GMR에는 아예
존재하지 않는 방식이며, 계속 축/부호가 꼬이는 근본 원인이었다. **이 문서 및 이후 작업은
그 방식을 쓰지 않는다** (해당 코드/실험은 그대로 남겨두되 사용하지 않음).

## 이 프로젝트의 기존 position-IK 경로 (GMR과 동일한 설계, 이미 구현되어 있음)

`--backend mocapapi --mocap-drive ik` (기본값) 가 곧 GMR 스타일이다:
`RBY1UpperBodyRetargeter.update()` → `_solve_arm_position_block()` /
`_solve_wrist_orientation_block()`.

### 1. BVH 본 이름 해석 (`scripts/upper_body_retarget_vedo.py:_MOCAP_NAMES`)

| 내부 키 | 후보 BVH 본 이름 | 비고 |
|---|---|---|
| `hips` | `Hips` | 골반, Kabsch 정렬 기준점 |
| `chest` | `Spine3` → `Spine2` → `Spine1` → `Spine` | 존재하는 첫 이름 사용 |
| `head` | `Head` | |
| `left_shoulder` | `LeftShoulder` | |
| `left_elbow` | `LeftForeArm` | **본의 근위(proximal) 관절 위치 = 팔꿈치** |
| `left_wrist` | `LeftHand` | **본의 근위 관절 위치 = 손목** |
| `right_shoulder` | `RightShoulder` | |
| `right_elbow` | `RightForeArm` | |
| `right_wrist` | `RightHand` | |

GMR의 `LeftElbow`/`LeftWrist`(Xsens/Lafan1 명명)와 물리적으로 동일한 지점 —
명명 규칙만 다르고 본질은 같다.

각 값은 `GetJointGlobalPosition`으로 읽은 **월드 좌표계 위치**다 (로컬 아님).
`left_wrist_rotation`/`right_wrist_rotation`/`head_rotation`도 `GetJointGlobalRotation`
(월드 회전행렬)로 읽는다 — GMR도 `quat_fk`로 계산한 글로벌 회전을 쓴다.

### 2. 로봇 링크 매칭 (`robot_hand/upper_body_retarget.py:ROBOT_LANDMARK_LINKS`)

| BVH 키 (사람) | 로봇 링크 | IK 블록 |
|---|---|---|
| `chest` | `link_torso_5` | (torso pinned 모드에서는 미사용) |
| `head` | `link_head_2` | head_0/head_1 (2-DOF) |
| `left_shoulder` | `link_left_arm_0` | 왼팔 position 블록의 기준점 (구동 안 함, 어깨 자체가 origin) |
| `left_elbow` | `link_left_arm_3` | 왼팔 position 블록 타깃 |
| `left_wrist` | `link_left_arm_6` | 왼팔 position 블록 타깃 + orientation 블록 타깃 |
| `right_shoulder` | `link_right_arm_0` | 오른팔 동일 |
| `right_elbow` | `link_right_arm_3` | 오른팔 동일 |
| `right_wrist` | `link_right_arm_6` | 오른팔 동일 |

### 3. 팔 IK 블록 구조 (3-1-3, torso 고정 모드 기준)

```
어깨(spherical, arm_0/1/2) — 팔꿈치(revolute, arm_3) — 손목(spherical, arm_4/5/6)
```

- **Position 블록** (`_solve_arm_position_block`, side당 arm_0~arm_6 전체):
  팔꿈치 위치 + 손목 위치 타깃을 동시에 만족하도록 7-DOF(어깨3+팔꿈치1+손목3) 전체를
  IK로 푼다. 타깃은 `_targets()`가 계산한 사람 팔꿈치/손목의 (Kabsch 정렬 + 선택적
  `--bvh-scale` 길이 스케일 적용된) 로봇 좌표계 위치.
- **Orientation 블록** (`_solve_wrist_orientation_block`, arm_4/5/6만):
  손목의 **글로벌 회전행렬**을 보정 IK 타깃으로 추가 — position 블록이 이미 준 손목
  위치는 유지한 채 손목 3-DOF만 미세조정해서 손 방향까지 맞춘다.
- `joint_prior` (= 예전에 만들던 `_bvh_joint_angle_hints`) 는 후보 분기 선택용 보조
  점수일 뿐, 이 문서의 매핑 방식과는 무관하고 현재 사용 안 함.

### 4. 헤드 회전

Axis Neuron은 Y-up: 헤드 로컬 X = pitch, Y = yaw. RBY1은 Z가 yaw(`head_0`), Y가
pitch(`head_1`). `update()` 안에서 `neutral_head_rotation.T @ current` 소각 근사로
직접 변환 (별도 문서화 불필요할 만큼 짧고 안정적으로 이미 검증됨).

### 5. 스케일링

GMR의 `human_scale_table`(사람 전체 키 기준 uniform-ish 스케일)과 달리, 이 프로젝트는
`--bvh-scale` 옵션으로 **세그먼트별** (`{side}_upper`, `{side}_lower`, `{side}_shoulder`,
`head`) 신장 비율을 계산해서, 사람이 팔을 완전히 펴지 않았을 때 로봇 팔도 비례해서
덜 뻗도록 한다 (`_measure_human_lengths` / `_extension_ratio`).

## 결론 / 적용 상태

이 매핑은 **이미 코드로 존재**하며 (`--mocap-drive ik`가 기본값), 로컬 오일러 직접
매핑(`--mocap-drive bvh-joints`)을 켜지 않는 한 항상 이 경로가 사용된다. 즉 "GMR 방식
상반신 적용"은 실질적으로 이미 되어 있는 상태이고, 이번 작업은 그 사실을 GMR의
`ik_match_table`과 대조해서 검증하고 이 문서로 정리한 것. 하반신은 로봇에 다리 DOF가
없으므로 대상 아님.
