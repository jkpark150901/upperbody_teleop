# 교시(시연) 데이터 요구사항

이 문서는 LeRobot으로 모방학습(SmolVLA, ACT)을 할 때 필요한 교시 데이터의 형식과 수집 기준을 정리한 것입니다. 관절 공간(joint space) action을 기준으로 합니다.

---

## 1. 저장 형식: LeRobotDataset v3.0

- 수치 데이터는 parquet, 카메라 영상은 mp4, 메타데이터는 JSON/parquet로 저장합니다.
- 파일을 직접 만들지 말고 `LeRobotDataset` API(`create` → `add_frame` → `save_episode` → `finalize`)로 생성합니다. 이 API가 디렉토리 구조, `stats.json`, 영상 인코딩을 자동으로 처리합니다.

```
my_dataset/
├── meta/
│   ├── info.json                            # 스키마: fps, features, 경로 템플릿
│   ├── stats.json                           # feature별 mean/std/min/max (정규화용, 필수)
│   ├── tasks.parquet                        # task 문장 ↔ task_index
│   └── episodes/chunk-000/file-000.parquet  # 에피소드별 위치 정보
├── data/chunk-000/file-000.parquet          # 프레임 단위 데이터
└── videos/<camera_key>/chunk-000/file-000.mp4
```

**에피소드와 chunk/file의 차이**
- **에피소드**: 시연 한 번입니다. 의미 단위이고, 학습할 때 샘플링도 에피소드를 기준으로 합니다.
- **file**: 여러 에피소드를 이어 붙인 저장 단위입니다. data는 약 100MB, video는 약 200MB가 넘으면 다음 파일로 넘어갑니다.
- **chunk**: file을 최대 1000개까지 묶는 폴더입니다.
- 어떤 에피소드가 어느 파일의 어느 구간에 있는지는 `meta/episodes/*.parquet`에 기록됩니다. 사용자가 직접 관리할 필요는 없습니다.

---

## 2. 프레임 단위 필드

| 키 | dtype / shape | 내용 | 필수 |
|---|---|---|---|
| `observation.state` | float32, `(N_state,)` | **측정된** 관절 위치. 필요하면 속도나 그리퍼 상태도 포함 | ✅ |
| `action` | float32, `(N_action,)` | **명령한** 관절 목표 위치(absolute) | ✅ |
| `observation.images.<cam>` | video, `(H, W, 3)` uint8 | 카메라 RGB 영상 | ✅ (1개 이상) |
| `task` | str | 자연어 작업 지시문 | ✅ |

`timestamp`, `frame_index`, `episode_index`, `index`, `task_index`는 자동으로 생성됩니다.

### action 작성 규칙
- **action에는 목표 위치(command)를 넣고, state에는 측정값을 넣습니다.** 목표 위치는 텔레옵 리더 암, 리타겟팅 결과, 컨트롤러 setpoint 같은 값입니다. action과 state가 사실상 같은 값이면 정책은 의미 있는 동작을 배우지 못합니다.
- 값은 **absolute**(목표 위치 그 자체)로 저장합니다. relative 변환이 필요하면 학습할 때 processor가 처리합니다(pi0 계열의 `use_relative_actions`).
- 단위와 관절 순서는 모든 에피소드에서 같아야 합니다. `names`에 관절 이름을 기록해 두기를 권장합니다.
- **추론할 때 로봇에 보내는 명령은 학습한 action과 같은 공간이어야 합니다.** joint로 학습했다면 joint 명령을 그대로 보내면 됩니다.
- 하체를 별도 locomotion 컨트롤러가 맡는다면, action에는 상체 관절과 손만 넣고 하체 명령은 고수준 값(예: base velocity)으로 넣는 구성을 고려합니다.

### task 문장
- SmolVLA는 언어를 입력으로 실제로 사용합니다. ACT는 무시합니다.
- 구체적이고 일관되게 씁니다. 예: `"pick up the red cup and place it in the box"`
- 같은 작업에는 같은 문장을 씁니다. 작업이 여러 종류라면 문장으로 구분합니다.

---

## 3. 정책별 제약

| 항목 | SmolVLA | ACT |
|---|---|---|
| state/action 차원 | **각각 32 이하** (`max_state_dim`, `max_action_dim`. 부족한 부분은 0으로 채움) | 제한 없음 |
| 카메라 수 | 최대 3개 | 제한 없음 (1개 이상) |
| 카메라 이름 | 사전학습 모델 기준 `camera1~3`. `--rename_map`으로 연결 | 자유 |
| 이미지 해상도 | 자유 (내부에서 512×512로 패딩·리사이즈) | 자유 |
| 언어 | 사용함 | 사용 안 함 |
| 예측 길이 | 50스텝 chunk | 100스텝 chunk (기본값) |

**SmolVLA 학습 명령 예시.** `lerobot/svla_so101_pickplace`로 동작을 확인한 형태입니다.

```bash
uv run lerobot-train \
  --policy.path=lerobot/smolvla_base \
  --dataset.repo_id=<repo_id> --dataset.root=<로컬경로> \
  --rename_map='{"observation.images.<cam_a>": "observation.images.camera1", "observation.images.<cam_b>": "observation.images.camera2"}' \
  --policy.empty_cameras=1 \
  --batch_size=64 --steps=20000 \
  --output_dir=outputs/train/my_smolvla \
  --policy.device=cuda --policy.push_to_hub=false
```

`empty_cameras` 값은 3에서 실제 카메라 수를 뺀 값입니다.

---

## 4. 수집 기준 (권장)

| 항목 | 권장값 / 기준 |
|---|---|
| fps | 30 (SmolVLA의 50스텝 chunk는 약 1.7초 분량) |
| 에피소드 수 | task당 최소 50개 |
| 카메라 구성 | 위에서 보는 시점(전역) + 손목 카메라 |
| 카메라 위치 | 모든 에피소드에서 고정. 바뀌면 일반화 성능이 떨어짐 |
| 동기화 | 같은 프레임의 state, action, 이미지는 같은 시점에 기록 |

**시연 품질**
- 정책은 **시연과 얼마나 비슷한지만** 학습합니다. 보상이나 성공 여부는 loss에 들어가지 않습니다. 망설임, 실패, 불필요한 동작도 그대로 따라 하므로, 실패한 에피소드는 저장하지 않거나 나중에 삭제합니다.
- 같은 상황에서는 같은 전략을 씁니다. 왼쪽으로 돌기와 오른쪽으로 돌기가 섞이면 ACT 같은 회귀 모델은 그 평균을 출력할 수 있습니다.
- 물체의 위치나 자세, 조명, 배경은 실제로 쓸 범위 안에서 다양하게 바꿔 줍니다.
- 조금 벗어난 상태에서 복구하는 시연을 일부 넣어 두면 실행 중 누적 오차에 더 강해집니다.

**물리적 타당성**
- 시연은 실제 로봇이나 정상적인 텔레옵으로 수집하므로, 불가능한 자세는 데이터에 들어가지 않습니다.
- 하지만 정책이 출력하는 값이 항상 타당하다는 보장은 없습니다. 자기충돌, 관절 한계 같은 검사는 학습이 아니라 **실행 시점의 안전 필터**에서 처리합니다.
  - LeRobot이 기본으로 제공하는 것: 관절별 범위 clip, `max_relative_target`
  - LeRobot이 제공하지 않는 것: 자기충돌 검사

---

## 5. 생성 코드 예시

```python
from lerobot.datasets.lerobot_dataset import LeRobotDataset

ds = LeRobotDataset.create(
    repo_id="me/my_humanoid",
    root="/data/my_humanoid",
    fps=30,
    robot_type="my_humanoid",
    features={
        "observation.state": {"dtype": "float32", "shape": (N_STATE,), "names": STATE_NAMES},
        "action":            {"dtype": "float32", "shape": (N_ACT,),   "names": ACTION_NAMES},
        "observation.images.head":  {"dtype": "video", "shape": (480, 640, 3),
                                     "names": ["height", "width", "channels"]},
        "observation.images.wrist": {"dtype": "video", "shape": (480, 640, 3),
                                     "names": ["height", "width", "channels"]},
    },
)

for episode in raw_episodes:
    for step in episode:
        ds.add_frame({
            "observation.state": step.measured_q,        # 측정값
            "action": step.commanded_q,                  # 목표값
            "observation.images.head": step.head_rgb,    # HxWx3 uint8
            "observation.images.wrist": step.wrist_rgb,
            "task": episode.instruction,
        })
    ds.save_episode()
ds.finalize()
```

---

## 6. 수집 후 체크리스트

- [ ] `meta/info.json`의 fps, features, shape가 의도한 값과 같은가
- [ ] `meta/stats.json`이 있고, action/state 값의 범위가 정상인가
- [ ] `lerobot-dataset-viz`로 몇 개 에피소드를 열어 영상과 관절 궤적이 맞는지 확인했는가
- [ ] action과 state가 같은 값이 아닌가 (action이 state보다 시간상 앞서 있어야 함)
- [ ] 실패하거나 중단된 에피소드를 제거했는가 (`lerobot-edit-dataset`)
- [ ] task 문장의 오타나 표기가 통일되어 있는가
- [ ] (SmolVLA) state/action 차원이 32 이하이고 카메라가 3개 이하인가
- [ ] 5~10스텝 짜리 짧은 학습을 돌려 오류 없이 loss가 나오는가
