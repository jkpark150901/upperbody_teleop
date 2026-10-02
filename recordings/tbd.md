# lerobot_ds 수정 사항

이 문서는 `lerobot_ds`를 LeRobotDataset v3.0 형식으로 맞추기 위해 고쳐야 할 항목만 정리한 것입니다. 비교 기준은 공식 데이터셋 `lerobot/svla_so101_pickplace`이고, 실제 로딩 테스트 결과를 바탕으로 했습니다.

**현재 상태**: `LeRobotDataset(..., root="lerobot_ds")`로 불러오면 다음 오류로 실패합니다.

```
TypeError: DatasetInfo.__init__() missing 1 required positional argument: 'codebase_version'
```

**권장 해결 방법**: 메타데이터를 손으로 고치기보다, 원본 데이터를 `LeRobotDataset.create(use_videos=False)` → `add_frame` → `save_episode` → `finalize`로 다시 생성합니다. 이렇게 하면 아래 1~5번이 모두 자동으로 처리됩니다.

---

## 1. `meta/info.json`

| 항목 | 현재 | 수정 |
|---|---|---|
| `codebase_version` | 없음 | `"v3.0"` |
| `fps` | `30.0` (float) | `30` (int) |
| `total_tasks` | 없음 | `1` |
| `chunks_size` | 없음 | `1000` |
| `data_files_size_in_mb` | 없음 | `100` |
| `video_files_size_in_mb` | 없음 | `200` |
| `data_path` | 없음 | `"data/chunk-{chunk_index:03d}/file-{file_index:03d}.parquet"` |
| `video_path` | 없음 | `null` (image 방식) |
| `splits` | 없음 | `{"train": "0:2"}` |
| features: `timestamp`, `frame_index`, `episode_index`, `index`, `task_index` | 없음 | 추가 (아래 예시) |
| features: 카메라 3개의 `dtype` | `"video"` | `"image"` |
| features: 카메라 3개의 `shape` | `[null, null, 3]` | 실제 해상도. 예: `[480, 640, 3]` |
| features: 카메라의 `note` | 있음 | 삭제 |

추가해야 할 기본 feature는 다음과 같습니다.

```json
"timestamp":     {"dtype": "float32", "shape": [1], "names": null},
"frame_index":   {"dtype": "int64",   "shape": [1], "names": null},
"episode_index": {"dtype": "int64",   "shape": [1], "names": null},
"index":         {"dtype": "int64",   "shape": [1], "names": null},
"task_index":    {"dtype": "int64",   "shape": [1], "names": null}
```

## 2. `data/chunk-000/file-000.parquet`: 이미지 컬럼

- 지금은 카메라 컬럼이 아예 없습니다.
- `observation.images.cam_right_wrist`, `observation.images.cam_left_wrist`, `observation.images.cam_head` 세 컬럼을 추가하고, 프레임마다 PNG 이미지를 넣습니다.
- 그 밖의 컬럼(state, action, timestamp, index 계열)은 현재 상태 그대로 정상입니다.

## 3. `meta/episodes/chunk-000/file-000.parquet`

현재는 `episode_index`, `length`, `task_index` 세 컬럼만 있습니다. 아래 컬럼이 필요합니다.

| 컬럼 | 내용 | ep 0 | ep 1 |
|---|---|---|---|
| `episode_index` | 에피소드 번호 | 0 | 1 |
| `tasks` | task 문자열 **리스트** | `["pick up the cloth"]` | `["pick up the cloth"]` |
| `length` | 프레임 수 | 5 | 6 |
| `data/chunk_index`, `data/file_index` | 데이터 파일 위치 | 0, 0 | 0, 0 |
| `dataset_from_index`, `dataset_to_index` | 전체 기준 프레임 범위 (끝은 미포함) | 0, 5 | 5, 11 |
| `meta/episodes/chunk_index`, `meta/episodes/file_index` | 이 메타 파일의 위치 | 0, 0 | 0, 0 |
| `stats/<feature>/{min,max,mean,std,count}` | 에피소드별 통계 | | |

`task_index` 컬럼은 이 파일에 필요 없습니다.

## 4. `meta/tasks.parquet`

- **현재**: `task_index`와 `task`가 둘 다 일반 컬럼입니다.
- **수정**: task 문자열을 DataFrame의 **index**로 두고, 컬럼은 `task_index` 하나만 둡니다.

```python
pd.DataFrame({"task_index": [0]}, index=pd.Index(["pick up the cloth"], name="task")).to_parquet(path)
```

## 5. `meta/stats.json`

- `observation.state`와 `action`에 `count`를 추가합니다.
- 카메라 3개, `timestamp`, `frame_index`, `episode_index`, `index`, `task_index`에도 `min`, `max`, `mean`, `std`, `count` 항목을 추가합니다.
- 이미지 통계는 채널별 `(3, 1, 1)` 형태이고, 값은 0~1 범위로 계산합니다.

---

## 학습 시 주의 (SmolVLA)

- state와 action이 58차원이라 SmolVLA의 기본 상한 32를 넘습니다.
- `--policy.max_state_dim=58 --policy.max_action_dim=58`로 늘려야 합니다.
- 이 경우 `smolvla_base`의 state/action projection 층은 사전학습 가중치를 그대로 쓰지 못할 가능성이 큽니다. 아직 검증하지 않았습니다.
- 다른 방법은 손가락 관절을 축약해 32차원 이하로 줄이는 것입니다.

## 수정 후 확인

```bash
uv run python -c "
from lerobot.datasets.lerobot_dataset import LeRobotDataset
ds = LeRobotDataset('local/lerobot_ds', root='lerobot_ds')
print(ds); print({k: getattr(v, 'shape', v) for k, v in ds[0].items()})
"
```
