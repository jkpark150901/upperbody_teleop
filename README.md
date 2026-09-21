# Upperbody Teleop

Perception Neuron(전신 모캡) + SenseGlove(손가락) → RBY1 상반신 + DG5F 손 URDF 리타게팅
→ (녹화 / 재생 / 원격 서버 전송). 실행 명령 모음은 [RUN_GUIDE.md](RUN_GUIDE.md), 이 문서는
전체 구조와 "무엇을 어떻게 쓰는지"를 한 곳에 정리한 것.

```
Axis Studio(BVH/UDP 7002) ─▶ MocapApi ──┐
                                        ├─▶ upper_body_retarget_vedo.py ─▶ vedo 시각화 (rby1_dg5f.urdf)
SenseGlove ─▶ senseglove_bridge.exe ────┘        │  ├─▶ --record  녹화(.jsonl)
        (TCP 8850)                                │  └─▶ --send    관절각 UDP/TCP 전송 ─▶ 원격 서버
                                                  └─ 팔 3-1-3 닫힌형 IK + 손가락 flexion 매핑
```

## 설치

```
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

- 실기: MocapApi(`MocapApi/` 번들) + Axis Studio 스트리밍(UDP, 기본 포트 7002), SenseGlove는 SenseCom +
  `senseglove_bridge/` 빌드 (아래 "실기 준비" / [RUN_GUIDE.md](RUN_GUIDE.md) §0-1).
- 하드웨어 없이도 `--backend mock`, `--backend replay`로 전부 돌려볼 수 있음.
- Windows에서 이 저장소는 `.venv\Scripts\python.exe`로 실행할 것 (가상환경 활성화가 안 된 셸의 `python`은
  Microsoft Store 스텁일 수 있음).

## 시각화만 하기 (전송/녹화 없이)

```powershell
# 1) 리타게팅 결과 (RBY1 + DG5F 손), 가짜 데이터 — 하드웨어 필요 없음
python scripts\upper_body_retarget_vedo.py --backend mock --hand-backend mock

# 2) 저장된 녹화 재생
python scripts\upper_body_retarget_vedo.py --backend replay --replay-file recordings\session1.jsonl     # 통합 녹화(팔+손가락)
python scripts\upper_body_retarget_vedo.py --backend replay --replay-file demo.jsonl --show-skeleton      # 전신 스켈레톤 녹화 + 옆에 원본 스켈레톤

# 3) 실시간 모캡 (손 없이) / 모캡 + 글러브
python scripts\upper_body_retarget_vedo.py --backend mocapapi --port 7002
python scripts\upper_body_retarget_vedo.py --backend mocapapi --hand-backend bridge --show-skeleton

# 4) 리타게팅 없이 원본 데이터만
python scripts\mocap_skeleton_vedo.py --port 7002                 # 전신 raw 스켈레톤 (--backend mock|replay 도 가능)
python scripts\mocap_hand_raw_plot.py --port 7002                 # LeftHand/RightHand 위치·회전 raw 그래프 (Tk)
python scripts\hand_retarget_vedo.py --backend mock               # SenseGlove → DG5F 손만 (팔 없음), --backend bridge 로 실기

# 5) 팔 IK 단독 테스트 (손목 위치만 타겟)
python scripts\wrist_ik_test_vedo.py --backend synthetic
```

`--show-skeleton`은 오른쪽에 원본 스켈레톤 패널을 따로 띄움(카메라는 패널별로 독립 조작). 실시간(`mocapapi`)과
전신 스켈레톤 포맷 녹화에서만 동작하고, 통합 녹화·mock에는 원본 스켈레톤이 없어서 경고 후 무시됨.

## 실기 사용 흐름

1. Axis Studio에서 포즈 캘리브레이션 + UDP 스트리밍 켜기 (포즈 보정은 여기서 하는 게 전제).
2. (글러브 쓸 때) SenseCom 실행 → `senseglove_bridge\senseglove_bridge.exe` 를 별도 창에 상시 실행
   (`listening on 127.0.0.1:8850` 확인). 안 떠 있으면 `--hand-backend bridge`는 `ConnectionRefusedError`.
3. `upper_body_retarget_vedo.py --backend mocapapi ...` 실행. 첫 프레임에서 `--calibration-pose`
   (기본 차렷) 기준으로 즉시 캘리브레이션 — 처음엔 그 자세로 서 있을 것.
4. 필요하면 다중 자세 캘리브레이션(아래)으로 팔 길이/관절 범위 차이 보정.

### 다중 자세 캘리브레이션 (팔 스케일 보정)

인간과 로봇의 팔 길이·관절 범위가 달라 생기는 오차를 줄이기 위한 옵션. 실행 중 자세를 잡고 **Tk 컨트롤 패널
버튼**(vedo 창 옆에 별도 창으로 뜸) 또는 키로:

| 동작 | 버튼 | 키 |
|---|---|---|
| 지금 프레임을 차렷/T포즈/만세로 버퍼에 저장 | 차렷/T포즈/만세 캡처 | `1` / `2` / `3` |
| 저장된 자세들로 캘리브레이션 확정 (+ BVH 스케일 보정 ON) | 확정 | `C` |
| 버퍼 비우기 | 버퍼 초기화 | `0` |
| 현재 자세로 즉시 단일 재캘리브레이션 | 즉시 단일 자세 재캘리브레이션 | `R` |

- 여러 자세를 함께 Kabsch로 맞추고, 각 팔 구간 길이는 자세들 중 **가장 펴진 값(최댓값)**으로 잡음.
- 만세 = 양팔 수직 위(`left_arm_1=π`, `right_arm_1=-π`, FK로 검증한 값).
- 타이머 자동 진행은 예전에 추적이 깨져서 뺐음 — 키/버튼 즉시 캡처만 지원.

## 녹화 / 재생

```powershell
python scripts\upper_body_retarget_vedo.py --backend mocapapi --hand-backend bridge --record recordings\session1.jsonl
python scripts\upper_body_retarget_vedo.py --backend replay --replay-file recordings\session1.jsonl
```

| 포맷 | 만든 곳 | 내용 | 재생 |
|---|---|---|---|
| `upper_body_combined_v1` | `upper_body_retarget_vedo.py --record` | 랜드마크 9개 위치 + head/양 손목 회전 + 손 flexion | 팔 IK + 손가락 모두 재생 |
| 전신 스켈레톤 | `mocap_skeleton_vedo.py --record` | 모든 관절 위치 (cm) — 회전 없음 | 팔 IK만 (손목 방향 블록은 스킵), `--show-skeleton` 가능 |

`--backend replay`가 헤더를 보고 자동 판별. 파일 끝이 잘려 있어도(프로세스 강제 종료) 깨진 줄은 건너뜀.
모캡 좌표는 **cm** (Hips y≈93) — 리타게팅은 방향 벡터만 써서 무관하지만 직접 다룰 땐 주의.

## 원격 서버로 관절각 전송

```powershell
python scripts\articulation_receiver.py --proto udp --port 9100                       # 원격(또는 로컬 테스트)
python scripts\upper_body_retarget_vedo.py --backend mocapapi --hand-backend bridge --send udp://192.168.0.20:9100
python scripts\upper_body_retarget_vedo.py --backend mocapapi --send tcp://192.168.0.20:9100 --send-hz 50
```

- 보내는 값: `rby1_dg5f.urdf` 회전 관절 **58개**(토르소 2 + 팔 7×2 + 머리 2 + 손가락 20×2), rad.
- 메시지 2종(JSON): `layout`(관절 이름 순서 + crc32 id) 과 `frame`(`seq`, `t_send`, `t_sample`, `layout`, `q[]`,
  `hands{left,right}`). 프레임엔 값만 순서대로 넣어 UDP 데이터그램 하나(~0.5KB)에 들어가게 함.
  `layout`은 UDP는 2초마다 재전송, TCP는 연결 시 1회. 수신 측은 layout id가 일치하는 프레임만 적용.
- UDP: 데이터그램 1개 = 메시지 1개. TCP: 줄바꿈 구분 JSON, 이쪽이 클라이언트(서버가 listen, 늦게 떠도 1초마다 재접속).
- 전송은 별도 스레드 — 링크가 느리거나 끊겨도 UI/IK는 안 멈추고 오래된 프레임만 버림. 속도는 `--send-hz`(기본 30, `--hz` 이하).
- `hands.left/right == false` 면 그 손가락 값은 글러브 데이터 없이 0으로 채운 것.
- 수신 서버 구현은 `scripts/articulation_receiver.py`와 `teleop/articulation_protocol.py`(`decode_message`) 참고.
  이미 정해진 메시지 형식이 있으면 `encode_frame()`만 바꾸면 됨.

## 리타게팅 알고리즘 요약

- **입력**: 랜드마크 9개(hips/chest/head/양 어깨·팔꿈치·손목) 위치 + head·양 손목 **월드 회전**
  (`GetJointGlobalRotation` — `get_local_rotation()`은 부모 기준 상대값이라 팔뚝이 움직이면 손목 델타가 오염됨).
- **캘리브레이션**: chest→head/어깨 벡터를 Kabsch로 인간→로봇 회전에 맞춤. 로봇 팔은 인간 세그먼트의 **방향 벡터 × 로봇 자체 길이**로 목표를 만듦(스케일 프리).
- **팔 IK (3-1-3, 닫힌형)**: 어깨(arm_0/1/2, 축이 한 점에서 교차하는 구면관절) — 위팔 조준 + 어깨-팔꿈치-손목 평면 법선으로
  스윙 결정 후 ZXZ 오일러 분해(축0이 기울어져 있어 고정 좌표변환 후 분해). 팔꿈치(arm_3) 각도는 atan2로 직접 계산. 손목(arm_4/5/6, Z-Y-Z)은
  손목 회전 목표를 ZYZ 분해. 후보(스윙 부호 × 오일러 2해)가 여러 개면 관절 한계 침범이 가장 적은 것, 동률이면 이전 값에 가까운 것.
  이전의 수치(DLS) 방식은 로컬 미니멈에 빠져 폐기. 몸통 고정 시 어깨 위치는 구조적으로 목표와 조금 어긋남(정상), 손목/머리 오차가 실질 지표.
- **손가락**: SenseGlove flexion(0~1)을 DG5F 손가락별 굽힘 관절 3개에 같이 매핑 + 인접 손가락 캡슐 자체충돌 회피(`robot_hand/hand_retarget.py`).
  캡슐은 `robot_hand/urdf/dg5f_{left,right}.urdf`에만 있어서 손 리타게팅은 그 URDF로 계산하고 각도만 통합 URDF FK에 병합(관절 이름 동일).

## 알려진 사실 / 결정 사항

- **URDF `*_hand_mount` 오프셋**: z=-0.1548 → **-0.1087** 로 수정. 원래 값은 손목 링크 메시 끝과 손 마운트 메시 사이에 4.61cm 빈 공간을 남겼음
  (두 메시 정점을 손목 로컬 좌표계에서 비교해 산출). 손 내부 조인트(mount→base→palm)는 원래부터 서로 겹쳐서 그대로.
- **롤/피치/요가 이상해 보이는 건** 센서 부착 위치 때문에 Axis Studio가 내보내는 BVH 자체가 틀어진 것 — 이 코드에서 보정할 문제가 아님.
- **vedo 버튼/슬라이더 사용 불가**: 이 환경 VTK에 `vtkRenderer.AddActor2D`가 없어 `add_button()`이 AttributeError. 그래서 캘리브레이션 GUI는 별도 Tk 창.
- **vedo `actor.vertices = 더_큰_배열`은 렌더 개수를 늘리지 못함** (좌표 버퍼만 교체, 셀 배열은 생성 시 크기 그대로). 실제 최종 크기로 한 번에 만들 것.
- **폴링/렌더 분리**: `poll_next_event()`는 한 tick에 끝까지 비워야 함(1회만 부르면 ~30초 밀리는 백로그 발생 — 실기에서 확인). 렌더는 `--render-hz`로 별도 제한.
- **콘솔 인코딩**: 한국어 Windows(cp949) 콘솔에서 `print()`에 em-dash 같은 특수 문자를 쓰면 크래시. 콘솔 출력은 한글+ASCII만.
- 개발용 `scripts/dev_dry_run.py`: 창 없이 다른 vedo 스크립트의 `main()`을 N tick 돌리고 스크린샷 저장 (`-- ` 뒤 인자는 대상 스크립트로 전달).

## 저장소 구조

```
scripts/
  upper_body_retarget_vedo.py   메인: 모캡(+글러브) → RBY1+DG5F 리타게팅 시각화 / --record / --send / 다중 자세 캘리브레이션 GUI
  articulation_receiver.py      --send 수신 확인용 리시버 (UDP/TCP)
  mocap_skeleton_vedo.py        전신 raw 스켈레톤 뷰어 + 녹화/재생
  mocap_hand_raw_plot.py        손 위치/회전 raw 그래프 (Tk)
  mocap_joint_plot.py / mocap_torso_raw_plot.py / udp_raw_probe.py   raw 스트림 진단
  wrist_ik_test_vedo.py         손목 위치 IK 단독 테스트 (합성/실시간 입력, 입력-달성 비교 차트)
  hand_retarget_vedo.py         SenseGlove → DG5F 손만 (팔 없음)
  senseglove_monitor.py / bridge_throughput_check.py   글러브/브릿지 진단
  extract_hand_urdf.py          rby1_dg5f.urdf → 손 전용 URDF(캡슐 충돌 포함) 재생성
  dev_dry_run.py                창 없이 스모크 테스트 + 스크린샷
robot_hand/
  upper_body_retarget.py        3-1-3 닫힌형 팔 IK + 캘리브레이션 (RBY1UpperBodyRetargeter)
  hand_retarget.py              flexion → DG5F 관절각 + 자체충돌 회피
  urdf_fk.py / mesh_loader.py   URDF 파서·FK (only= 로 필요한 링크만) / .dae 로더
  urdf/dg5f_{left,right}.urdf   손 전용 URDF
teleop/
  articulation_protocol.py      관절각 전송 프로토콜 + UDP/TCP 송신기
  geometry.py                   Pose/쿼터니언
  (udp_protocol.py, teleop_state.py, calibration.py — 아래 레거시 파이프라인용)
devices/                        PN/SenseGlove 리더
senseglove_bridge/              SGCore → JSON/TCP C++ 브릿지
rby1_dg5f.urdf, rby1/, dg5f/    로봇 URDF와 메시
recordings/                     녹화 파일 (session1.jsonl 등)
```

---

# 레거시: MuJoCo UDP 파이프라인 (Phase 0-4)

아래는 URDF 이전 단계에서 만든 원래 파이프라인(PN 손목/가슴 + SenseGlove → UDP → MuJoCo 축 마커)의 문서. 지금의
로봇 URDF 리타게팅과는 별개로 남아 있음.

### Try it now (single machine, mock hardware)

Everything defaults to synthetic mock data, so the full pipeline runs without
any device attached -- run these in two terminals from the project root:

```
python mujoco_teleop/teleop_receiver_viewer.py
python scripts/teleop_sender.py --backend mock
```

The viewer window shows: a chest axis triad (fixed anchor, live orientation),
two wrist target axis triads (red=X, green=Y, blue=Z), and per-hand finger
joint points (yellow=left, cyan=right). Press `r` in the viewer to recenter,
`space` to pause.

`python scripts/preview_axis_markers.py --out preview.png` renders one frame
offscreen to a PNG without needing a live UDP link or a GUI window -- useful
for quickly checking `mujoco_teleop/markers.py` changes.

### Sanity-check device data before touching MuJoCo

Before setting up the UDP link and the second (MuJoCo) machine at all, check
that Perception Neuron + SenseGlove data itself looks right, straight from
the readers, using a lightweight [vedo](https://vedo.embl.es/) window
instead of MuJoCo -- no network, no second machine:

```
python scripts/vedo_local_preview.py --backend mock   # or --backend real once hardware is wired up
```

Same visual language as the MuJoCo viewer (RGB axis triads for chest/wrists,
yellow/cyan points for left/right finger joints), but plotting the raw,
un-retargeted reader output directly -- exactly what `teleop_sender.py`
would send. Close the window to stop. Once this looks right, move on to the
two-machine UDP setup above.

### Run with real hardware

On the device PC (Perception Neuron + SenseGlove attached):

```
python scripts/teleop_sender.py --pn-backend mocapapi --sg-backend bridge --config configs/teleop.yaml
```

(`--backend real` also works as shorthand for `mocapapi` + `bridge`.) PN
and SenseGlove backends are independent flags because they need different
levels of trust -- see below.

On the MuJoCo PC:

```
python mujoco_teleop/teleop_receiver_viewer.py --config configs/teleop.yaml
```

Edit `configs/teleop.yaml`'s `network.target_ip` (device PC -> MuJoCo PC's
real address) and `network.listen_ip`/`listen_port` first.

**Neither MocapApi nor SenseCom/a physical glove is available on the
machine this was written on**, so real hardware itself is still unverified.
The SDKs themselves are no longer guesswork, though:

- **Perception Neuron / `MocapApiPNReader`**: MocapApi ships a real, SWIG-
  generated Python wrapper (confirmed from its public repo), so
  `import MocapApi` is expected to actually work once the SDK is installed
  -- the main risk is the *joint names* (`JOINT_NAME_CHEST` etc. in
  `pn_reader.py`), which depend on your Axis Studio actor template.
- **SenseGlove**: there is **no official Python binding for SGCore**, only
  C++/C#, so `--sg-backend bridge` is the path, not a fallback:
  **`senseglove_bridge/senseglove_bridge.cpp`** streams `HandPose` as JSON
  over a local socket for `SenseGloveJSONBridgeReader` to read. This one is
  no longer "best-effort against docs" -- the real SenseGlove-API SDK
  (v2.102.1) and a real, running SenseCom were both downloaded and used:
  the helper **compiles and links against the real `sgcore.lib`**, and
  when run against actual SenseCom (with no physical glove attached) it
  correctly detects SenseCom, listens on its socket, and correctly reports
  zero connected devices -- then a real `SenseGloveJSONBridgeReader`
  connected to it end-to-end. See `senseglove_bridge/README.md` for exact
  build steps (the SDK is bundled at `senseglove_bridge/vendor/`, MIT
  licensed, so no separate download is needed). The one thing that
  couldn't be tested without hardware is an actual physical glove --
  `flexion`/`joint_positions` values it reports once you have one attached
  are worth a sanity check before trusting them fully.

### Device PC setup

The device PC needs Python (`requirements-device.txt`: `numpy` + `pyyaml`
+ `vedo` for the local sanity-check preview -- no MuJoCo) plus, for
`--backend real`, vendor software that is
**not** installable via pip. None of it is on the machine this code was
written on, so treat the exact download/version as something to confirm on
the actual device PC, not as verified fact:

**Perception Neuron:**
- *Axis Studio* (Noitom) -- the capture/calibration app; must be running and
  streaming (BVH/Calc over UDP/TCP) for anything downstream to receive data.
- *MocapApi SDK* (Noitom) -- provides `MocapApi.py` (a SWIG-generated
  wrapper) plus the native `MocapApi.dll`. Both need to be on `PYTHONPATH`
  (or next to `devices/perception_neuron/pn_reader.py`) so `import MocapApi`
  succeeds. Official repo: https://github.com/pnmocap/MocapApi
- USB/sensor-hub drivers for the Perception Neuron hardware itself (comes
  with the Axis Studio installer).

**SenseGlove:**
- *SenseCom* -- the background service that discovers/connects the gloves;
  must be running before anything else can see any device. Download from
  https://github.com/Adjuvo/SenseCom/releases (the `Win/` installer inside).
- *SenseGlove-API SDK* -- already bundled at `senseglove_bridge/vendor/`
  (v2.102.1, MIT licensed), so building `senseglove_bridge.cpp` needs no
  separate download; see its README for the exact `cl.exe` command. Only
  grab a different version from https://github.com/Adjuvo/SenseGlove-API
  if your glove needs one.

Before trusting real hardware end to end:
- PN: run `python -m devices.perception_neuron.pn_reader --backend mocapapi`
  standalone from the project root and confirm it prints sane chest/wrist
  values -- the docstring lists what to check (mainly joint names) if it
  doesn't.
- SenseGlove: build `senseglove_bridge/senseglove_bridge.exe` against the
  real SDK, run it, and confirm it prints "Client connected" once you point
  `python -m devices.senseglove.sg_reader --backend bridge` at it -- then
  check the flexion values it prints move the way your fingers do.

### Layout

```
devices/
  perception_neuron/   PN reader (mock + MocapApi backends)
  senseglove/           SenseGlove reader (mock + SGCore-guess + JSON-bridge backends)
senseglove_bridge/
  senseglove_bridge.cpp   real C++ helper: SGCore HandPose -> JSON over TCP (device PC)
teleop/
  geometry.py            Pose / quaternion utilities
  teleop_state.py         PN + SenseGlove -> HumanTeleopState aggregation
  udp_protocol.py         JSON-over-UDP wire format
  calibration.py          recenter + wrist retargeting (plan section 7)
robot_hand/
  urdf/dg5f_{left,right}.urdf  DG5F hand only, extracted from rby1_dg5f.urdf
                               (scripts/extract_hand_urdf.py) -- no arm/torso,
                               no mesh files (not present in this repo)
  urdf_fk.py               minimal URDF parser + forward kinematics
  hand_retarget.py          SenseGlove flexion -> DG5F finger joint angles
                             (plan section 9, Phase 1 simple mapping)
mujoco_teleop/
  scene_axis.xml           empty ground-plane scene (no robot body yet)
  markers.py               axis triad / point mjvScene geom helpers
  teleop_receiver_viewer.py  UDP receiver + passive viewer (MuJoCo PC entry point)
scripts/
  teleop_sender.py          PN + SenseGlove -> UDP (device PC entry point)
  vedo_local_preview.py     local raw-data sanity check, no MuJoCo/UDP (device PC)
  senseglove_monitor.py     standalone tkinter UI: per-hand connection status
                             + live finger flexion bars (no MuJoCo/vedo)
  extract_hand_urdf.py      (re)generates robot_hand/urdf/dg5f_{left,right}.urdf
  hand_retarget_vedo.py     SenseGlove -> DG5F hand URDF retargeting, vedo skeleton view
  preview_axis_markers.py   offline single-frame PNG smoke test (MuJoCo side)
configs/teleop.yaml       network + retargeting + backend config
```

### Notes

- Raw PN/SenseGlove data is intentionally **not** sent over UDP (plan
  section 12 wants raw teleop input recorded, but that should happen on the
  sender side directly from the readers once a recorder exists -- keeping it
  off the network keeps packets small and the link easy to debug).
- Recentering/retargeting runs on the receiver (MuJoCo) side, since that's
  where "robot workspace" and the viewer's `r` key naturally live; the
  sender only streams raw torso/wrist/hand data.
- `configs/teleop.yaml`'s `sides.*.marker_origin` stands in for the real
  robot EE pose at calibration time (plan section 7.1) until a URDF exists --
  swap it out once robot arm IK (plan Phase 5) is wired up.
