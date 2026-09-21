# 실행 가이드

이 저장소에서 지금까지 만든 도구들의 실행 방법 모음. 순서대로 따라가면
SenseGlove 데이터 수신 확인 → 로봇 손(DG5F) URDF 리타게팅/시각화까지 갈 수 있다.

---

## 0. 사전 준비 (실제 SenseGlove 하드웨어 쓸 때만)

- SenseCom 실행 + 글러브 페어링/캘리브레이션 완료
- `senseglove_bridge.exe` 빌드 (최초 1회, 또는 `senseglove_bridge.cpp` 수정 시):

```powershell
cd senseglove_bridge
& "C:\Program Files (x86)\Microsoft Visual Studio\2022\BuildTools\Common7\Tools\Launch-VsDevShell.ps1" -Arch amd64
cl /EHsc /std:c++17 /MD senseglove_bridge.cpp /I vendor\include /Fe:senseglove_bridge.exe /link Ws2_32.lib vendor\lib_win64_msvc143_release\sgcore.lib
```

## 1. 브릿지 실행 (실제 글러브 쓸 때는 항상 먼저, 별도 창에서 계속 띄워둠)

```powershell
cd senseglove_bridge
.\senseglove_bridge.exe
```

- `listening on 127.0.0.1:8850 ...` 뜨는지 확인 — 안 뜨면 아래 `--backend bridge`/`bridge_throughput_check.py` 전부 `ConnectionRefusedError`.
- 2초마다 `[rate] achieved=... tick_work: avg=...ms max=...ms | left: connected=X/N ... | right: connected=X/N ...` 로그 출력. 브릿지 자체의 실제 루프 속도 + 손별 연결/성공 비율을 보여줌 (다운스트림 파이썬/렌더링과 무관).
- 10초마다 `[diag] device type=... connected=... isRight=...`로 SenseCom이 보는 원시 디바이스 상태도 덤프.

## 2. 브릿지 데이터 처리량만 순수 측정 (렌더링/리타게팅 전혀 안 거침)

```powershell
python scripts\bridge_throughput_check.py --seconds 15
```

- `avg_rate`: 초당 완성된 손 포즈(flexion 5개 + joint_positions 20개 전부 포함한 한 줄) 개수. 관절 하나당이 아니라 **손 전체 스냅샷 기준**.
- `per-hand line counts`, `inter-line gap p50/p99/max`로 어느 손이 얼마나 자주 끊기는지, 순간 정지가 있었는지 확인.
- 이 스크립트가 느리면 브릿지/SGCore/글러브 쪽 문제, 정상인데 아래 4번 vedo만 느리면 렌더링 문제.

## 3. 손가락 데이터 확인용 UI (연결 상태 + flexion 막대그래프)

```powershell
python scripts\senseglove_monitor.py --backend mock      # 가짜 데이터
python scripts\senseglove_monitor.py --backend bridge     # 실제 글러브 (1번 먼저 실행 필요)
```

손별 LIVE(초록)/STALE(주황)/NO DATA(빨강) 상태 + 마지막 수신 후 경과 시간 + 손가락 5개 flexion 값.

## 4. 손 URDF 재추출 (rby1_dg5f.urdf 바뀌었을 때만)

```powershell
python scripts\extract_hand_urdf.py
```

`rby1_dg5f.urdf`(RBY1 상반신 전체)에서 `right_hand_*`/`left_hand_*` 링크·조인트만 뽑아
`robot_hand/urdf/dg5f_{left,right}.urdf` 생성. 손가락 세그먼트 링크마다
자기충돌 검사용 `<collision><capsule/></collision>`도 함께 authored됨.

## 5. SenseGlove → DG5F 손 리타게팅 + URDF 기반 시각화 (관절각 표시)

```powershell
python scripts\hand_retarget_vedo.py --backend mock                    # 가짜 데이터, 양손, mesh 렌더링(기본값)
python scripts\hand_retarget_vedo.py --backend mock --hand right       # 한쪽만
python scripts\hand_retarget_vedo.py --backend mock --render capsule   # 가벼운 capsule 렌더링
python scripts\hand_retarget_vedo.py --backend bridge                  # 실제 글러브 (1번 먼저 실행 필요)
python scripts\hand_retarget_vedo.py --backend bridge --hz 90 --render-hz 30   # poll/retarget은 빠르게, 렌더만 30Hz로 캡
```

- SenseGlove flexion → DG5F 손가락 4-DOF 체인 관절각으로 리타게팅 (계획서 9장 Phase 1).
- `--render mesh`(기본값): 각 링크의 실제 URDF `<visual><mesh>`(repo 루트 `dg5f/meshes/`의 `.dae`, `robot_hand/mesh_loader.py`가 trimesh+pycollada로 로드)를 파트별 원래 재질 색상 그대로 렌더링. `--render capsule`: 각 링크의 URDF `<collision><capsule>` 형상(반지름·길이)만 가볍게 렌더링 — mesh 모드보다 오브젝트가 단순해서 저사양에서 더 빠름.
- 인접 손가락(엄지-검지, 검지-중지, 중지-약지, 약지-소지) 자기충돌을 각 손가락의 굽힘 세그먼트(_2~_4, 손바닥에 고정된 밑동 _1은 항상 정지해 있어 검사 대상에서 제외)의 capsule-capsule 거리로 검사, 충돌 직전까지만 굽히도록 clamp. 클램프가 걸린 손가락은 **빨간색**으로 표시됨 (mesh 모드에서는 클램프가 풀리면 그 링크의 실제 재질 색으로 복원됨).
- capsule 반지름은 `scripts\extract_hand_urdf.py`가 실제 `dg5f/meshes/.../collision/*.STL`에서 계산한 값(그 링크 축 기준 정점 거리의 중앙값) — `rby1_dg5f.urdf`가 바뀌어 재추출할 때마다(4번) 같이 갱신됨.
- `--hz`(poll+리타게팅 틱 속도)와 `--render-hz`(vedo `plotter.render()` 캡, 기본 30)가 분리되어 있음 — `plotter.render()`가 프레임당 제일 비싼 호출이라, poll을 아무리 빠르게 해도 화면은 `--render-hz`로만 갱신됨(눈으로는 30Hz 이상 차이 안 남).
- 콘솔에 2초마다 `poll=.. Hz / render=.. Hz / compute / render_cost` 프레임 타이밍 출력. `poll`은 낮은데 `compute`가 크면 리타게팅 계산 병목, `render_cost`가 크면 렌더링 병목.
- **알려진 이슈**: 렌더링 자체가 vedo/VTK 오브젝트 개별 갱신 오버헤드로 낮게 나올 수 있음 (mock 백엔드로도 동일하게 재현됨 → 브릿지/데이터 문제 아니라 렌더링 문제로 확인됨). `--render-hz`로 낮춰서 완화 가능하지만, 근본적으로 더 빠르게 하려면 손 전체를 배치 오브젝트 1~2개로 합치는 추가 최적화 필요.

## 6. Perception Neuron (IMU 모캡) 손 데이터 실시간 확인

Axis Neuron Pro가 UDP로 BVH 스트림을 보내고 있어야 함 (기본 포트 7002, `mocap7002.txt` 참고). MocapApi는 이 SDK가 넘겨주는 마지막(최신) 프레임만 쓰도록 `disable_event_cache()`를 걸어두고 있음 — 오래된 프레임이 쌓여서 지연이 커지는 걸 SDK 레벨에서 막기 위함.

```powershell
python scripts\udp_raw_probe.py --port 7002                       # MocapApi 없이 순수 UDP 패킷 pkt/s만 확인
python scripts\mocap_joint_plot.py --port 7002                    # 아무 관절이나 골라서 XYZ 확인 (관절 이름 탐색용)
python scripts\mocap_hand_raw_plot.py --port 7002                 # LeftHand/RightHand만, RAW 위치/회전 그래프
python scripts\mocap_hand_raw_plot.py --port 7002 --render-hz 20  # 화면 갱신만 더 낮춰서 비교
```

- `mocap_hand_raw_plot.py`는 리타게팅/URDF/캘리브레이션을 전혀 거치지 않고 SDK가 주는 값을 그대로 그림 — 자세와 무관하게 손 위치/회전이 실시간으로 반영되는지 스트리밍 파이프라인만 검증하는 용도.
- poll(≈90Hz까지 가능)과 화면 갱신(`--render-hz`, 기본 30)이 분리되어 있음 — 캔버스를 매 poll마다 다시 그리면 Tk 이벤트 큐에 redraw가 밀려서 "수신 Hz는 높은데 화면은 계속 지연되는" 현상이 생기므로, 화면은 사람 눈이 구분 못 하는 수준(약 30Hz)으로만 갱신.
- 상태줄에 수신 Hz / frame interval / jitter / batch max 표시. `batch max`가 계속 1보다 크면 poll 호출 사이에 이벤트가 쌓이고 있다는 뜻(백로그 의심).

```powershell
python scripts\upper_body_retarget_vedo.py --backend mock                              # 가짜 데이터로 파이프라인 확인
python scripts\upper_body_retarget_vedo.py --backend mocapapi --port 7002              # 실제 모캡, RBY1 상반신 URDF 시각화
python scripts\upper_body_retarget_vedo.py --backend mocapapi --hz 90 --render-hz 30   # poll/IK는 빠르게, 렌더만 30Hz로 캡
```

- PN 손목(`LeftHand`/`RightHand`) 포함 상반신 랜드마크를 RBY1 URDF 관절각으로 IK 리타게팅해서 vedo로 시각화. 실행 후 차렷 자세로 서 있다가(또는 T-pose면 `--calibration-pose tpose`) 첫 프레임이 캘리브레이션 기준이 됨. `1`/`2`=재캘리브레이션(차렷/T-pose), `R`=반복.
- 여기도 `--hz`(poll+IK)와 `--render-hz`(vedo 렌더 캡)가 분리되어 있음 — IK 계산 자체가 반복적 최적화라 다소 무거우니, 화면이 버벅이면 `--render-hz`를 낮추기 전에 콘솔의 프레임 타이밍부터 확인.

### 6-1. 리타게팅 중 원본 데이터 녹화 + SenseGlove 손 데이터 동시 취득 → 손 포함 URDF 재생

```powershell
# 실제 모캡 + 실제 글러브(브릿지) 동시 사용, 녹화까지
python scripts\upper_body_retarget_vedo.py --backend mocapapi --hand-backend bridge --record recordings\session1.jsonl

# 가짜 데이터로 먼저 흐름만 확인
python scripts\upper_body_retarget_vedo.py --backend mock --hand-backend mock --record recordings\test.jsonl

# 녹화 재생 -- 팔 IK + 손가락(SenseGlove flexion)까지 그 녹화 안에 같이 들어있어서 별도 --hand-backend 없이 재생됨
python scripts\upper_body_retarget_vedo.py --backend replay --replay-file recordings\session1.jsonl
```

- `--hand-backend`(`none`/`mock`/`sgcore`/`bridge`)로 `hand_retarget_vedo.py`와 같은 SenseGlove 리더를 상반신 모캡과 동시에 폴링 — 로봇 팔+손이 rby1_dg5f.urdf 한 화면에서 같이 움직임(손가락 색은 mesh 자체 재질, 자기충돌로 클램프되면 `hand_retarget_vedo.py`와 달리 색은 안 바뀜, 각도만 제한됨).
- `--record PATH`는 이 스크립트 자체의 새 포맷(`CombinedRecorder`, `upper_body_combined_v1`) — 매 프레임 랜드마크 9개 위치 + head/양쪽 손목 회전 + 손 flexion을 한 줄씩 저장. `mocap_skeleton_vedo.py`의 원본 전신 스켈레톤 녹화(`demo.jsonl` 등)와는 다른 파일이고, `--backend replay`가 헤더를 보고 자동으로 구분해서 예전 스켈레톤 녹화도 그대로 재생됨(그 경우만 오른쪽 패널에 원본 스켈레톤 비교가 뜸).
- 재생 중에 다시 `--hand-backend`를 주면 녹화된 손 데이터 대신 그 라이브 소스가 우선 적용됨(팔은 항상 녹화/재생 쪽 기준).

### 6-2. 리타게팅 결과(로봇 관절각)를 원격 서버로 전송

```powershell
# 원격 서버: 수신 확인용 (또는 이 프로토콜을 받는 자체 서버)
python scripts\articulation_receiver.py --proto udp --port 9100

# 이쪽: 관절각 스트리밍
python scripts\upper_body_retarget_vedo.py --backend mocapapi --hand-backend bridge --send udp://192.168.0.20:9100
python scripts\upper_body_retarget_vedo.py --backend mocapapi --send tcp://192.168.0.20:9100 --send-hz 50
```

- 보내는 값: `rby1_dg5f.urdf` 회전 관절 58개(토르소 2 + 팔 7x2 + 머리 2 + 손가락 20x2)의 각도(rad). 프레임에는 값만 순서대로 담기고, 관절 이름 순서는 별도 `layout` 메시지(2초마다 재전송, TCP는 연결 시 1회)로 전달 — 수신 측은 layout id가 일치하는 프레임만 적용하면 됨. 포맷 상세는 `teleop/articulation_protocol.py`.
- UDP는 데이터그램 1개 = 메시지 1개, TCP는 줄바꿈 구분 JSON(이쪽이 클라이언트, 서버가 listen; 서버가 늦게 떠도 1초마다 재접속). 전송은 별도 스레드라 링크가 느리거나 끊겨도 UI/IK 루프는 안 멈추고 오래된 프레임만 버려짐.
- `hands.left/right`가 false면 그쪽 손가락은 글러브 데이터 없이 0(펴짐)으로 채운 값.

## 7. (참고) 그 외 기존 도구

```powershell
python mujoco_teleop\teleop_receiver_viewer.py       # MuJoCo 쪽 (PN+SenseGlove 마커, 로봇 팔 없음)
python scripts\teleop_sender.py --backend mock        # 위 뷰어에 UDP로 보내는 송신측
python scripts\vedo_local_preview.py --backend mock   # PN+SenseGlove 원본 raw 데이터만 (로봇 손 없이)
```

---

## 전형적인 순서

**SenseGlove(손가락) 쪽**
1. (하드웨어 쓸 때) 브릿지 실행 → **2번**으로 데이터 자체가 정상 속도로 오는지 확인
2. **3번**으로 손가락 데이터가 잘 들어오는지 확인
3. **5번**으로 로봇 손에 리타게팅 + 자기충돌 회피가 되는지 확인

**Perception Neuron(IMU 모캡, 손/상반신 위치) 쪽**
1. **6번**의 `udp_raw_probe.py`로 UDP 패킷 자체가 들어오는지 확인
2. **6번**의 `mocap_hand_raw_plot.py`로 손 위치/회전이 자세와 무관하게 실시간 반영되는지 확인
3. 문제없으면 `upper_body_retarget_vedo.py`로 RBY1 URDF 리타게팅까지 확인

## 문제 생겼을 때 어디를 봐야 하는지

| 증상 | 확인할 것 |
|---|---|
| `ConnectionRefusedError` | 브릿지 exe가 안 떠 있음 (1번부터) |
| `senseglove_bridge.exe`가 "SenseCom not running" 반복 | SenseCom 자체가 꺼져있거나 막 재시작됨 |
| 특정 손만 `connected=0`이 계속 나옴 | 그 손 글러브의 실제 BLE 연결 문제 (SenseCom 창에서 직접 확인) |
| `[rate] achieved`가 60Hz보다 많이 낮음, `tick_work`이 큼 | 브릿지 루프 자체 병목 (SGCore 호출 비용) |
| 브릿지/`bridge_throughput_check.py`는 빠른데 vedo만 느림 | 렌더링 병목 (5번의 "알려진 이슈" 참고, `--render-hz`로 완화) |
| Axis Neuron 쪽 fps/Hz는 정상인데 화면이 계속 지연됨 | 렌더 콜이 poll 루프에 묶여 있는지 확인 — `mocap_hand_raw_plot.py`/`upper_body_retarget_vedo.py`/`hand_retarget_vedo.py`는 이미 `--render-hz`로 분리해둠 |
| `mocap_hand_raw_plot.py` 상태줄의 `batch max`가 계속 큼 | poll 호출 사이 이벤트 백로그 — poll 주기를 더 짧게(더 자주 `after`/타이머 호출) |
