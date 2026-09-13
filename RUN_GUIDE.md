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
python scripts\hand_retarget_vedo.py --backend mock                    # 가짜 데이터, 양손
python scripts\hand_retarget_vedo.py --backend mock --hand right       # 한쪽만
python scripts\hand_retarget_vedo.py --backend bridge                  # 실제 글러브 (1번 먼저 실행 필요)
```

- SenseGlove flexion → DG5F 손가락 4-DOF 체인 관절각으로 리타게팅 (계획서 9장 Phase 1).
- 렌더링은 각 링크의 실제 URDF `<collision><capsule>` 형상(반지름·길이)을 그대로 사용 — 임의의 스켈레톤이 아니라 URDF 자체를 시각화.
- 인접 손가락(엄지-검지, 검지-중지, 중지-약지, 약지-소지) 자기충돌을 capsule-capsule 거리로 검사, 충돌 직전까지만 굽히도록 clamp. 클램프가 걸린 손가락은 **빨간색**으로 표시됨.
- 콘솔에 2초마다 `achieved Hz / compute / render` 프레임 타이밍 출력.
- **알려진 이슈**: 렌더링 자체가 vedo/VTK 오브젝트 개별 갱신 오버헤드로 ~10Hz 정도밖에 안 나옴 (mock 백엔드로도 동일하게 재현됨 → 브릿지/데이터 문제 아니라 렌더링 문제로 확인됨). 더 빠르게 하려면 손 전체를 배치 오브젝트 1~2개로 합치는 추가 최적화 필요.

## 6. (참고) 그 외 기존 도구

```powershell
python mujoco_teleop\teleop_receiver_viewer.py       # MuJoCo 쪽 (PN+SenseGlove 마커, 로봇 팔 없음)
python scripts\teleop_sender.py --backend mock        # 위 뷰어에 UDP로 보내는 송신측
python scripts\vedo_local_preview.py --backend mock   # PN+SenseGlove 원본 raw 데이터만 (로봇 손 없이)
```

---

## 전형적인 순서

1. (하드웨어 쓸 때) 브릿지 실행 → **2번**으로 데이터 자체가 정상 속도로 오는지 확인
2. **3번**으로 손가락 데이터가 잘 들어오는지 확인
3. **5번**으로 로봇 손에 리타게팅 + 자기충돌 회피가 되는지 확인

## 문제 생겼을 때 어디를 봐야 하는지

| 증상 | 확인할 것 |
|---|---|
| `ConnectionRefusedError` | 브릿지 exe가 안 떠 있음 (1번부터) |
| `senseglove_bridge.exe`가 "SenseCom not running" 반복 | SenseCom 자체가 꺼져있거나 막 재시작됨 |
| 특정 손만 `connected=0`이 계속 나옴 | 그 손 글러브의 실제 BLE 연결 문제 (SenseCom 창에서 직접 확인) |
| `[rate] achieved`가 60Hz보다 많이 낮음, `tick_work`이 큼 | 브릿지 루프 자체 병목 (SGCore 호출 비용) |
| 브릿지/`bridge_throughput_check.py`는 빠른데 vedo만 느림 | 렌더링 병목 (5번의 "알려진 이슈" 참고) |
