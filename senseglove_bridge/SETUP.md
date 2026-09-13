# SenseGlove 장치 PC 설정 가이드

`senseglove_bridge.exe`를 빌드해서 SenseGlove 데이터를
[`SenseGloveJSONBridgeReader`](../devices/senseglove/sg_reader.py)로
흘려보내기까지의 전체 절차. 처음부터 순서대로 따라가면 됨.

---

## 0. 준비물

- [ ] PC에 블루투스 있는지 확인 (없으면 USB 블루투스 동글 필요)
- [ ] Visual Studio (C++ 빌드 도구 포함) 설치돼 있는지 확인
- [ ] 이 저장소를 받아서 `senseglove_bridge/vendor/` 폴더가 있는지 확인
      (SDK 헤더+라이브러리가 이미 들어있음, 별도 다운로드 불필요)

---

## 1. 블루투스

SenseGlove Nova / Nova2는 **평상시 동작에 유선 모드가 없음** — USB-C는
충전·펌웨어 업데이트 전용. Nova는 Bluetooth Classic(SPP), Nova2 v2.x
펌웨어는 BLE를 사용.

PC에 블루투스가 없으면 USB 블루투스 동글 하나 꽂으면 됨. SenseGlove
공식 문서엔 특정 모델 추천이 없음 — Bluetooth 4.0 이상 지원하고
Windows 10/11 정식 드라이버가 있는 제품이면 대체로 무난함.

---

## 2. SenseCom 설치

SenseCom은 글러브를 찾아서 연결해주는 백그라운드 서비스. 이게 없으면
아무것도 안 됨.

1. https://github.com/Adjuvo/SenseCom/releases 에서 최신 릴리스 열기
2. `Win/SenseCom_install_<버전>.exe` 다운로드해서 실행, 설치 마법사 따라가기
3. 설치 중 Visual C++ 재배포 패키지 관련 실패 메시지가 뜰 수 있는데,
   이미 설치돼 있으면 무시해도 됨

> **참고**: 인스톨러 대신 GitHub 저장소 소스 zip을 받아서
> `Win/SenseCom_Win_Latest/SenseCom.exe`를 압축만 풀어서 바로 실행하는
>것도 됨 (테스트 시 이 방법 사용함). 단, Unity 앱이라 `SenseCom.exe`
> 옆에 **`SenseCom_Data` 폴더가 통째로 같이 있어야 함** — `.exe`만
> 따로 복사하면 "Application folder ... There should be 'SenseCom_Data'
> folder next to the executable" 에러가 남. 압축 풀 때 와일드카드로
> 부분만 풀면(`unzip "Win/SenseCom_Win_Latest/*"` 등) 하위 폴더가
> 통째로 빠질 수 있으니 전체를 다 풀어야 함. 이 방법은 트레이 아이콘도
> 없고 자동 시작 등록도 안 되므로, 계속 쓸 거면 정식 인스톨러 권장.

---

## 3. 글러브 페어링

1. SenseCom 실행 (설치했으면 트레이 아이콘에서, 아니면 `SenseCom.exe` 직접 실행)
2. Windows 설정 → 블루투스 및 기타 디바이스 → 디바이스 추가에서 글러브 페어링
   (글러브 자체 이름으로 목록에 나타남)
3. 목록에 안 보이면: 블루투스 디바이스 검색 모드를 **"Basic"에서
   "Advanced"로** 변경 후 재시도
4. SenseCom 창에서 글러브가 연결된 상태로 표시되는지 확인

---

## 4. `senseglove_bridge.exe` 빌드

SDK는 이미 `vendor/`에 들어있어서 추가 다운로드 필요 없음.

일반 PowerShell 창에서 (VS가 설치된 경로/버전에 맞게 경로 수정):

```powershell
& "C:\Program Files\Microsoft Visual Studio\<버전>\<에디션>\Common7\Tools\Launch-VsDevShell.ps1" -Arch amd64
cd <repo>\senseglove_bridge
cl /EHsc /std:c++17 /MD senseglove_bridge.cpp /I vendor\include /Fe:senseglove_bridge.exe /link Ws2_32.lib vendor\lib_win64_msvc143_release\sgcore.lib
copy vendor\lib_win64_msvc143_release\sgcore.dll .
copy vendor\lib_win64_msvc143_release\sgconnect.dll .
```

`Launch-VsDevShell.ps1`은 그 PowerShell 창에서 `cl`을 쓸 수 있게
환경변수를 잡아주는 스크립트 — 창을 새로 열 때마다 다시 실행해야 함.
아니면 시작 메뉴에서 "x64 Native Tools Command Prompt for VS"로 열면
처음부터 `cl`이 바로 됨.

### 빌드 중 겪을 수 있는 에러

| 에러 | 원인 / 해결 |
|---|---|
| `'cl' 용어가 인식되지 않습니다` | VS 개발자 환경이 로드 안 됨 → 위 `Launch-VsDevShell.ps1` 먼저 실행 |
| `use of undefined type 'SGCore::Kinematics::Vect3D'` | `HandPose.hpp`는 `Vect3D`를 전방 선언만 함 → `#include <SenseGlove/Core/Vect3D.hpp>` 필요 (이미 소스에 포함돼 있음) |
| `LNK2038 mismatch detected for 'RuntimeLibrary'` | 벤더 `.lib`은 동적 CRT(`/MD`)로 빌드됨 → 컴파일 시 `/MD` 플래그 필수 |
| `output filename matches input filename` | `/Fe`와 파일명 사이에 공백 있으면 안 됨 → `/Fe:이름.exe` (콜론) 형식으로 |

---

## 5. 실행 확인

1. SenseCom이 켜져 있고 글러브가 페어링된 상태에서:

   ```powershell
   .\senseglove_bridge.exe
   ```

   `listening on 127.0.0.1:8850 -- waiting for the Python side ...`가
   뜨면 정상. `SenseCom not running yet...` 에러가 뜨면 2~3단계로
   돌아가서 SenseCom/페어링 확인.

2. 다른 창에서 (저장소 루트, 가상환경 활성화):

   ```powershell
   python -m devices.senseglove.sg_reader --backend bridge
   ```

   `senseglove_bridge.exe` 쪽에 `Client connected` 뜨고, Python 쪽에
   flexion 값이 출력되면서 손가락 움직임에 따라 값이 변하는지 확인.

---

## 6. 전체 파이프라인에 연결

여기까지 되면 전체 teleop 파이프라인에 꽂는다:

```powershell
# 이 PC (장치 PC)
python scripts\teleop_sender.py --sg-backend bridge --pn-backend mock ...
```

```powershell
# MuJoCo PC
python mujoco_teleop\teleop_receiver_viewer.py
```

옵션/설정은 최상위 `README.md` 참고.

---

## 참고: JSON 와이어 포맷

`senseglove_bridge.exe`가 한 줄에 하나씩(손마다, ~60Hz) 내보내는 형식:

```json
{"hand": "left" | "right",
 "flexion": [thumb, index, middle, ring, pinky],
 "joint_positions": [[[x,y,z], ...4 joints...], ...5 fingers...]}
```

- `flexion`: 0(펌)~1(오므림)
- `joint_positions`: 미터 단위, 손목 기준 상대 좌표
