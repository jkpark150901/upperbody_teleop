# CloudXR + Meta Quest 3 설치 및 서버 실행 가이드 (Windows)

## 목표

Meta Quest 3에 별도 APK를 설치하지 않고, Quest Browser의 CloudXR.js/WebXR를 이용해 Windows PC의 CloudXR Runtime에 연결한다.

구성:

```text
Meta Quest 3
  └─ Meta Quest Browser
       └─ CloudXR.js / WebXR
            ↓ WebRTC
Windows PC
  ├─ CloudXR Runtime
  ├─ cloudxr-lovr-sample
  └─ OpenXR application
```

이 구성에서는 Quest Developer Mode / ADB / APK sideload가 필요하지 않다.

---

## 1. 필요한 것

### 하드웨어

- Meta Quest 3
- NVIDIA GPU가 장착된 Windows PC
- Quest와 PC가 같은 LAN/Wi-Fi에 연결
- 가능하면 Wi-Fi 6 이상 권장

### 소프트웨어

- Git
- CMake 3.10 이상
- Visual Studio 2022 이상
  - `Desktop development with C++`
- Node.js 20.19.0 이상
- npm
- NVIDIA GPU Driver
- CloudXR Runtime
- NVIDIA `cloudxr-lovr-sample`

확인:

```powershell
git --version
cmake --version
node --version
npm --version
nvidia-smi
```

Node.js는 최소 `v20.19.0` 이상이어야 한다.

---

## 2. 설치할 저장소

직접 clone할 저장소는 기본적으로 하나만 있으면 된다.

```powershell
cd C:\Users\admin
git clone https://github.com/NVIDIA/cloudxr-lovr-sample.git
cd cloudxr-lovr-sample
```

`cloudxr-lovr-sample`의 `build.bat`가 빌드 과정에서 아래 저장소를 자동으로 clone한다.

```text
NVIDIA/cloudxr-js-samples
```

따라서 `cloudxr-js-samples`를 별도로 clone할 필요는 없다.

---

## 3. CloudXR Runtime 다운로드

NGC CLI가 설치되어 있다면:

```powershell
ngc registry resource download-version nvidia/cloudxr-runtime:6.2.3
```

예:

```text
C:\Users\admin\cloudxr-runtime_v6.2.3
```

안에 다음 파일이 있어야 한다.

```text
CloudXR-6.2.3-Win64-sdk.zip
```

확인:

```powershell
Get-ChildItem C:\Users\admin\cloudxr-runtime_v6.2.3 -Recurse
```

---

## 4. Runtime 버전 지정

현재 `cloudxr-lovr-sample`의 기본값은 CloudXR Runtime 6.2.0 / CloudXR.js 6.2.0이다.

이미 Runtime 6.2.3을 받았다면 PowerShell에서:

```powershell
$env:CLOUDXR_RUNTIME_VERSION="6.2.3"
$env:CLOUDXR_JS_VERSION="6.2.0"
```

로 지정한다.

Runtime zip을 샘플의 캐시 폴더에 직접 넣어도 된다.

```powershell
cd C:\Users\admin\cloudxr-lovr-sample

New-Item -ItemType Directory -Force .\build\cloudxr

Copy-Item `
  C:\Users\admin\cloudxr-runtime_v6.2.3\CloudXR-6.2.3-Win64-sdk.zip `
  .\build\cloudxr\
```

---

## 5. 빌드

**관리자 권한 PowerShell/CMD에서 실행하지 않는 것을 권장한다.**

일반 PowerShell:

```powershell
cd C:\Users\admin\cloudxr-lovr-sample
.\build.bat
```

빌드 스크립트가 자동으로 수행하는 작업:

```text
CloudXR Runtime SDK 확인/압축 해제
↓
LÖVR clone
↓
CloudXR plugin 통합
↓
CMake configure/build
↓
CloudXR.js package 준비
↓
cloudxr-js-samples clone
↓
npm install
↓
React sample build
```

성공하면 `build\` 아래에 서버 실행 파일과 CloudXR.js 클라이언트가 준비된다.

---

## 6. 현재 발생 가능한 React 의존성 충돌

다음과 같은 오류가 발생할 수 있다.

```text
npm error ERESOLVE unable to resolve dependency tree

Found: react@19.3.0

Could not resolve dependency:
peer react >=19 <19.3
from @react-three/fiber
```

이 경우:

```powershell
cd C:\Users\admin\cloudxr-lovr-sample\build\cloudxr\cloudxr-js-samples\react
```

`package.json`에서:

```json
"react": "^19.2.5",
"react-dom": "^19.2.5"
```

를 다음처럼 정확한 버전으로 고정한다.

```json
"react": "19.2.5",
"react-dom": "19.2.5"
```

PowerShell로 한 번에 수정:

```powershell
(Get-Content package.json) `
-replace '"react": "\^19\.2\.5"', '"react": "19.2.5"' `
-replace '"react-dom": "\^19\.2\.5"', '"react-dom": "19.2.5"' |
Set-Content package.json
```

기존 설치 흔적 삭제:

```powershell
Remove-Item -Recurse -Force node_modules -ErrorAction SilentlyContinue
Remove-Item package-lock.json -ErrorAction SilentlyContinue
```

그 후 샘플 루트로 돌아가 재빌드:

```powershell
cd C:\Users\admin\cloudxr-lovr-sample
.\build.bat
```

---

## 7. CloudXR 서버 실행

Quest Browser용 WebRTC 프로파일로 실행:

```powershell
cd C:\Users\admin\cloudxr-lovr-sample
.\run.bat "--device-profile=auto-webrtc"
```

중요:

```text
--device-profile=auto-webrtc
```

를 사용해야 Quest Browser용 CloudXR.js dev server가 함께 실행된다.

Windows에서는 CloudXR.js dev server가 별도 CMD 창으로 열린다.

기본 웹 서버 포트:

```text
8080
```

기본 WebRTC signaling 포트:

```text
49100
```

---

## 8. Windows IP 확인

```powershell
ipconfig
```

Quest와 같은 네트워크 인터페이스의 IPv4 주소를 확인한다.

예:

```text
IPv4 Address . . . . . . . . . . : 192.168.0.15
```

그러면 Quest에서 접속할 주소는:

```text
http://192.168.0.15:8080/
```

---

## 9. Quest Browser 설정

Quest 3에서 기본 Meta Quest Browser를 연다.

HTTP 환경에서는 WebXR가 insecure origin을 막을 수 있으므로:

```text
chrome://flags
```

접속 후 다음 항목을 찾는다.

```text
Insecure origins treated as secure
```

또는:

```text
unsafely-treat-insecure-origin-as-secure
```

값에 PC 주소를 추가:

```text
http://192.168.0.15:8080
```

Enabled로 바꾸고 브라우저를 재시작한다.

그 다음:

```text
http://192.168.0.15:8080/
```

접속한다.

페이지가 뜨면 **Connect**를 누른다.

---

## 10. 정상 동작 확인

서버 콘솔에 다음과 비슷한 로그가 나오면 CloudXR Runtime 초기화가 성공한 것이다.

```text
NVIDIA CloudXR plugin loaded successfully
Created CloudXR Service
CloudXR Runtime initialized successfully
CloudXR device profile: auto-webrtc
```

Quest에서 연결되면:

```text
Quest Browser
    ↓
WebXR tracking
    ↓
CloudXR.js
    ↓
WebRTC
    ↓
CloudXR Runtime
    ↓
OpenXR application
```

경로가 완성된다.

---

## 11. 손 추적 사용

CloudXR Runtime을 통해 Quest의 tracking data가 서버 OpenXR 앱으로 전달된다.

OpenXR 앱에서는 손 추적 extension을 사용한다.

```cpp
XR_EXT_hand_tracking
```

주요 API:

```cpp
xrCreateHandTrackerEXT(...)
xrLocateHandJointsEXT(...)
```

최종 목표 구조:

```text
Quest 3
  └─ hand tracking
       ↓
CloudXR.js
       ↓
CloudXR Runtime
       ↓
XR_EXT_hand_tracking
       ↓
PC retargeter
       ↓
robot joint-space q
       ↓
MuJoCo / Isaac Sim
```

팔 관절 IMU 데이터는 PC에서 Quest hand tracking과 합친 후 로봇 joint space로 리타게팅한다.

---

## 12. Windows 방화벽

처음 실행할 때 Windows Defender Firewall 팝업이 뜨면 Private Network 접근을 허용한다.

최소한 다음 서비스가 막히지 않아야 한다.

```text
TCP 8080   : CloudXR.js dev server
49100      : WebRTC signaling (auto-webrtc 기본값)
```

CloudXR Runtime이 연결되지 않으면 먼저 Windows Firewall을 확인한다.

---

## 13. 자주 발생하는 문제

### `build.bat`을 못 찾음

현재 폴더 확인:

```powershell
pwd
dir *.bat
```

반드시:

```text
...\cloudxr-lovr-sample
```

에서 실행한다.

```powershell
.\build.bat
```

---

### Node.js 버전 오류

```text
ERROR: Node.js v20.19.0+ is required
```

확인:

```powershell
node -v
```

설치:

```powershell
winget install OpenJS.NodeJS.LTS
```

설치 후 PowerShell을 완전히 닫았다 다시 연다.

---

### React dependency `ERESOLVE`

위의 **6. 현재 발생 가능한 React 의존성 충돌** 절차로 React를 19.2.5에 고정한다.

---

### Quest에서 `WebXR not supported`

Quest Browser의:

```text
chrome://flags
```

에서 HTTP 주소를 secure origin으로 등록한다.

---

### `xrEnumerateInstanceExtensionProperties returned -51`

Windows에서 CloudXR 샘플을 **관리자 권한 터미널로 실행했을 때** 발생할 수 있다.

일반 사용자 PowerShell/CMD에서 실행한다.

---

### 8080 포트가 이미 사용 중

확인:

```powershell
netstat -ano | findstr :8080
```

이전 CloudXR.js dev server CMD 창이 남아 있으면 종료한다.

---

## 14. 최소 실행 명령 요약

새 설치 후:

```powershell
cd C:\Users\admin

git clone https://github.com/NVIDIA/cloudxr-lovr-sample.git
cd cloudxr-lovr-sample

$env:CLOUDXR_RUNTIME_VERSION="6.2.3"
$env:CLOUDXR_JS_VERSION="6.2.0"

.\build.bat
```

빌드 완료 후:

```powershell
.\run.bat "--device-profile=auto-webrtc"
```

PC IP 확인:

```powershell
ipconfig
```

Quest Browser:

```text
http://<PC_IP>:8080/
```

---

## 15. 참고

- CloudXR LÖVR sample  
  https://github.com/NVIDIA/cloudxr-lovr-sample

- CloudXR.js samples  
  https://github.com/NVIDIA/cloudxr-js-samples

- NVIDIA CloudXR documentation  
  https://docs.nvidia.com/cloudxr-sdk/latest/

- NVIDIA NGC CloudXR Runtime  
  https://catalog.ngc.nvidia.com/orgs/nvidia/resources/cloudxr-runtime

현재 목적에서는 `cloudxr-lovr-sample`을 먼저 성공적으로 실행해 Quest ↔ Windows CloudXR 연결을 검증한 뒤, 최종적으로 LÖVR 샘플을 자체 OpenXR + retargeting 프로그램으로 교체하면 된다.
