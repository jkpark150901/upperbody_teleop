// senseglove_bridge.cpp
//
// Vendor-SDK helper: reads SenseGlove HandPose via the official SGCore
// C++ API (https://github.com/Adjuvo/SenseGlove-API) and streams it as
// JSON lines over a local TCP socket, for
// devices/senseglove/sg_reader.py's SenseGloveJSONBridgeReader to consume.
// This exists because SGCore has no official Python binding -- see that
// file's module docstring.
//
// Every SGCore call below was checked against the SDK's actual public
// headers (HandLayer.hpp, HandPose.hpp, Vect3D.hpp, SenseCom.hpp) on
// https://github.com/Adjuvo/SenseGlove-API, not guessed.
//
// Build (Windows, MSVC "x64 Native Tools Command Prompt"), against a
// checkout of the SenseGlove-API SDK:
//
//   cl /EHsc /std:c++17 senseglove_bridge.cpp ^
//      /I "<SenseGlove-API>\include" ^
//      /link Ws2_32.lib "<SenseGlove-API>\lib\win\x64\SGCoreCpp.lib"
//
// (Exact .lib name/path depends on the SDK version -- check its `lib/`
// folder. `SGConnect.dll` and `SGCoreCpp.dll` from the same SDK must be
// next to the built .exe, or on PATH, at runtime.)
//
// Run SenseCom first (or let this program start it), then run this
// program, then run the Python side:
//   python -m devices.senseglove.sg_reader --backend bridge
// or as part of the full pipeline:
//   python scripts/teleop_sender.py --sg-backend bridge ...
//
// Wire format (one JSON object per line, per hand, per update):
//   {"hand":"left"|"right","flexion":[t,i,m,r,p],"joint_positions":[[[x,y,z]*4]*5]}
//   flexion: 0..1 (SGCore's HandPose::GetNormalizedFlexion, already normalized).
//   joint_positions: meters, relative to the wrist (SGCore reports
//   millimeters via HandPose::GetJointPositions -- converted here).

#include <winsock2.h>
#include <ws2tcpip.h>

#include <algorithm>
#include <chrono>
#include <cstdint>
#include <iostream>
#include <sstream>
#include <string>
#include <thread>

#include <SenseGlove/Core/DeviceList.hpp>
#include <SenseGlove/Core/HandLayer.hpp>
#include <SenseGlove/Core/HandPose.hpp>
#include <SenseGlove/Core/HapticGlove.hpp>
#include <SenseGlove/Core/Library.hpp>
#include <SenseGlove/Core/SenseCom.hpp>
#include <SenseGlove/Core/Vect3D.hpp>  // HandPose.hpp only forward-declares Vect3D

#pragma comment(lib, "Ws2_32.lib")

using namespace SGCore;

namespace {

constexpr int kPort = 8850;
constexpr double kUpdateHz = 60.0;
constexpr double kMmToM = 0.001;

std::string HandPoseToJson(bool rightHand, const HandPose& pose) {
    std::ostringstream ss;
    ss << "{\"hand\":\"" << (rightHand ? "right" : "left") << "\",";

    // GetNormalizedFlexion(bool bClamp01=true) overload returns all 5
    // fingers at once, thumb..pinky, already in [0,1].
    std::vector<float> flexion = pose.GetNormalizedFlexion(true);
    ss << "\"flexion\":[";
    for (size_t i = 0; i < flexion.size(); i++) {
        if (i) ss << ",";
        ss << flexion[i];
    }
    ss << "],";

    // GetJointPositions(): 5 (finger) x 4 (joint incl. fingertip) array of
    // Vect3D, millimeters, relative to the wrist.
    const auto& joints = pose.GetJointPositions();
    ss << "\"joint_positions\":[";
    for (size_t f = 0; f < joints.size(); f++) {
        if (f) ss << ",";
        ss << "[";
        for (size_t j = 0; j < joints[f].size(); j++) {
            if (j) ss << ",";
            const auto& p = joints[f][j];
            ss << "[" << (p.GetX() * kMmToM) << "," << (p.GetY() * kMmToM) << ","
               << (p.GetZ() * kMmToM) << "]";
        }
        ss << "]";
    }
    ss << "]}";
    return ss.str();
}

// Retries forever instead of giving up -- ScanningActive() has been observed
// to report a transient false even while SenseCom.exe is alive and showing a
// device list (e.g. right after SenseCom itself was restarted), so a single
// failed attempt here doesn't mean SenseCom is actually down.
void WaitForSenseCom() {
    int attempt = 0;
    while (!SenseCom::ScanningActive()) {
        if (attempt == 0) {
            std::cout << "SenseCom not running yet -- starting it..." << std::endl;
            if (!SenseCom::StartupSenseCom()) {
                std::cerr << "Could not start SenseCom via StartupSenseCom() "
                             "(it may already be running under a different "
                             "registration) -- will keep polling ScanningActive()."
                          << std::endl;
            }
        }
        if (attempt > 0 && attempt % 25 == 0) {
            std::cout << "[diag] still waiting for SenseCom, attempt=" << attempt << std::endl;
        }
        std::this_thread::sleep_for(std::chrono::milliseconds(200));
        attempt++;
    }
}

}  // namespace

int main() {
    WaitForSenseCom();

    std::cout << "[diag] SGCore::Library::Version()=" << Library::Version()
              << " BackendVersion()=" << Library::BackendVersion()
              << " SGConnectVersion()=" << Library::SGConnectVersion() << std::endl;
    std::cout << "[diag] DeviceList::SenseComRunning()=" << DeviceList::SenseComRunning()
              << " ActiveDevices()=" << DeviceList::ActiveDevices() << std::endl;
    std::cout << "[diag] HandLayer::GlovesConnected()=" << HandLayer::GlovesConnected()
              << " DeviceConnected(left)=" << HandLayer::DeviceConnected(false)
              << " DeviceConnected(right)=" << HandLayer::DeviceConnected(true) << std::endl;
    for (const auto& dev : DeviceList::GetDevices()) {
        std::cout << "[diag] device type=" << SGDevice::ToString(dev->GetDeviceType())
                  << " id=" << dev->GetDeviceId() << " addr=" << dev->GetAddress()
                  << " connected=" << dev->IsConnected();
        auto glove = std::dynamic_pointer_cast<HapticGlove>(dev);
        if (glove) {
            std::cout << " isRight=" << glove->IsRight();
        } else {
            std::cout << " (not castable to HapticGlove)";
        }
        std::cout << std::endl;
    }

    WSADATA wsaData;
    if (WSAStartup(MAKEWORD(2, 2), &wsaData) != 0) {
        std::cerr << "WSAStartup failed" << std::endl;
        return 1;
    }

    SOCKET listenSock = socket(AF_INET, SOCK_STREAM, IPPROTO_TCP);
    if (listenSock == INVALID_SOCKET) {
        std::cerr << "socket() failed: " << WSAGetLastError() << std::endl;
        WSACleanup();
        return 1;
    }

    // Loopback-only by default -- the Python-side reader connects from the
    // same machine. If the MuJoCo teleop_sender.py process ever runs on a
    // different machine than this helper, change INADDR_LOOPBACK and point
    // SenseGloveJSONBridgeReader's --sg-bridge-host at this machine's IP.
    sockaddr_in addr{};
    addr.sin_family = AF_INET;
    addr.sin_addr.s_addr = htonl(INADDR_LOOPBACK);
    addr.sin_port = htons(kPort);

    if (bind(listenSock, reinterpret_cast<sockaddr*>(&addr), sizeof(addr)) == SOCKET_ERROR) {
        std::cerr << "bind() failed: " << WSAGetLastError() << std::endl;
        closesocket(listenSock);
        WSACleanup();
        return 1;
    }
    listen(listenSock, 1);

    std::cout << "senseglove_bridge listening on 127.0.0.1:" << kPort
              << " -- waiting for the Python side (SenseGloveJSONBridgeReader) "
                 "to connect..."
              << std::endl;

    const auto period = std::chrono::duration<double>(1.0 / kUpdateHz);
    int tickCount = 0;

    // Outer loop: accept a client, stream until it disconnects (or a
    // send() fails), then go back to waiting for the next one -- this
    // process is meant to be left running across repeated Python-side
    // reconnects (dev iteration, or the sender restarting) rather than
    // exiting after the first client drops.
    while (true) {
        SOCKET clientSock = INVALID_SOCKET;
        int waitTicks = 0;
        while (clientSock == INVALID_SOCKET) {
            fd_set readfds;
            FD_ZERO(&readfds);
            FD_SET(listenSock, &readfds);
            timeval tv{0, 200000};  // 200ms
            int sel = select(0, &readfds, nullptr, nullptr, &tv);
            if (sel > 0 && FD_ISSET(listenSock, &readfds)) {
                clientSock = accept(listenSock, nullptr, nullptr);
                if (clientSock == INVALID_SOCKET) {
                    std::cerr << "accept() failed: " << WSAGetLastError()
                              << " -- retrying" << std::endl;
                    std::this_thread::sleep_for(std::chrono::milliseconds(200));
                }
                continue;
            }

            // Touch the API so the connection stays warm while idle; discard results.
            HandPose tmpPose;
            HandLayer::GetHandPose(false, tmpPose);
            HandLayer::GetHandPose(true, tmpPose);

            if (waitTicks % 5 == 0) {  // ~1s
                std::cout << "[diag] waiting for client, tick=" << waitTicks
                          << " DeviceConnected(left)=" << HandLayer::DeviceConnected(false)
                          << " DeviceConnected(right)=" << HandLayer::DeviceConnected(true)
                          << std::endl;
            }
            waitTicks++;
        }
        std::cout << "Client connected. Streaming HandPose at " << kUpdateHz
                  << " Hz. Ctrl+C to stop." << std::endl;

        // Wall-clock-gated rate report (every ~2s), NOT tick-count-gated --
        // if the loop itself is running slower than kUpdateHz (e.g. a
        // blocking SGCore call), a tick-count-based gate would silently
        // under-report, hiding exactly the stall this is meant to catch.
        auto reportWindowStart = std::chrono::steady_clock::now();
        auto lastDeviceDump = std::chrono::steady_clock::now();
        int windowTicks = 0;
        int windowConnected[2] = {0, 0};   // [left, right]
        int windowPoseOk[2] = {0, 0};
        int windowSent[2] = {0, 0};
        double windowMaxTickMs = 0.0;
        double windowSumTickMs = 0.0;

        bool clientAlive = true;
        while (clientAlive) {
            auto tickStart = std::chrono::steady_clock::now();

            for (bool rightHand : {false, true}) {
                int idx = rightHand ? 1 : 0;
                // GetHandPose() alone already returns false when the
                // device isn't connected, so a separate DeviceConnected()
                // call first was pure overhead -- doubling the per-hand
                // HandLayer call count for no behavior difference (verified
                // against real hardware: with both calls present,
                // pose_ok/sent tracked connected 1:1 in the [rate] log,
                // meaning DeviceConnected() was never catching anything
                // GetHandPose() itself didn't already reject). Dropping it
                // was the cheapest lever on tick_work (measured avg 26-37ms
                // against a 60Hz/16.7ms budget) without touching behavior.
                HandPose pose;
                if (!HandLayer::GetHandPose(rightHand, pose)) {
                    continue;
                }
                windowConnected[idx]++;
                windowPoseOk[idx]++;

                std::string line = HandPoseToJson(rightHand, pose) + "\n";
                int sent = send(clientSock, line.c_str(), static_cast<int>(line.size()), 0);
                if (sent == SOCKET_ERROR) {
                    std::cerr << "send() failed (client disconnected): "
                              << WSAGetLastError() << " -- waiting for a new client"
                              << std::endl;
                    clientAlive = false;
                    break;
                }
                windowSent[idx]++;
            }

            auto tickWorkElapsed = std::chrono::steady_clock::now() - tickStart;
            double tickWorkMs = std::chrono::duration<double, std::milli>(tickWorkElapsed).count();
            windowMaxTickMs = (std::max)(windowMaxTickMs, tickWorkMs);
            windowSumTickMs += tickWorkMs;
            windowTicks++;
            tickCount++;

            auto now = std::chrono::steady_clock::now();
            double windowS = std::chrono::duration<double>(now - reportWindowStart).count();
            if (windowS >= 2.0) {
                double achievedHz = windowTicks / windowS;
                std::cout << "[rate] achieved=" << achievedHz << " Hz (target=" << kUpdateHz << ") "
                          << "tick_work: avg=" << (windowSumTickMs / windowTicks) << "ms "
                          << "max=" << windowMaxTickMs << "ms | "
                          << "left: connected=" << windowConnected[0] << "/" << windowTicks
                          << " pose_ok=" << windowPoseOk[0] << " sent=" << windowSent[0] << " | "
                          << "right: connected=" << windowConnected[1] << "/" << windowTicks
                          << " pose_ok=" << windowPoseOk[1] << " sent=" << windowSent[1]
                          << std::endl;
                reportWindowStart = now;
                windowTicks = 0;
                windowConnected[0] = windowConnected[1] = 0;
                windowPoseOk[0] = windowPoseOk[1] = 0;
                windowSent[0] = windowSent[1] = 0;
                windowMaxTickMs = 0.0;
                windowSumTickMs = 0.0;
            }
            if (std::chrono::duration<double>(now - lastDeviceDump).count() >= 10.0) {
                for (const auto& dev : DeviceList::GetDevices()) {
                    std::cout << "  [diag] device type=" << SGDevice::ToString(dev->GetDeviceType())
                              << " id=" << dev->GetDeviceId() << " connected=" << dev->IsConnected();
                    auto glove = std::dynamic_pointer_cast<HapticGlove>(dev);
                    if (glove) {
                        std::cout << " isRight=" << glove->IsRight();
                    } else {
                        std::cout << " (not castable to HapticGlove)";
                    }
                    std::cout << std::endl;
                }
                lastDeviceDump = now;
            }

            auto elapsed = std::chrono::steady_clock::now() - tickStart;
            auto remaining = period - elapsed;
            if (remaining > std::chrono::duration<double>(0)) {
                std::this_thread::sleep_for(remaining);
            }
        }

        closesocket(clientSock);
    }

    closesocket(listenSock);
    WSACleanup();
    return 0;
}
