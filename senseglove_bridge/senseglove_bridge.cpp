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

#include <chrono>
#include <cstdint>
#include <iostream>
#include <sstream>
#include <string>
#include <thread>

#include <SenseGlove/Core/HandLayer.hpp>
#include <SenseGlove/Core/HandPose.hpp>
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

bool EnsureSenseComRunning() {
    if (SenseCom::ScanningActive()) {
        return true;
    }
    std::cout << "SenseCom not running yet -- starting it..." << std::endl;
    if (!SenseCom::StartupSenseCom()) {
        std::cerr << "Could not start SenseCom. Make sure it has been run "
                     "at least once (or start it manually) and try again."
                  << std::endl;
        return false;
    }
    for (int i = 0; i < 100 && !SenseCom::ScanningActive(); i++) {
        std::this_thread::sleep_for(std::chrono::milliseconds(100));
    }
    return SenseCom::ScanningActive();
}

}  // namespace

int main() {
    if (!EnsureSenseComRunning()) {
        return 1;
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

    SOCKET clientSock = accept(listenSock, nullptr, nullptr);
    if (clientSock == INVALID_SOCKET) {
        std::cerr << "accept() failed: " << WSAGetLastError() << std::endl;
        closesocket(listenSock);
        WSACleanup();
        return 1;
    }
    std::cout << "Client connected. Streaming HandPose at " << kUpdateHz
              << " Hz. Ctrl+C to stop." << std::endl;

    const auto period = std::chrono::duration<double>(1.0 / kUpdateHz);
    bool clientAlive = true;

    while (clientAlive) {
        auto tickStart = std::chrono::steady_clock::now();

        for (bool rightHand : {false, true}) {
            if (!HandLayer::DeviceConnected(rightHand)) {
                continue;
            }

            HandPose pose;
            if (!HandLayer::GetHandPose(rightHand, pose)) {
                continue;
            }

            std::string line = HandPoseToJson(rightHand, pose) + "\n";
            int sent = send(clientSock, line.c_str(), static_cast<int>(line.size()), 0);
            if (sent == SOCKET_ERROR) {
                std::cerr << "send() failed (client likely disconnected): "
                          << WSAGetLastError() << std::endl;
                clientAlive = false;
                break;
            }
        }

        auto elapsed = std::chrono::steady_clock::now() - tickStart;
        auto remaining = period - elapsed;
        if (remaining > std::chrono::duration<double>(0)) {
            std::this_thread::sleep_for(remaining);
        }
    }

    closesocket(clientSock);
    closesocket(listenSock);
    WSACleanup();
    return 0;
}
