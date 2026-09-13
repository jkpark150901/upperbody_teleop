# senseglove_bridge

Vendor-SDK helper for the device PC: reads SenseGlove `HandPose` via the
official SGCore C++ API and streams it as JSON over a local TCP socket for
`devices/senseglove/sg_reader.py`'s `SenseGloveJSONBridgeReader` to consume
(see that file's module docstring for why -- SGCore has no official Python
binding).

**→ See [SETUP.md](SETUP.md) for the full step-by-step procedure** (Bluetooth,
installing SenseCom, pairing the glove, building, running, troubleshooting
table for every error hit while building this).

## Status: built and run against the real SDK, just not real gloves

`vendor/` bundles the actual SenseGlove-API SDK v2.102.1
(https://github.com/Adjuvo/SenseGlove-API, MIT-licensed, license copied to
`vendor/LICENSE`) -- headers in `vendor/include/`, the prebuilt Windows
x64/MSVC143 release `sgcore.lib`/`sgcore.dll`/`sgconnect.lib`/`sgconnect.dll`
in `vendor/lib_win64_msvc143_release/`. `senseglove_bridge.cpp` was:

1. Compiled and linked against these real libs with MSVC (`cl /MD`) --
   clean, no errors.
2. Run against a real, actually-running SenseCom process with no physical
   glove attached: it correctly detected SenseCom via
   `SenseCom::ScanningActive()`, started listening on `127.0.0.1:8850`, and
   correctly reported zero connected devices (`HandLayer::DeviceConnected`
   false for both hands) rather than crashing or making anything up.
3. Connected to end-to-end from the real Python
   `SenseGloveJSONBridgeReader` over that TCP socket, confirming the wire
   protocol works both directions with the real SDK in the loop.

**What's not tested: an actual physical glove.** That's the one thing that
genuinely can't be verified without hardware. Everything else in this
pipeline has been.

If your actual glove needs a different SDK version, grab it from
https://github.com/Adjuvo/SenseGlove-API/releases and swap the contents of
`vendor/include` and `vendor/lib_win64_msvc143_release` -- the API calls
used here (`HandLayer::GetHandPose`, `HandPose::GetNormalizedFlexion`,
`HandPose::GetJointPositions`) have been stable since at least v2.x.

## Wire format

One JSON object per line, per hand, per update (~60 Hz):

```
{"hand": "left" | "right",
 "flexion": [thumb, index, middle, ring, pinky],           // 0..1
 "joint_positions": [[[x,y,z], ...4 joints...], ...5 fingers...]}  // meters, relative to wrist
```
