# Upperbody Teleop

Implements plan Phase 0-4 (`perception_neuron_senseglove_mujoco_teleop_plan.md`):
Perception Neuron (wrist/chest) + SenseGlove (fingers) -> UDP -> MuJoCo, drawn
as RGB axis triads (wrist/chest) and point markers (finger joints). No
URDF/robot body yet -- that's the next phase once a robot model is available.

Two machines, per plan section 5:

```
[Device PC: PN + SenseGlove]  --UDP-->  [MuJoCo PC]
  scripts/teleop_sender.py                mujoco_teleop/teleop_receiver_viewer.py
```

## Install

```
python -m venv .venv
.venv\Scripts\activate           # Windows; `source .venv/bin/activate` on Linux
pip install -r requirements.txt  # single machine, both sides (mock testing)
```

For the real two-PC deployment, use a venv on each machine with the
matching file instead of the combined one: `requirements-device.txt` on
the PN/SenseGlove PC (no MuJoCo), `requirements-mujoco.txt` on the MuJoCo
PC (no vedo/vendor SDKs). See "Device PC setup" below for the non-pip
vendor software the device PC also needs.

## Try it now (single machine, mock hardware)

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

## Sanity-check device data before touching MuJoCo

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

## Run with real hardware

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

## Device PC setup

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

## Layout

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

## Notes

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
#   u p p e r b o d y _ t e l e o p  
 