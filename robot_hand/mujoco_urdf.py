"""Auto-generates a MuJoCo-loadable copy of a URDF.

MuJoCo's URDF compiler has no Collada decoder ("no decoder found for mesh
file ...dae", confirmed by trying to load rby1_dg5f.urdf directly) -- see
mesh_loader.py's docstring for why trimesh/pycollada already sits between
this repo's .dae visual meshes and anything that renders them. This mirrors
robot_hand/anydex_retarget.py's _pinocchio_compatible_urdf pattern: convert
what the target consumer can't read, write a modified URDF copy pointing at
the converted files, cache by source mtime so repeat runs are instant.

Collision geometry is already .STL (MuJoCo-native, see rby1_dg5f.urdf) and
is left untouched -- only <visual><mesh filename="...dae"> entries are
rewritten.
"""

from __future__ import annotations

import pathlib
import xml.etree.ElementTree as ET

import mujoco
import trimesh

from robot_hand.mesh_loader import resolve_mesh_path

_ROOT = pathlib.Path(__file__).resolve().parent.parent
_OUT_URDF_DIR = _ROOT / "outputs" / "mujoco_urdf"
_OUT_MESH_DIR = _ROOT / "outputs" / "mujoco_meshes"


def _convert_mesh(dae_path: pathlib.Path) -> pathlib.Path:
    """.dae -> .obj, cached by source mtime. Some DG5F parts bundle several
    differently-colored sub-meshes in one .dae (a trimesh.Scene with
    multiple geometries -- see mesh_loader.py); URDF's <visual><mesh> is one
    geometry per link, so those are concatenated into a single mesh here.
    That loses per-part material color, which matters for vedo's rendering
    but not for MuJoCo control -- geometry only."""
    rel = dae_path.relative_to(_ROOT) if dae_path.is_relative_to(_ROOT) else pathlib.Path(dae_path.name)
    out_path = (_OUT_MESH_DIR / rel).with_suffix(".obj")
    if out_path.exists() and out_path.stat().st_mtime >= dae_path.stat().st_mtime:
        return out_path
    out_path.parent.mkdir(parents=True, exist_ok=True)
    scene = trimesh.load(str(dae_path))
    mesh = scene.dump(concatenate=True) if isinstance(scene, trimesh.Scene) else scene
    mesh.export(str(out_path))
    return out_path


def mujoco_compatible_urdf(urdf_path) -> pathlib.Path:
    """Returns a URDF path MuJoCo can load: same content as `urdf_path`,
    except every .dae mesh reference is repointed at a converted .obj copy.
    Cached in outputs/mujoco_urdf/ -- safe to call every run."""
    urdf_path = pathlib.Path(urdf_path).resolve()
    out_path = _OUT_URDF_DIR / f"{urdf_path.stem}_mujoco.urdf"
    src_mtime = urdf_path.stat().st_mtime
    if out_path.exists() and out_path.stat().st_mtime >= src_mtime:
        return out_path

    tree = ET.parse(urdf_path)
    root = tree.getroot()
    n_converted = 0
    for mesh_elem in root.iter("mesh"):
        filename = mesh_elem.get("filename")
        if filename is None:
            continue
        src = resolve_mesh_path(filename)
        if filename.lower().endswith(".dae"):
            # MuJoCo can't read this format at all -- swap in the converted copy.
            src = _convert_mesh(src)
            n_converted += 1
        # Already-STL collision meshes need no format conversion, but their
        # relative "dg5f/meshes/..." paths were relative to urdf_path's own
        # directory (repo root) -- out_path lives elsewhere
        # (outputs/mujoco_urdf/), so every reference is rewritten absolute,
        # converted or not.
        mesh_elem.set("filename", src.as_posix())

    _OUT_URDF_DIR.mkdir(parents=True, exist_ok=True)
    tree.write(out_path, encoding="utf-8", xml_declaration=True)
    print(f"[mujoco_urdf] {urdf_path.name}: converted {n_converted} .dae mesh(es) -> {out_path}")
    return out_path


def _joint_efforts(urdf_path: pathlib.Path) -> dict:
    """URDF <joint><limit effort="..."> per joint name -- MuJoCo's URDF
    importer has no <actuator> concept (URDF doesn't have one), so this
    reads the original URDF directly rather than trying to recover it from
    the compiled/converted MJCF spec."""
    efforts = {}
    for joint_elem in ET.parse(urdf_path).getroot().iter("joint"):
        name = joint_elem.get("name")
        limit = joint_elem.find("limit")
        if name and limit is not None and limit.get("effort") is not None:
            efforts[name] = float(limit.get("effort"))
    return efforts


_RIGID_JOINTS = ("torso_hp", "torso_5")  # the body/torso -- must stay physically fixed, not just servoed


_ACTUATED_MJCF_PATH = _OUT_URDF_DIR / "rby1_dg5f_actuated.xml"


def _build_actuated_spec(
    urdf_path,
    heavy_kp: float = 200.0,
    heavy_kv: float = 20.0,
    light_kp: float = 150.0,
    light_kv: float = 10.0,
    heavy_effort_threshold: float = 40.0,
    rigid_kp: float = 20000.0,
    rigid_kv: float = 1000.0,
) -> tuple[mujoco.MjSpec, dict]:
    """Builds the actuated MjSpec (position actuator per revolute joint,
    same gaintype/biastype MJCF's <position kp kv> shorthand expands to,
    plus rest-pose self-collision excludes) -- see actuated_mjcf_path()/
    build_actuated_model() below for the cached-file wrapper around this;
    call this directly only if you need the uncompiled mujoco.MjSpec
    itself (e.g. to add more to it before compiling).

    torso_hp/torso_5 (_RIGID_JOINTS) get a much stiffer kp/kv/damping than
    every other joint, effectively a rigid weld rather than a torque-
    limited servo (no forcerange cap either) -- this project's IK has
    always treated the torso as fixed (see RBY1UpperBodyRetargeter.update(),
    self.q[:2] pinned to reference_q unless --torso retarget), so the
    physics body has to actually stay put under the arms' own weight, not
    merely drift back toward it. Every other joint gets an actuator too,
    including head_0/head_1 -- there's no reason to leave them unactuated
    (free-swinging under gravity) just because this project's convention
    is to never *command* them away from 0; a position actuator holding
    ctrl=0 (MuJoCo's default, and what MujocoJointWriter.write_ctrl leaves
    it at by skipping those names) is exactly the "frozen head" behavior
    the kinematic version had, just physically enforced instead of merely
    never-written.

    Non-rigid gains are split into two tiers by the joint's own URDF
    effort limit (heavy: shoulder/elbow, light: wrist/fingers) as a
    starting point -- not tuned against the real robot's actual servo
    response. forcerange is capped at that same URDF effort limit either
    way, so the actuator can't apply more torque than the real joint is
    rated for.
    """
    urdf_path = pathlib.Path(urdf_path)
    xml_path = mujoco_compatible_urdf(urdf_path)
    effort = _joint_efforts(urdf_path)

    # Fresh, throwaway spec/model just to find which body pairs already
    # touch at rest -- MjSpec.compile() is NOT safe to call twice on the
    # same spec object (confirmed: recompiling after spec.add_exclude(...)
    # on an already-compiled spec didn't remove that contact, it *added*
    # ones -- ncon went 109 -> 317 on an otherwise-identical minimal repro).
    # So the probe gets its own spec, never touched again, and the real
    # model below gets a second, separate fresh spec that's compiled
    # exactly once, actuators/excludes and all.
    #
    # Tried giving every probe geom a small extra margin so a pair right at
    # the edge of contact would reliably count as "touching" -- that made
    # it worse: with margin set, the probe's own left_hand_ll_dg_palm body
    # came out at a wildly different xpos than the real model's (~0.6m off
    # in Z, matching neither a valid rest pose nor a plausible IK pose),
    # meaning geom.margin mutation on this spec/mujoco version corrupts
    # something in body placement, not just contact distance. Left at
    # default margin (0) instead, plus one confirmed-fragile pair added by
    # name below rather than trusting the probe to always catch it.
    probe_model = mujoco.MjSpec.from_file(str(xml_path)).compile()
    probe_data = mujoco.MjData(probe_model)
    mujoco.mj_forward(probe_model, probe_data)
    baseline_pairs = set()
    for i in range(probe_data.ncon):
        c = probe_data.contact[i]
        b1 = mujoco.mj_id2name(probe_model, mujoco.mjtObj.mjOBJ_BODY, probe_model.geom_bodyid[c.geom1])
        b2 = mujoco.mj_id2name(probe_model, mujoco.mjtObj.mjOBJ_BODY, probe_model.geom_bodyid[c.geom2])
        if b1 and b2:
            baseline_pairs.add(tuple(sorted((b1, b2))))
    # Confirmed touching at rest on real hardware but missed by the probe
    # above (capsule pairs that sit right at the edge of MuJoCo's contact-
    # generation threshold, so whether the probe catches them varies run to
    # run) -- added by name for both hands rather than chasing the probe's
    # flakiness further. thumb-base(1_2)/palm: ~1.7mm. ring/pinky
    # (4_*/5_3,5_4): fingers resting close enough together that the
    # SelfCollisionMonitor diagnostic kept flagging it every run.
    for side in ("left", "right"):
        prefix = "ll" if side == "left" else "rl"
        h = f"{side}_hand_{prefix}_dg"
        baseline_pairs.add(tuple(sorted((f"{h}_palm", f"{h}_1_2"))))
        for ring_seg in (1, 2, 3, 4):
            for pinky_seg in (3, 4):
                baseline_pairs.add(tuple(sorted((f"{h}_4_{ring_seg}", f"{h}_5_{pinky_seg}"))))

    spec = mujoco.MjSpec.from_file(str(xml_path))

    # URDF carries no armature/damping (confirmed 0.0 on every joint), which
    # is exactly the recipe for a stiff position actuator to go unstable on
    # a light-inertia link (fingertips, the head/camera mount) -- confirmed
    # the hard way (QACC NaN at the head joints within the first few
    # physics steps, even with zero contacts). armature (added rotor
    # inertia) and joint damping are the standard MuJoCo fix, on top of
    # switching off the default explicit-Euler integrator for one that
    # handles damped/stiff systems better.
    spec.option.integrator = mujoco.mjtIntegrator.mjINT_IMPLICITFAST
    tier_params = {
        "rigid": dict(kp=rigid_kp, kv=rigid_kv, armature=0.2, damping=rigid_kv),
        "heavy": dict(kp=heavy_kp, kv=heavy_kv, armature=0.05, damping=0.5),
        "light": dict(kp=light_kp, kv=light_kv, armature=0.01, damping=0.1),
    }
    joint_tiers: dict = {}
    for joint in list(spec.joints):
        if joint.type != mujoco.mjtJoint.mjJNT_HINGE:
            continue
        e = effort.get(joint.name, light_kp)
        if joint.name in _RIGID_JOINTS:
            tier, e = "rigid", 1e6
        else:
            tier = "heavy" if e >= heavy_effort_threshold else "light"
        joint_tiers[joint.name] = tier
        p = tier_params[tier]
        joint.armature = p["armature"]
        joint.damping = [p["damping"], 0.0, 0.0]
        gainprm = [0.0] * 10
        gainprm[0] = p["kp"]
        biasprm = [0.0] * 10
        biasprm[1] = -p["kp"]
        biasprm[2] = -p["kv"]
        lo, hi = joint.range
        spec.add_actuator(
            name=f"act_{joint.name}",
            gaintype=mujoco.mjtGain.mjGAIN_FIXED, gainprm=gainprm,
            biastype=mujoco.mjtBias.mjBIAS_AFFINE, biasprm=biasprm,
            trntype=mujoco.mjtTrn.mjTRN_JOINT, target=joint.name,
            ctrllimited=True, ctrlrange=[lo, hi],
            forcelimited=True, forcerange=[-e, e],
        )

    # The collision capsules/STLs badly overlap other links *within the
    # same arm/hand* even at rest -- confirmed penetration depths up to
    # ~5cm between e.g. a forearm link and its own fingers (these capsules
    # were authored for hand_retarget.py's own Python-side finger self-
    # collision check, a coarse early-warning margin, not fitted for literal
    # MuJoCo contact dynamics). mj_forward-only (no mj_step) never cared,
    # but real contact dynamics do: resolving that much penetration blows
    # up into a QACC-NaN explosion within the first few physics steps
    # (confirmed the hard way).
    #
    # An earlier version of this used group-based contype/conaffinity
    # (disable ALL intra-limb contact) instead -- robust, but it also
    # disabled genuine finger-vs-*different*-finger self-collision within
    # one hand, which is exactly the case this needs to catch (confirmed
    # on real hardware: fingers crossing into each other went undetected).
    # Excluding only the specific body pairs already touching at rest
    # (baseline_pairs, from the probe model above) is more precise -- at
    # rest, different fingers don't touch each other at all (only each
    # finger's own forearm/wrist-adjacent overlap does), so finger-vs-
    # finger contact stays live while the persistent rest-pose noise is
    # still suppressed. This is less bulletproof against pathological/
    # extreme poses than the group-mask version was (a finger bent far past
    # its real range could still find a new, unexcluded overlap with its
    # own wrist and reproduce the same instability) -- acceptable since
    # normal finger ROM shouldn't reach those poses, not because the
    # underlying oversized capsules actually got fixed.
    for b1, b2 in baseline_pairs:
        spec.add_exclude(name=f"exclude_{b1}_{b2}", bodyname1=b1, bodyname2=b2)

    # Training-data cameras: head + both wrists, per the "camera on the
    # head and each wrist, no other head DOF" assumption this whole
    # project has used since the first MuJoCo pass (see module docstrings
    # elsewhere -- e.g. scripts/upper_body_retarget_mujoco.py's "no head
    # DOF at all, a physically fixed camera"). The rough box housing
    # marker itself now lives in rby1_dg5f.urdf directly (an extra
    # <visual> on each of these three links) so it shows up in any
    # URDF-based tool, not just this MuJoCo-specific pipeline -- only the
    # actual <camera> (MuJoCo-only concept, mujoco.Renderer can render
    # from it, confirmed working headless) is added here. xyaxes points
    # the camera's view (its own -Z) down the mount link's local +X
    # (forward, this project's established convention -- see
    # bvh_upper_body_mapping.md) with +Z (up) as the camera's own +Y,
    # i.e. upright and forward-facing regardless of which of the three
    # links it's on.
    for cam_name, link_name in (
        ("cam_head", "link_head_2"),
        ("cam_left_wrist", "link_left_arm_6"),
        ("cam_right_wrist", "link_right_arm_6"),
    ):
        spec.body(link_name).add_camera(
            name=cam_name, pos=[0.045, 0.0, 0.0],
            xyaxes=[0.0, -1.0, 0.0, 0.0, 0.0, 1.0],
            fovy=60.0,
        )

    return spec, {"joint_tiers": joint_tiers, "tier_params": tier_params}


def _collapse_into_default_classes(xml_text: str, joint_tiers: dict, tier_params: dict) -> str:
    """Rewrites spec.to_xml()'s output so kp/kv/damping/armature live in
    ONE <default class="..."> block per tier (heavy/light/rigid) instead of
    baked onto every individual <joint>/<general> element.

    Needed because MjSpec.to_xml() always serializes concrete resolved
    values for every element regardless of any class assigned in Python
    (confirmed: setting joint.classname before to_xml() still emits
    damping="0" armature="0" inline, which -- explicit attributes always
    outrank a class in MJCF -- shadows the class's values entirely). This
    reaches into the XML text after the fact instead: strip the per-element
    damping/armature/gainprm/biasprm and point each element at its tier's
    class, so changing one <default> block's numbers is the only thing
    needed to retune every joint in that tier at once.
    """
    root = ET.fromstring(xml_text)

    default_root = root.find("default")
    if default_root is None:
        default_root = ET.Element("default")
        root.insert(0, default_root)
    for tier, p in tier_params.items():
        tier_default = ET.SubElement(default_root, "default", {"class": tier})
        ET.SubElement(tier_default, "joint", {
            "damping": str(p["damping"]), "armature": str(p["armature"]),
        })
        ET.SubElement(tier_default, "general", {
            "biastype": "affine",
            "gainprm": f"{p['kp']} 0 0",
            # kp appears in both gainprm[0] and biasprm[1] for this
            # <position>-style actuator (biasprm = [0, -kp, -kv]) -- change
            # BOTH together, MJCF has no expressions/variables to keep them
            # in sync automatically.
            "biasprm": f"0 {-p['kp']} {-p['kv']}",
            "ctrllimited": "true", "forcelimited": "true",
        })

    worldbody = root.find("worldbody")
    for joint_el in worldbody.iter("joint"):
        tier = joint_tiers.get(joint_el.get("name"))
        if tier is None:
            continue
        joint_el.set("class", tier)
        joint_el.attrib.pop("damping", None)
        joint_el.attrib.pop("armature", None)

    actuator_root = root.find("actuator")
    if actuator_root is not None:
        for act_el in actuator_root:
            name = act_el.get("name", "")
            joint_name = name[len("act_"):] if name.startswith("act_") else act_el.get("joint")
            tier = joint_tiers.get(joint_name)
            if tier is None:
                continue
            act_el.set("class", tier)
            for attr in ("gainprm", "biasprm", "gaintype", "biastype", "ctrllimited", "forcelimited"):
                act_el.attrib.pop(attr, None)

    ET.indent(root, space="  ")
    return ET.tostring(root, encoding="unicode")


def actuated_mjcf_path(urdf_path, force: bool = False, **gain_kwargs) -> pathlib.Path:
    """Cached MJCF file for _build_actuated_spec's model -- write it once,
    then edit kp/kv/damping/armature directly in the saved XML's single
    <default class="heavy|light|rigid"> block (see
    _collapse_into_default_classes) and re-run without going through
    Python at all (or this whole rebuild) each time.

    Cached by source URDF mtime, same convention as mujoco_compatible_urdf:
    regenerated only if missing, `force=True`, or urdf_path changed since.
    Editing the saved XML by hand bumps *its own* mtime newer than the
    source URDF, so those edits are picked up (and preserved -- not
    silently overwritten) on every later call, exactly like re-running
    after tweaking values in the file is supposed to work.
    """
    urdf_path = pathlib.Path(urdf_path)
    if not force and _ACTUATED_MJCF_PATH.exists() and _ACTUATED_MJCF_PATH.stat().st_mtime >= urdf_path.stat().st_mtime:
        return _ACTUATED_MJCF_PATH
    spec, meta = _build_actuated_spec(urdf_path, **gain_kwargs)
    xml_text = _collapse_into_default_classes(spec.to_xml(), meta["joint_tiers"], meta["tier_params"])
    _ACTUATED_MJCF_PATH.parent.mkdir(parents=True, exist_ok=True)
    _ACTUATED_MJCF_PATH.write_text(xml_text, encoding="utf-8")
    print(f"[mujoco_urdf] wrote actuated MJCF -> {_ACTUATED_MJCF_PATH}")
    return _ACTUATED_MJCF_PATH


def build_actuated_model(urdf_path, force: bool = False, **gain_kwargs) -> mujoco.MjModel:
    """Loads actuated_mjcf_path()'s cached file (building/writing it first
    if missing or stale) instead of re-running _build_actuated_spec's
    MjSpec construction on every call -- much faster on repeat runs, and
    the whole point is that the saved file is the thing to hand-edit for
    quick kp/kv/damping/etc. iteration, not these Python defaults."""
    path = actuated_mjcf_path(urdf_path, force=force, **gain_kwargs)
    return mujoco.MjModel.from_xml_path(str(path))
