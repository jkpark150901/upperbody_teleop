"""Optional AnyDexRetarget backend for Quest/OpenXR -> DG5F.

This module keeps our live input path unchanged.  It only adapts the raw
OpenXR 26-joint hand skeleton into the 21-keypoint hand layout AnyDex uses
internally, then maps AnyDex qpos back to this repo's URDF joint-name dict.
"""

from __future__ import annotations

import pathlib
import sys
import xml.etree.ElementTree as ET
from typing import Dict, Optional

import numpy as np
import yaml

_ROOT = pathlib.Path(__file__).resolve().parent.parent
_ANYDEX_ROOT = _ROOT / "third_party" / "AnyDexRetarget"
_DEFAULT_CONFIG = {
    "left": _ROOT / "configs" / "anydex" / "dg5f_left_vector_quest3.yaml",
    "right": _ROOT / "configs" / "anydex" / "dg5f_right_vector_quest3.yaml",
}
_SANITIZED_URDF_DIR = _ROOT / "outputs" / "anydex_urdf"

# OpenXR 26-joint layout from devices.quest_hand.quest_hand_reader:
# 0 palm, 1 wrist, thumb 2..5, index 6..10, middle 11..15,
# ring 16..20, pinky 21..25.
_OPENXR_TO_MEDIAPIPE_21 = [
    1,              # wrist
    2, 3, 4, 5,    # thumb CMC/MCP/IP/tip
    7, 8, 9, 10,   # index MCP/PIP/DIP/tip
    12, 13, 14, 15,
    17, 18, 19, 20,
    22, 23, 24, 25,
]


def openxr_to_mediapipe21(xr_joints: np.ndarray) -> np.ndarray:
    """Return OpenXR hand joints as a MediaPipe-style (21, 3) array."""
    arr = np.asarray(xr_joints, dtype=float)
    if arr.shape[0] != 26 or arr.shape[1] < 3:
        raise ValueError(f"expected OpenXR joints shape (26, >=3), got {arr.shape}")
    return arr[_OPENXR_TO_MEDIAPIPE_21, :3].copy()


def _load_anydex_retargeter():
    if not _ANYDEX_ROOT.exists():
        raise RuntimeError(
            f"AnyDexRetarget is missing at {_ANYDEX_ROOT}. "
            "Clone https://github.com/qqsq12321/AnyDexRetarget.git there first."
        )
    anydex_root = str(_ANYDEX_ROOT)
    if anydex_root not in sys.path:
        sys.path.insert(0, anydex_root)
    try:
        from anydexretarget import Retargeter
    except ModuleNotFoundError as exc:
        if exc.name in {"nlopt", "pin", "pinocchio"}:
            raise RuntimeError(
                "AnyDexRetarget dependency is missing. Install its runtime deps "
                "in the drt env, at least: conda install -n drt -c conda-forge nlopt"
            ) from exc
        raise
    return Retargeter


def _load_config(path: pathlib.Path) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    robot = config.setdefault("robot", {})
    urdf_path = pathlib.Path(robot["urdf_path"])
    if not urdf_path.is_absolute():
        urdf_path = (_ROOT / urdf_path).resolve()
    robot["urdf_path"] = str(_pinocchio_compatible_urdf(urdf_path))
    return config


def _pinocchio_compatible_urdf(urdf_path: pathlib.Path) -> pathlib.Path:
    """Write a collision-free URDF copy for Pinocchio/urdfdom.

    scripts/extract_hand_urdf.py intentionally authors custom capsule
    collisions for our local collision checker and capsule renderer.  urdfdom
    does not understand that extension and prints one error per link while
    Pinocchio loads the model.  AnyDex only needs kinematics, so the cleanest
    adapter is to remove collision tags in a generated copy.
    """
    urdf_path = urdf_path.resolve()
    _SANITIZED_URDF_DIR.mkdir(parents=True, exist_ok=True)
    out_path = _SANITIZED_URDF_DIR / f"{urdf_path.stem}_pinocchio.urdf"

    src_mtime = urdf_path.stat().st_mtime
    if out_path.exists() and out_path.stat().st_mtime >= src_mtime:
        return out_path

    tree = ET.parse(urdf_path)
    root = tree.getroot()
    for link in root.findall("link"):
        for collision in list(link.findall("collision")):
            link.remove(collision)
    tree.write(out_path, encoding="utf-8", xml_declaration=True)
    return out_path


class AnyDexDg5fRetargeter:
    """DG5F wrapper around AnyDexRetarget's Retargeter."""

    def __init__(self, side: str, config_path: Optional[pathlib.Path] = None):
        side = side.lower()
        if side not in ("left", "right"):
            raise ValueError(f"side must be left/right, got {side!r}")
        self.side = side
        self.config_path = pathlib.Path(config_path) if config_path else _DEFAULT_CONFIG[side]

        retargeter_cls = _load_anydex_retargeter()
        config = _load_config(self.config_path)
        self.retargeter = retargeter_cls.from_config(config, hand_side=side)
        self.joint_names = [str(name) for name in self.retargeter.optimizer.robot.dof_joint_names]
        self.last_info: Dict[str, object] = {}

    def retarget_openxr_joints(self, xr_joints: np.ndarray, apply_filter: bool = True) -> Dict[str, float]:
        keypoints = openxr_to_mediapipe21(xr_joints)
        qpos, info = self.retargeter.retarget_verbose(keypoints, apply_filter=apply_filter)
        self.last_info = info
        return {name: float(value) for name, value in zip(self.joint_names, qpos)}

    def reset(self) -> None:
        self.retargeter.reset()
