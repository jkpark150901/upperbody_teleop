"""Loads the DG5F's visual mesh files (Collada .dae, at repo-root dg5f/)
into vedo actors for scripts/hand_retarget_vedo.py's --render mesh mode.

vedo/VTK has no built-in Collada reader (vedo.load() fails on .dae), so
this goes through trimesh+pycollada instead, which also gets each
sub-mesh's material color -- some DG5F parts (e.g. the palm) bundle several
differently-colored sub-meshes in one .dae file (a trimesh.Scene with
multiple geometries), so one URDF <visual><mesh> can become more than one
vedo.Mesh, grouped into a single vedo.Assembly.

URDF mesh filenames follow a "../dg5f/meshes/<part>/visual/<name>.dae"
convention (see rby1_dg5f.urdf / scripts/extract_hand_urdf.py) that assumes
the URDF sits in a directory that's a sibling of dg5f/ -- true on the
original robot description package, not true here (robot_hand/urdf/ isn't
a sibling of the repo-root dg5f/). resolve_mesh_path() locates the fixed
"dg5f/meshes/..." suffix instead of trusting the literal ".." count.
"""

from __future__ import annotations

import pathlib
from typing import List, Tuple

import trimesh
import vedo

_REPO_ROOT = pathlib.Path(__file__).parent.parent
_MESH_MARKER = "dg5f/meshes/"
_RBY1_MESH_MARKER = "rby1ub_src/urdf/meshes/"
_DEFAULT_RGBA = (200, 200, 200, 255)


def resolve_mesh_path(urdf_filename: str) -> pathlib.Path:
    normalized = urdf_filename.replace("\\", "/")
    idx = normalized.find(_MESH_MARKER)
    if idx != -1:
        path = _REPO_ROOT / normalized[idx:]
    else:
        idx = normalized.find(_RBY1_MESH_MARKER)
        if idx == -1:
            raise ValueError(
                "mesh filename does not contain a supported dg5f or rby1 mesh path: "
                f"{urdf_filename}"
            )
        path = _REPO_ROOT / "rby1" / normalized[idx:]
    if not path.is_file():
        raise FileNotFoundError(f"resolved {urdf_filename!r} -> {path}, which doesn't exist")
    return path


def _geometry_rgba(geometry: trimesh.Trimesh) -> Tuple[int, int, int, int]:
    material = getattr(geometry.visual, "material", None)
    color = getattr(material, "baseColorFactor", None)
    if color is None:
        color = getattr(material, "main_color", None)
    if color is None:
        return _DEFAULT_RGBA
    return tuple(int(v) for v in color)


def load_mesh_parts(urdf_filename: str) -> List[Tuple["vedo.Mesh", Tuple[int, int, int, int]]]:
    """(vedo.Mesh, rgba) pairs for one URDF <visual><mesh>, in the mesh's
    own local frame (every DG5F .dae checked has an identity scene-graph
    transform on each of its geometries, so raw vertices are used as-is)."""
    path = resolve_mesh_path(urdf_filename)
    scene = trimesh.load(str(path))
    geometries = scene.geometry.values() if isinstance(scene, trimesh.Scene) else [scene]

    parts = []
    for geometry in geometries:
        rgba = _geometry_rgba(geometry)
        mesh = vedo.Mesh([geometry.vertices, geometry.faces])
        parts.append((mesh, rgba))
    return parts
