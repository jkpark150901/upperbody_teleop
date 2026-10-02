"""Shared live-updating 3D skeleton graph (matplotlib): points + bone
edges, each point labeled with position (x,y,z) and, if that joint's
orientation is known, orientation as Euler angles (rx,ry,rz, degrees) --
exactly what hand_process.py and mocap_process.py's own windows are for
("스켈레톤들의 x,y,z,rx,ry,rz를 표시"). Not run alongside MuJoCo's own
viewer in the same process (that combination crashed before -- see
landmark_graph_viewer.py's docstring for why), which is no longer a
concern now that retargeting and physics are three separate processes.
"""
from __future__ import annotations

from typing import Dict, Optional, Sequence, Tuple

import numpy as np


def rotmat_to_euler_xyz_deg(r: np.ndarray) -> np.ndarray:
    """Intrinsic X-Y-Z Euler angles (degrees) from a 3x3 rotation matrix --
    display only (no retargeting math reads this back), so any standard
    convention is fine as long as it's applied consistently."""
    sy = float(np.clip(r[0, 2], -1.0, 1.0))
    ry = np.arcsin(sy)
    if abs(sy) < 0.9999999:
        rx = np.arctan2(-r[1, 2], r[2, 2])
        rz = np.arctan2(-r[0, 1], r[0, 0])
    else:
        rx = np.arctan2(r[2, 1], r[1, 1])
        rz = 0.0
    return np.degrees([rx, ry, rz])


class SkeletonGraphWindow:
    def __init__(
        self,
        title: str,
        joint_names: Sequence[str],
        edges: Sequence[Tuple[str, str]],
        *,
        show_labels: bool = True,
    ):
        import matplotlib
        matplotlib.use("TkAgg")
        import matplotlib.pyplot as plt
        self._plt = plt
        self.joint_names = list(joint_names)
        self.edges = list(edges)
        self.show_labels = show_labels

        plt.ion()
        self.fig = plt.figure(title, figsize=(7, 7))
        self.ax = self.fig.add_subplot(111, projection="3d")
        self.fig.show()

    def update(
        self,
        positions: Dict[str, np.ndarray],
        rotations: Optional[Dict[str, Optional[np.ndarray]]] = None,
    ) -> None:
        rotations = rotations or {}
        names = [n for n in self.joint_names if n in positions]
        if not names:
            return
        pts = np.array([positions[n] for n in names], dtype=float)

        ax = self.ax
        ax.cla()
        ax.scatter(pts[:, 0], pts[:, 1], pts[:, 2], c="red", s=30, depthshade=False)
        for a, b in self.edges:
            if a not in positions or b not in positions:
                continue
            p0, p1 = positions[a], positions[b]
            ax.plot([p0[0], p1[0]], [p0[1], p1[1]], [p0[2], p1[2]], c="steelblue", lw=1.5)

        if self.show_labels:
            for name, p in zip(names, pts):
                r = rotations.get(name)
                if r is not None:
                    rx, ry, rz = rotmat_to_euler_xyz_deg(np.asarray(r))
                    text = f"{name}\n{p[0]:.3f},{p[1]:.3f},{p[2]:.3f}\n{rx:.0f},{ry:.0f},{rz:.0f}"
                else:
                    text = f"{name}\n{p[0]:.3f},{p[1]:.3f},{p[2]:.3f}"
                ax.text(p[0], p[1], p[2], text, size=6)

        ax.set_title("x,y,z (m) / rx,ry,rz (deg) per joint")
        center = pts.mean(axis=0)
        radius = max(1e-3, float(np.abs(pts - center).max()))
        ax.set_xlim(center[0] - radius, center[0] + radius)
        ax.set_ylim(center[1] - radius, center[1] + radius)
        ax.set_zlim(center[2] - radius, center[2] + radius)
        self.fig.canvas.draw_idle()
        self.fig.canvas.flush_events()

    def close(self) -> None:
        self._plt.close(self.fig)
