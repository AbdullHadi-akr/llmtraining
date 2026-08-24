#!/usr/bin/env python3
"""Plot the solid geometry of the OP battery cell from the measurement grids.

Key facts (see prompt 007):
- We have THREE planar 11x11 = 121-point grids, each at a fixed x:
    Cell Center  x = 0.0        (symmetry plane)
    JR1 Center   x = +0.01079   (jelly-roll monitor plane)
    Gehaeusewand x = +0.02190   (housing wall)
  each spanning y in [-0.099, 0.099], z in [-0.052, 0.052].
- The REAL solid is mirror-symmetric about the cell-center plane (x = 0).
  So physically there is also:
    JR2          x = -0.01079   (mirror of JR1)
    Gehaeusewand2 x = -0.02190  (mirror of the housing wall)
  Their field values are the mirror image of the x>0 grids, so for TRAINING we
  only use the three x>=0 grids we measured. The mirrored planes are drawn here
  purely for geometry visualization.
- The 8 corner points P1..P8 of the cell box are NOT needed: the extreme points
  of the grids already define the solid bounding box.

Output: battery_surrogate_agenticWorkflow_PINN/artifacts/op01_pinn/solid_geometry.png
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401  (registers 3d projection)

PINN_ROOT = Path(__file__).parent.parent
COORD_DIR = PINN_ROOT / "data" / "coordinates"

GRIDS = {
    "Cell Center (x=0, symmetry plane)": ("Coordinates - Grid Cell Center.csv", "C2", "o"),
    "JR1 Center (x=+0.0108)":            ("Coordinates - Grid JR1 Center.csv",  "C1", "^"),
    "Gehaeusewand (x=+0.0219)":          ("Coordinates - Grid Gehäusewand.csv", "C3", "s"),
}
MIRROR = {
    "JR2 (mirror of JR1, x=-0.0108)":          ("Coordinates - Grid JR1 Center.csv",  "C1"),
    "Gehaeusewand2 (mirror, x=-0.0219)":       ("Coordinates - Grid Gehäusewand.csv", "C3"),
}


def _load(name: str) -> np.ndarray:
    return pd.read_csv(COORD_DIR / name).values[:, :3].astype(float)


def _box_edges(ax, lo, hi, **kw) -> None:
    """Draw the 12 edges of the axis-aligned box [lo, hi]."""
    (x0, y0, z0), (x1, y1, z1) = lo, hi
    corners = np.array([[x0, y0, z0], [x1, y0, z0], [x1, y1, z0], [x0, y1, z0],
                        [x0, y0, z1], [x1, y0, z1], [x1, y1, z1], [x0, y1, z1]])
    edges = [(0, 1), (1, 2), (2, 3), (3, 0), (4, 5), (5, 6), (6, 7), (7, 4),
             (0, 4), (1, 5), (2, 6), (3, 7)]
    for a, b in edges:
        ax.plot(*zip(corners[a], corners[b]), **kw)


def main() -> None:
    out_dir = PINN_ROOT / "artifacts" / "op01_pinn"
    out_dir.mkdir(parents=True, exist_ok=True)

    # measured grids (x >= 0) + mirrored grids (x < 0) for the full solid extent
    all_pts = []
    fig = plt.figure(figsize=(11, 8))
    ax = fig.add_subplot(111, projection="3d")

    for label, (fname, color, marker) in GRIDS.items():
        pts = _load(fname)
        all_pts.append(pts)
        ax.scatter(pts[:, 0], pts[:, 1], pts[:, 2], c=color, marker=marker,
                   s=18, depthshade=True, label=label)

    for label, (fname, color) in MIRROR.items():
        pts = _load(fname).copy()
        pts[:, 0] *= -1.0  # mirror about x = 0 (cell-center plane)
        all_pts.append(pts)
        ax.scatter(pts[:, 0], pts[:, 1], pts[:, 2], c=color, marker="x",
                   s=18, alpha=0.5, label=label)

    P = np.concatenate(all_pts, axis=0)
    lo = P.min(axis=0)
    hi = P.max(axis=0)
    _box_edges(ax, lo, hi, color="k", lw=1.2, alpha=0.7)

    # cell-center symmetry plane at x = 0
    yy, zz = np.meshgrid(np.linspace(lo[1], hi[1], 2), np.linspace(lo[2], hi[2], 2))
    ax.plot_surface(np.zeros_like(yy), yy, zz, color="C2", alpha=0.08)

    ax.set_xlabel("x [m]"); ax.set_ylabel("y [m]"); ax.set_zlabel("z [m]")
    ax.set_title(
        "OP solid geometry from grid extreme points\n"
        f"bounding box: x[{lo[0]:.4f}, {hi[0]:.4f}]  "
        f"y[{lo[1]:.4f}, {hi[1]:.4f}]  z[{lo[2]:.4f}, {hi[2]:.4f}] m"
    )
    ax.legend(fontsize=8, loc="upper left")
    try:
        ax.set_box_aspect((hi - lo))  # true aspect ratio
    except Exception:
        pass
    fig.tight_layout()
    out = out_dir / "solid_geometry.png"
    fig.savefig(out, dpi=130)
    plt.close(fig)

    print("grids: 3 measured (x>=0) + 2 mirrored (x<0)")
    print("solid bounding box (from grid extremes):")
    print(f"  x: [{lo[0]:.5f}, {hi[0]:.5f}]  ({hi[0]-lo[0]:.5f} m)")
    print(f"  y: [{lo[1]:.5f}, {hi[1]:.5f}]  ({hi[1]-lo[1]:.5f} m)")
    print(f"  z: [{lo[2]:.5f}, {hi[2]:.5f}]  ({hi[2]-lo[2]:.5f} m)")
    print(f"written: {out}")


if __name__ == "__main__":
    main()
