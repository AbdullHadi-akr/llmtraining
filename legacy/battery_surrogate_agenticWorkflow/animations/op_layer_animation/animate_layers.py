"""Animate the three thermal grid layers of an OP bundle over time.

Each cached OP bundle stores a temperature field ``T`` with shape
``(n_time, n_sensors)`` where the sensors are the concatenation of three
thermal grid layers:

* ``cc``   - Grid Cell Center
* ``g``    - Grid Gehaeusewand (housing wall)
* ``jr1c`` - Grid JR1 Center

Every layer holds 121 sensors (an 11x11 grid). This module loads a bundle
straight from its ``.npz`` cache file and builds a matplotlib animation with
one filled-contour panel per layer, sharing a common colour scale so the three
layers can be compared frame by frame as time advances.

The animation is written to a GIF (no ffmpeg needed) and the path is returned
so a notebook or script can display it inline.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np


LAYER_LABELS: dict[str, str] = {
    "cc": "Cell Center",
    "g": "Gehaeusewand",
    "jr1c": "JR1 Center",
}
LAYER_ORDER: tuple[str, ...] = ("cc", "g", "jr1c")


def load_op_arrays(npz_path: str | Path) -> dict[str, np.ndarray]:
    """Load the arrays needed for the animation from an OP ``.npz`` cache file."""

    npz_path = Path(npz_path)
    if not npz_path.exists():
        raise FileNotFoundError(f"OP cache file not found: {npz_path}")

    with np.load(npz_path, allow_pickle=True) as data:
        arrays = {
            "T": np.asarray(data["T"], dtype=np.float32),
            "t_fast": np.asarray(data["t_fast"], dtype=np.float32),
            "xyz": np.asarray(data["xyz"], dtype=np.float32),
            "layer": np.asarray(data["layer"]),
        }
    return arrays


def _varying_axes(xyz: np.ndarray) -> tuple[int, int]:
    """Return the indices of the two coordinate axes with the most variation."""

    spread = xyz.std(axis=0)
    # The two axes with the largest spread define the plane of the layer.
    return tuple(int(i) for i in np.argsort(spread)[-2:][::-1])  # type: ignore[return-value]


def build_layer_animation(
    npz_path: str | Path,
    out_path: str | Path,
    n_frames: int = 60,
    fps: int = 12,
    levels: int = 30,
    cmap: str = "inferno",
    dpi: int = 90,
) -> Path:
    """Build and save a GIF animating the three layers of one OP over time.

    Parameters
    ----------
    npz_path:
        Path to the OP ``.npz`` cache file (e.g. ``data_cache/OP01.npz``).
    out_path:
        Destination GIF path.
    n_frames:
        Number of time frames sampled evenly across the simulation.
    fps:
        Frames per second in the output GIF.
    levels:
        Number of contour levels per panel.
    cmap:
        Matplotlib colormap name.
    dpi:
        Output resolution.

    Returns
    -------
    Path
        The written GIF path.
    """

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.animation import FuncAnimation, PillowWriter

    arrays = load_op_arrays(npz_path)
    temps = arrays["T"]
    times = arrays["t_fast"]
    xyz = arrays["xyz"]
    layer = arrays["layer"]

    n_time = temps.shape[0]
    frame_idx = np.linspace(0, n_time - 1, num=min(n_frames, n_time)).astype(int)

    # Group sensor columns and coordinates by layer.
    layer_names = np.asarray([str(v) for v in layer])
    panels: list[dict] = []
    for name in LAYER_ORDER:
        mask = layer_names == name
        if not mask.any():
            continue
        cols = np.where(mask)[0]
        coords = xyz[cols]
        ax_a, ax_b = _varying_axes(coords)
        panels.append(
            {
                "name": name,
                "cols": cols,
                "u": coords[:, ax_a],
                "v": coords[:, ax_b],
            }
        )

    if not panels:
        raise ValueError("No known thermal layers found in bundle.")

    # Shared colour scale across all layers and sampled frames.
    sampled = temps[frame_idx]
    vmin = float(np.nanmin(sampled))
    vmax = float(np.nanmax(sampled))
    contour_levels = np.linspace(vmin, vmax, levels)

    fig, axes = plt.subplots(1, len(panels), figsize=(5 * len(panels), 5))
    if len(panels) == 1:
        axes = [axes]

    def draw(frame_number: int):
        t_index = frame_idx[frame_number]
        mappable = None
        for ax, panel in zip(axes, panels):
            ax.clear()
            values = temps[t_index, panel["cols"]]
            mappable = ax.tricontourf(
                panel["u"],
                panel["v"],
                values,
                levels=contour_levels,
                cmap=cmap,
                extend="both",
            )
            ax.set_title(f"{LAYER_LABELS.get(panel['name'], panel['name'])}")
            ax.set_aspect("equal")
            ax.set_xticks([])
            ax.set_yticks([])
        fig.suptitle(
            f"{Path(npz_path).stem}  |  t = {times[t_index]:8.1f} s"
            f"  ({frame_number + 1}/{len(frame_idx)})",
            fontsize=13,
        )
        return mappable

    first = draw(0)
    cbar = fig.colorbar(first, ax=axes, fraction=0.046, pad=0.04)
    cbar.set_label("Temperature (degC)")

    anim = FuncAnimation(fig, draw, frames=len(frame_idx), blit=False)

    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    anim.save(out_path, writer=PillowWriter(fps=fps), dpi=dpi)
    plt.close(fig)
    return out_path


if __name__ == "__main__":
    import argparse

    here = Path(__file__).resolve()
    workflow_root = here.parents[2]  # .../battery_surrogate_agenticWorkflow

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--op",
        default="OP01",
        help="OP id to animate (default: OP01).",
    )
    parser.add_argument(
        "--frames", type=int, default=60, help="Number of time frames (default: 60)."
    )
    parser.add_argument("--fps", type=int, default=12, help="GIF frames per second.")
    args = parser.parse_args()

    npz = workflow_root / "data_cache" / f"{args.op}.npz"
    out = here.parent / f"{args.op}_layers.gif"
    written = build_layer_animation(npz, out, n_frames=args.frames, fps=args.fps)
    print(f"Wrote animation: {written}")
