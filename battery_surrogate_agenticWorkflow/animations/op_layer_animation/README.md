# OP Layer Animation

Animate the three thermal grid layers of an OP bundle over time.

Each cached OP bundle (`data_cache/OPxx.npz`) stores a temperature field `T`
with shape `(n_time, n_sensors)`. The sensors are three thermal grid layers of
121 points each (an 11×11 grid per layer):

| Layer | Label | Meaning |
|-------|-------|---------|
| `cc`   | Cell Center   | Grid at the cell center |
| `g`    | Gehäusewand   | Grid at the housing wall |
| `jr1c` | JR1 Center    | Grid at the jelly-roll 1 center |

## Files

- `animate_layers.py` — reusable `build_layer_animation(...)` helper plus a CLI.
- `animate_op_layers.ipynb` — notebook that loads OP01, animates the three
  layers side by side, displays the animation inline, and saves a GIF.
- `OP01_layers.gif` — generated animation (created after running the notebook).

## Run it

Open `animate_op_layers.ipynb` and run all cells. It produces:

- An inline GIF of the three layers evolving over the full simulation.
- A static snapshot figure at start / middle / end.

To animate a different OP, change `OP_ID` in the notebook (e.g. `"OP02"`), or
use the CLI once a non-WSL Python terminal is available:

```
python animate_layers.py --op OP01 --frames 60 --fps 12
```
