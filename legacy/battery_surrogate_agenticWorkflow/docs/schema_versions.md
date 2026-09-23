# Schema Versions

This file records the cache schema versions used by the workflow package.
The build step should refuse to write a cache if the current version is not listed here.

## v1

- First packaged workflow shape.
- One OP bundle with fixed arrays for electrical and thermal outputs.
- `sim_config` stored as one flat numeric block.
- No per-channel profile split in the cache.

## v2

- OP input config is split into scalar values plus per-channel profile arrays.
- Sentinel values from `Inputsignale.csv` are stored in metadata instead of being guessed.
- Name lists live in bundle metadata rather than in separate arrays.
- `soc_start` can be marked as derived instead of being forced into a float array.

## v3

- Adds the wall path: `wall_ts`, a `{name: (time, values)}` map carrying
  `q_solid_to_fluid` (W, from `*_Heat Transfer.csv`) and `fluid_out_temp`
  (C, from `*_Temperaturen.csv`).
- **Each series keeps its own time axis.** These two files do not share the
  axis of `*_Heat Source.csv`. Assuming they did is what broke
  `GridCNN/tools/balance_check.py` on 22.09. (`fp and xp are not of the same
  length`) after the same assumption had passed silently elsewhere. `meta`
  records, per series, the length, the span, and whether the axis happens to
  equal `t_slow` -- recorded, never equalised.
- `fluid_props` is read **by column name** instead of by position, and the
  names ride along in `fluid_props_names`. The v2 contract guessed the order
  ("typically density, specific heat, thermal conductivity"); the export is
  Conductivity, Density, Specific Heat. A file with no named hit still falls
  back to the positional read, and then the names say `unbenannt_*`.
- Not added, because they were already there -- checked on 22.09.:
  `mdot` is the canonical channel `fluid_mass_flow`, `cp_fluid` is in
  `fluid_props`, and `total_w` is `q_source[:, 2]`.
- Reading is backwards compatible: a v2 bundle loads with an empty `wall_ts`
  and empty `fluid_props_names`, which means "not rebuilt yet" rather than a
  silent zero.
