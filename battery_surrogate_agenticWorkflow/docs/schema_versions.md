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
