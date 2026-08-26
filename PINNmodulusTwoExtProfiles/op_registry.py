"""The OP01-OP16 operating-point table and the split the benchmark ranks on.

Why this file exists
--------------------
``PINNmodulusTwo`` trains on OP01-OP05 and holds out OP06/OP07 -- seven operating
points that are all *constant*: one C-rate, one fluid inlet temperature and one
volume flow, held for the whole run. From OP08 on, the drivers become
**profiles**: they vary in time. Which OP carries which profile is not visible
anywhere in the cached ``.npz`` filename and only partly in its metadata, so the
plan sheet is transcribed here once, in code, and everything downstream reads it
from here.

Two things follow from having the table:

1. The split is no longer "take the first five and hold out the next two". With
   profiles the interesting question is what a held-out OP asks the model to do
   that training never showed it, so the OPs are grouped into TIERS by exactly
   that (see ``TIER_*`` below) and the benchmark reports per tier.
2. A held-out OP whose *driver values* sit outside the training range is an
   extrapolation, not a generalisation test, and saying so up front is more
   useful than discovering it in the error plot. ``data.coverage_report`` checks
   the numbers; this file states the intent.

The numbers below are the plan sheet, NOT measurements read back from the
bundles. They are used for reporting, tiering and sanity checks -- never to
build a feature. Every feature the model sees comes from the cached ``.npz``
via ``data.py``, so a wrong row here can mislabel a plot but cannot silently
corrupt training. ``data.profile_report()`` prints what the bundles actually
contain next to what this table claims, which is how a mismatch surfaces.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Sequence

# Config channels that the upstream assembly can deliver as a time series
# (``legacy/.../data/assemble.py``: CANONICAL_CHANNELS that accept a profile
# sentinel). Everything else is a scalar for the whole run.
PROFILE_CHANNELS = ("c_rate", "cell_current", "fluid_inlet_temp", "fluid_mass_flow")

# ---- tiers ------------------------------------------------------------------
# Tiers grade how hard a held-out OP is, by what it asks of the model rather
# than by whether it happens to contain a profile.
TIER_IN = "T0-in-time"       # a training OP, scored on its own timeline
TIER_INTERP = "T1-interp"    # unseen settings, inside the trained envelope
TIER_PROFILE = "T2-profile"  # unseen OP whose profile TYPES were trained on
TIER_EXTRAP = "T3-extrap"    # a driver value or profile type outside training

TIER_ORDER = (TIER_IN, TIER_INTERP, TIER_PROFILE, TIER_EXTRAP)

TIER_MEANING = {
    TIER_IN: "training OP, scored on its own timeline (in-sample unless "
             "--holdout-tail is set)",
    TIER_INTERP: "held-out OP, every driver inside the trained range",
    TIER_PROFILE: "held-out OP with profiles whose TYPE appears in training",
    TIER_EXTRAP: "held-out OP outside the trained envelope: a driver value "
                 "beyond the trained range, or a profile type never trained on",
}


@dataclass(frozen=True)
class OPSpec:
    """One row of the plan sheet."""

    op_id: str
    art: str                     # the sheet's "Art" column, verbatim in German
    charge: str                  # "CH" (charge) / "DCH" / "WLTP"
    c_rate: float | None         # None where the sheet says "Test Data"
    start_temp_c: float | None   # solid initial temperature
    fluid_temp_c: float | None   # None when it is a profile
    volume_flow_lpm: float | None  # None when it is a profile
    profiles: tuple = ()         # which PROFILE_CHANNELS vary in time
    tier: str = TIER_IN
    note: str = ""

    @property
    def has_profile(self) -> bool:
        return bool(self.profiles)


# ---- the plan sheet ---------------------------------------------------------
# "Training" block: OP01-OP16, all charge (CH), V_max 4.35 V, SOC 10-90 %.
# OP17-OP19 are the mini-module measurement comparison and are NOT here; see
# MEASUREMENT_OPS at the bottom for why.
OPS: Dict[str, OPSpec] = {
    s.op_id: s
    for s in [
        # --- constant drivers ------------------------------------------------
        OPSpec("OP01", "CC", "CH", 2.0, 25.0, 25.0, 15.0, tier=TIER_IN,
               note="reference point of the whole set"),
        OPSpec("OP02", "CC", "CH", 2.0, 15.0, 15.0, 15.0, tier=TIER_IN),
        OPSpec("OP03", "CC", "CH", 2.0, 30.0, 30.0, 15.0, tier=TIER_IN),
        OPSpec("OP04", "CC", "CH", 2.0, 25.0, 25.0, 30.0, tier=TIER_IN),
        OPSpec("OP05", "CC", "CH", 2.0, 40.0, 40.0, 30.0, tier=TIER_IN),
        OPSpec("OP06", "CC", "CH", 2.0, 25.0, 25.0, 0.0, tier=TIER_INTERP,
               note="no coolant flow; the flow=0 regime is trained via OP07/OP14"),
        OPSpec("OP07", "CC", "CH", 2.0, 10.0, 10.0, 0.0, tier=TIER_IN,
               note="coldest constant start with no flow"),
        # --- profile drivers -------------------------------------------------
        OPSpec("OP08", "CC mit Fluidtemperaturprofil", "CH", 3.0, 25.0, None, 15.0,
               profiles=("fluid_inlet_temp",), tier=TIER_IN,
               note="first fluid-temperature profile"),
        OPSpec("OP09", "CC mit Fluidtemperaturprofil", "CH", 2.5, 15.0, None, 15.0,
               profiles=("fluid_inlet_temp",), tier=TIER_PROFILE,
               note="C-rate 2.5 interpolates between the trained 2 and 3"),
        OPSpec("OP10", "CC-CV (Strominput aus Batemo vorsimuliert)", "CH",
               2.0, 25.0, 25.0, 15.0,
               profiles=("cell_current",), tier=TIER_IN,
               note="first current profile: the CV phase tapers the current"),
        OPSpec("OP11", "CC-CV_anode (Strominput aus Batemo vorsimuliert)", "CH",
               3.0, 15.0, 15.0, 15.0,
               profiles=("cell_current",), tier=TIER_IN,
               note="anode-limited CV taper, a differently shaped current profile"),
        OPSpec("OP12", "CC mit Fluidtemperaturprofil und CC-CV", "CH",
               2.0, 25.0, None, 15.0,
               profiles=("cell_current", "fluid_inlet_temp"), tier=TIER_IN,
               note="two profiles at once; trained so the compound case is not new"),
        OPSpec("OP13", "CC mit Fluidtemperaturprofil und CC-CV_anode", "CH",
               4.0, 25.0, None, 15.0,
               profiles=("cell_current", "fluid_inlet_temp"), tier=TIER_EXTRAP,
               note="C-rate 4 is above every trained C-rate -> extrapolation"),
        OPSpec("OP14", "CC", "CH", 2.0, 0.0, 0.0, 0.0, tier=TIER_IN,
               note="coldest start in the set, no flow"),
        OPSpec("OP15", "CC mit Fluidtemperaturprofil und Volumenstromprofil "
                       "und CC-CV", "CH", 2.0, 25.0, None, None,
               profiles=("cell_current", "fluid_inlet_temp", "fluid_mass_flow"),
               tier=TIER_EXTRAP,
               note="the volume-flow PROFILE type appears nowhere in training"),
        OPSpec("OP16", "CC", "CH", 2.0, 25.0, 25.0, 90.0, tier=TIER_EXTRAP,
               note="sheet says 15*6 l/min = 90, three times the trained maximum"),
    ]
}

ALL_OPS: tuple = tuple(OPS)

# OP17-OP19 of the sheet ("Abgleich mit Minimodul-Test") compare against measured
# mini-module data instead of the Batemo/StarCCM+ simulations: their C-rate,
# temperatures and flow all read "Test Data", and OP19 is a synthetic WLTP drive
# cycle. They are a different validation exercise -- measurement vs. simulation,
# partly discharge, partly a drive cycle -- and are deliberately out of scope
# here. Nothing in this extension reads them.
MEASUREMENT_OPS = ("OP17", "OP18", "OP19")


# ---- the split --------------------------------------------------------------
# Rules the split follows, so it can be argued with rather than just used:
#
# * Every profile TYPE that a selection OP relies on must occur in training --
#   otherwise the val number measures extrapolation and selection optimises for
#   the wrong thing.
# * Selection never touches the extrapolation tier. Choosing a configuration on
#   OP13/OP15/OP16 would turn the only out-of-envelope evidence into a fitted
#   quantity.
# * Every one of OP01-OP16 is used exactly once, so nothing is quietly dropped.
DEFAULT_TRAIN_OPS = ("OP01", "OP02", "OP03", "OP04", "OP05", "OP07",
                     "OP08", "OP10", "OP11", "OP12", "OP14")
DEFAULT_VAL_OPS = ("OP06", "OP09")             # what the benchmark RANKS on
DEFAULT_TEST_OPS = ("OP13", "OP15", "OP16")    # report only, never selected on


def spec(op_id: str) -> OPSpec:
    try:
        return OPS[op_id]
    except KeyError:
        extra = (f"  {op_id} is a mini-module measurement OP and is out of scope "
                 f"for this extension.\n" if op_id in MEASUREMENT_OPS else "")
        raise KeyError(
            f"unknown operating point {op_id!r}.\n{extra}"
            f"  known: {', '.join(ALL_OPS)}"
        ) from None


def tier_of(op_id: str) -> str:
    return spec(op_id).tier


def profiles_of(op_id: str) -> tuple:
    return spec(op_id).profiles


def profile_ops() -> List[str]:
    return [o for o in ALL_OPS if OPS[o].has_profile]


def constant_ops() -> List[str]:
    return [o for o in ALL_OPS if not OPS[o].has_profile]


def describe(op_id: str) -> str:
    s = spec(op_id)
    def _f(v, unit, profile):
        return "profile" if profile else ("?" if v is None else f"{v:g}{unit}")
    return (
        f"{s.op_id}  {s.tier:<11}  C={_f(s.c_rate, 'C', 'c_rate' in s.profiles)}"
        f"  T0={_f(s.start_temp_c, 'C', False)}"
        f"  T_fluid={_f(s.fluid_temp_c, 'C', 'fluid_inlet_temp' in s.profiles)}"
        f"  V̇={_f(s.volume_flow_lpm, 'l/min', 'fluid_mass_flow' in s.profiles)}"
        f"  | {s.art}"
    )


def check_split(train: Sequence[str], val: Sequence[str],
                test: Sequence[str]) -> List[str]:
    """Warnings about a split, from the plan sheet alone (no data needed).

    Returns a list of human-readable lines; an empty list means the split obeys
    the rules above. This is intentionally advisory: a deliberately hard split is
    a legitimate choice, it just should not be an accidental one.
    """
    warnings: List[str] = []
    seen = list(train) + list(val) + list(test)
    dupes = sorted({o for o in seen if seen.count(o) > 1})
    if dupes:
        warnings.append(f"OP(s) used in more than one role: {', '.join(dupes)}")
    unknown = [o for o in seen if o not in OPS]
    if unknown:
        warnings.append(f"OP(s) not in the plan sheet: {', '.join(unknown)}")
    unused = [o for o in ALL_OPS if o not in seen]
    if unused:
        warnings.append(f"OP(s) in the sheet but in no role: {', '.join(unused)}")

    trained_profiles = {p for o in train if o in OPS for p in OPS[o].profiles}
    for op_id in val:
        if op_id not in OPS:
            continue
        new = [p for p in OPS[op_id].profiles if p not in trained_profiles]
        if new:
            warnings.append(
                f"selection OP {op_id} carries a profile type never trained on "
                f"({', '.join(new)}) - the ranking would then be driven by "
                f"extrapolation, which is not what selection should optimise."
            )
        if OPS[op_id].tier == TIER_EXTRAP:
            warnings.append(
                f"selection OP {op_id} is tier {TIER_EXTRAP}; selecting on the "
                f"extrapolation tier spends the only out-of-envelope evidence."
            )
    if not any(OPS[o].has_profile for o in train if o in OPS):
        warnings.append(
            "no training OP carries a profile - every profile OP held out is "
            "then an extrapolation test, and the driver-rate channels this "
            "extension adds never see a non-zero rate during training."
        )
    return warnings


def split_summary(train: Sequence[str], val: Sequence[str],
                  test: Sequence[str]) -> List[str]:
    """Readable block describing a split; used by the benchmark header."""
    def _block(name: str, ops: Sequence[str]) -> List[str]:
        lines = [f"{name} ({len(ops)}):"]
        for op_id in ops:
            lines.append("    " + (describe(op_id) if op_id in OPS
                                   else f"{op_id}  (not in the plan sheet)"))
        return lines

    lines = _block("train", train) + _block("val (selection)", val) \
        + _block("test (report only)", test)
    lines.append("tiers: " + "; ".join(f"{t} = {TIER_MEANING[t]}"
                                       for t in TIER_ORDER))
    warn = check_split(train, val, test)
    if warn:
        lines.append("split warnings:")
        lines += [f"    ! {w}" for w in warn]
    return lines


if __name__ == "__main__":
    print("Operating points (plan sheet, no data needed):")
    for op_id in ALL_OPS:
        print("  " + describe(op_id))
    print(f"\n  constant drivers: {', '.join(constant_ops())}")
    print(f"  with profiles   : {', '.join(profile_ops())}")
    print(f"  out of scope    : {', '.join(MEASUREMENT_OPS)} "
          f"(mini-module measurement comparison)")
    print()
    print("\n".join(split_summary(DEFAULT_TRAIN_OPS, DEFAULT_VAL_OPS,
                                  DEFAULT_TEST_OPS)))
