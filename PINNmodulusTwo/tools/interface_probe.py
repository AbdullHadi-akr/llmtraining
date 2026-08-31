"""Wie viele Kollokationspunkte liegen an einer Materialgrenze -- und zaehlt der
fehlende ``(grad lambda) . grad T``-Term im Inneren ueberhaupt?

Beantwortet die beiden Messungen aus ``ARCHITECTURE.md`` 4.1, OHNE etwas zu
trainieren: kein Modulus, kein torch, keine GPU. Messung 1 braucht nur numpy und
``data_cache/OP*.npz`` und liest daraus ``xyz``, ``layer`` und ``T``. Messung 2
braucht zusaetzlich ``material_properties/`` und wird sonst uebersprungen.

    python3 PINNmodulusTwo/tools/interface_probe.py

Warum das vor jeder Codeaenderung kommt
---------------------------------------
``physics.heat_residual`` rechnet ``Fo : grad^2 T``, die nicht-konservative Form.
Der Term ``(grad lambda) . grad T`` fehlt. Innerhalb einer Materialregion ist das
eine Naeherung, deren Fehler messbar ist (Messung 2). An einer Regionsgrenze ist
es keine Naeherung mehr: dort ist ``T`` nur ``C^0``, ``grad^2 T`` existiert nicht,
und ``train.py`` zieht seine Kollokationspunkte trotzdem gleichverteilt ueber
ALLE Punkte. Ob das Rand- oder Hauptsache ist, haengt allein daran, wie gross der
betroffene Anteil ist -- das ist Messung 1.

Die beiden Zahlen entscheiden zwischen den Optionen A/B/C in ``ARCHITECTURE.md``
4.1. Ohne sie waere jede Aenderung am Physik-Term geraten.

Was hier NICHT gemessen wird
----------------------------
Ob das Modell besser wird. Das ist eine Trainingsfrage und haengt an Schritt A
aus ``README_MODEL_CRITIQUE.md``, nicht an dieser Sonde.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

_HERE = Path(__file__).resolve().parent
_PROJECT = _HERE.parent
_ROOT = _PROJECT.parent

# Same search order as data.py / data_probe.py, most specific first.
_CACHE_CANDIDATES = (
    _PROJECT / "data_cache",
    _ROOT / "data_cache",
    _ROOT / "legacy" / "battery_surrogate_agenticWorkflow" / "data_cache",
    _ROOT / "battery_surrogate_agenticWorkflow" / "data_cache",
)


def _resolve_cache(explicit: str | None) -> Path:
    if explicit:
        p = Path(explicit).expanduser().resolve()
        if not p.is_dir():
            sys.exit(f"[ABORT] --data-cache {p} is not a directory")
        return p
    for c in _CACHE_CANDIDATES:
        if c.is_dir() and any(c.glob("*.npz")):
            return c
    sys.exit("[ABORT] no data_cache with OP*.npz found. Looked in:\n  "
             + "\n  ".join(str(c) for c in _CACHE_CANDIDATES)
             + "\nPass --data-cache <path> if it lives elsewhere.")


def _knn(xyz: np.ndarray, k: int) -> np.ndarray:
    """Indices of the k nearest neighbours of every point, excluding itself.

    363 points -- the full (P, P) distance matrix is 1 MB and needs no kd-tree.
    """
    d2 = ((xyz[:, None, :] - xyz[None, :, :]) ** 2).sum(-1)
    np.fill_diagonal(d2, np.inf)
    return np.argsort(d2, axis=1)[:, :k]


def measure_interface_fraction(xyz: np.ndarray, layer: np.ndarray,
                               k: int) -> dict:
    """Messung 1: Anteil der Punkte mit einem Nachbarn aus einer anderen Region.

    Das ist die Definition, die Option A umsetzen wuerde: ein Punkt ist
    Grenzpunkt, wenn unter seinen k naechsten Nachbarn eine andere Region
    vorkommt. Sie haengt an ``k`` -- deshalb gibt die Sonde mehrere ``k`` aus,
    damit sichtbar ist, ob das Ergebnis daran haengt oder robust ist.
    """
    nn = _knn(xyz, k)
    same = layer[nn] == layer[:, None]
    is_interface = ~same.all(axis=1)
    per_region = {}
    for lab in np.unique(layer):
        m = layer == lab
        per_region[str(lab)] = {
            "n": int(m.sum()),
            "n_interface": int(is_interface[m].sum()),
            "frac": float(is_interface[m].mean()),
        }
    return {
        "k": k,
        "n_points": int(len(layer)),
        "n_interface": int(is_interface.sum()),
        "frac": float(is_interface.mean()),
        "per_region": per_region,
        "mask": is_interface,
    }


def _quadratic_derivs(xyz: np.ndarray, values: np.ndarray, nn: np.ndarray,
                      keep: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Local weighted least squares: fit a quadratic, read off grad and Hessian.

    The grid is scattered, so there is no stencil to apply. A quadratic in 3D has
    10 coefficients [1, x, y, z, x^2, y^2, z^2, xy, xz, yz]; with >= 10 usable
    neighbours the fit is determined and its linear/quadratic coefficients ARE
    the derivatives at the centre point (the expansion is around that point).

    ``keep`` masks neighbours that must not enter the fit -- for the
    within-region measurement that is every neighbour from another material,
    because fitting a smooth polynomial across a jump is exactly the error this
    whole probe is about.

    Returns ``(grad (P,3), lap_diag (P,3))``; rows that had too few usable
    neighbours come back as NaN and are excluded downstream.
    """
    P = len(xyz)
    grad = np.full((P, 3), np.nan)
    lap = np.full((P, 3), np.nan)
    for i in range(P):
        idx = nn[i][keep[i]]
        if len(idx) < 10:
            continue
        d = xyz[idx] - xyz[i]
        # Scale-free weighting: nearer neighbours dominate, but nothing is hard
        # cut off. h is the local spacing, so the weight is geometry-independent.
        h = np.linalg.norm(d, axis=1).max() + 1e-30
        w = np.exp(-(np.linalg.norm(d, axis=1) / h) ** 2)
        A = np.column_stack([
            np.ones(len(idx)), d[:, 0], d[:, 1], d[:, 2],
            0.5 * d[:, 0] ** 2, 0.5 * d[:, 1] ** 2, 0.5 * d[:, 2] ** 2,
            d[:, 0] * d[:, 1], d[:, 0] * d[:, 2], d[:, 1] * d[:, 2],
        ])
        b = values[idx] - values[i]
        Aw = A * w[:, None]
        try:
            coef, *_ = np.linalg.lstsq(Aw, b * w, rcond=None)
        except np.linalg.LinAlgError:
            continue
        grad[i] = coef[1:4]
        lap[i] = coef[4:7]          # the 0.5*d^2 basis makes these the plain second derivatives
    return grad, lap


def measure_gradlambda_term(xyz, layer, T, lam_iso, nn, interface_mask) -> dict:
    """Messung 2: RMS(|grad lambda| * |grad T|) / RMS(lambda * lap T), im Inneren.

    Nur ueber Nicht-Grenzpunkte und nur mit Nachbarn derselben Region -- die
    Frage ist ausdruecklich, ob der fehlende Term INNERHALB eines Materials
    zaehlt. An der Grenze ist er keine Korrektur, sondern eine Distribution, und
    dort ist die Antwort nicht 'Term nachruesten', sondern 'PDE gilt nicht'.
    """
    same_region = layer[nn] == layer[:, None]
    interior = ~interface_mask
    if interior.sum() == 0:
        return {"error": "no interior points at this k"}

    gl, _ = _quadratic_derivs(xyz, lam_iso, nn, same_region)
    gT, lT = _quadratic_derivs(xyz, T, nn, same_region)

    ok = interior & np.isfinite(gl).all(1) & np.isfinite(gT).all(1) \
        & np.isfinite(lT).all(1)
    if ok.sum() == 0:
        return {"error": "no point had >= 10 same-region neighbours"}

    cross = np.abs((gl[ok] * gT[ok]).sum(1))          # |(grad lam).(grad T)|
    main = np.abs(lam_iso[ok] * lT[ok].sum(1))        # |lam * lap T|
    r_cross = float(np.sqrt((cross ** 2).mean()))
    r_main = float(np.sqrt((main ** 2).mean()))
    return {
        "n_used": int(ok.sum()),
        "rms_cross": r_cross,
        "rms_main": r_main,
        "ratio": float(r_cross / (r_main + 1e-30)),
    }


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--data-cache", default=None)
    ap.add_argument("--op", default="OP01",
                    help="OP whose grid and temperatures are probed")
    ap.add_argument("--k", type=int, nargs="+", default=[6, 12, 18],
                    help="neighbour counts for the interface definition")
    ap.add_argument("--n-times", type=int, default=20,
                    help="time slices averaged for measurement 2")
    cli = ap.parse_args()

    cache = _resolve_cache(cli.data_cache)
    f = cache / f"{cli.op}.npz"
    if not f.exists():
        sys.exit(f"[ABORT] {f} not found")
    npz = np.load(f, allow_pickle=True)
    xyz = np.asarray(npz["xyz"], dtype=np.float64)
    layer = np.asarray(npz["layer"])
    T = np.asarray(npz["T"], dtype=np.float64)

    print("=" * 70)
    print(f"INTERFACE PROBE -- {cli.op}   ({cache})")
    print("=" * 70)
    print(f"  points: {len(layer)}   regions: "
          + ", ".join(f"{lab}={int((layer == lab).sum())}"
                      for lab in np.unique(layer)))
    print()

    # ---- Messung 1 -----------------------------------------------------------
    print("Messung 1 -- Anteil grenzflaechennaher Punkte")
    print("-" * 70)
    m1 = {}
    for k in cli.k:
        r = measure_interface_fraction(xyz, layer, k)
        m1[k] = r
        detail = "  ".join(f"{lab}:{v['n_interface']}/{v['n']}"
                           for lab, v in r["per_region"].items())
        print(f"  k={k:<3d} {r['n_interface']:>4d}/{r['n_points']} "
              f"= {100 * r['frac']:5.1f} %   [{detail}]")
    frac = m1[cli.k[0]]["frac"]
    print()
    if frac < 0.10:
        print(f"  -> {100*frac:.1f} %: Randsache. Option A (Grenzpunkte aus dem")
        print("     batch_phys-Sampling nehmen) reicht, B lohnt kaum.")
    elif frac < 0.35:
        print(f"  -> {100*frac:.1f} %: spuerbar. Option A klar sinnvoll; ob B")
        print("     noetig ist, entscheidet der MAE-Effekt von A.")
    else:
        print(f"  -> {100*frac:.1f} %: Hauptsache, nicht Randsache. Option A wirft")
        print("     einen grossen Teil des Physik-Terms weg -- dann ist B (echte")
        print("     Flusskopplung) die eigentlich richtige Antwort.")
    print()

    # ---- Messung 2 -----------------------------------------------------------
    print("Messung 2 -- RMS((grad lam).(grad T)) / RMS(lam * lap T), im Inneren")
    print("-" * 70)
    try:
        sys.path.insert(0, str(_PROJECT))
        from materials import load_material_properties
    except Exception as e:                                  # noqa: BLE001
        print(f"  uebersprungen: material_properties/ nicht ladbar ({e})")
        print("  Messung 1 oben ist davon unabhaengig und bleibt gueltig.")
        return 0

    props = load_material_properties(layer=layer)
    lam = np.asarray(props["lambda_tensor"], dtype=np.float64)
    lam_iso = (lam[:, 0, 0] + lam[:, 1, 1] + lam[:, 2, 2]) / 3.0

    k = cli.k[0]
    nn = _knn(xyz, max(k, 18))          # the LSQ fit needs >= 10 usable rows
    mask = m1[k]["mask"]
    t_idx = np.linspace(0, len(T) - 1, min(cli.n_times, len(T))).astype(int)
    ratios = []
    for ti in t_idx:
        r = measure_gradlambda_term(xyz, layer, T[ti], lam_iso, nn, mask)
        if "error" in r:
            print(f"  {r['error']}")
            return 0
        ratios.append(r["ratio"])
    ratios = np.array(ratios)
    print(f"  k={k}, {len(t_idx)} Zeitpunkte, "
          f"{measure_gradlambda_term(xyz, layer, T[t_idx[0]], lam_iso, nn, mask)['n_used']} "
          f"Innenpunkte")
    print(f"  Verhaeltnis: median {np.median(ratios):.4f}   "
          f"p90 {np.percentile(ratios, 90):.4f}   max {ratios.max():.4f}")
    print()
    if np.median(ratios) < 0.01:
        print("  -> < 1 %: der fehlende Term ist im Inneren irrelevant. lambda")
        print("     fix lassen, Option C entfaellt.")
    else:
        print(f"  -> {100*np.median(ratios):.1f} %: nicht vernachlaessigbar. Option C")
        print("     (grad lambda . grad T fuer die glatte Variation innerhalb")
        print("     einer Region) wird diskutabel -- an Grenzen bleibt sie falsch.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
