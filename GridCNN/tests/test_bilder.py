"""Rauchtest fuer ``tools/bilder.py``: aus JSON in der Form, die Training und
``nachmessen.py`` schreiben, entstehen die PNGs -- ohne GPU und ohne Daten."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pytest

pytest.importorskip("matplotlib")
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "tools"))
import bilder as B  # noqa: E402


def _kurve(rng, n=50, form=(3, 11, 11)):
    t = np.linspace(0, 100, n)
    mae = np.abs(rng.normal(1, 0.2, n))
    return {"t_s": t.tolist(), "split_t_s": 80.0, "mae": mae.tolist(),
            "abs_min": (mae * 0.1).tolist(), "abs_q25": (mae * 0.6).tolist(),
            "abs_q50": mae.tolist(), "abs_q75": (mae * 1.4).tolist(),
            "abs_max": (mae * 3).tolist(), "bias": (mae - 1).tolist(),
            "fehler_min": (-mae * 2).tolist(),
            "fehler_q25": (-mae * 0.5).tolist(),
            "fehler_q75": (mae * 0.5).tolist(),
            "fehler_max": (mae * 2).tolist(),
            "T_wahr_mittel": (30 + t / 5).tolist(),
            "T_wahr_min": (28 + t / 5).tolist(),
            "T_wahr_max": (35 + t / 4).tolist(),
            "T_modell_mittel": (31 + t / 5).tolist(),
            "T_modell_min": (29 + t / 5).tolist(),
            "T_modell_max": (36 + t / 4).tolist(),
            "karte_form": list(form),
            "karte_mae": np.abs(rng.normal(1, 0.3, int(np.prod(form))))
                           .tolist(),
            "karte_bias": rng.normal(0, 1, int(np.prod(form))).tolist()}


def _profil(rng, ops, kurven=True):
    out = {}
    for op in ops:
        segs = np.abs(rng.normal(2, 0.5, 6))
        out[op] = {"mae": float(segs.mean()), "bias": 0.3, "drift": 1.1,
                   "mae_segmente": segs.tolist(),
                   "bias_segmente": (segs - 2).tolist()}
        if kurven:
            out[op]["kurve"] = _kurve(rng)
    return out


def _lauf(verz: Path, rng, *, insample: bool) -> None:
    val = ["OP06", "OP09"]
    je = {}
    for s in range(3):
        d = verz / f"seed{s}"
        d.mkdir(parents=True)
        hist = [{"epoch": e, "data": 1 / e, "grad_norm": 1.0 + e % 3,
                 "saturated": 0, "saturated_max": 100,
                 **({"val_mae_C": {o: 5.0 / e for o in val},
                     "val_detail": _profil(rng, val, kurven=False)}
                    if e % 2 == 0 else {})} for e in range(1, 11)]
        (d / "history.json").write_text(json.dumps(hist))
        (d / "metrics.json").write_text(json.dumps({
            "val_mae_C_berichtet": {o: 5.0 + s for o in val},
            "fehlerprofil": _profil(rng, val, kurven=False),
            "triviale_latten_C": {"OP06": {"mittel": 10.8},
                                  "OP09": {"mittel": 7.76}}}))
        (d / "model.pt").write_bytes(b"")
        je[f"seed{s}/model.pt"] = _profil(rng, val)
        je[f"seed{s}/model_best.pt"] = _profil(rng, val)
    (verz / "nachgemessen.json").write_text(json.dumps({"je_checkpoint": je}))
    if insample:
        train = [f"OP{i:02d}" for i in (1, 2, 3, 4, 5, 7, 8, 10, 11, 12, 14)]
        je = {f"seed{s}/model.pt": _profil(rng, train) for s in range(3)}
        (verz / "nachgemessen_insample.json").write_text(
            json.dumps({"je_checkpoint": je}))


def test_aus_den_json_entstehen_die_bilder(tmp_path):
    rng = np.random.default_rng(0)
    _lauf(tmp_path / "A", rng, insample=True)
    _lauf(tmp_path / "B-exp", rng, insample=False)
    aus = tmp_path / "bilder"
    assert B.main([str(tmp_path / "A"), str(tmp_path / "B-exp"),
                   "--aus", str(aus)]) == 0
    namen = {p.name for p in aus.glob("*.png")}
    erwartet = {"A_lernkurven.png", "A_bias_epochen.png",
                "A_halte_sensoren.png", "A_halte_vorzeichen.png",
                "A_halte_temperatur.png", "A_halte_karte.png",
                "A_halte_profil.png", "A_insample_sensoren_seed0.png",
                "A_insample_karte.png", "B-exp_halte_sensoren.png",
                "vergleich_latte.png", "vergleich_profil.png",
                "vergleich_zeit.png"}
    assert erwartet <= namen, erwartet - namen
    assert all((aus / n).stat().st_size > 5_000 for n in erwartet)


def test_ohne_zeitreihen_gibt_es_nur_das_profil(tmp_path):
    """Eine alte nachgemessen.json (vor dem 25.09.) traegt keine ``kurve``:
    dann das Profil und ein Hinweis, kein Absturz."""
    rng = np.random.default_rng(1)
    verz = tmp_path / "A"
    verz.mkdir()
    je = {f"seed{s}/model.pt": _profil(rng, ["OP06"], kurven=False)
          for s in range(2)}
    (verz / "nachgemessen.json").write_text(json.dumps({"je_checkpoint": je}))
    aus = tmp_path / "bilder"
    assert B.main([str(verz), "--aus", str(aus)]) == 0
    assert (aus / "A_halte_profil.png").exists()
    assert not (aus / "A_halte_sensoren.png").exists()


def test_model_best_bleibt_draussen():
    """``model_best.pt`` ist auf der Haltemenge ausgewaehlt -- in den Bildern
    steht nur der Stand der letzten Epoche."""
    js = B.je_seed({"je_checkpoint": {"seed0/model.pt": {"OP06": 1},
                                      "seed0/model_best.pt": {"OP06": 2}}})
    assert js == {"seed0": {"OP06": 1}}
