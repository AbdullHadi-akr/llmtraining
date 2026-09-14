"""Der Loeser ohne Netz.

Zwei Sorten Zusagen werden hier bewacht:

* **physikalische** -- Diffusion glaettet, Energie geht nicht verloren, wo
  keine Senke ist, eine Quelle heizt;
* **handwerkliche** -- ein divergierender Lauf wird als solcher gemeldet und
  laeuft nicht stumm bis ``inf``.

Die zweite Sorte klingt nebensaechlich und ist es nicht: der Fahrplan sagt
ausdruecklich, Divergenz in Stufe 3 werde **nicht mit dem Netz uebertuencht**.
Dafuer muss sie erst einmal zuverlaessig auffallen.
"""

from __future__ import annotations

import numpy as np
import pytest
import torch

import grid as gridmod
import physics as phys
import solve


def _fo(layout, kappa=1e-5):
    fo = torch.zeros(*layout.shape, 3, 3, dtype=torch.float64)
    for i in range(3):
        fo[..., i, i] = kappa
    return fo


def _zero_src(layout):
    return torch.zeros(layout.shape, dtype=torch.float64)


# ---------------------------------------------------------------------------
# Physik
# ---------------------------------------------------------------------------
def test_diffusion_glaettet(layout, random_field):
    """Faellt die Varianz nicht, hat der Stern ein falsches Vorzeichen."""
    fo = _fo(layout)
    dt = 0.2 * phys.cfl_limit(layout, fo)
    res = solve.rollout(layout=layout, tn0=random_field, n_steps=20, dt_n=dt,
                        fo_field=fo, qsrc_n=_zero_src(layout))
    assert res.stable
    assert res.tn[-1].var() < res.tn[0].var()


def test_konstantes_feld_bleibt_konstant(layout):
    """Ohne Quelle und mit adiabaten Raendern darf nichts passieren.

    Das prueft alle drei Paddings auf einmal: jede falsche Geisterschicht
    wuerde hier ein Gefaelle erfinden, wo keines ist.
    """
    f = torch.full(layout.shape, 3.25, dtype=torch.float64)
    fo = _fo(layout)
    res = solve.rollout(layout=layout, tn0=f, n_steps=50,
                        dt_n=0.2 * phys.cfl_limit(layout, fo),
                        fo_field=fo, qsrc_n=_zero_src(layout))
    assert res.stable
    assert np.abs(res.tn - 3.25).max() < 1e-9


def test_adiabat_laeuft_gegen_eine_konstante(layout, random_field):
    """Ohne Quelle und ohne Senke muss das Feld flach werden.

    DAS ist die adiabate Zusage -- nicht, dass das Mittel erhalten bleibt.
    Siehe den naechsten Test: der Stern ist nicht konservativ, und das ist eine
    bewusste Folge zweier dokumentierter Entscheidungen.
    """
    fo = _fo(layout)
    res = solve.rollout(layout=layout, tn0=random_field, n_steps=4000,
                        dt_n=0.2 * phys.cfl_limit(layout, fo),
                        fo_field=fo, qsrc_n=_zero_src(layout))
    assert res.stable
    assert res.tn[-1].std() < 1e-3 * res.tn[0].std()


def test_der_stern_ist_nicht_konservativ_und_die_drift_saettigt(layout,
                                                                random_field):
    """Der gemessene Preis der nicht-konservativen Form -- festgeschrieben.

    Gerechnet wird ``Fo : grad^2 T`` mit ortskonstantem Fo, und das y/z-Padding
    ist knotenzentriertes ``reflect`` ohne Halbzellgewichte am Rand. Beides ist
    Absicht (README Sec. 5, physics.py-Kopf), beides macht den diskreten
    Operator nicht bilanztreu: die Spaltensummen sind am Rand ungleich null.

    Die Folge ist eine **saettigende** Drift des Mittels -- sie waechst, solange
    das Feld ungleichmaessig ist, und hoert auf, sobald es flach ist. Genau das
    wird hier festgenagelt, mit zwei Zusagen:

    * die Drift bleibt klein gegen die Streuung des Anfangsfeldes,
    * und sie WAECHST NICHT weiter, wenn man viermal so lange rollt.

    Waechst sie doch, leckt ein Rand echte Energie, und das waere ein Fehler
    statt einer bekannten Naeherung. Gemessen am 14.09.: ~3.6 % von ``std``.
    """
    fo = _fo(layout)
    dt = 0.2 * phys.cfl_limit(layout, fo)
    start = float(random_field.mean()), float(random_field.std())

    def drift(n):
        r = solve.rollout(layout=layout, tn0=random_field, n_steps=n, dt_n=dt,
                          fo_field=fo, qsrc_n=_zero_src(layout))
        return abs(r.tn[-1].mean() - start[0])

    kurz, lang = drift(400), drift(1600)
    assert kurz < 0.05 * start[1], f"Drift {kurz:.4g} zu gross"
    # Saettigung: viermal so lange darf nicht nennenswert mehr driften.
    assert lang < 1.1 * kurz, f"Drift waechst weiter: {kurz:.4g} -> {lang:.4g}"


def test_quelle_heizt(layout):
    """Eine positive Quelle auf JR1 muss die Temperatur dort anheben."""
    f = torch.zeros(layout.shape, dtype=torch.float64)
    src = torch.zeros(layout.shape, dtype=torch.float64)
    src[1] = 1.0
    fo = _fo(layout)
    res = solve.rollout(layout=layout, tn0=f, n_steps=10,
                        dt_n=0.2 * phys.cfl_limit(layout, fo),
                        fo_field=fo, qsrc_n=src)
    assert res.stable
    assert res.tn[-1].mean() > 0.0


# ---------------------------------------------------------------------------
# Stabilitaet und CFL
# ---------------------------------------------------------------------------
def test_unter_der_cfl_schranke_stabil(layout, random_field):
    fo = _fo(layout)
    res = solve.rollout(layout=layout, tn0=random_field, n_steps=200,
                        dt_n=0.4 * phys.cfl_limit(layout, fo),
                        fo_field=fo, qsrc_n=_zero_src(layout))
    assert res.stable
    assert res.max_abs < 10.0


def test_ueber_der_cfl_schranke_divergiert_und_meldet_es(layout, random_field):
    """Divergenz muss auffallen, nicht stumm bis inf laufen.

    Der Fahrplan sagt: Divergenz in Stufe 3 wird nicht mit dem Netz
    uebertuencht. Dafuer muss sie zuverlaessig gemeldet werden.
    """
    fo = _fo(layout)
    res = solve.rollout(layout=layout, tn0=random_field, n_steps=500,
                        dt_n=5.0 * phys.cfl_limit(layout, fo),
                        fo_field=fo, qsrc_n=_zero_src(layout))
    assert not res.stable
    assert res.diverged_at is not None
    assert "DIVERGIERT" in res.summary()
    # Abgebrochen statt durchgerechnet: die Trajektorie endet beim Abbruch.
    assert res.tn.shape[0] == res.diverged_at + 1


def test_cfl_schranke_wird_mitprotokolliert(layout, random_field):
    fo = _fo(layout)
    dt = 0.2 * phys.cfl_limit(layout, fo)
    res = solve.rollout(layout=layout, tn0=random_field, n_steps=5, dt_n=dt,
                        fo_field=fo, qsrc_n=_zero_src(layout))
    assert res.dt_used == pytest.approx(dt)
    assert res.cfl_dt_max == pytest.approx(phys.cfl_limit(layout, fo))
    assert "dt_max" in res.summary()


# ---------------------------------------------------------------------------
# Buchfuehrung
# ---------------------------------------------------------------------------
def test_adiabater_lauf_wird_als_ablation_markiert(layout, random_field):
    """Ohne Wandterm ist das keine Physik-Latte -- und muss es auch sagen.

    Diese Notiz ist der Unterschied zwischen einem ehrlichen Zwischenstand und
    einer Zahl, die spaeter als Latte zitiert wird, obwohl die Gehaeusewand
    gar keine Senke hatte.
    """
    fo = _fo(layout)
    res = solve.rollout(layout=layout, tn0=random_field, n_steps=3,
                        dt_n=0.2 * phys.cfl_limit(layout, fo),
                        fo_field=fo, qsrc_n=_zero_src(layout))
    assert not res.wall_used
    assert any("adiabat" in n for n in res.notes)
    assert "Wandterm AUS" in res.summary()


def test_trajektorie_kommt_flach_zurueck(layout, random_field):
    """``op_metrics`` erwartet ``(n_t, n_points)`` -- wie ``data.OPData.Tn``."""
    fo = _fo(layout)
    res = solve.rollout(layout=layout, tn0=random_field, n_steps=7,
                        dt_n=0.2 * phys.cfl_limit(layout, fo),
                        fo_field=fo, qsrc_n=_zero_src(layout))
    assert res.tn.shape == (8, layout.n_points)
    np.testing.assert_allclose(
        res.tn[0], gridmod.to_flat(random_field, layout).numpy(), atol=1e-12)


def test_wall_ohne_scaling_faellt_laut(layout, random_field):
    """``wall`` und ``scaling`` gehoeren zusammen -- sonst fehlt die Bruecke."""
    w = phys.WallModel(
        layout, phys.UCurve.from_samples([0.0], [50.0]), cp_fluid=3500.0,
        lam_xx_wall=torch.tensor(200.0, dtype=torch.float64))
    with pytest.raises(ValueError, match="gehoeren zusammen"):
        solve.rollout(layout=layout, tn0=random_field, n_steps=1, dt_n=1e-3,
                      fo_field=_fo(layout), qsrc_n=_zero_src(layout), wall=w)


def test_zeitabhaengige_quelle_wird_schrittweise_gelesen(layout):
    """``Qsrc`` darf eine Folge sein -- ``data.OPData.Qsrc`` ist genau das."""
    f = torch.zeros(layout.shape, dtype=torch.float64)
    src = torch.zeros(4, *layout.shape, dtype=torch.float64)
    src[2:, 1] = 1.0                      # Quelle schaltet erst spaet ein
    fo = _fo(layout)
    res = solve.rollout(layout=layout, tn0=f, n_steps=4,
                        dt_n=0.2 * phys.cfl_limit(layout, fo),
                        fo_field=fo, qsrc_n=src)
    assert res.stable
    assert abs(res.tn[2].mean()) < 1e-12          # noch kalt
    assert res.tn[-1].mean() > 0.0                # danach geheizt
