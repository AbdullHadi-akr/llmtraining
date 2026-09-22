"""Tests fuer die Trainingsschleife.

Der wichtigste Test ist ``test_der_gradient_ueberquert_die_historie_nicht``.
Er haelt die Eigenschaft fest, aus der der ganze Hebel von Stufe 5 folgt: weil
der Gradient nur **einen** Schritt weit laeuft, kann er den Spaetfehler (O13)
strukturell nicht erreichen -- egal wie lange man trainiert. Wer diese
Schleife optimiert und die Eigenschaft dabei wegraeumt, hat Stufe 5 unmoeglich
gemacht, ohne dass eine Trainingskurve es zeigen wuerde.
"""

from __future__ import annotations

import numpy as np
import pytest
import torch

import model as M
import train as T


# ---------------------------------------------------------------------------
# Hilfen
# ---------------------------------------------------------------------------
@pytest.fixture
def statics(layout):
    rng = np.random.default_rng(0)
    n = layout.n_points
    rho = np.where(np.arange(n) % 3 == 1, 2800.0, 2500.0)
    return M.build_static_maps(layout, lam=rng.random((n, 3, 3)) + 1.0,
                               rho=rho, cp=np.full(n, 900.0))


@pytest.fixture
def op(layout):
    """Ein synthetischer Betriebspunkt -- kein Cache noetig."""
    torch.manual_seed(0)
    nx, ny, nz = layout.shape
    n_t = 40
    fo = torch.zeros(nx, ny, nz, 3, 3)
    fo[..., 0, 0], fo[..., 1, 1], fo[..., 2, 2] = 3e-3, 2e-3, 2.5e-3
    fo[1, ..., 0, 1] = 4e-4
    qsrc = torch.zeros(n_t, nx, ny, nz)
    qsrc[:, 1] = 0.05
    return T.OPTensors(
        op_id="OP99",
        tn_seq=torch.randn(n_t, nx, ny, nz),
        tn_ic=torch.randn(nx, ny, nz),
        qsrc=qsrc, fo=fo,
        config=torch.randn(n_t, 7), forcing=torch.randn(n_t, 11),
        dtn=0.01, split_t=30, n_t=n_t)


@pytest.fixture
def net(layout):
    torch.manual_seed(0)
    n = M.GridCNN(layout)
    with torch.no_grad():           # sonst ist g_theta identisch null
        n.correction.head.weight.normal_(0.0, 0.05)
    return n


# ---------------------------------------------------------------------------
# Die Eigenschaft, die der Fahrplan schuetzt
# ---------------------------------------------------------------------------
def test_der_rollout_traegt_keinen_graphen(net, op, statics):
    """``@torch.no_grad()`` -- der Puffer ist eingefroren, nicht nur gemeint."""
    traj, _ = T.rollout(net, op, statics, lag1=5, lag2=20)
    assert traj.requires_grad is False
    assert traj.grad_fn is None
    assert traj.shape == (op.n_t, *op.tn_ic.shape)


def test_der_gradient_ueberquert_die_historie_nicht(net, op, statics):
    """Der Gradient laeuft genau **einen** Schritt weit.

    Konkret geprueft: stoert man die Trajektorie an einem Zeitpunkt, der weder
    Anker noch einer der beiden Lags ist, aendert sich der Verlust **nicht**.
    Waere die Rekurrenz durchgaengig differenzierbar, haenge jeder spaetere
    Schritt an jedem frueheren, und die Stoerung schlueg durch.

    Genau deshalb ist truncated BPTT (Stufe 5) die einzige Aenderung, die den
    Spaetfehler erreichen kann -- und genau deshalb steht dieser Test hier.
    """
    traj, _ = T.rollout(net, op, statics, lag1=5, lag2=20)
    idx = torch.tensor([25])
    kw = dict(lag1=5, lag2=20)

    vorher = float(T.data_loss(net, op, statics, traj, idx, **kw).detach())
    gestoert = traj.clone()
    # 25 ist der Anker, 20 und 5 sind die Lags. 12 ist keines davon.
    gestoert[12] += 10.0
    nachher = float(T.data_loss(net, op, statics, gestoert, idx, **kw).detach())
    assert vorher == nachher, "die Stoerung schlaegt durch -- Gradient laeuft weiter"

    # Gegenprobe: der Anker selbst MUSS durchschlagen, sonst misst der Test nichts
    anker = traj.clone()
    anker[25] += 10.0
    assert float(T.data_loss(net, op, statics, anker, idx, **kw).detach()) != vorher


def test_die_lags_schlagen_durch(net, op, statics):
    """Die zweite Gegenprobe: beide Lags haengen wirklich dran."""
    traj, _ = T.rollout(net, op, statics, lag1=5, lag2=20)
    idx = torch.tensor([25])
    kw = dict(lag1=5, lag2=20)
    basis = float(T.data_loss(net, op, statics, traj, idx, **kw).detach())
    for lag_t in (20, 5):                       # 25-5 = 20, 25-20 = 5
        g = traj.clone()
        g[lag_t] += 10.0
        assert float(T.data_loss(net, op, statics, g, idx, **kw).detach()) != basis, lag_t


def test_die_historie_kommt_aus_dem_puffer_nicht_aus_den_labels(net, op, statics):
    """Wo Teacher Forcing entstehen wuerde, wenn man ``tn_seq`` einsetzte."""
    traj, _ = T.rollout(net, op, statics, lag1=5, lag2=20)
    idx = torch.tensor([10, 15])
    kw = dict(lag1=5, lag2=20)
    basis = float(T.data_loss(net, op, statics, traj, idx, **kw).detach())

    # Labels VOR dem Ziel veraendern: sie sind Historie nur im Teacher-Forcing-Fall
    op.tn_seq[5] += 100.0
    op.tn_seq[9] += 100.0
    assert float(T.data_loss(net, op, statics, traj, idx, **kw).detach()) == basis

    # das Ziel selbst muss sehr wohl durchschlagen
    op.tn_seq[11] += 100.0
    assert float(T.data_loss(net, op, statics, traj, idx, **kw).detach()) != basis


def test_der_gradient_erreicht_die_gewichte(net, op, statics):
    traj, _ = T.rollout(net, op, statics, lag1=5, lag2=20)
    loss = T.data_loss(net, op, statics, traj, torch.tensor([5, 11]),
                       lag1=5, lag2=20)
    loss.backward()
    for name, p in net.named_parameters():
        assert p.grad is not None and torch.isfinite(p.grad).all(), name


# ---------------------------------------------------------------------------
# Historie am Rand
# ---------------------------------------------------------------------------
def test_die_lags_werden_bei_null_geklemmt(op):
    """Am Anfang gibt es keine Vergangenheit -- geklemmt, nicht abgeschnitten."""
    traj = torch.randn(30, *op.tn_ic.shape)
    t0, t1, t2 = T.history_at(traj, torch.tensor([2]), 5, 20)
    assert torch.equal(t0, traj[2:3])
    assert torch.equal(t1, traj[0:1])       # 2-5 -> 0
    assert torch.equal(t2, traj[0:1])       # 2-20 -> 0


def test_ausserhalb_des_randes_wird_nicht_geklemmt(op):
    traj = torch.randn(30, *op.tn_ic.shape)
    t0, t1, t2 = T.history_at(traj, torch.tensor([25]), 5, 20)
    assert torch.equal(t1, traj[20:21])
    assert torch.equal(t2, traj[5:6])


# ---------------------------------------------------------------------------
# Der Saettigungszaehler
# ---------------------------------------------------------------------------
def test_ein_weglaufender_rollout_wird_gezaehlt(layout, op, statics):
    """Stille saehe hier aus wie langsame Konvergenz. Das ist der teuerste Irrtum.

    Der Bias wird auf einen festen, viel zu grossen Wert gesetzt statt zufaellig
    gezogen: ``g_theta`` ist dann konstant 500, die Rate also ``L + Q + 500``,
    und bei ``dt = 0.01`` waechst das Feld garantiert um 5 je Schritt. Mit
    zufaelligen Gewichten haenge der Test daran, wie die Fixtures den
    RNG-Zustand hinterlassen -- und ein Test, der von der Reihenfolge seiner
    Fixtures abhaengt, misst nicht das, was in seinem Namen steht.
    """
    net = M.GridCNN(layout)
    with torch.no_grad():
        net.correction.head.bias.fill_(500.0)
    _, sat = T.rollout(net, op, statics, lag1=5, lag2=20, clamp=10.0)
    assert sat > 0


def test_ohne_clamp_wird_nichts_gezaehlt(net, op, statics):
    _, sat = T.rollout(net, op, statics, lag1=5, lag2=20, clamp=0.0)
    assert sat == 0


# ---------------------------------------------------------------------------
# Diagnose
# ---------------------------------------------------------------------------
def test_ein_flachgelaufenes_modell_faellt_am_streuungsverhaeltnis_auf(op):
    """Die triviale Loesung: konstant in Ort und Zeit. Beide Residuen sind dann 0."""
    flach = torch.zeros(op.n_t, *op.tn_ic.shape)
    s, t = T.spread_ratios(flach, op.tn_seq)
    assert s < 1e-6 and t < 1e-6


def test_ein_strukturiertes_modell_hat_ein_verhaeltnis_nahe_eins(op):
    s, t = T.spread_ratios(op.tn_seq.clone(), op.tn_seq)
    assert abs(s - 1.0) < 1e-5 and abs(t - 1.0) < 1e-5


def test_bei_init_ist_das_physik_residuum_exakt_null(layout, op, statics):
    """``head`` ist null -> ``g_theta`` ist null -> die Rate IST die Physik."""
    frisch = M.GridCNN(layout)
    traj, _ = T.rollout(frisch, op, statics, lag1=5, lag2=20)
    lp, verh = T.physics_loss(frisch, op, statics, traj, torch.tensor([5, 12]),
                              lag1=5, lag2=20)
    assert float(lp.detach()) == 0.0
    assert float(verh) == 0.0


def test_nach_dem_lernen_ist_das_residuum_ungleich_null(net, op, statics):
    traj, _ = T.rollout(net, op, statics, lag1=5, lag2=20)
    lp, verh = T.physics_loss(net, op, statics, traj, torch.tensor([5, 12]),
                              lag1=5, lag2=20)
    assert float(lp.detach()) > 0.0 and float(verh) > 0.0


# ---------------------------------------------------------------------------
# Der Wandterm
# ---------------------------------------------------------------------------
def test_wall_loss_summiert_ueber_die_knotenflaechen():
    q = torch.full((2, 11, 11), 100.0)      # W/m^2
    flaeche = 0.0206 / 121                  # A / (ny*nz)
    ziel = torch.full((2,), 100.0 * 0.0206)
    assert float(T.wall_loss(q, ziel, flaeche)) < 1e-8


def test_der_wandterm_faellt_laut_aus_statt_zu_raten(net, op, statics):
    """Stufe 2 ist offen. Ein geratenes U waere ein freier Parameter."""
    wall = object()
    with pytest.raises(NotImplementedError, match="U\\(V_dot\\)"):
        T._wall_ghost(op.tn_ic.unsqueeze(0), wall, op, 0)


# ---------------------------------------------------------------------------
# Eine Epoche
# ---------------------------------------------------------------------------
def test_eine_epoche_veraendert_die_gewichte(layout, op, statics):
    torch.manual_seed(0)
    net = M.GridCNN(layout)
    vorher = [p.detach().clone() for p in net.parameters()]
    opt = torch.optim.Adam(net.parameters(), lr=1e-3)
    st = T.train_epoch(net, T.stack_ops([op]), statics, opt, inner_steps=3, batch_t=4,
                       lag1=5, lag2=20, w_data=1.0, w_phys=0.0, w_wall=0.0,
                       clamp=50.0, rng=np.random.default_rng(0))
    assert st.data > 0.0 and st.n_ops == 1
    assert any(not torch.equal(a, b)
               for a, b in zip(vorher, net.parameters()))


def test_der_physik_strafterm_laesst_sich_zuschalten(layout, op, statics):
    torch.manual_seed(0)
    net = M.GridCNN(layout)
    opt = torch.optim.Adam(net.parameters(), lr=1e-3)
    st = T.train_epoch(net, T.stack_ops([op]), statics, opt, inner_steps=2, batch_t=4,
                       lag1=5, lag2=20, w_data=1.0, w_phys=0.1, w_wall=0.0,
                       clamp=50.0, rng=np.random.default_rng(0))
    assert not np.isnan(st.phys)


def test_ein_abgeschalteter_term_ist_nan_und_nicht_null(layout, op, statics):
    """Eine Konvergenzkurve soll eine Luecke zeigen, keine flache Linie."""
    torch.manual_seed(0)
    net = M.GridCNN(layout)
    opt = torch.optim.Adam(net.parameters(), lr=1e-3)
    st = T.train_epoch(net, T.stack_ops([op]), statics, opt, inner_steps=2, batch_t=4,
                       lag1=5, lag2=20, w_data=1.0, w_phys=0.0, w_wall=0.0,
                       clamp=50.0, rng=np.random.default_rng(0))
    assert np.isnan(st.phys) and np.isnan(st.wall)
    assert "--" in st.line(1)


def test_w_wall_ohne_stufe_2_wird_gemeldet(layout, op, statics):
    torch.manual_seed(0)
    net = M.GridCNN(layout)
    opt = torch.optim.Adam(net.parameters(), lr=1e-3)
    st = T.train_epoch(net, T.stack_ops([op]), statics, opt, inner_steps=1, batch_t=2,
                       lag1=5, lag2=20, w_data=1.0, w_phys=0.0, w_wall=1.0,
                       clamp=50.0, rng=np.random.default_rng(0))
    assert any("Stufe 2" in n for n in st.notes)


def test_labels_hinter_split_t_werden_nicht_gefittet(layout, op, statics):
    """Sonst wird aus einer ausgehaltenen Zahl eine Trainingszahl."""
    torch.manual_seed(0)
    net = M.GridCNN(layout)
    opt = torch.optim.Adam(net.parameters(), lr=1e-3)
    gesehen = []
    echt = T.data_loss

    def spion(n, o, s, traj, idx, **kw):
        gesehen.append(idx.max().item() + 1)     # das Ziel ist idx+1
        return echt(n, o, s, traj, idx, **kw)

    T.data_loss = spion
    try:
        T.train_epoch(net, T.stack_ops([op]), statics, opt, inner_steps=40, batch_t=8,
                      lag1=5, lag2=20, w_data=1.0, w_phys=0.0, w_wall=0.0,
                      clamp=50.0, rng=np.random.default_rng(0))
    finally:
        T.data_loss = echt
    assert max(gesehen) <= op.split_t - 1, f"Ziel {max(gesehen)} >= split_t {op.split_t}"


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def test_ohne_cache_bricht_es_mit_hinweis_ab(tmp_path, capsys):
    rc = T.main(["--cache", str(tmp_path / "gibtsnicht"), "--seeds", "3"])
    assert rc == 2
    assert "Kein Cache" in capsys.readouterr().err


def test_ein_seed_wird_angemeckert(tmp_path, capsys):
    T.main(["--cache", str(tmp_path / "gibtsnicht"), "--seeds", "1"])
    assert "Ein Seed ist keine Streuung" in capsys.readouterr().err


def test_die_praesentationsgroesse_ist_ueber_die_cli_erreichbar():
    a = T.build_argparser().parse_args(["--width", "64", "--blocks", "4"])
    assert (a.width, a.blocks) == (64, 4)


# ---------------------------------------------------------------------------
# Der T4-Hebel: ein Rollout statt elf
# ---------------------------------------------------------------------------
def _op_wie(op, seed, n_t=None):
    """Ein zweiter OP mit anderen Zahlen, optional anderer Laenge."""
    torch.manual_seed(seed)
    n_t = op.n_t if n_t is None else n_t
    nx, ny, nz = op.tn_ic.shape
    q = torch.zeros(n_t, nx, ny, nz)
    q[:, 1] = 0.05
    return T.OPTensors(
        op_id=f"OP{seed}", tn_seq=torch.randn(n_t, nx, ny, nz),
        tn_ic=torch.randn(nx, ny, nz), qsrc=q, fo=op.fo,
        config=torch.randn(n_t, 7), forcing=torch.randn(n_t, 11),
        dtn=op.dtn, split_t=min(30, n_t - 1), n_t=n_t)


# Die Schranke fuer den float64-Vergleich gebatcht gegen einzeln: relativ zur
# Feldamplitude. 1e-12 ist grob 1e4 Rundungseinheiten von float64 und damit sechs
# Groessenordnungen unter der float32-Abweichung (~1e-6 relativ), die derselbe
# Vergleich zeigt. Das ist die Aussage, auf die es ankommt -- **Batchen aendert
# das Experiment nicht** -- und sie haelt auf jeder Maschine.
#
# ⚠ NICHT ``torch.equal``. Bis zum 22.09. stand hier Bitgleichheit, und sie war
# **nicht portabel**: derselbe Test lief lokal gruen und fiel auf dem
# GitHub-Runner (main, 163b21c, 17.09., ``2 failed, 97 passed`` -- main war
# deswegen fuenf Tage rot, und der Benchmark-Schritt dahinter wurde stillschweigend
# uebersprungen). torch waehlt den Faltungsalgorithmus nach Batchgroesse UND
# Maschine; Gleitkommaaddition ist nicht assoziativ, also ist Bitgleichheit
# zwischen Batch 1 und Batch 3 durch nichts garantiert -- auch in float64 nicht.
# Sie trat ein, solange die Algorithmen zufaellig uebereinstimmten. Ein Test, der
# auf so einen Zufall baut, misst die Maschine und nicht den Code.
REL_F64 = 1e-12


def test_gebatcht_rollt_in_float64_wie_einzeln(layout):
    """**Die Zusage, auf der die T4-Optimierung steht.**

    Elf OPs zusammen zu rollen ist nur dann eine Beschleunigung und keine
    Aenderung, wenn dabei dasselbe herauskommt. Waere es nicht so, waere jede
    spaetere Zahl mit der Batchgroesse verwechselbar -- und das faellt in einer
    Trainingskurve nicht auf.

    Geprueft wird in float64 gegen :data:`REL_F64`. In float32 weichen die beiden
    Wege um ~1e-6 relativ ab, und zwar nicht wegen eines Fehlers: die
    Faltungsbibliothek waehlt fuer Batch 1 einen anderen Algorithmus als fuer
    Batch 3. In float64 faellt der Unterschied um sechs Groessenordnungen -- was
    genau belegt, dass es Rechenreihenfolge ist und keine Physik. Der Test daneben
    misst die float32-Groessenordnung, damit sie nicht unbemerkt waechst.
    """
    d = torch.float64
    torch.manual_seed(0)
    net = M.GridCNN(layout).to(d)
    with torch.no_grad():
        net.correction.head.weight.normal_(0.0, 0.05)
    statics = _statics_dtype(layout, d)
    ops = [_op_dtype(layout, d, s) for s in (0, 1, 2)]
    kw = dict(lag1=5, lag2=20)

    einzeln = [T.rollout(net, o, statics, **kw)[0] for o in ops]
    gebatcht, _ = T.rollout_batched(net, T.stack_ops(ops), statics, **kw)

    for i, e in enumerate(einzeln):
        abw = float((gebatcht[:e.shape[0], i] - e).abs().max())
        skala = float(e.abs().max())
        assert abw <= REL_F64 * skala, (
            f"OP {i}: gebatcht weicht um {abw:.3e} ab, Feldamplitude {skala:.3e} "
            f"-> relativ {abw / skala:.3e} > {REL_F64:.0e}. Das ist zu viel fuer "
            f"Rechenreihenfolge in float64 -- Batchen aendert hier das Ergebnis."
        )


def test_in_float32_bleibt_die_abweichung_im_rundungsrauschen(net, op, statics):
    """Die Gegenprobe zum Test darueber: gross genug, um sie zu bemerken?"""
    ops = [op, _op_wie(op, 1), _op_wie(op, 2)]
    kw = dict(lag1=5, lag2=20)
    einzeln = [T.rollout(net, o, statics, **kw)[0] for o in ops]
    gebatcht, _ = T.rollout_batched(net, T.stack_ops(ops), statics, **kw)
    abw = max(float((gebatcht[:e.shape[0], i] - e).abs().max())
              for i, e in enumerate(einzeln))
    # z-gescorte Temperaturen liegen bei O(1); 1e-4 waere schon sichtbar.
    assert abw < 1e-4, f"float32-Abweichung {abw} ist zu gross fuer Rundung"


def _statics_dtype(layout, dtype):
    rng = np.random.default_rng(0)
    n = layout.n_points
    return M.build_static_maps(
        layout, lam=rng.random((n, 3, 3)) + 1.0,
        rho=np.where(np.arange(n) % 3 == 1, 2800.0, 2500.0),
        cp=np.full(n, 900.0), dtype=dtype)


def _op_dtype(layout, dtype, seed, n_t=40):
    torch.manual_seed(seed)
    nx, ny, nz = layout.shape
    fo = torch.zeros(nx, ny, nz, 3, 3, dtype=dtype)
    fo[..., 0, 0], fo[..., 1, 1], fo[..., 2, 2] = 3e-3, 2e-3, 2.5e-3
    q = torch.zeros(n_t, nx, ny, nz, dtype=dtype)
    q[:, 1] = 0.05
    return T.OPTensors(
        op_id=f"OP{seed}", tn_seq=torch.randn(n_t, nx, ny, nz, dtype=dtype),
        tn_ic=torch.randn(nx, ny, nz, dtype=dtype), qsrc=q, fo=fo,
        config=torch.randn(n_t, 7, dtype=dtype),
        forcing=torch.randn(n_t, 11, dtype=dtype),
        dtn=0.01, split_t=min(30, n_t - 1), n_t=n_t)


def test_verschieden_lange_ops_lassen_sich_stapeln(net, op, statics):
    """Die kuerzeren werden aufgefuellt -- gelesen wird ueber ihr Ende nie."""
    kurz = _op_wie(op, 3, n_t=25)
    b = T.stack_ops([op, kurz])
    assert b.n_max == op.n_t
    assert b.config.shape == (2, op.n_t, 7)
    # aufgefuellt wird durch Wiederholen der letzten Zeile, nicht mit Nullen
    assert torch.equal(b.config[1, 25], kurz.config[24])
    assert torch.equal(b.config[1, -1], kurz.config[24])


def test_der_kurze_op_rollt_trotzdem_richtig(layout, net, op, statics):
    """Auffuellen darf den gueltigen Teil des kurzen OP nicht veraendern.

    Geprueft wird wie beim Schwestertest oben: **float64, ohne Toleranz**, und
    daneben die float32-Groessenordnung.

    ⚠ Bis zum 22.09. stand hier nur ein float32-Vergleich gegen ``< 1e-6``.
    Diese Schranke widersprach dem Test daneben, der ~8e-6 als die normale
    float32-Abweichung ausweist -- sie hielt bei 1.2e-6 rein zufaellig. Als in
    ``conftest.py`` ``dy``/``dz`` auf die gemessenen Werte korrigiert wurden,
    stieg die Abweichung auf 1.9e-6 und der Test fiel, obwohl sich am Verhalten
    nichts geaendert hatte. Eine Schranke, die auf eine Korrektur in der
    fuenften Stelle der Gitterweite reagiert, misst Rundung und nicht Padding.

    In float64 faellt die Differenz um sechs Groessenordnungen, auch fuer den
    gepaddeten OP; geprueft gegen :data:`REL_F64` und nicht auf Bitgleichheit --
    die ist zwischen zwei Batchgroessen durch nichts garantiert. Warum, steht bei
    :data:`REL_F64`.
    """
    d = torch.float64
    torch.manual_seed(0)
    netz = M.GridCNN(layout).to(d)
    with torch.no_grad():
        netz.correction.head.weight.normal_(0.0, 0.05)
    stat64 = _statics_dtype(layout, d)
    lang64 = _op_dtype(layout, d, 0)
    kurz64 = _op_dtype(layout, d, 3, n_t=25)
    kw = dict(lag1=5, lag2=20)

    allein = T.rollout(netz, kurz64, stat64, **kw)[0]
    zusammen, _ = T.rollout_batched(
        netz, T.stack_ops([lang64, kurz64]), stat64, **kw)
    abw = float((zusammen[:25, 1] - allein).abs().max())
    skala = float(allein.abs().max())
    assert abw <= REL_F64 * skala, (
        f"gepaddet weicht um {abw:.3e} ab, Feldamplitude {skala:.3e} "
        f"-> relativ {abw / skala:.3e} > {REL_F64:.0e}"
    )

    # Die Gegenprobe in float32: gross genug, um sie zu bemerken, aber im
    # Rundungsrauschen -- dieselbe Schranke wie beim Schwestertest.
    kurz32 = _op_wie(op, 3, n_t=25)
    allein32 = T.rollout(net, kurz32, statics, **kw)[0]
    zus32, _ = T.rollout_batched(net, T.stack_ops([op, kurz32]), statics, **kw)
    abw = float((zus32[:25, 1] - allein32).abs().max())
    assert abw < 1e-4, f"float32-Abweichung {abw} ist zu gross fuer Rundung"


def test_verschiedene_zeitschritte_fallen_laut_aus(op):
    """Gemeinsam rollen setzt dasselbe dt voraus -- sonst rollen sie ungleich weit."""
    anders = _op_wie(op, 4)
    anders.dtn = 0.02
    with pytest.raises(ValueError, match="verschiedene Zeitschritte"):
        T.stack_ops([op, anders])


def test_saettigung_wird_je_op_gezaehlt_nicht_je_batchgroesse(layout, op, statics):
    """Sonst haenge die gemeldete Zahl an der Batchgroesse statt am Verhalten."""
    net = M.GridCNN(layout)
    with torch.no_grad():
        net.correction.head.bias.fill_(500.0)
    kw = dict(lag1=5, lag2=20, clamp=10.0)
    _, einer = T.rollout(net, op, statics, **kw)
    _, drei = T.rollout_batched(
        net, T.stack_ops([op, _op_wie(op, 1), _op_wie(op, 2)]), statics, **kw)
    assert drei > einer, "der Zaehler sieht die anderen OPs nicht"
    assert drei <= 3 * einer + 3


def test_eine_leere_op_liste_faellt_laut_aus():
    with pytest.raises(ValueError, match="leere OP-Liste"):
        T.stack_ops([])


def test_die_cli_kennt_device():
    """Es fehlte komplett -- ein Lauf auf der T4 braucht es."""
    a = T.build_argparser().parse_args(["--device", "cuda:0"])
    assert a.device == "cuda:0"
    assert T.build_argparser().parse_args([]).device == "ask"


# ---------------------------------------------------------------------------
# Die Ablationsflags -- eine Uebersetzung, an einer Stelle
# ---------------------------------------------------------------------------
def _args(*argv):
    return T.build_argparser().parse_args(list(argv))


def test_no_coord_maps_kommt_an_BEIDEN_stellen_an():
    """Karten und erste Faltung muessen zusammen schmal werden, nicht eine.

    Die haeufigste Art, diese Ablation kaputtzumachen, ist, nur eine der beiden
    Breiten umzustellen. Dann faellt es beim ersten Forward auf -- aber erst
    nach dem Laden der Daten, also Minuten spaeter und auf der Maschine.
    """
    kw = T.modell_kwargs(_args("--no-coord-maps"))
    assert kw["static"]["coord_maps"] is False
    assert kw["net"]["n_static"] == M.CH_STATIC_OHNE_KOORD

    vor = T.modell_kwargs(_args())
    assert vor["static"]["coord_maps"] is True
    assert vor["net"]["n_static"] == M.CH_STATIC


def test_die_uebersetzung_baut_ein_netz_das_zu_den_karten_passt(layout):
    """Ende zu Ende: was modell_kwargs sagt, laeuft auch durch."""
    import numpy as np
    n = layout.n_points
    rng = np.random.default_rng(0)
    for argv, breite in (((), 44), (("--no-coord-maps",), 42)):
        kw = T.modell_kwargs(_args(*argv))
        statics = M.build_static_maps(
            layout, lam=rng.random((n, 3, 3)) + 1.0, rho=np.full(n, 2500.0),
            cp=np.full(n, 900.0), **kw["static"])
        netz = M.GridCNN(layout, **kw["net"])
        nx, ny, nz = layout.shape
        torch.manual_seed(0)
        state = M.state_channels(*(torch.randn(2, nx, ny, nz) for _ in range(3)))
        drivers = M.driver_channels(torch.randn(2, 7), torch.randn(2, 11), ny, nz)
        x = M.assemble_input(state, statics, drivers)
        assert x.shape[1] == breite
        assert netz.correction(x).shape == (2, nx, ny, nz)


def test_der_konfigurationsname_trifft_die_vier_arme():
    """A/B/C/D wie in der Fahrplantabelle -- damit im Log steht, was lief."""
    assert T.konfigurationsname(_args()).startswith("B")
    assert T.konfigurationsname(_args("--no-physics")).startswith("A")
    assert T.konfigurationsname(_args("--w-phys", "0.1")).startswith("C")
    assert T.konfigurationsname(_args("--no-coord-maps")).startswith("D")
    # A schlaegt D: ohne Physik ist die Kartenfrage nicht mehr dieselbe Frage.
    assert T.konfigurationsname(
        _args("--no-physics", "--no-coord-maps")).startswith("A")


# ---------------------------------------------------------------------------
# Stufe 5 -- truncated BPTT. Die Aenderung, wegen der der 22.09. kein
# Ergebnis war: ein Schritt trainierte, achttausend wurden gemessen.
# ---------------------------------------------------------------------------
def _zwei_schritte_mit_abgeschnittener_mitte(net, op, statics, traj, idx,
                                             lag1, lag2):
    """Dasselbe Zweischritt-Fenster wie ``rollout_loss(k=2)``, aber der
    Zwischenzustand ist ``detach()``-ed.

    Vorwaerts ist das Zeichen fuer Zeichen dieselbe Zahl -- ``detach`` aendert
    keinen Wert. Rueckwaerts ist es der alte Zustand: der Gradient kommt nicht
    ueber den ersten Schritt hinaus. Die Referenz, gegen die sich beweisen
    laesst, dass ``rollout_loss`` genau das nicht mehr tut.
    """
    t0 = traj[idx]
    ny, nz = t0.shape[2:]

    def einen(zustand, j):
        x = M.assemble_input(
            M.state_channels(zustand,
                             traj[(idx + j - lag1).clamp(min=0)],
                             traj[(idx + j - lag2).clamp(min=0)]),
            statics,
            M.driver_channels(op.config[idx + j], op.forcing[idx + j], ny, nz))
        p = net.step(zustand, x, dt_n=op.dtn, fo_field=op.fo,
                     qsrc=op.qsrc[idx + j], ghost_hi=M.adiabatic_ghost(zustand))
        return p, torch.mean((p - op.tn_seq[idx + j + 1]) ** 2)

    p0, l0 = einen(t0, 0)
    _, l1 = einen(p0.detach(), 1)          # <- hier wird die Zeit gekappt
    return (l0 + l1) / 2


def test_rollout_loss_bei_k_eins_ist_exakt_data_loss(net, op, statics):
    """``--tbptt 1`` muss den Lauf vom 22.09. reproduzieren, nicht aehneln."""
    traj, _ = T.rollout(net, op, statics, lag1=5, lag2=20)
    idx = torch.tensor([7, 18, 25])
    kw = dict(lag1=5, lag2=20)
    a = T.data_loss(net, op, statics, traj, idx, **kw)
    b = T.rollout_loss(net, op, statics, traj, idx, k=1, **kw)
    assert float(a.detach()) == float(b.detach())


def test_der_gradient_ueberquert_die_historie_bei_k_groesser_eins_sehr_wohl(
        net, op, statics):
    """Die Zusage von Stufe 5, als Gegenstueck zum Ein-Schritt-Test oben.

    Vorwaerts sind beide Fassungen dieselbe Zahl; rueckwaerts darf die
    abgeschnittene NICHT denselben Gradienten liefern. Waere sie es, liefe der
    Gradient trotz ``k = 2`` nur einen Schritt weit -- und Stufe 5 waere
    gebaut, ohne zu wirken.
    """
    traj, _ = T.rollout(net, op, statics, lag1=5, lag2=20)
    idx = torch.tensor([25])
    kw = dict(lag1=5, lag2=20)

    voll = T.rollout_loss(net, op, statics, traj, idx, k=2, **kw)
    gekappt = _zwei_schritte_mit_abgeschnittener_mitte(
        net, op, statics, traj, idx, 5, 20)
    assert float(voll.detach()) == pytest.approx(float(gekappt.detach()), rel=1e-12), \
        "die beiden Fassungen sind vorwaerts verschieden -- der Test misst nichts"

    g_voll = torch.autograd.grad(voll, list(net.parameters()), retain_graph=True)
    g_kapp = torch.autograd.grad(gekappt, list(net.parameters()))
    assert any(not torch.allclose(a, b) for a, b in zip(g_voll, g_kapp)), \
        "derselbe Gradient wie mit abgeschnittener Mitte -- k>1 wirkt nicht"


def test_ein_systematischer_drift_faellt_erst_ueber_mehrere_schritte_auf(layout,
                                                                         op,
                                                                         statics):
    """Warum ``data_loss`` den Lauf vom 22.09. nicht retten konnte.

    Ein Netz mit konstanter Vorspannung ``c`` auf flachen Labels macht je
    Schritt denselben winzigen Fehler ``dt*c``. Nach ``j`` Schritten steht
    ``(j+1)*dt*c`` da, der quadratische Fehler waechst also mit ``(j+1)^2``:

        L(k) = (dt*c)^2 * mean_j (j+1)^2      ->  L(4) = 7.5 * L(1)

    ``data_loss`` sieht davon **nur** ``L(1)``. Genau dieser Faktor ist der
    Unterschied zwischen "der Verlust ist klein" und "der Rollout ist weg".
    """
    net = M.GridCNN(layout, use_physics=False)
    with torch.no_grad():
        net.correction.head.weight.zero_()
        net.correction.head.bias.fill_(0.25)      # g_theta == 0.25, ueberall

    op.tn_seq = torch.zeros_like(op.tn_seq)
    traj = torch.zeros(op.n_t, *op.tn_ic.shape)
    idx = torch.tensor([9])
    kw = dict(lag1=5, lag2=20)

    eins = float(T.rollout_loss(net, op, statics, traj, idx, k=1, **kw).detach())
    vier = float(T.rollout_loss(net, op, statics, traj, idx, k=4, **kw).detach())
    assert eins == pytest.approx((op.dtn * 0.25) ** 2, rel=1e-5)
    assert vier == pytest.approx(7.5 * eins, rel=1e-5)


def test_das_fenster_greift_nicht_ueber_split_t(layout, op, statics):
    """Sonst wird aus der ausgehaltenen Zahl eine Trainingszahl -- mit k
    schlimmer als mit einem Schritt, weil das Fenster k Ziele weit reicht."""
    torch.manual_seed(0)
    net = M.GridCNN(layout)
    opt = torch.optim.Adam(net.parameters(), lr=1e-3)
    gesehen = []
    echt = T.rollout_loss

    def spion(n, o, s, traj, idx, *, k, **kw):
        gesehen.append(idx.max().item() + k)      # das letzte Ziel ist idx+k
        return echt(n, o, s, traj, idx, k=k, **kw)

    T.rollout_loss = spion
    try:
        T.train_epoch(net, T.stack_ops([op]), statics, opt, inner_steps=40,
                      batch_t=8, lag1=5, lag2=20, w_data=1.0, w_phys=0.0,
                      w_wall=0.0, clamp=50.0, rng=np.random.default_rng(0),
                      tbptt=6)
    finally:
        T.rollout_loss = echt
    assert gesehen, "der k>1-Pfad wurde gar nicht genommen"
    assert max(gesehen) <= op.split_t - 1, \
        f"Ziel {max(gesehen)} >= split_t {op.split_t}"


def test_ein_zu_kurzer_op_kuerzt_das_fenster_statt_zu_fallen(layout, statics):
    """Ein OP mit split_t = 3 traegt kein Fenster von 16 -- k wird gekuerzt."""
    torch.manual_seed(0)
    nx, ny, nz = layout.shape
    kurz = T.OPTensors(
        op_id="OP_KURZ", tn_seq=torch.randn(6, nx, ny, nz),
        tn_ic=torch.randn(nx, ny, nz), qsrc=torch.zeros(6, nx, ny, nz),
        fo=torch.zeros(nx, ny, nz, 3, 3), config=torch.randn(6, 7),
        forcing=torch.randn(6, 11), dtn=0.01, split_t=3, n_t=6)
    net = M.GridCNN(layout)
    opt = torch.optim.Adam(net.parameters(), lr=1e-3)
    st = T.train_epoch(net, T.stack_ops([kurz]), statics, opt, inner_steps=3,
                       batch_t=2, lag1=5, lag2=20, w_data=1.0, w_phys=0.0,
                       w_wall=0.0, clamp=50.0, rng=np.random.default_rng(0),
                       tbptt=16)
    assert np.isfinite(st.data)


# ---------------------------------------------------------------------------
# Stabilitaet: Clipping und der Endlichkeitswaechter
# ---------------------------------------------------------------------------
def _aenderungsnorm(vorher, net):
    return float(torch.sqrt(sum(((a - b.detach()) ** 2).sum()
                                for a, b in zip(vorher, net.parameters()))))


def test_clip_grad_begrenzt_den_schritt(layout, op, statics):
    """Mit SGD ist der Schritt genau ``lr * g`` -- geklemmt also hoechstens
    ``lr * clip``. Mit Adam waere derselbe Test blind: der normiert ohnehin."""
    def einmal(clip):
        torch.manual_seed(0)
        net = M.GridCNN(layout)
        with torch.no_grad():
            net.correction.head.weight.normal_(0.0, 0.05)
        vorher = [p.detach().clone() for p in net.parameters()]
        opt = torch.optim.SGD(net.parameters(), lr=1.0)
        st = T.train_epoch(net, T.stack_ops([op]), statics, opt, inner_steps=1,
                           batch_t=8, lag1=5, lag2=20, w_data=1.0, w_phys=0.0,
                           w_wall=0.0, clamp=50.0,
                           rng=np.random.default_rng(0), clip_grad=clip)
        return _aenderungsnorm(vorher, net), st

    geklemmt, st_k = einmal(1e-4)
    frei, st_f = einmal(0.0)
    assert geklemmt <= 1e-4 * 1.001, geklemmt
    assert frei > geklemmt, "ohne Schranke war der Schritt nicht groesser"
    # Die Norm wird in BEIDEN Faellen gemessen -- auch ohne Schranke.
    assert np.isfinite(st_k.grad_norm) and np.isfinite(st_f.grad_norm)
    assert st_k.grad_norm == pytest.approx(st_f.grad_norm, rel=1e-5), \
        "gemeldet wird die Norm VOR dem Klemmen, sonst sagt sie nichts"


def test_nicht_endliche_updates_werden_verworfen_und_gezaehlt(layout, op,
                                                              statics):
    """Ein einziges NaN vergiftet sonst den ganzen Lauf, und die Kurve zeigte
    nur eine flache Linie -- nicht zu unterscheiden von Konvergenz."""
    torch.manual_seed(0)
    net = M.GridCNN(layout)
    op.tn_seq = torch.full_like(op.tn_seq, float("inf"))
    vorher = [p.detach().clone() for p in net.parameters()]
    opt = torch.optim.Adam(net.parameters(), lr=1e-3)
    st = T.train_epoch(net, T.stack_ops([op]), statics, opt, inner_steps=5,
                       batch_t=4, lag1=5, lag2=20, w_data=1.0, w_phys=0.0,
                       w_wall=0.0, clamp=50.0, rng=np.random.default_rng(0))
    assert st.uebersprungen == 5
    assert all(torch.equal(a, b) for a, b in zip(vorher, net.parameters())), \
        "ein nicht-endliches Update ist in die Gewichte gelaufen"
    assert "[UEBERSPRUNGEN] 5" in st.line(1)


def test_die_epochenzeile_traegt_k_und_die_gradientennorm(layout, op, statics):
    torch.manual_seed(0)
    net = M.GridCNN(layout)
    opt = torch.optim.Adam(net.parameters(), lr=1e-3)
    st = T.train_epoch(net, T.stack_ops([op]), statics, opt, inner_steps=2,
                       batch_t=4, lag1=5, lag2=20, w_data=1.0, w_phys=0.0,
                       w_wall=0.0, clamp=50.0, rng=np.random.default_rng(0),
                       tbptt=3)
    zeile = st.line(7)
    assert "k   3" in zeile and "|g|" in zeile


# ---------------------------------------------------------------------------
# Die drei Zahlen, die geraten statt hergeleitet waren
# ---------------------------------------------------------------------------
def test_lags_kommen_aus_der_zeit_nicht_aus_der_schrittzahl(op):
    """5 und 20 Schritte sind bei subsample 2 genau 1 s und 4 s. Bei
    subsample 10 waeren dieselben Zahlen 5 s und 20 s -- eine andere
    Historie, ohne dass jemand etwas geaendert haette."""
    assert T.lags_aufloesen(2, None, None)[:2] == (5, 20)
    assert T.lags_aufloesen(10, None, None)[:2] == (1, 4)
    assert T.lags_aufloesen(20, None, None)[:2] == (1, 2)
    # Von Hand gesetzt gewinnt, und das steht auch im Text.
    l1, l2, text = T.lags_aufloesen(10, 5, 20)
    assert (l1, l2) == (5, 20) and "von Hand" in text


def test_clamp_auto_kommt_aus_den_labels(op):
    """Die geerbten 50 sind ±480 C und fangen nichts ab, was noch zu retten
    waere. 'auto' bindet die Schranke an die groesste echte Auslenkung."""
    op.tn_seq = torch.full_like(op.tn_seq, 0.0)
    op.tn_seq[3] = 2.4
    wert, text = T.clamp_aufloesen("auto", [op], faktor=3.0, T_sigma=9.602)
    assert wert == 8.0 and "auto" in text          # ceil(3 * 2.4) = 8
    assert T.clamp_aufloesen("aus", [op], faktor=3.0, T_sigma=9.602)[0] == 0.0
    assert T.clamp_aufloesen("12.5", [op], faktor=3.0, T_sigma=9.602)[0] == 12.5


def test_das_curriculum_erreicht_sein_ziel_und_bleibt_dort():
    """Die zweite Haelfte laeuft auf voller Laenge -- sonst kaeme der
    berichtete Median von einem wandernden Ziel."""
    werte = [T.tbptt_bei(e, 4, 16, 40) for e in range(1, 41)]
    assert werte[0] == 4 and werte[-1] == 16
    assert werte == sorted(werte), "das Fenster darf nicht schrumpfen"
    assert all(v == 16 for v in werte[19:]), "ab der Haelfte muss k stehen"
    # Kein Curriculum verlangt, kein Curriculum geliefert.
    assert [T.tbptt_bei(e, 16, 16, 40) for e in (1, 20, 40)] == [16, 16, 16]


def test_berichtet_wird_der_median_nicht_das_beste():
    """Das Beste waere auf der Haltemenge ausgewaehlt -- und die Haltemenge
    ist hier die ganze Messung."""
    punkte = [(10, {"OP06": 30.0}), (20, {"OP06": 3.0}),   # der Ausreisser
              (30, {"OP06": 9.0}), (40, {"OP06": 11.0}),
              (50, {"OP06": 10.0}), (60, {"OP06": 12.0})]
    assert T.median_ueber(punkte) == {"OP06": 11.0}        # Median von 10, 12
    assert T.median_ueber([]) == {}


def test_die_val_mae_gebatcht_trifft_die_einzeln_gerollte(net, op, statics):
    """Gebatcht ist ein Tempohebel, kein Experiment -- die Zahl muss stehen."""
    kurz = T.OPTensors(
        op_id="OP_KURZ", tn_seq=op.tn_seq[:25].clone(),
        tn_ic=op.tn_ic.clone(), qsrc=op.qsrc[:25].clone(), fo=op.fo,
        config=op.config[:25].clone(), forcing=op.forcing[:25].clone(),
        dtn=op.dtn, split_t=20, n_t=25)
    kw = dict(lag1=5, lag2=20, clamp=10.0)
    gebatcht = T.val_mae(net, [op, kurz], statics, T_sigma=9.602, **kw)
    for einzeln in (op, kurz):
        traj, _ = T.rollout(net, einzeln, statics, **kw)
        erwartet = float((traj - einzeln.tn_seq).abs().mean().item() * 9.602)
        assert gebatcht[einzeln.op_id] == pytest.approx(erwartet, rel=1e-4)


def test_der_protokollname_sagt_wie_trainiert_wurde():
    """Ohne ihn steht in einem halben Jahr eine val-MAE in einer Tabelle und
    niemand weiss, ob sie mit k=1 oder k=16 entstanden ist."""
    args = T.build_argparser().parse_args([])
    assert "k=4->16" in T.protokollname(args)
    alt = T.build_argparser().parse_args(
        ["--tbptt", "1", "--tbptt-start", "1", "--clip-grad", "0",
         "--lr-plan", "konstant"])
    name = T.protokollname(alt)
    assert "k=1" in name and "clip=aus" in name and "konstant" in name


def test_das_verdikt_nennt_einen_unlesbaren_lauf_beim_namen():
    """Der Lauf vom 22.09., durch die neue Schlusstafel geschickt.

    27.11 ± 22.26 C gegen eine Latte von 7.7: der Mittelwert sieht nach einem
    Ergebnis aus, die Streuung sagt, dass es keines ist. Genau das musste man
    damals aus drei Dokumenten zusammensuchen.
    """
    alle = {"OP06": [51.95, 20.43, 8.95]}
    latten = {"OP06": {"mittelwert": 7.70, "persistenz": 14.0}}
    zeilen, kenn = T.zusammenfassung(alle, latten)
    text = "\n".join(zeilen)
    assert kenn["lesbar"] is False
    assert kenn["seed_streuung_C"]["OP06"] > T.LESBARKEIT_C
    assert "KEIN ERGEBNIS" in text and "NICHTS ranken" in text
    assert kenn["guete"]["OP06"] == pytest.approx(27.11 / 7.70, rel=1e-3)


def test_das_verdikt_trennt_gelernt_von_geraten():
    """Dieselbe Streuung, zwei verschiedene Befunde -- und beide sind echt."""
    latten = {"OP06": {"mittelwert": 7.70, "persistenz": 14.0}}
    _, gut = T.zusammenfassung({"OP06": [4.0, 4.3, 4.2]}, latten)
    assert gut["lesbar"] is True and gut["guete"]["OP06"] < 1.0

    zeilen, schlecht = T.zusammenfassung({"OP06": [8.0, 8.3, 8.2]}, latten)
    assert schlecht["lesbar"] is True and schlecht["guete"]["OP06"] > 1.0
    assert "negatives" in "\n".join(zeilen)


def test_ohne_latten_sagt_das_verdikt_dass_es_nichts_sagen_kann():
    zeilen, kenn = T.zusammenfassung({"OP06": [4.0, 4.1, 4.2]}, {})
    assert kenn["lesbar"] is False and kenn["guete"] == {}
    assert "keine Latten" in "\n".join(zeilen)
