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
    """Ein geratenes U waere ein freier Parameter -- also lieber laut fallen.

    ⚠ Die Begruendung hat sich am 22.09., abends, GEAENDERT, und der Test
    haelt jetzt die neue fest. Bis dahin stand hier "Stufe 2 ist offen" --
    das stimmte seit dem Rebuild auf Schema v3 nicht mehr. ``U(V_dot)`` ist
    kalibriert; was fehlt, ist die Verdrahtung (``q_wall_meas`` bleibt None,
    ``t_in``/``mdot`` je Zeitschritt fehlen in ``op_tensoren``).

    Der Test prueft deshalb die **Verdrahtung** als Grund, nicht die
    Kalibrierung -- sonst haelt er eine Behauptung fest, die falsch ist.
    """
    wall = object()
    with pytest.raises(NotImplementedError, match="nicht verdrahtet"):
        T._wall_ghost(op.tn_ic.unsqueeze(0), wall, op, 0)
    with pytest.raises(NotImplementedError, match="q_wall_meas"):
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


# ---------------------------------------------------------------------------
# Das Fehlerprofil ueber die Trajektorie -- der Befund vom 22.09., abends
# ---------------------------------------------------------------------------
def _driftendes_netz(layout, c: float):
    """Ein Netz, dessen Rollout linear wegdriftet: ``T_k = k * dt * c``.

    Kopfgewichte null, Kopf-Bias ``c``, keine Physik -- also ist die Rate
    ueberall exakt ``c``, und der Fehler gegen flache Labels waechst linear
    mit dem Zeitschritt. Genau die Form, die der Plot vom 22.09. auf OP06
    zeigt, nur in geschlossener Form statt gemessen.
    """
    net = M.GridCNN(layout, use_physics=False)
    with torch.no_grad():
        net.correction.head.weight.zero_()
        net.correction.head.bias.fill_(c)
    return net


def _flacher_op(layout, n_t: int = 40, dtn: float = 0.01):
    nx, ny, nz = layout.shape
    return T.OPTensors(
        op_id="OP_FLACH", tn_seq=torch.zeros(n_t, nx, ny, nz),
        tn_ic=torch.zeros(nx, ny, nz), qsrc=torch.zeros(n_t, nx, ny, nz),
        fo=torch.zeros(nx, ny, nz, 3, 3), config=torch.zeros(n_t, 7),
        forcing=torch.zeros(n_t, 11), dtn=dtn, split_t=n_t - 1, n_t=n_t)


def test_das_profil_findet_den_spaetfehler_den_der_mittelwert_verdeckt(
        layout, statics):
    """Der Befund, wegen dem es diese Messung gibt.

    ``OP06 6.57 C`` sah am 22.09. nach einer Zahl aus. Der Plot ueber
    dieselbe Trajektorie zeigte 1.1 C in der Mitte und 15 C am Ende. Hier
    dasselbe in geschlossener Form: der Fehler waechst linear, also ist

        MAE(letztes Viertel) / MAE(gesamt) = 34.5 / 19.5 = 1.769

    und genau das muss ``drift`` melden.
    """
    op = _flacher_op(layout)
    net = _driftendes_netz(layout, 0.25)
    d = T.val_auswertung(net, [op], statics, lag1=5, lag2=20, clamp=0.0,
                         T_sigma=10.0, segmente=4)["OP_FLACH"]

    schritt = 0.01 * 0.25 * 10.0                       # dt * c * T_sigma
    assert d["mae"] == pytest.approx(schritt * 19.5, rel=1e-5)
    assert d["mae_segmente"][0] == pytest.approx(schritt * 4.5, rel=1e-5)
    assert d["mae_segmente"][-1] == pytest.approx(schritt * 34.5, rel=1e-5)
    assert d["drift"] == pytest.approx(34.5 / 19.5, rel=1e-5)
    assert d["drift"] > T.DRIFT_SCHWELLE, "der Spaetfehler muss auffallen"


def test_der_bias_sagt_zu_warm_oder_zu_kalt(layout, statics):
    """Ohne Vorzeichen ist ein feldweiter Pegelfehler nicht von Streuung zu
    unterscheiden -- und genau das zeigt der Plot vom 22.09. nicht."""
    op = _flacher_op(layout)
    kw = dict(lag1=5, lag2=20, clamp=0.0, T_sigma=10.0, segmente=4)
    warm = T.val_auswertung(_driftendes_netz(layout, 0.25), [op], statics,
                            **kw)["OP_FLACH"]
    kalt = T.val_auswertung(_driftendes_netz(layout, -0.25), [op], statics,
                            **kw)["OP_FLACH"]
    assert warm["bias"] > 0 and kalt["bias"] < 0
    assert warm["bias"] == pytest.approx(-kalt["bias"], rel=1e-5)
    # Der Betrag ist derselbe -- die MAE allein koennte die beiden nie trennen.
    assert warm["mae"] == pytest.approx(kalt["mae"], rel=1e-5)
    assert warm["bias_segmente"][-1] > warm["bias_segmente"][0]


def test_ein_gleichmaessiger_fehler_hat_drift_nahe_eins(layout, statics):
    """Die Gegenprobe: ohne Spaetfehler darf nichts gemeldet werden."""
    op = _flacher_op(layout)
    net = _driftendes_netz(layout, 0.0)           # exakt null Rate
    op.tn_seq = torch.full_like(op.tn_seq, 0.5)   # konstanter Versatz
    op.tn_ic = torch.full_like(op.tn_ic, 0.5)
    d = T.val_auswertung(net, [op], statics, lag1=5, lag2=20, clamp=0.0,
                         T_sigma=10.0, segmente=4)["OP_FLACH"]
    assert d["mae"] == pytest.approx(0.0, abs=1e-6)
    assert not np.isfinite(d["drift"]) or d["drift"] <= T.DRIFT_SCHWELLE


def test_val_mae_ist_genau_die_mae_aus_der_auswertung(net, op, statics):
    """Zwei Wege zu derselben Zahl driften auseinander -- also gibt es einen."""
    kw = dict(lag1=5, lag2=20, clamp=10.0, T_sigma=9.602)
    schlank = T.val_mae(net, [op], statics, **kw)
    voll = T.val_auswertung(net, [op], statics, **kw)
    assert schlank == {k: v["mae"] for k, v in voll.items()}


def test_das_profil_wird_ueber_messpunkte_und_seeds_gemedianed():
    """Ein Ausreisser-Messpunkt darf das Profil nicht kippen."""
    mach = lambda m, segs: {"mae": m, "bias": -m, "drift": 2.0,            # noqa: E731
                            "mae_segmente": segs,
                            "bias_segmente": [-v for v in segs]}
    # Neun Messpunkte -> das letzte Drittel sind drei. Die Ausreisser liegen
    # davor und duerfen genau deshalb nichts aendern.
    punkte = [(e, {"OP06": mach(99.0, [99.0, 99.0])}) for e in (10, 20, 30)]
    punkte += [(e, {"OP06": mach(0.1, [0.1, 0.1])}) for e in (40, 50, 60)]
    punkte += [(70, {"OP06": mach(6.0, [3.0, 9.0])}),
               (80, {"OP06": mach(7.0, [4.0, 10.0])}),
               (90, {"OP06": mach(8.0, [5.0, 11.0])})]
    p = T.median_profil(punkte)["OP06"]
    assert p["mae"] == pytest.approx(7.0)                    # Median von 6,7,8
    assert p["mae_segmente"] == pytest.approx([4.0, 10.0])

    ueber = T.profil_ueber_seeds([
        {"OP06": mach(5.0, [2.0, 8.0])},
        {"OP06": mach(7.0, [4.0, 10.0])},
        {"OP06": mach(9.0, [6.0, 12.0])}])["OP06"]
    assert ueber["mae"] == pytest.approx(7.0)
    assert ueber["mae_segmente"] == pytest.approx([4.0, 10.0])
    assert T.profil_ueber_seeds([]) == {} and T.median_profil([]) == {}


def test_die_schlusstafel_nennt_den_spaetfehler_beim_namen():
    latten = {"OP06": {"mittelwert": 10.80, "persistenz": 16.67}}
    profil = {"OP06": {"mae": 6.57, "bias": -3.4, "drift": 2.01,
                       "mae_segmente": [4.6, 1.7, 4.3, 4.6, 8.1, 13.2],
                       "bias_segmente": [-1.0, -0.5, -2.0, -3.0, -6.0, -12.9]}}
    zeilen, kenn = T.zusammenfassung({"OP06": [6.5, 6.6, 6.7]}, latten,
                                     profil=profil)
    text = "\n".join(zeilen)
    assert "O13" in text and "NICHT gleichmaessig" in text
    assert kenn["drift"]["OP06"] == pytest.approx(2.01)

    # Und die Gegenprobe: ohne Spaetfehler keine Warnung.
    flach = {"OP06": dict(profil["OP06"], drift=1.02,
                          mae_segmente=[6.5] * 6, bias_segmente=[-0.1] * 6)}
    zeilen2, _ = T.zusammenfassung({"OP06": [6.5, 6.6, 6.7]}, latten,
                                   profil=flach)
    assert "gleichmaessig verteilt" in "\n".join(zeilen2)
    assert "O13" not in "\n".join(zeilen2)


def test_das_fenster_muss_lag2_erreichen():
    """23.09.: k=16 bei subsample 2 liegt hinter lag2=20 -- kein Update lief
    durch die Rueckkopplung ueber lag2. Im POC (subsample 10) lag lag2=4 im
    Fenster."""
    w = T.fensterwarnung(4, 16, 5, 20, subsample=2)
    assert w and "NIE" in w and "--tbptt-start 20 --tbptt 80" in w
    assert T.fensterwarnung(4, 16, 1, 4, subsample=10) is None
    assert T.fensterwarnung(20, 80, 5, 20, subsample=2) is None


def test_der_protokollname_nennt_k_in_sekunden():
    args = T.build_argparser().parse_args(["--subsample", "2"])
    assert "(0.8->3.2 s)" in T.protokollname(args)


def test_nachmessen_trifft_die_auswertung_im_lauf(tmp_path, layout, net, op,
                                                  statics):
    """Checkpoint nachgemessen == dieselbe Zahl wie im Lauf."""
    pfad = tmp_path / "model.pt"
    torch.save(net.state_dict(), pfad)
    kw = dict(lag1=5, lag2=20, clamp=10.0, T_sigma=9.602)
    nach = T.profil_aus_checkpoint(pfad, layout, {}, [op], statics, **kw)
    vor = T.val_auswertung(net, [op], statics, **kw)
    assert nach["OP99"]["mae"] == pytest.approx(vor["OP99"]["mae"])
    assert nach["OP99"]["drift"] == pytest.approx(vor["OP99"]["drift"])


# ---------------------------------------------------------------------------
# 23.09.: das Fenster in Sekunden, und das Nachmessen aus Gewichten
# ---------------------------------------------------------------------------
def test_das_fenster_muss_lag2_erreichen():
    """Lauf 16: k=16 bei subsample 2 liegt hinter lag2=20 -- kein Update lief
    durch die Rueckkopplung ueber lag2. Im POC (subsample 10) lag lag2=4 im
    Fenster 4->16, und dort meldet sich nichts."""
    w = T.fensterwarnung(4, 16, 5, 20, subsample=2)
    assert w is not None and "NIE" in w
    assert "--tbptt-start 40 --tbptt 160" not in w       # nicht bei dt=0.1 s
    assert "--tbptt-start 20 --tbptt 80" in w            # 4->16 s bei dt=0.2 s
    assert T.fensterwarnung(4, 16, 1, 4, subsample=10) is None
    assert T.fensterwarnung(20, 80, 5, 20, subsample=2) is None
    # lag2 erreicht, aber das Startfenster nicht einmal lag1.
    w = T.fensterwarnung(4, 80, 5, 20, subsample=2)
    assert w is not None and "lag1" in w


def test_der_protokollname_nennt_k_in_sekunden():
    """Zwei Logs mit 'k=4->16' waren 4->16 s und 0.8->3.2 s -- ohne dass es
    einer Zeile anzusehen war."""
    zwei = T.build_argparser().parse_args(["--subsample", "2"])
    zehn = T.build_argparser().parse_args(["--subsample", "10"])
    assert "k=4->16 (0.8->3.2 s)" in T.protokollname(zwei)
    assert "k=4->16 (4->16 s)" in T.protokollname(zehn)


def test_nachmessen_trifft_die_auswertung_im_lauf(tmp_path, layout, net, op,
                                                  statics):
    """Ein gespeichertes model.pt nachgemessen ist dieselbe Zahl wie im Lauf
    -- sonst waere das nachgeholte Profil von Lauf 16 nichts wert."""
    pfad = tmp_path / "model.pt"
    torch.save(net.state_dict(), pfad)
    kw = dict(lag1=5, lag2=20, clamp=10.0, T_sigma=9.602)
    nach = T.profil_aus_checkpoint(pfad, layout, {}, [op], statics, **kw)
    vor = T.val_auswertung(net, [op], statics, **kw)
    for schl in ("mae", "bias", "drift"):
        assert nach["OP99"][schl] == pytest.approx(vor["OP99"][schl], rel=1e-6)
    assert nach["OP99"]["mae_segmente"] == pytest.approx(
        vor["OP99"]["mae_segmente"], rel=1e-6)


def test_die_fruehphase_wird_eine_zahl():
    """Lauf 16, Seed 0: gesaettigt bis ep 8 (1.4 %), danach nicht mehr."""
    verlauf = [{"epoch": e, "saturated": s, "saturated_max": 88286}
               for e, s in [(1, 0), (2, 36379), (3, 54871), (7, 88111),
                            (8, 1220), (9, 0), (38, 500)]]
    f = T.fruehphase(verlauf)
    assert f["epochen"] == 4 and f["letzte"] == 8      # 500/88286 < 1 %
    # ein spaeter Sturm zeigt sich auch
    verlauf.append({"epoch": 40, "saturated": 9000, "saturated_max": 88286})
    assert T.fruehphase(verlauf)["letzte"] == 40
    # aeltere history.json ohne saturated_max
    assert T.fruehphase([{"epoch": 3, "saturated": 1}])["letzte"] == 3
    assert T.fruehphase([])["letzte"] == 0


def test_die_tafel_steht_aus_den_metrics_auch_ueber_zwei_prozesse():
    """Lauf 16: Seeds 0-1 und Seed 2 in zwei Prozessen, die Tafel im Log
    kannte nur Seed 2. Aus den metrics.json steht sie ueber alle drei."""
    lat = {"OP06": {"mittelwert": 10.8009, "persistenz": 16.6788},
           "OP09": {"mittelwert": 7.7625, "persistenz": 18.5493}}
    metriken = [{"konfiguration": "A", "protokoll": "k=4->16",
                 "val_mae_C_berichtet": {"OP06": a, "OP09": b},
                 "triviale_latten_C": lat}
                for a, b in [(5.5798, 7.3909), (8.3833, 10.6178),
                             (5.9703, 7.7978)]]
    alle, latten, protokolle = T.tafel_aus_metrics(metriken)
    assert alle["OP09"] == [7.3909, 10.6178, 7.7978] and len(protokolle) == 1
    zeilen, k = T.zusammenfassung(alle, latten)
    assert k["guete"]["OP06"] == pytest.approx(0.615, abs=1e-3)
    assert k["guete"]["OP09"] == pytest.approx(1.108, abs=1e-3)
    assert not k["lesbar"]
    # gemischte Laeufe fallen auf
    metriken[2]["protokoll"] = "k=20->80"
    assert len(T.tafel_aus_metrics(metriken)[2]) == 2


def test_nachmessen_laeuft_von_vorn_bis_hinten(tmp_path, monkeypatch, capsys,
                                               layout, op, statics):
    """Das Werkzeug einmal ganz, mit dem Datenpfad gestubbt -- es laeuft
    sonst nur auf der Maschine mit data_cache/, und dort soll es nicht an
    etwas Banalem scheitern."""
    import importlib.util
    import json
    from pathlib import Path
    from types import SimpleNamespace

    pfad = Path(T.__file__).resolve().parent / "tools" / "nachmessen.py"
    spec = importlib.util.spec_from_file_location("nachmessen", pfad)
    nm = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(nm)

    args = T.build_argparser().parse_args(["--no-physics"])
    torch.manual_seed(1)
    net = M.GridCNN(layout, **T.modell_kwargs(args)["net"])
    lat = {"OP99": {"mittelwert": 9.0, "persistenz": 12.0}}
    laeufe = tmp_path / "A"
    for n, mae in enumerate((5.0, 6.0, 7.0)):
        d = laeufe / f"seed{n}"
        d.mkdir(parents=True)
        torch.save(net.state_dict(), d / "model.pt")
        (d / "metrics.json").write_text(json.dumps(
            {"konfiguration": "A", "protokoll": "k=20->80",
             "val_mae_C_berichtet": {"OP99": mae},
             "triviale_latten_C": lat}))
        (d / "history.json").write_text(json.dumps(
            [{"epoch": 1, "saturated": 50, "saturated_max": 100},
             {"epoch": 2, "saturated": 0, "saturated_max": 100}]))

    bundle = SimpleNamespace(T_sigma=9.602, T_mu=33.0, T_span_ref=1605.2)
    monkeypatch.setattr(T, "lade_datensatz",
                        lambda a, dev: (bundle, [op], [op], layout, statics))
    monkeypatch.setattr(T, "_pinn_module", lambda name: SimpleNamespace(
        DEFAULT_TRAIN_OPS=["OP99"], DEFAULT_VAL_OPS=["OP99"]))
    monkeypatch.setattr(T, "resolve_device", lambda spec: torch.device("cpu"))

    rc = nm.main(["--no-physics", "--subsample", "2", "--device", "cpu",
                  "--cache", str(tmp_path), "--laeufe", str(laeufe)])
    out = capsys.readouterr().out
    assert rc == 0
    assert out.count("FRUEHPHASE: 1 Epoche(n)") == 3
    assert "3/3 Seed(s) unterbieten sie" in out
    assert "Median ueber 3 Seed(s)" in out
    erg = json.loads((laeufe / "nachgemessen.json").read_text())
    assert set(erg["je_checkpoint"]) == {f"seed{n}/model.pt" for n in range(3)}
    erwartet = T.val_auswertung(net, [op], statics, lag1=5, lag2=20,
                                clamp=erg["clamp"], T_sigma=9.602)
    assert erg["median_model_pt"]["OP99"]["mae"] == pytest.approx(
        erwartet["OP99"]["mae"], rel=1e-6)
    # Die Zeitreihen fuer tools/bilder.py stehen mit drin (25.09.).
    k = erg["je_checkpoint"]["seed0/model.pt"]["OP99"]["kurve"]
    assert len(k["t_s"]) == op.n_t and "abs_q75" in k and "karte_mae" in k

    # --json: eine zweite Messung (z. B. in-sample) ueberschreibt die erste
    # nicht.
    ziel = laeufe / "nachgemessen_insample.json"
    assert nm.main(["--no-physics", "--subsample", "2", "--device", "cpu",
                    "--cache", str(tmp_path), "--laeufe", str(laeufe),
                    "--json", str(ziel)]) == 0
    assert ziel.exists() and (laeufe / "nachgemessen.json").exists()


def test_die_cfl_zeile_empfiehlt_kein_kleineres_subsample_mehr():
    """Lauf 16/17: 110x ueber der Schranke bei subsample 2 -- und selbst die
    Rohabtastung (subsample 1) laege 55x darueber. Die Materialdaten sind
    echt. Ein kleineres --subsample ist also kein Weg, und die Zeile darf ihn
    nicht mehr empfehlen."""
    text = T.cfl_text(0.000124595, 1.12998e-06, 1605.2, 2)
    assert "110.3x" in text and "55.1x" in text
    assert "loest das NICHT" in text and "Integrator" in text


def test_mit_exaktem_schritt_warnt_die_cfl_zeile_nicht_mehr():
    """Lauf 18 hat --integrator exp. Die Zeile darf dort nicht mehr "B/C/D
    braucht einen eigenen Integrator" sagen -- er ist ja da."""
    text = T.cfl_text(0.000124595, 1.12998e-06, 1605.2, 2, integrator="exp")
    assert "110.3x" in text and "exakt" in text
    assert not text.startswith("!!") and "eigenen Integrator" not in text
    unter = T.cfl_text(1e-7, 1e-6, 1605.2, 2)
    assert unter.startswith("[CFL]") and "!!" not in unter


# ---------------------------------------------------------------------------
# PR #51: Schalter, Laufname, Physik-Residuum zum exakten Schritt
# ---------------------------------------------------------------------------
def test_der_laufname_ueberschreibt_lauf_17_nicht():
    """Bis PR #51 hiess jedes Verzeichnis nur nach dem Arm -- ein Lauf mit
    kompakten Karten haette die Gewichte von Lauf 17 ueberschrieben."""
    assert T.laufname(_args("--no-physics")) == "A"
    assert T.laufname(_args()) == "B"
    assert T.laufname(_args("--no-physics", "--karten", "kompakt",
                            "--treiber", "film")) == "A-kompakt-film"
    assert T.laufname(_args("--integrator", "exp")) == "B-exp"
    assert T.laufname(_args("--no-physics", "--integrator", "exp",
                            "--w-phys", "0.1")) == "A-exp-wphys0.1"


def test_kompakte_karten_kennen_ihre_breite_erst_nach_dem_bauen(layout):
    n = layout.n_points
    rng = np.random.default_rng(0)
    args = _args("--karten", "kompakt", "--treiber", "film",
                 "--integrator", "exp")
    vorher = T.modell_kwargs(args)
    assert vorher["static"]["kompakt"] is True
    assert vorher["net"]["n_static"] is None
    statics = M.build_static_maps(
        layout, lam=rng.random((n, 3, 3)) + 1.0, rho=np.full(n, 2500.0),
        cp=np.full(n, 900.0), **vorher["static"])
    kw = T.modell_kwargs(args, statics)
    assert kw["net"]["n_static"] == statics.n_channels
    netz = M.GridCNN(layout, **kw["net"])
    nx, ny, nz = layout.shape
    state = M.state_channels(*(torch.randn(2, nx, ny, nz) for _ in range(3)))
    drv = M.driver_channels(torch.randn(2, 7), torch.randn(2, 11), ny, nz)
    x = M.assemble_input(state, statics, drv)
    assert netz.correction(x).shape == (2, nx, ny, nz)


def test_das_physik_residuum_zum_exakten_schritt(layout, op, statics):
    """B/C/D: das Residuum ist genau g -- bei Kopf null also null.
    A: g minus die Sekante des exakten Schritts -- braucht kein Label."""
    b = M.GridCNN(layout, integrator="exp")
    a = M.GridCNN(layout, use_physics=False, integrator="exp")
    traj = op.tn_seq.clone()
    idx = torch.arange(5, 13)
    kw = dict(lag1=2, lag2=4)
    verlust_b, _ = T.physics_loss(b, op, statics, traj, idx, **kw)
    assert float(verlust_b.detach()) == 0.0
    verlust_a, _ = T.physics_loss(a, op, statics, traj, idx, **kw)
    sek = a.physik(op.fo, op.dtn).sekante(traj[idx], op.qsrc[idx])
    assert float(verlust_a.detach()) == pytest.approx(float((sek ** 2).mean()),
                                             rel=1e-5)
    # Der Euler-Default bleibt, wie er war.
    alt = M.GridCNN(layout)
    verlust_alt, _ = T.physics_loss(alt, op, statics, traj, idx, **kw)
    assert float(verlust_alt.detach()) == 0.0


@pytest.mark.parametrize("argv", [
    ("--integrator", "exp"),
    ("--no-physics", "--integrator", "exp", "--w-phys", "0.1"),
    ("--no-physics", "--treiber", "film"),
])
def test_eine_epoche_laeuft_mit_den_neuen_schaltern(layout, op, statics, argv):
    args = _args(*argv)
    netz = M.GridCNN(layout, **T.modell_kwargs(args, statics)["net"])
    with torch.no_grad():
        netz.correction.head.weight.normal_(0.0, 0.05)
    vorher = [p.detach().clone() for p in netz.parameters()]
    opt = torch.optim.Adam(netz.parameters(), lr=1e-3)
    st = T.train_epoch(netz, T.stack_ops([op]), statics, opt, inner_steps=2,
                       batch_t=4, lag1=2, lag2=4, w_data=1.0,
                       w_phys=args.w_phys, w_wall=0.0, clamp=10.0,
                       rng=np.random.default_rng(0), tbptt=3, clip_grad=1.0)
    assert np.isfinite(st.data)
    assert any(not torch.equal(v, p) for v, p in zip(vorher, netz.parameters()))


def test_die_kartenschalter_kommen_beim_laden_an(monkeypatch, layout):
    """Der Probelauf vom 23.09. abends: ``--karten kompakt`` stand im Log,
    aber ``statics_aus_bundle`` reichte nur ``coord_maps`` weiter -- das Netz
    bekam weiter 17 Karten. Jeder Schluessel aus modell_kwargs()["static"]
    muss bei build_static_maps ankommen."""
    from types import SimpleNamespace
    gesehen = {}

    def fang(layout_, **kw):
        gesehen.update(kw)
        return "statics"

    monkeypatch.setattr(M, "build_static_maps", fang)
    monkeypatch.setattr(T, "_pinn_module", lambda name: SimpleNamespace(
        load_material_properties=lambda layer: {"lambda_tensor": None}))
    bundle = SimpleNamespace(region=np.zeros(layout.n_points), rho=None,
                             Cp=None)
    kw = T.modell_kwargs(_args("--karten", "kompakt", "--no-coord-maps"))
    T.statics_aus_bundle(bundle, layout, device=None, **kw["static"])
    for schluessel, wert in kw["static"].items():
        assert gesehen[schluessel] == wert


# ---------------------------------------------------------------------------
# 25.09.: die Zeitreihe fuer die Bilder
# ---------------------------------------------------------------------------
def test_ohne_kurven_bleibt_die_auswertung_wie_sie_war(net, op, statics):
    """Das Training ruft ``val_auswertung`` ohne ``kurven`` -- dort darf sich
    nichts aendern, sonst sind Lauf 17 und Lauf 18/19 nicht mehr dieselbe
    Messung."""
    kw = dict(lag1=5, lag2=20, clamp=10.0, T_sigma=9.602)
    ohne = T.val_auswertung(net, [op], statics, **kw)["OP99"]
    mit = T.val_auswertung(net, [op], statics, kurven=True, T_mu=33.0,
                           T_span_ref=1605.2, **kw)["OP99"]
    assert "kurve" not in ohne
    for schl in ("mae", "bias", "drift", "mae_segmente", "bias_segmente"):
        assert mit[schl] == ohne[schl]


def test_die_zeitreihe_trifft_die_zahlen_aus_dem_log(net, op, statics):
    """Die Kurve ist dieselbe Messung wie die Zeile im Log, nur aufgeloest:
    ihr Mittel ist die MAE, die Quantile liegen in der richtigen Reihenfolge,
    und die Karte je Punkt mittelt auf dieselbe Zahl."""
    kw = dict(lag1=5, lag2=20, clamp=10.0, T_sigma=9.602, kurven=True,
              T_mu=33.0, T_span_ref=1605.2)
    p = T.val_auswertung(net, [op], statics, **kw)["OP99"]
    k = p["kurve"]
    assert len(k["t_s"]) == op.n_t               # 40 < KURVE_PUNKTE: alle
    assert k["t_s"][1] == pytest.approx(op.dtn * 1605.2, rel=1e-4)
    assert k["split_t_s"] == pytest.approx(op.split_t * op.dtn * 1605.2,
                                           rel=1e-4)
    assert np.mean(k["mae"]) == pytest.approx(p["mae"], abs=1e-3)
    assert np.mean(k["bias"]) == pytest.approx(p["bias"], abs=1e-3)
    assert np.mean(k["karte_mae"]) == pytest.approx(p["mae"], abs=1e-3)
    assert k["karte_form"] == list(op.tn_ic.shape)
    reihe = np.array([k[s] for s in ("abs_min", "abs_q25", "abs_q50",
                                     "abs_q75", "abs_max")])
    assert np.all(np.diff(reihe, axis=0) >= -1e-4)
    reihe = np.array([k[s] for s in ("fehler_min", "fehler_q25",
                                     "fehler_q75", "fehler_max")])
    assert np.all(np.diff(reihe, axis=0) >= -1e-4)
    # Daten in C: der Versatz T_mu kommt zurueck, die Differenz ist der Bias.
    diff = np.subtract(k["T_modell_mittel"], k["T_wahr_mittel"])
    assert diff == pytest.approx(k["bias"], abs=1e-3)


def test_lange_trajektorien_werden_ausgeduennt(net, layout, statics):
    """8040 Schritte bei subsample 2: die JSON bekommt hoechstens
    KURVE_PUNKTE Stuetzstellen je Reihe, nicht jeden Schritt."""
    nx, ny, nz = layout.shape
    n_t = 3 * T.KURVE_PUNKTE + 7
    lang = T.OPTensors(
        op_id="OPLANG", tn_seq=torch.zeros(n_t, nx, ny, nz),
        tn_ic=torch.zeros(nx, ny, nz), qsrc=torch.zeros(n_t, nx, ny, nz),
        fo=torch.zeros(nx, ny, nz, 3, 3), config=torch.zeros(n_t, 7),
        forcing=torch.zeros(n_t, 11), dtn=0.01, split_t=n_t // 2, n_t=n_t)
    k = T.val_auswertung(net, [lang], statics, lag1=5, lag2=20, clamp=10.0,
                         T_sigma=1.0, kurven=True)["OPLANG"]["kurve"]
    assert len(k["t_s"]) <= T.KURVE_PUNKTE
    assert all(len(k[s]) == len(k["t_s"]) for s in ("mae", "abs_max",
                                                    "T_wahr_mittel"))


def test_nachmessen_reicht_die_kurven_durch(tmp_path, layout, net, op,
                                            statics):
    pfad = tmp_path / "model.pt"
    torch.save(net.state_dict(), pfad)
    p = T.profil_aus_checkpoint(pfad, layout, {}, [op], statics, lag1=5,
                                lag2=20, clamp=10.0, T_sigma=9.602,
                                kurven=True, T_mu=33.0, T_span_ref=1.0)
    assert "kurve" in p["OP99"]
