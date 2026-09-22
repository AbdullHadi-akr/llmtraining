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


def test_gebatcht_rollt_in_float64_BIT_FUER_BIT_wie_einzeln(layout):
    """**Die Zusage, auf der die T4-Optimierung steht.**

    Elf OPs zusammen zu rollen ist nur dann eine Beschleunigung und keine
    Aenderung, wenn dabei dasselbe herauskommt. Waere es nicht so, waere jede
    spaetere Zahl mit der Batchgroesse verwechselbar -- und das faellt in einer
    Trainingskurve nicht auf.

    Geprueft wird in **float64 ohne Toleranz**. In float32 weichen die beiden
    Wege um ~8e-6 ab, und zwar nicht wegen eines Fehlers: die
    Faltungsbibliothek waehlt fuer Batch 1 einen anderen Algorithmus als fuer
    Batch 3, und Gleitkommaaddition ist nicht assoziativ. In float64
    verschwindet der Unterschied **exakt** -- was genau belegt, dass es
    Rechenreihenfolge ist und keine Physik. Der Test daneben misst die
    float32-Groessenordnung, damit sie nicht unbemerkt waechst.
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
        assert torch.equal(gebatcht[:e.shape[0], i], e), f"OP {i}"


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

    In float64 ist die Differenz **exakt null**, auch fuer den gepaddeten OP.
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
    assert torch.equal(zusammen[:25, 1], allein)

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
