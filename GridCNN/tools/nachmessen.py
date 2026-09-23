#!/usr/bin/env python3
"""Fehlerprofil aus gespeicherten Gewichten -- ohne nachzutrainieren.

Warum es das gibt
-----------------
Am 23.09. lief Konfiguration A auf voller Aufloesung (``laeufe/16``) mit
einem ``train.py`` von VOR PR #47: drei gueltige Seeds, aber ohne das
Fehlerprofil -- kein ``drift``, kein Bias. Die Gewichte liegen noch unter
``GridCNN/artifacts/A/seed*/``. Das Profil nachzuholen ist eine Minute
Rollout je Seed, kein Lauf.

Aufruf
------
**Dieselben Flags wie der Lauf**, plus ``--laeufe``::

    python3 GridCNN/tools/nachmessen.py --no-physics --subsample 2 \\
        --device cuda --cache data_cache --laeufe GridCNN/artifacts/A

Gemessen wird je Seed ``model.pt`` (Stand der letzten Epoche) und, falls
vorhanden, ``model_best.pt``. Die Lags und ``--clamp auto`` loesen sich wie im
Lauf auf.

Dazu, ohne Rollout: die **Schlusstafel ueber alle Seeds** aus den
``metrics.json`` (auch wenn die Seeds in getrennten Prozessen liefen) und die
**Fruehphase** je Seed aus ``history.json``.

Die Probe, dass es dieselbe Messung ist
---------------------------------------
Die MAE von ``model.pt`` muss die Zeile ``letztes ep 60`` aus dem Log treffen
(bis auf Rundung). Trifft sie nicht, passen Checkpoint und Flags nicht
zusammen -- dann ist das Profil nichts wert.

⚠ ``model.pt`` ist EIN Stand, die berichtete Zahl ist ein Median ueber das
letzte Drittel. Das Profil hier ist eine Diagnose (wo sitzt der Fehler, zu
warm oder zu kalt), keine Kopfzahl.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))

import train as T  # noqa: E402


def main(argv: list | None = None) -> int:
    p = T.build_argparser()
    p.description = __doc__.splitlines()[0]
    p.add_argument("--laeufe", type=Path, required=True,
                   help="Verzeichnis mit seed*/model.pt, z. B. "
                        "GridCNN/artifacts/A")
    args = p.parse_args(argv)
    if not args.cache.exists():
        print(f"!! Kein Cache unter {args.cache}.", file=sys.stderr)
        return 2
    seeds = sorted(d for d in args.laeufe.glob("seed*")
                   if (d / "model.pt").exists())
    if not seeds:
        print(f"!! Kein seed*/model.pt unter {args.laeufe}.", file=sys.stderr)
        return 2

    reg = T._pinn_module("op_registry")
    args.ops = args.ops or list(reg.DEFAULT_TRAIN_OPS)
    args.val_ops = args.val_ops or list(reg.DEFAULT_VAL_OPS)
    device = T.resolve_device(args.device)
    args.lag1, args.lag2, lag_text = T.lags_aufloesen(
        args.subsample, args.lag1, args.lag2)
    print(f"[konfiguration] {T.konfigurationsname(args)}")
    print(f"[lags] {lag_text}")

    bundle, train, val, layout, statics = T.lade_datensatz(args, device)
    args.clamp, clamp_text = T.clamp_aufloesen(
        args.clamp, train + val, faktor=args.clamp_faktor,
        T_sigma=bundle.T_sigma)
    print(f"[clamp] {clamp_text}")
    net_kw = T.modell_kwargs(args)["net"]
    kw = dict(lag1=args.lag1, lag2=args.lag2, clamp=args.clamp,
              T_sigma=bundle.T_sigma, device=device)

    ergebnis, letzte = {}, []
    for d in seeds:
        for name in ("model.pt", "model_best.pt"):
            pfad = d / name
            if not pfad.exists():
                continue
            profil = T.profil_aus_checkpoint(pfad, layout, net_kw, val,
                                             statics, **kw)
            ergebnis[f"{d.name}/{name}"] = profil
            if name == "model.pt":
                letzte.append(profil)
            mae = "  ".join(f"{k} {v['mae']:.4f}"
                            for k, v in sorted(profil.items()))
            print(f"\n[{d.name}/{name}] MAE {mae}")
            for z in T.profil_zeilen(profil, einzug=f"[{d.name}]   "):
                print(z)

    # Die Tafel ueber ALLE Seeds aus den metrics.json -- bei Lauf 16 stand
    # im Log nur Seed 2, weil die Seeds in zwei Prozessen liefen.
    metriken = [json.loads((d / "metrics.json").read_text())
                for d in seeds if (d / "metrics.json").exists()]
    for d in seeds:
        h = d / "history.json"
        if h.exists():
            f = T.fruehphase(json.loads(h.read_text()))
            print(f"[{d.name}] FRUEHPHASE: {f['epochen']} Epoche(n) mit mehr "
                  f"als {100 * f['schwelle']:g} % am Clamp, letzte ep "
                  f"{f['letzte']}")
    tafel = None
    if metriken:
        alle, latten, protokolle = T.tafel_aus_metrics(metriken)
        if len(protokolle) > 1:
            print(f"!! Die Seeds stammen aus verschiedenen Laeufen: "
                  f"{sorted(map(str, protokolle))}", file=sys.stderr)
        zeilen, tafel = T.zusammenfassung(alle, latten)
        print(f"\n[tafel] aus {len(metriken)} metrics.json (berichtete Zahl "
              f"je Seed, nicht model.pt):")
        for z in zeilen:
            print(z)

    ueber = T.profil_ueber_seeds(letzte)
    print(f"\n[profil] model.pt, Median ueber {len(letzte)} Seed(s):")
    for z in T.profil_zeilen(ueber, einzug="          "):
        print(z)
    print("\nProbe: die MAE je model.pt muss die Zeile 'letztes ep ...' im "
          "Log treffen.")
    ziel = args.laeufe / "nachgemessen.json"
    ziel.write_text(json.dumps(
        {"konfiguration": T.konfigurationsname(args),
         "subsample": args.subsample, "lag1": args.lag1, "lag2": args.lag2,
         "clamp": float(args.clamp), "je_checkpoint": ergebnis,
         "median_model_pt": ueber, "tafel": tafel}, indent=2))
    print(f"  -> {ziel}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
