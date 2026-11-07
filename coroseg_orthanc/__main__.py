"""Pont Orthanc ↔ CoroSeg.

    python -m coroseg_orthanc watch                 # service : surveille Orthanc (auto/label) ou reçoit en DICOM (dicom)
    python -m coroseg_orthanc process <ID étude>    # traite une étude tout de suite (ID Orthanc)
    python -m coroseg_orthanc series <ID étude>     # liste les séries candidates et celle qui serait choisie
    python -m coroseg_orthanc demo                  # envoie un fantôme dans Orthanc et le traite (sans licence)
    python -m coroseg_orthanc check                 # teste la connexion à Orthanc, CPU/GPU, licence

Configuration par variables d'environnement (voir coroseg_orthanc/config.py et deploy/coroseg.env).
"""

from __future__ import annotations

import argparse
import json
import sys

from .config import Config
from .orthanc import Orthanc
from .worker import Worker, eligible_series, log, process_study


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(prog="coroseg_orthanc")
    sub = ap.add_subparsers(dest="cmd", required=True)
    w = sub.add_parser("watch", help="surveille Orthanc en continu")
    w.add_argument("--depuis-le-debut", action="store_true", help="traite aussi les études déjà présentes")
    p = sub.add_parser("process", help="traite une étude (ID Orthanc)")
    p.add_argument("study")
    p.add_argument("--serie", default=None, help="ID Orthanc d'une série précise")
    s = sub.add_parser("series", help="liste les séries candidates d'une étude")
    s.add_argument("study")
    sub.add_parser("demo", help="envoie le fantôme dans Orthanc puis le traite (segmentation simulée)")
    sub.add_parser("check", help="teste la connexion à Orthanc et affiche le calcul utilisé (CPU/GPU)")
    a = ap.parse_args(argv)

    cfg = Config()
    from .device import set_threads

    set_threads(cfg.threads)
    o = Orthanc(cfg.orthanc_url, cfg.orthanc_user, cfg.orthanc_password)

    if a.cmd == "watch":
        if cfg.trigger == "dicom":
            from .receiver import serve

            serve(cfg)
        else:
            Worker(cfg).loop(from_start=a.depuis_le_debut)
        return 0
    if a.cmd == "check":
        from .device import describe, resolve_device

        import requests

        print(f"URL Orthanc : {cfg.orthanc_url}  |  utilisateur : {cfg.orthanc_user or '(aucun)'}  |  "
              f"mot de passe : {len(cfg.orthanc_password or '')} caractères")
        try:
            print(f"Orthanc {o.system()['Version']} joignable")
        except requests.HTTPError as e:
            if e.response is not None and e.response.status_code == 401:
                print("ERREUR 401 : Orthanc refuse les identifiants (ORTHANC_USER / ORTHANC_PASSWORD "
                      "dans /etc/coroseg/coroseg.env).")
            else:
                print(f"ERREUR : {e}")
            return 1
        except requests.RequestException as e:
            print(f"ERREUR : Orthanc injoignable ({e.__class__.__name__}) : vérifier ORTHANC_URL et le pare-feu.")
            return 1
        print(f"Calcul : {describe(resolve_device(cfg.device))}")
        print(f"Licence TotalSegmentator : {'renseignée' if cfg.license_number else 'ABSENTE (TOTALSEG_LICENSE)'}")
        if cfg.trigger == "dicom":
            print(f"Mode envoi manuel : AET {cfg.dicom_aet}, port {cfg.dicom_port}")
        else:
            print(f"Mode {cfg.trigger} : surveillance d'Orthanc")
        return 0
    if a.cmd == "series":
        print(json.dumps(eligible_series(o, a.study, cfg), indent=1, ensure_ascii=False))
        return 0
    if a.cmd == "process":
        wk = Worker(cfg)
        if a.serie:
            r = process_study(o, a.study, cfg, series_id=a.serie)
            print(json.dumps(r, indent=1, ensure_ascii=False))
        else:
            wk.handle(a.study, force=True)
        return 0
    if a.cmd == "demo":
        from .demo import phantom_dicom_files, phantom_segmenter

        files = phantom_dicom_files()
        study = None
        for b in files:
            study = o.upload(b)["ParentStudy"]
        log(f"Fantôme envoyé : {len(files)} coupes, étude {study}")
        r = process_study(o, study, cfg, segmenter=phantom_segmenter)
        o.add_label(study, cfg.label_done)
        log(f"Terminé en {r['seconds']} s : {len(r['instances'])} images dans {len(r['series'])} séries, "
            f"étude « FANTOME^CoroSeg » dans Orthanc.")
        return 0
    return 1


if __name__ == "__main__":
    sys.exit(main())
