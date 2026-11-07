"""Pont Orthanc ↔ CoroSeg.

    python -m coroseg_orthanc watch                 # service : surveille Orthanc et traite les coroscanners
    python -m coroseg_orthanc process <ID étude>    # traite une étude tout de suite (ID Orthanc)
    python -m coroseg_orthanc series <ID étude>     # liste les séries candidates et celle qui serait choisie
    python -m coroseg_orthanc demo                  # envoie un fantôme dans Orthanc et le traite (sans licence)

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
    a = ap.parse_args(argv)

    cfg = Config()
    o = Orthanc(cfg.orthanc_url, cfg.orthanc_user, cfg.orthanc_password)

    if a.cmd == "watch":
        Worker(cfg).loop(from_start=a.depuis_le_debut)
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
        print(json.dumps(r, indent=1, ensure_ascii=False))
        return 0
    return 1


if __name__ == "__main__":
    sys.exit(main())
