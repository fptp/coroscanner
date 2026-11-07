"""Traitement en ligne de commande / par lot.

    python -m coroseg examen1/ examen2.nii.gz -o resultats/ --licence aca_XXXX
    python -m coroseg --demo -o resultats/
    python -m coroseg ct.nii.gz --masque coronaires.nii.gz --aorte aorte.nii.gz -o resultats/
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from .io import load_mask_like, load_volume


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(prog="coroseg", description="Segmentation et étiquetage des coronaires sur coroscanner.")
    ap.add_argument("entrees", nargs="*", help="dossiers DICOM, archives .zip ou fichiers .nii(.gz)")
    ap.add_argument("-o", "--sortie", default="resultats_coroseg")
    ap.add_argument("--licence", default=None, help="numéro de licence TotalSegmentator (sinon celui déjà enregistré)")
    ap.add_argument("--device", default="gpu", help="gpu, gpu:1, cpu ou mps")
    ap.add_argument("--masque", default=None, help="masque coronaire existant (saute TotalSegmentator, 1 seule entrée)")
    ap.add_argument("--aorte", default=None, help="masque aortique existant (avec --masque)")
    ap.add_argument("--demo", action="store_true", help="traite le fantôme synthétique")
    ap.add_argument("--sans-maillages", action="store_true")
    a = ap.parse_args(argv)

    from .pipeline import analyse, run

    out_root = Path(a.sortie)
    jobs = [("demo", None)] if a.demo else [(Path(p).name.replace(".nii.gz", "").replace(".nii", ""), p) for p in a.entrees]
    if not jobs:
        ap.error("aucune entrée (ou utiliser --demo)")
    if a.masque and len(jobs) != 1:
        ap.error("--masque n'accepte qu'une seule entrée")

    status = 0
    for name, path in jobs:
        print(f"\n=== {name} ===")
        try:
            if path is None:
                from .phantom import make_phantom

                ct, cor, aorta, _ = make_phantom()
                res = analyse(ct, cor, aorta)
            else:
                ct = load_volume(path)
                if a.masque:
                    cor = load_mask_like(a.masque, ct) > 0
                    aorta = load_mask_like(a.aorte, ct) > 0 if a.aorte else None
                    res = analyse(ct, cor, aorta)
                else:
                    res = run(ct, out_root / name / "_totalseg", license_number=a.licence, device=a.device)
            res.save(out_root / name, meshes=not a.sans_maillages)
            for w in res.warnings:
                print("  ⚠", w)
            for row in res.summary_rows():
                print("  ", row)
            print(f"  → {out_root / name}")
        except Exception as e:  # on continue le lot
            status = 1
            print(f"  ERREUR : {e}", file=sys.stderr)
    return status


if __name__ == "__main__":
    sys.exit(main())
