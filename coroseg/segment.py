"""Segmentation par TotalSegmentator (tâche coronary_arteries + aorte)."""

from __future__ import annotations

import os
from pathlib import Path

import nibabel as nib
import numpy as np

from .io import load_mask_like

LICENSE_URL = "https://backend.totalsegmentator.com/license-academic/"


class SegmentationError(RuntimeError):
    pass


def totalsegmentator_available() -> bool:
    try:
        import totalsegmentator  # noqa: F401

        return True
    except Exception:
        return False


def _run(ct_path: Path, out_dir: Path, **kw) -> None:
    from totalsegmentator.python_api import totalsegmentator

    try:
        totalsegmentator(str(ct_path), str(out_dir), **kw)
    except SystemExit as e:  # TotalSegmentator appelle sys.exit() sur licence invalide
        raise SegmentationError(
            "TotalSegmentator s'est arrêté (licence absente ou invalide ?). "
            f"La tâche coronary_arteries nécessite une licence, gratuite pour un usage académique : {LICENSE_URL}"
        ) from e


def segment_ct(
    ct_img: nib.Nifti1Image,
    work_dir: str | os.PathLike,
    license_number: str | None = None,
    device: str = "gpu",
    log=print,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Renvoie (masque coronaires, masque aorte, masque cœur) alignés sur ct_img."""
    if not totalsegmentator_available():
        raise SegmentationError("TotalSegmentator n'est pas installé : pip install TotalSegmentator")

    work = Path(work_dir)
    work.mkdir(parents=True, exist_ok=True)
    ct_path = work / "ct_ras.nii.gz"
    nib.save(ct_img, ct_path)
    lic = license_number.strip() if license_number and license_number.strip() else None

    log("Segmentation des coronaires (TotalSegmentator, tâche coronary_arteries)…")
    _run(ct_path, work / "coronary", task="coronary_arteries", device=device, quiet=True, license_number=lic)
    cor_file = work / "coronary" / "coronary_arteries.nii.gz"
    if not cor_file.exists():
        raise SegmentationError(f"Fichier de sortie introuvable : {cor_file}")

    log("Segmentation de l'aorte et du cœur (repérage des ostia)…")
    _run(ct_path, work / "total", task="total", roi_subset=["aorta", "heart"], device=device, quiet=True)

    cor = load_mask_like(cor_file, ct_img) > 0
    aorta = load_mask_like(work / "total" / "aorta.nii.gz", ct_img) > 0
    heart_f = work / "total" / "heart.nii.gz"
    heart = load_mask_like(heart_f, ct_img) > 0 if heart_f.exists() else np.zeros_like(cor)
    if cor.sum() == 0:
        raise SegmentationError("Le masque coronaire est vide : vérifier qu'il s'agit bien d'un coroscanner injecté.")
    return cor, aorta, heart
