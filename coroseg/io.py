"""Lecture des coroscanners (DICOM ou NIfTI) et mise en orientation canonique RAS."""

from __future__ import annotations

import os
import tempfile
import zipfile
from dataclasses import dataclass
from pathlib import Path

import nibabel as nib
import numpy as np


@dataclass
class DicomSeries:
    series_uid: str
    folder: str
    files: list[str]
    description: str
    n_slices: int

    @property
    def label(self) -> str:
        return f"{self.description or '(sans description)'} — {self.n_slices} coupes"


def _meta(reader, key: str) -> str:
    try:
        return reader.GetMetaData(key).strip() if reader.HasMetaDataKey(key) else ""
    except Exception:
        return ""


def list_dicom_series(folder: str | os.PathLike) -> list[DicomSeries]:
    """Liste toutes les séries DICOM d'un dossier (récursivement), triées par nombre de coupes."""
    import SimpleITK as sitk

    found: list[DicomSeries] = []
    for dirpath, _dirs, _files in os.walk(folder):
        try:
            uids = sitk.ImageSeriesReader.GetGDCMSeriesIDs(dirpath)
        except Exception:
            continue
        for uid in uids or ():
            files = list(sitk.ImageSeriesReader.GetGDCMSeriesFileNames(dirpath, uid))
            if len(files) < 10:  # topogrammes, scouts, pages de dose…
                continue
            desc = ""
            try:
                r = sitk.ImageFileReader()
                r.SetFileName(files[0])
                r.LoadPrivateTagsOff()
                r.ReadImageInformation()
                parts = [_meta(r, "0008|103e"), _meta(r, "0020|0011")]
                phase = _meta(r, "0020|9241") or _meta(r, "0018|9241")
                desc = " / ".join(p for p in parts if p)
                if phase:
                    desc += f" (phase {phase})"
            except Exception:
                pass
            found.append(DicomSeries(uid, dirpath, files, desc, len(files)))
    found.sort(key=lambda s: s.n_slices, reverse=True)
    return found


def load_dicom_series(series: DicomSeries) -> nib.Nifti1Image:
    """Charge une série DICOM en image NIfTI (valeurs en UH), orientation canonique RAS."""
    import SimpleITK as sitk

    reader = sitk.ImageSeriesReader()
    reader.SetFileNames(series.files)
    img = reader.Execute()
    with tempfile.TemporaryDirectory() as td:
        p = os.path.join(td, "ct.nii.gz")
        sitk.WriteImage(img, p)
        nimg = nib.load(p)
        nimg = nib.Nifti1Image(np.asarray(nimg.dataobj), nimg.affine, nimg.header)
    return to_canonical(nimg)


def to_canonical(img: nib.Nifti1Image) -> nib.Nifti1Image:
    """Réoriente en RAS+ et convertit en int16 (UH)."""
    img = nib.as_closest_canonical(img)
    data = np.asarray(img.dataobj, dtype=np.float32)
    data = np.clip(np.round(data), -32768, 32767).astype(np.int16)
    out = nib.Nifti1Image(data, img.affine)
    out.header.set_xyzt_units("mm")
    return out


def load_volume(path: str | os.PathLike, series_index: int = 0) -> nib.Nifti1Image:
    """Charge un fichier .nii/.nii.gz, un dossier DICOM ou une archive .zip de DICOM."""
    path = Path(path)
    if path.is_dir():
        series = list_dicom_series(path)
        if not series:
            raise ValueError(f"Aucune série DICOM exploitable trouvée dans {path}")
        return load_dicom_series(series[series_index])
    name = path.name.lower()
    if name.endswith((".nii", ".nii.gz")):
        return to_canonical(nib.load(str(path)))
    if name.endswith(".zip"):
        td = tempfile.mkdtemp(prefix="coroseg_zip_")
        with zipfile.ZipFile(path) as z:
            z.extractall(td)
        return load_volume(td, series_index)
    raise ValueError(f"Format non reconnu : {path.name} (attendu : dossier DICOM, .zip, .nii ou .nii.gz)")


def load_mask_like(path: str | os.PathLike, ref: nib.Nifti1Image) -> np.ndarray:
    """Charge un masque NIfTI et vérifie qu'il est dans la même grille que l'image de référence."""
    m = nib.as_closest_canonical(nib.load(str(path)))
    data = np.asarray(m.dataobj)
    if data.shape != ref.shape or not np.allclose(m.affine, ref.affine, atol=1e-2):
        from nibabel.processing import resample_from_to

        m = resample_from_to(m, ref, order=0)
        data = np.asarray(m.dataobj)
    return data


def spacing_of(img: nib.Nifti1Image) -> np.ndarray:
    return np.sqrt((img.affine[:3, :3] ** 2).sum(axis=0))
