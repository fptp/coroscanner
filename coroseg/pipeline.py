"""Chaîne complète : volume → segmentation → étiquetage → lignes centrales → exports."""

from __future__ import annotations

import csv
import json
import time
import zipfile
from dataclasses import dataclass, field
from pathlib import Path

import nibabel as nib
import numpy as np

from . import LABELS, __version__
from .centerline import Centerline
from .labeling import label_coronaries
from .mesh import mask_to_mesh, write_stl


def narrowing_profile(cl: Centerline, half_window_mm: float = 15.0, gap_mm: float = 3.0, edge_mm: float = 4.0) -> np.ndarray:
    """Rétrécissement relatif (%) du diamètre du MASQUE par rapport à une référence locale.

    Référence « interpolée » (comme en QCA) : moyenne des 80e percentiles des diamètres en amont
    et en aval du point (fenêtres de gap_mm à half_window_mm), ce qui limite l'effet de l'effilement normal.
    Indicatif seulement : ce n'est pas une mesure de sténose luminale validée.
    """
    d = 2 * cl.radius
    s = cl.s
    out = np.full(len(d), np.nan)
    for i in range(len(d)):
        if s[i] < edge_mm or s[i] > s[-1] - edge_mm:
            continue
        ds = s - s[i]
        refs = [np.percentile(d[m], 80) for m in ((ds >= -half_window_mm) & (ds < -gap_mm),
                                                  (ds > gap_mm) & (ds <= half_window_mm)) if m.sum() >= 3]
        if not refs:
            continue
        ref = float(np.mean(refs))
        out[i] = max(0.0, 100 * (1 - d[i] / ref)) if ref > 0 else np.nan
    return out


@dataclass
class CoroResult:
    ct: nib.Nifti1Image  # volume canonique RAS, int16 (UH)
    coronary: np.ndarray  # bool
    aorta: np.ndarray  # bool
    labels: np.ndarray  # uint8 (cf. coroseg.LABELS)
    centerlines: list[Centerline]
    warnings: list[str] = field(default_factory=list)
    ostia_mm: dict = field(default_factory=dict)
    timings: dict = field(default_factory=dict)

    @property
    def affine(self):
        return self.ct.affine

    @property
    def hu(self) -> np.ndarray:
        return np.asarray(self.ct.dataobj)

    def summary_rows(self) -> list[dict]:
        rows = []
        for cl in self.centerlines:
            d = 2 * cl.radius
            pct = narrowing_profile(cl)
            i = int(np.nanargmax(pct)) if np.isfinite(pct).any() else None
            rows.append({
                "Vaisseau": cl.name,
                "Longueur (mm)": round(cl.length, 1),
                "Ø moyen (mm)": round(float(d.mean()), 2),
                "Rétrécissement max (%)": None if i is None else round(float(pct[i])),
                "à (mm de l'origine)": None if i is None else round(float(cl.s[i]), 1),
                "Ø au rétrécissement (mm)": None if i is None else round(float(d[i]), 2),
            })
        return rows

    def volumes_ml(self) -> dict[str, float]:
        v = float(np.prod(np.sqrt((self.affine[:3, :3] ** 2).sum(0)))) / 1000
        return {LABELS[k][0]: round(float((self.labels == k).sum()) * v, 3) for k in LABELS if (self.labels == k).any()}

    # ---------------------------------------------------------------------------------------------
    def save(self, out_dir: str | Path, meshes: bool = True) -> Path:
        out = Path(out_dir)
        out.mkdir(parents=True, exist_ok=True)
        nib.save(nib.Nifti1Image(self.labels.astype(np.uint8), self.affine), out / "coronaires_labels.nii.gz")
        nib.save(nib.Nifti1Image(self.coronary.astype(np.uint8), self.affine), out / "coronaires_masque.nii.gz")
        (out / "centerlines.json").write_text(json.dumps([c.to_dict() for c in self.centerlines], indent=1))
        with open(out / "centerlines.csv", "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["vaisseau", "label", "s_mm", "x_mm", "y_mm", "z_mm", "rayon_mm"])
            for c in self.centerlines:
                for s, p, r in zip(c.s, c.points, c.radius):
                    w.writerow([c.name, c.label, f"{s:.2f}", f"{p[0]:.3f}", f"{p[1]:.3f}", f"{p[2]:.3f}", f"{r:.3f}"])
        report = {
            "coroseg_version": __version__,
            "labels": {k: {"code": v[0], "nom": v[1]} for k, v in LABELS.items()},
            "volumes_ml": self.volumes_ml(),
            "vaisseaux": self.summary_rows(),
            "ostia_mm_RAS": self.ostia_mm,
            "avertissements": self.warnings,
            "durees_s": self.timings,
            "avertissement_usage": "Outil de recherche, non validé pour un usage clinique.",
        }
        (out / "rapport.json").write_text(json.dumps(report, indent=1, ensure_ascii=False))
        if meshes:
            mdir = out / "maillages"
            mdir.mkdir(exist_ok=True)
            for k, (code, _, _) in LABELS.items():
                m = mask_to_mesh(self.labels == k, self.affine)
                if m is not None:
                    write_stl(mdir / f"{k:02d}_{code.replace(' ', '').replace('.', '')}.stl", *m, name=code)
            m = mask_to_mesh(self.coronary, self.affine)
            if m is not None:
                write_stl(mdir / "arbre_complet.stl", *m, name="coronaires")
        return out

    def zip(self, out_dir: str | Path) -> Path:
        out = Path(out_dir)
        zpath = out.parent / f"{out.name}.zip"
        with zipfile.ZipFile(zpath, "w", zipfile.ZIP_DEFLATED) as z:
            for p in out.rglob("*"):
                if p.is_file():
                    z.write(p, p.relative_to(out))
        return zpath


def analyse(ct: nib.Nifti1Image, coronary: np.ndarray, aorta: np.ndarray | None, log=print, timings=None) -> CoroResult:
    t0 = time.time()
    log("Squelettisation, lignes centrales et identification des vaisseaux…")
    lab = label_coronaries(coronary, aorta, ct.affine, log=log)
    timings = dict(timings or {})
    timings["etiquetage"] = round(time.time() - t0, 1)
    return CoroResult(ct, coronary.astype(bool), (aorta if aorta is not None else np.zeros_like(coronary)).astype(bool),
                      lab.label_map, lab.centerlines, lab.warnings, lab.ostia_mm, timings)


def run(ct: nib.Nifti1Image, work_dir, license_number=None, device="gpu", log=print) -> CoroResult:
    from .segment import segment_ct

    t0 = time.time()
    cor, aorta, _heart = segment_ct(ct, work_dir, license_number=license_number, device=device, log=log)
    t_seg = round(time.time() - t0, 1)
    return analyse(ct, cor, aorta, log=log, timings={"segmentation": t_seg})
