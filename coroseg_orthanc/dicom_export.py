"""Conversion des résultats CoroSeg en objets DICOM renvoyés dans Orthanc.

- DICOM SEG (une segment par vaisseau) : superposable dans OHIF, Stone Web Viewer, 3D Slicer, Weasis…
- série « fusion » : coupes axiales du scanner avec les coronaires colorées (Secondary Capture, RGB),
  lisible par n'importe quelle visionneuse, même sans support du SEG ;
- série « CPR » : une image par vaisseau principal (CPR redressée sous 4 angles + profil de calibre) ;
- image « rapport » : tableau des mesures, avertissements, vue antérieure de l'arbre étiqueté.
"""

from __future__ import annotations

import datetime as dt
import io
from dataclasses import dataclass
from pathlib import Path

import nibabel as nib
import numpy as np
import pydicom
from PIL import Image, ImageDraw, ImageFont
from pydicom.dataset import Dataset, FileMetaDataset
from pydicom.encaps import encapsulate
from pydicom.uid import JPEGBaseline8Bit, ExplicitVRLittleEndian, generate_uid

from coroseg import LABELS, __version__
from coroseg.cpr import straightened_cpr, window
from coroseg.io import DicomSeries, list_dicom_series, load_dicom_series
from coroseg.pipeline import CoroResult, narrowing_profile

SC_SOP_CLASS = "1.2.840.10008.5.1.4.1.1.7"  # Secondary Capture Image Storage
MANUFACTURER = "CoroSeg"
DISCLAIMER = "Outil de recherche - non valide pour le diagnostic"

# codes SNOMED-CT des segments (catégorie : structure anatomique)
_SCT = {
    1: ("3227004", "Left main coronary artery"),
    2: ("59438005", "Left anterior descending coronary artery"),
    3: ("57396003", "Circumflex coronary artery"),
    4: ("13647002", "Right coronary artery"),
    5: ("41801008", "Coronary artery"),
    6: ("41801008", "Coronary artery"),
    7: ("41801008", "Coronary artery"),
    8: ("41801008", "Coronary artery"),
    9: ("41801008", "Coronary artery"),
}

_PATIENT_STUDY_TAGS = [
    "SpecificCharacterSet", "PatientName", "PatientID", "IssuerOfPatientID", "PatientBirthDate", "PatientSex",
    "PatientAge", "PatientWeight", "PatientSize", "StudyInstanceUID", "StudyDate", "StudyTime", "StudyID",
    "AccessionNumber", "ReferringPhysicianName", "StudyDescription", "InstitutionName",
]


# ----------------------------------------------------------------------------------------------------
# lecture de la série source
# ----------------------------------------------------------------------------------------------------
@dataclass
class SourceSeries:
    datasets: list[Dataset]  # même ordre que l'axe 0 de `hu`
    hu: np.ndarray  # (n_coupes, lignes, colonnes), UH
    sitk_img: object  # SimpleITK.Image de la série source
    ct: nib.Nifti1Image  # même volume en RAS canonique (entrée de CoroSeg)
    description: str


def _phase_key(ds: Dataset):
    for kw in ("NominalPercentageOfCardiacPhase", "TriggerTime", "TemporalPositionIdentifier", "AcquisitionNumber"):
        v = ds.get(kw)
        if v not in (None, ""):
            return kw, float(v)
    return None, None


def load_source(folder: str | Path, log=print) -> SourceSeries:
    """Charge la série CT téléchargée ; si elle contient plusieurs phases, garde la plus proche de 75 %."""
    import SimpleITK as sitk

    series = list_dicom_series(folder)
    if not series:
        raise ValueError("aucune série DICOM exploitable dans les fichiers reçus")
    ser = series[0]
    files = ser.files
    heads = [pydicom.dcmread(f, stop_before_pixels=True) for f in files]
    pos = [tuple(np.round(np.asarray(h.ImagePositionPatient, float), 2)) for h in heads]
    if len(set(pos)) < len(pos):  # positions dupliquées → série multiphase
        groups: dict = {}
        for f, h in zip(files, heads):
            kw, v = _phase_key(h)
            groups.setdefault(v, []).append(f)
        if len(groups) < 2:
            raise ValueError("coupes à positions dupliquées sans tag de phase : impossible de reconstituer le volume")
        kw = _phase_key(heads[0])[0]
        target = 75.0 if kw == "NominalPercentageOfCardiacPhase" else None
        key = min(groups, key=lambda g: abs(g - target)) if target is not None else max(groups, key=lambda g: len(groups[g]))
        log(f"Série multiphase ({len(groups)} phases selon {kw}) : phase {key:g} retenue.")
        sub = Path(folder) / f"_phase_{key:g}"
        sub.mkdir(exist_ok=True)
        for f in groups[key]:
            (sub / Path(f).name).write_bytes(Path(f).read_bytes())
        ser = list_dicom_series(sub)[0]
        files = ser.files

    reader = sitk.ImageSeriesReader()
    reader.SetFileNames(files)
    img = reader.Execute()
    hu = sitk.GetArrayFromImage(img).astype(np.float32)
    datasets = [pydicom.dcmread(f, stop_before_pixels=True) for f in files]
    ct = load_dicom_series(DicomSeries(ser.series_uid, ser.folder, files, ser.description, len(files)))
    return SourceSeries(datasets, hu, img, ct, ser.description)


def nifti_to_sitk(data: np.ndarray, affine: np.ndarray):
    """Tableau (i,j,k) + affine RAS → image SimpleITK (repère LPS)."""
    import SimpleITK as sitk

    img = sitk.GetImageFromArray(np.ascontiguousarray(np.asarray(data).transpose(2, 1, 0)))
    lps = np.diag([-1.0, -1.0, 1.0]) @ affine[:3, :]
    sp = np.sqrt((lps[:, :3] ** 2).sum(0))
    img.SetSpacing(tuple(float(s) for s in sp))
    img.SetDirection(tuple(float(v) for v in (lps[:, :3] / sp).flatten()))
    img.SetOrigin(tuple(float(v) for v in lps[:, 3]))
    return img


def labels_on_source_grid(labels: np.ndarray, affine: np.ndarray, src: SourceSeries) -> np.ndarray:
    """Ramène la carte de labels (RAS canonique) dans la grille des coupes DICOM d'origine : (n, lignes, colonnes)."""
    import SimpleITK as sitk

    lab = nifti_to_sitk(labels.astype(np.uint8), affine)
    out = sitk.Resample(lab, src.sitk_img, sitk.Transform(), sitk.sitkNearestNeighbor, 0, sitk.sitkUInt8)
    return sitk.GetArrayFromImage(out)


# ----------------------------------------------------------------------------------------------------
# outils communs
# ----------------------------------------------------------------------------------------------------
def _now():
    n = dt.datetime.now()
    return n.strftime("%Y%m%d"), n.strftime("%H%M%S")


def _font(size: int):
    for p in ("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", "/usr/share/fonts/dejavu/DejaVuSans.ttf",
              "DejaVuSans.ttf", "arial.ttf"):
        try:
            return ImageFont.truetype(p, size)
        except OSError:
            continue
    try:
        return ImageFont.load_default(size)
    except TypeError:  # Pillow < 10.1
        return ImageFont.load_default()


def _base_dataset(ref: Dataset, series_uid: str, series_number: int, description: str, instance_number: int,
                  institution: str = "") -> Dataset:
    d, t = _now()
    ds = Dataset()
    for kw in _PATIENT_STUDY_TAGS:
        if kw in ref:
            setattr(ds, kw, ref.get(kw))
    ds.SpecificCharacterSet = "ISO_IR 192"  # UTF-8 (accents)
    if institution:
        ds.InstitutionName = institution
    ds.SOPClassUID = SC_SOP_CLASS
    ds.SOPInstanceUID = generate_uid()
    ds.Modality = "OT"
    ds.ConversionType = "WSD"
    ds.SeriesInstanceUID = series_uid
    ds.SeriesNumber = series_number
    ds.SeriesDescription = description
    ds.SeriesDate, ds.SeriesTime = d, t
    ds.ContentDate, ds.ContentTime = d, t
    ds.InstanceNumber = instance_number
    ds.ImageType = ["DERIVED", "SECONDARY"]
    ds.Manufacturer = MANUFACTURER
    ds.SoftwareVersions = __version__
    ds.ImageComments = DISCLAIMER
    ds.BurnedInAnnotation = "YES"
    src = Dataset()
    src.ReferencedSOPClassUID = ref.SOPClassUID
    src.ReferencedSOPInstanceUID = ref.SOPInstanceUID
    ds.SourceImageSequence = [src]
    return ds


def _set_rgb(ds: Dataset, rgb: np.ndarray, jpeg: bool) -> None:
    rgb = np.ascontiguousarray(rgb.astype(np.uint8))
    ds.Rows, ds.Columns = rgb.shape[:2]
    ds.SamplesPerPixel = 3
    ds.PlanarConfiguration = 0
    ds.BitsAllocated = ds.BitsStored = 8
    ds.HighBit = 7
    ds.PixelRepresentation = 0
    meta = FileMetaDataset()
    meta.MediaStorageSOPClassUID = ds.SOPClassUID
    meta.MediaStorageSOPInstanceUID = ds.SOPInstanceUID
    if jpeg:
        buf = io.BytesIO()
        Image.fromarray(rgb, "RGB").save(buf, format="JPEG", quality=92, subsampling=1)  # 4:2:2
        ds.PhotometricInterpretation = "YBR_FULL_422"
        ds.LossyImageCompression = "01"
        ds.LossyImageCompressionMethod = "ISO_10918_1"
        ds.PixelData = encapsulate([buf.getvalue()])
        ds["PixelData"].VR = "OB"
        ds["PixelData"].is_undefined_length = True
        meta.TransferSyntaxUID = JPEGBaseline8Bit
    else:
        ds.PhotometricInterpretation = "RGB"
        ds.PixelData = rgb.tobytes()
        meta.TransferSyntaxUID = ExplicitVRLittleEndian
    ds.file_meta = meta


def to_bytes(ds: Dataset) -> bytes:
    buf = io.BytesIO()
    ds.save_as(buf, enforce_file_format=True)
    return buf.getvalue()


def _lut():
    lut = np.zeros((256, 3), np.float32)
    for k, (_, _, rgb) in LABELS.items():
        lut[k] = rgb
    return lut


# ----------------------------------------------------------------------------------------------------
# DICOM SEG
# ----------------------------------------------------------------------------------------------------
def make_seg(src: SourceSeries, lab_src: np.ndarray, series_number: int = 9001) -> Dataset:
    import highdicom as hd
    from pydicom.sr.codedict import codes
    from pydicom.sr.coding import Code

    present = [k for k in LABELS if (lab_src == k).any()]
    if not present:
        raise ValueError("aucun vaisseau segmenté")
    remap = np.zeros(256, np.uint8)
    for i, k in enumerate(present, start=1):
        remap[k] = i
    pixel = remap[lab_src]

    algo = hd.AlgorithmIdentificationSequence(
        name="CoroSeg (TotalSegmentator coronary_arteries)", version=__version__,
        family=codes.cid7162.ArtificialIntelligence)
    category = Code("123037004", "SCT", "Anatomical Structure")
    descs = []
    for i, k in enumerate(present, start=1):
        short, name, rgb = LABELS[k]
        sct = _SCT[k]
        descs.append(hd.seg.SegmentDescription(
            segment_number=i,
            segment_label=f"{short} - {name}"[:64],
            segmented_property_category=category,
            segmented_property_type=Code(sct[0], "SCT", sct[1]),
            algorithm_type=hd.seg.SegmentAlgorithmTypeValues.AUTOMATIC,
            algorithm_identification=algo,
            tracking_uid=hd.UID(),
            tracking_id=short,
            display_color=hd.color.CIELabColor.from_rgb(*rgb),
        ))

    seg = hd.seg.Segmentation(
        source_images=src.datasets,
        pixel_array=pixel,
        segmentation_type=hd.seg.SegmentationTypeValues.BINARY,
        segment_descriptions=descs,
        series_instance_uid=hd.UID(),
        series_number=series_number,
        sop_instance_uid=hd.UID(),
        instance_number=1,
        manufacturer=MANUFACTURER,
        manufacturer_model_name="CoroSeg",
        software_versions=__version__,
        device_serial_number="1",
        series_description="CoroSeg - SEG coronaires",
        content_description="Segmentation automatique des coronaires (recherche)",
        content_label="CORONAIRES",
        omit_empty_frames=True,
    )
    seg.SpecificCharacterSet = "ISO_IR 192"
    return seg


# ----------------------------------------------------------------------------------------------------
# série fusion (coupes axiales colorées)
# ----------------------------------------------------------------------------------------------------
def make_fusion_series(src: SourceSeries, lab_src: np.ndarray, mode: str = "roi", alpha: float = 0.55,
                       wl: float = 300, ww: float = 900, jpeg: bool = True, series_number: int = 9002,
                       institution: str = "", margin: int = 10) -> list[Dataset]:
    n = lab_src.shape[0]
    if mode == "all":
        idx = range(n)
    else:
        z = np.nonzero(lab_src.reshape(n, -1).any(1))[0]
        if len(z) == 0:
            return []
        idx = range(max(z.min() - margin, 0), min(z.max() + margin + 1, n))
    lut = _lut()
    uid = generate_uid()
    out = []
    for num, k in enumerate(idx, start=1):
        gray = window(src.hu[k], wl, ww).astype(np.float32)
        rgb = np.repeat(gray[..., None], 3, -1)
        m = lab_src[k] > 0
        if m.any():
            rgb[m] = (1 - alpha) * rgb[m] + alpha * lut[lab_src[k][m]]
        ref = src.datasets[k]
        ds = _base_dataset(ref, uid, series_number, "CoroSeg - fusion coronaires", num, institution)
        for kw in ("FrameOfReferenceUID", "ImagePositionPatient", "ImageOrientationPatient", "PixelSpacing",
                   "SliceThickness", "SliceLocation", "PatientPosition"):
            if kw in ref:
                setattr(ds, kw, ref.get(kw))
        _set_rgb(ds, rgb, jpeg)
        out.append(ds)
    return out


# ----------------------------------------------------------------------------------------------------
# série CPR
# ----------------------------------------------------------------------------------------------------
def _cpr_panel(res: CoroResult, cl, wl: float, ww: float, angles=(0, 45, 90, 135), px: float = 0.3,
               hw: float = 12.0, scale: int = 2) -> np.ndarray:
    hu = res.hu.astype(np.float32)
    cprs = [window(straightened_cpr(hu, res.affine, cl, a, hw, px), wl, ww) for a in angles]
    h, w = cprs[0].shape
    gap, head, prof_w = 6, 70, 160
    W = len(angles) * (w * scale + gap) + prof_w + 3 * gap
    H = h * scale + head + 30
    canvas = Image.new("RGB", (W, H), (0, 0, 0))
    color = LABELS.get(cl.label, ("", "", (255, 255, 255)))[2]
    dr = ImageDraw.Draw(canvas)
    f1, f2 = _font(26), _font(15)
    dr.text((gap, 8), f"CPR {cl.name}", fill=color, font=f1)
    dr.text((gap, 42), f"longueur {cl.length:.0f} mm  -  ostium en haut  -  W{ww:g} L{wl:g}", fill=(200, 200, 200), font=f2)
    x = gap
    for a, c in zip(angles, cprs):
        im = Image.fromarray(c).resize((w * scale, h * scale), Image.BILINEAR).convert("RGB")
        canvas.paste(im, (x, head))
        dr.text((x + 4, head + 4), f"{a}°", fill=(255, 255, 0), font=f2)
        x += w * scale + gap
    # profil de calibre aligné sur l'axe du vaisseau
    d = 2 * cl.radius
    pct = narrowing_profile(cl)
    x0, x1 = x + gap, x + gap + prof_w
    dmax = max(5.0, float(np.nanmax(d)) * 1.1)
    rows = head + (np.interp(cl.s, [0, cl.length], [0, h * px]) / px * scale)
    xs = x0 + d / dmax * (x1 - x0)
    dr.rectangle([x0, head, x1, head + h * scale], outline=(90, 90, 90))
    for g in range(1, int(dmax) + 1):
        gx = x0 + g / dmax * (x1 - x0)
        dr.line([(gx, head), (gx, head + h * scale)], fill=(40, 40, 40))
    dr.line(list(zip(xs.tolist(), rows.tolist())), fill=color, width=2)
    dr.text((x0, head - 20), f"Ø (mm, 0-{dmax:.0f})", fill=(200, 200, 200), font=f2)
    if np.isfinite(pct).any():
        i = int(np.nanargmax(pct))
        y = rows[i]
        dr.line([(gap, y), (x1, y)], fill=(255, 60, 60), width=1)
        dr.text((x0, min(y + 4, H - 40)), f"-{pct[i]:.0f}% à {cl.s[i]:.0f} mm\nØ {d[i]:.1f} mm", fill=(255, 90, 90), font=f2)
    dr.text((gap, H - 24), DISCLAIMER, fill=(150, 150, 150), font=f2)
    return np.asarray(canvas)


def make_cpr_series(src: SourceSeries, res: CoroResult, labels=(2, 3, 4, 8), include_branches: bool = False,
                    wl: float = 300, ww: float = 900, jpeg: bool = True, series_number: int = 9003,
                    institution: str = "") -> list[Dataset]:
    uid = generate_uid()
    out = []
    cls = [c for c in res.centerlines if c.label in labels or include_branches]
    if not cls:  # pas de bifurcation trouvée : au moins le TC
        cls = [c for c in res.centerlines if c.label == 1]
    for num, cl in enumerate(cls, start=1):
        if cl.length < 10:
            continue
        rgb = _cpr_panel(res, cl, wl, ww)
        ds = _base_dataset(src.datasets[0], uid, series_number, "CoroSeg - CPR", num, institution)
        ds.DerivationDescription = f"CPR redressee {cl.name}"
        _set_rgb(ds, rgb, jpeg)
        out.append(ds)
    return out


# ----------------------------------------------------------------------------------------------------
# rapport (image)
# ----------------------------------------------------------------------------------------------------
def _anterior_view(res: CoroResult, size: int = 520) -> Image.Image:
    lab = res.labels
    sp = np.sqrt((res.affine[:3, :3] ** 2).sum(0))
    m = lab > 0
    if not m.any():
        return Image.new("RGB", (size, size))
    # vue de face : rayons le long de y (antérieur = j max en RAS), premier voxel rencontré
    rev = m[:, ::-1, :]
    hit = rev.any(1)
    first = rev.argmax(1)
    j = lab.shape[1] - 1 - first
    ii, kk = np.nonzero(hit)
    col = np.zeros(hit.shape + (3,), np.float32)
    depth = first[ii, kk].astype(np.float32)
    shade = 1 - 0.6 * (depth - depth.min()) / max(np.ptp(depth), 1)
    col[ii, kk] = _lut()[lab[ii, j[ii, kk], kk]] * shade[:, None]
    img = col.transpose(1, 0, 2)[::-1, ::-1]  # lignes = z (haut = crânial), colonnes = x (droite patient à gauche)
    im = Image.fromarray(img.astype(np.uint8))
    w_mm, h_mm = lab.shape[0] * sp[0], lab.shape[2] * sp[2]
    s = size / max(w_mm, h_mm)
    return im.resize((max(int(w_mm * s), 1), max(int(h_mm * s), 1)), Image.BILINEAR)


def make_report_image(src: SourceSeries, res: CoroResult, series_number: int = 9004, jpeg: bool = True,
                      institution: str = "", series_description: str = "") -> Dataset:
    W, H = 1400, 1000
    im = Image.new("RGB", (W, H), (12, 12, 16))
    dr = ImageDraw.Draw(im)
    f_t, f, f_s = _font(30), _font(18), _font(15)
    dr.text((30, 20), "CoroSeg - segmentation automatique des coronaires", fill=(255, 255, 255), font=f_t)
    dr.text((30, 60), f"Série source : {series_description or src.description}   |   CoroSeg {__version__}",
            fill=(180, 180, 180), font=f_s)

    view = _anterior_view(res, 520)
    im.paste(view, (W - view.width - 30, 110))
    dr.text((W - view.width - 30, 110 + view.height + 6), "vue antérieure (droite du patient à gauche)",
            fill=(160, 160, 160), font=f_s)
    y = 110 + view.height + 40
    for k, (code, name, rgb) in LABELS.items():
        if (res.labels == k).any():
            dr.rectangle([W - 550, y, W - 534, y + 16], fill=rgb)
            dr.text((W - 525, y - 2), f"{code} - {name}", fill=(220, 220, 220), font=f_s)
            y += 22

    cols = [("Vaisseau", 0), ("Long. (mm)", 230), ("Ø moy. (mm)", 345), ("Rétréc. max", 470), ("à (mm)", 590),
            ("Ø (mm)", 670)]
    y = 110
    for c, x in cols:
        dr.text((30 + x, y), c, fill=(255, 220, 120), font=f)
    y += 32
    for row in res.summary_rows():
        vals = [row["Vaisseau"], row["Longueur (mm)"], row["Ø moyen (mm)"],
                "" if row["Rétrécissement max (%)"] is None else f"{row['Rétrécissement max (%)']} %",
                row["à (mm de l'origine)"] or "", row["Ø au rétrécissement (mm)"] or ""]
        for (c, x), v in zip(cols, vals):
            dr.text((30 + x, y), str(v), fill=(230, 230, 230), font=f)
        y += 26
        if y > 700:
            dr.text((30, y), "…", fill=(230, 230, 230), font=f)
            break
    y += 20
    vols = res.volumes_ml()
    if vols:
        dr.text((30, y), "Volumes (mL) : " + ", ".join(f"{k} {v:.2f}" for k, v in vols.items()), fill=(200, 200, 200), font=f_s)
        y += 30
    if res.warnings:
        dr.text((30, y), "Avertissements :", fill=(255, 150, 80), font=f)
        y += 28
        for w in res.warnings:
            for line in _wrap(w, 85):
                dr.text((40, y), line, fill=(255, 190, 140), font=f_s)
                y += 20
    dr.text((30, H - 60), "Le « rétrécissement » est calculé sur le diamètre du masque segmenté : repère de lecture,",
            fill=(160, 160, 160), font=f_s)
    dr.text((30, H - 40), "pas une quantification de sténose.  " + DISCLAIMER + ".", fill=(160, 160, 160), font=f_s)

    ds = _base_dataset(src.datasets[0], generate_uid(), series_number, "CoroSeg - rapport", 1, institution)
    _set_rgb(ds, np.asarray(im), jpeg)
    return ds


def _wrap(text: str, n: int) -> list[str]:
    words, lines, cur = text.split(), [], ""
    for w in words:
        if len(cur) + len(w) + 1 > n:
            lines.append(cur)
            cur = w
        else:
            cur = f"{cur} {w}".strip()
    return lines + ([cur] if cur else [])


# ----------------------------------------------------------------------------------------------------
# lignes centrales (DICOM SR 3D) : exploitables par un viewer (CPR interactive, VR)
# ----------------------------------------------------------------------------------------------------
# Codes privés (schéma 99COROSEG) qui identifient les éléments du SR ; le viewer les cherche par CodeValue.
CL_CODES = {
    "root": ("CL-ROOT", "Lignes centrales coronaires CoroSeg"),
    "vessel": ("CL-VESSEL", "Vaisseau"),
    "name": ("CL-NAME", "Nom du vaisseau"),
    "label": ("CL-LABEL", "Code CoroSeg du vaisseau"),
    "line": ("CL-LINE", "Ligne centrale"),
    "radius": ("CL-RADIUS", "Rayons du masque (mm)"),
    "length": ("CL-LENGTH", "Longueur"),
}


def _cl_code(key):
    from pydicom.sr.coding import Code

    v, m = CL_CODES[key]
    return Code(v, "99COROSEG", m)


def make_centerline_sr(src: SourceSeries, res: CoroResult, step_mm: float = 1.0, series_number: int = 9005,
                       institution: str = "") -> Dataset:
    """Comprehensive 3D SR : pour chaque vaisseau, une POLYLINE SCOORD3D en coordonnées patient (LPS, mm).

    Structure (codes 99COROSEG) :
      CONTAINER CL-ROOT
        CONTAINER CL-VESSEL                       (un par ligne centrale : TC→IVA, TC→Cx, CD, branches…)
          TEXT     CL-NAME    « TC → IVA »
          TEXT     CL-LABEL   « IVA »  (TC, IVA, Cx, CD, Diag, Marg, Br. CD, Bissec.)
          SCOORD3D CL-LINE    POLYLINE, points tous les `step_mm` mm, de l'ostium vers la distalité
          TEXT     CL-RADIUS  rayons du masque (mm) aux mêmes points, séparés par des espaces
          NUM      CL-LENGTH  longueur (mm)
    """
    import highdicom as hd
    from pydicom.sr.codedict import codes

    ref = src.datasets[0]
    for_uid = ref.FrameOfReferenceUID
    root = hd.sr.ContainerContentItem(name=_cl_code("root"), template_id=None)
    items = []
    for cl in res.centerlines:
        if cl.length < 5 or len(cl.points) < 2:
            continue
        s_new = np.arange(0.0, cl.length + 1e-6, step_mm)
        if s_new[-1] < cl.length - 1e-3:
            s_new = np.append(s_new, cl.length)
        pts = np.stack([np.interp(s_new, cl.s, cl.points[:, k]) for k in range(3)], axis=1)
        rad = np.interp(s_new, cl.s, cl.radius)
        lps = pts * np.array([-1.0, -1.0, 1.0])  # RAS (CoroSeg) → LPS (DICOM)
        v = hd.sr.ContainerContentItem(name=_cl_code("vessel"), relationship_type="CONTAINS")
        v.ContentSequence = [
            hd.sr.TextContentItem(name=_cl_code("name"), value=cl.name, relationship_type="CONTAINS"),
            hd.sr.TextContentItem(name=_cl_code("label"), value=LABELS.get(cl.label, ("?",))[0],
                                  relationship_type="CONTAINS"),
            hd.sr.Scoord3DContentItem(name=_cl_code("line"), graphic_type=hd.sr.GraphicTypeValues3D.POLYLINE,
                                      graphic_data=lps.astype(np.float32), frame_of_reference_uid=for_uid,
                                      relationship_type="CONTAINS"),
            hd.sr.TextContentItem(name=_cl_code("radius"), value=" ".join(f"{r:.2f}" for r in rad),
                                  relationship_type="CONTAINS"),
            hd.sr.NumContentItem(name=_cl_code("length"), value=round(float(cl.length), 1),
                                 unit=codes.UCUM.Millimeter, relationship_type="CONTAINS"),
        ]
        items.append(v)
    if not items:
        raise ValueError("aucune ligne centrale à exporter")
    root.ContentSequence = items
    sr = hd.sr.Comprehensive3DSR(
        evidence=src.datasets,
        content=root,
        series_instance_uid=hd.UID(),
        series_number=series_number,
        sop_instance_uid=hd.UID(),
        instance_number=1,
        manufacturer=MANUFACTURER,
        institution_name=institution or None,
        is_complete=True,
    )
    sr.SeriesDescription = "CoroSeg - lignes centrales"
    sr.SpecificCharacterSet = "ISO_IR 192"
    sr.SoftwareVersions = __version__
    return sr
