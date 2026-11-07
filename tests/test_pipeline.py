"""Tests sur fantôme synthétique :  python -m pytest -q"""

import numpy as np
import pytest

from coroseg import LABELS
from coroseg.cpr import cross_section, straightened_cpr
from coroseg.labeling import label_coronaries
from coroseg.phantom import STENOSIS, make_phantom
from coroseg.pipeline import analyse


@pytest.fixture(scope="module")
def demo():
    ct, cor, aorta, vessels = make_phantom()
    return ct, cor, aorta, vessels, analyse(ct, cor, aorta, log=lambda *a: None)


def _labels_along(res, poly):
    inv = np.linalg.inv(res.affine)
    poly = np.asarray(poly, float)
    pts = [a + (b - a) * t for a, b in zip(poly[:-1], poly[1:]) for t in np.linspace(0.15, 0.85, 5)]
    v = np.round(np.array(pts) @ inv[:3, :3].T + inv[:3, 3]).astype(int)
    return res.labels[tuple(v.T)]


def test_every_vessel_gets_expected_label(demo):
    *_, vessels, res = demo
    for name, (poly, _, _, lab) in vessels.items():
        got = _labels_along(res, poly)
        assert (got == lab).mean() >= 0.9, f"{name}: attendu {LABELS[lab][0]}, obtenu {np.unique(got)}"


def test_whole_mask_labelled(demo):
    _, cor, *_, res = demo
    assert np.all(res.labels[cor] > 0)
    assert not np.any(res.labels[~cor])


def test_stenosis_found_on_lad(demo):
    *_, res = demo
    row = next(r for r in res.summary_rows() if r["Vaisseau"] == "TC → IVA")
    assert row["Rétrécissement max (%)"] >= 35
    lad = next(c for c in res.centerlines if c.name == "TC → IVA")
    assert abs(row["à (mm de l'origine)"] - (10 + STENOSIS[1] * (lad.length - 10))) < 8


def test_left_right_swap_when_mirrored(demo):
    """Si l'on inverse gauche/droite du patient, CD et TC doivent s'échanger de côté mais rester cohérents."""
    ct, cor, aorta, _, _ = demo
    lab = label_coronaries(cor[::-1], aorta[::-1], ct.affine, log=lambda *a: None)
    assert {1, 2, 3, 4} <= set(np.unique(lab.label_map))


def test_cpr_shapes(demo):
    *_, res = demo
    cl = res.centerlines[1]
    img = straightened_cpr(res.hu, res.affine, cl, angle_deg=45, half_width_mm=10, pixel_mm=0.5)
    assert img.shape == (int(np.ceil(cl.length / 0.5)), 41)
    assert img[:, 20].mean() > 300  # le centre de la CPR suit la lumière opacifiée
    xs = cross_section(res.hu, res.affine, cl, 5, half_width_mm=5, pixel_mm=0.5)
    assert xs.shape == (21, 21) and xs[10, 10] > 250


def test_save(tmp_path, demo):
    *_, res = demo
    out = res.save(tmp_path / "r")
    for f in ["coronaires_labels.nii.gz", "centerlines.json", "centerlines.csv", "rapport.json", "maillages/02_IVA.stl"]:
        assert (out / f).exists(), f


def test_dicom_roundtrip(tmp_path):
    sitk = pytest.importorskip("SimpleITK")
    from coroseg.io import list_dicom_series, load_dicom_series

    ct, *_ = make_phantom(spacing=1.0)
    arr = np.asarray(ct.dataobj)  # RAS (x,y,z)
    img = sitk.GetImageFromArray(np.ascontiguousarray(arr[::-1, ::-1, :].transpose(2, 1, 0)))  # → LPS, (z,y,x)
    img.SetSpacing((1.0, 1.0, 1.0))
    d = tmp_path / "dcm"
    d.mkdir()
    w = sitk.ImageFileWriter()
    w.KeepOriginalImageUIDOn()
    for z in range(img.GetDepth()):
        sl = img[:, :, z]
        for tag, val in [("0008|0060", "CT"), ("0020|000e", "1.2.3.4.5"), ("0008|103e", "CORO 75%"),
                         ("0020|0032", f"0\\0\\{z}"), ("0020|0037", "1\\0\\0\\0\\1\\0"), ("0020|0013", str(z))]:
            sl.SetMetaData(tag, val)
        w.SetFileName(str(d / f"{z:04d}.dcm"))
        w.Execute(sl)
    series = list_dicom_series(d)
    assert len(series) == 1 and series[0].n_slices == arr.shape[2]
    back = np.asarray(load_dicom_series(series[0]).dataobj)
    assert back.shape == arr.shape
    assert np.abs(back.astype(int) - arr).mean() < 1
