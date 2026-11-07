"""Tests du pont Orthanc (sans serveur Orthanc : conversion DICOM aller-retour sur le fantôme)."""

import io

import numpy as np
import pytest

pytest.importorskip("highdicom")
sitk = pytest.importorskip("SimpleITK")

import pydicom  # noqa: E402

from coroseg.pipeline import analyse  # noqa: E402
from coroseg_orthanc import dicom_export as dx  # noqa: E402
from coroseg_orthanc.demo import phantom_dicom_files, phantom_segmenter  # noqa: E402


@pytest.fixture(scope="module")
def processed(tmp_path_factory):
    d = tmp_path_factory.mktemp("dcm")
    for i, b in enumerate(phantom_dicom_files()):
        (d / f"{i:04d}.dcm").write_bytes(b)
    src = dx.load_source(d)
    cor, aorta = phantom_segmenter(src.ct, d)
    res = analyse(src.ct, cor, aorta, log=lambda *_: None)
    lab_src = dx.labels_on_source_grid(res.labels, res.affine, src)
    return src, res, lab_src


def test_labels_back_on_source_grid(processed):
    src, res, lab_src = processed
    assert lab_src.shape == src.hu.shape
    assert (lab_src > 0).sum() == (res.labels > 0).sum()  # réorientation pure : aucun voxel perdu
    # les coronaires sont opacifiées dans le fantôme (≈ 420 UH)
    assert np.median(src.hu[lab_src > 0]) > 300


def test_seg_roundtrip(processed):
    import highdicom as hd

    src, _res, lab_src = processed
    seg = dx.make_seg(src, lab_src)
    back = hd.seg.segread(io.BytesIO(dx.to_bytes(seg)))
    present = sorted(int(k) for k in np.unique(lab_src) if k)
    assert len(back.SegmentSequence) == len(present)
    vol = back.get_pixels_by_source_instance([d.SOPInstanceUID for d in src.datasets], combine_segments=True,
                                             skip_overlap_checks=True)
    remap = np.zeros(256, np.uint8)
    remap[present] = np.arange(1, len(present) + 1)
    assert np.array_equal(vol, remap[lab_src])


def test_secondary_captures(processed):
    src, res, lab_src = processed
    fusion = dx.make_fusion_series(src, lab_src, "roi")
    assert 0 < len(fusion) < len(src.datasets)
    cpr = dx.make_cpr_series(src, res)
    assert {d.DerivationDescription.split()[-1] for d in cpr} >= {"IVA", "Cx", "CD"}
    rep = dx.make_report_image(src, res)
    for ds in fusion[:2] + cpr + [rep]:
        back = pydicom.dcmread(io.BytesIO(dx.to_bytes(ds)))
        assert back.StudyInstanceUID == src.datasets[0].StudyInstanceUID
        assert back.Manufacturer == "CoroSeg"
