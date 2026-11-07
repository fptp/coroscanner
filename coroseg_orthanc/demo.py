"""Fantôme CoroSeg exporté en DICOM CT, pour tester la chaîne Orthanc sans examen réel ni licence."""

from __future__ import annotations

import io

import numpy as np
from pydicom.dataset import Dataset, FileMetaDataset
from pydicom.uid import CTImageStorage, ExplicitVRLittleEndian, generate_uid


def phantom_dicom_files(spacing: float = 0.5) -> list[bytes]:
    from coroseg.phantom import make_phantom

    from .dicom_export import nifti_to_sitk

    import SimpleITK as sitk

    ct, *_ = make_phantom(spacing=spacing)
    img = sitk.DICOMOrient(nifti_to_sitk(np.asarray(ct.dataobj), ct.affine), "LPS")
    arr = sitk.GetArrayFromImage(img)  # (z, y, x)
    d = np.array(img.GetDirection()).reshape(3, 3)
    study, series, frame = generate_uid(), generate_uid(), generate_uid()
    out = []
    for z in range(arr.shape[0]):
        ds = Dataset()
        ds.SpecificCharacterSet = "ISO_IR 192"
        ds.SOPClassUID = CTImageStorage
        ds.SOPInstanceUID = generate_uid()
        ds.PatientName = "FANTOME^CoroSeg"
        ds.PatientID = "COROSEG-DEMO"
        ds.PatientBirthDate = "19600101"
        ds.PatientSex = "O"
        ds.StudyInstanceUID, ds.SeriesInstanceUID, ds.FrameOfReferenceUID = study, series, frame
        ds.StudyDate = ds.SeriesDate = ds.ContentDate = "20260101"
        ds.StudyTime = ds.SeriesTime = ds.ContentTime = "120000"
        ds.StudyID, ds.AccessionNumber, ds.ReferringPhysicianName = "1", "DEMO1", ""
        ds.StudyDescription = "Coroscanner (fantome de demonstration)"
        ds.SeriesDescription = "CORO 75% 0.5mm"
        ds.Modality, ds.Manufacturer = "CT", "Fantome"
        ds.SeriesNumber, ds.InstanceNumber = 3, z + 1
        ds.ImageType = ["ORIGINAL", "PRIMARY", "AXIAL"]
        ds.PatientPosition = "HFS"
        ds.ImagePositionPatient = [round(float(v), 4) for v in img.TransformIndexToPhysicalPoint((0, 0, z))]
        ds.ImageOrientationPatient = [float(v) for v in (*d[:, 0], *d[:, 1])]
        ds.PixelSpacing = [spacing, spacing]
        ds.SliceThickness = spacing
        ds.KVP = 100
        ds.ConvolutionKernel = "B26f"
        ds.Rows, ds.Columns = arr.shape[1:]
        ds.SamplesPerPixel, ds.PhotometricInterpretation = 1, "MONOCHROME2"
        ds.BitsAllocated, ds.BitsStored, ds.HighBit, ds.PixelRepresentation = 16, 16, 15, 1
        ds.RescaleIntercept, ds.RescaleSlope, ds.RescaleType = 0, 1, "HU"
        ds.WindowCenter, ds.WindowWidth = 300, 900
        ds.PixelData = np.ascontiguousarray(arr[z].astype(np.int16)).tobytes()
        meta = FileMetaDataset()
        meta.MediaStorageSOPClassUID, meta.MediaStorageSOPInstanceUID = ds.SOPClassUID, ds.SOPInstanceUID
        meta.TransferSyntaxUID = ExplicitVRLittleEndian
        ds.file_meta = meta
        buf = io.BytesIO()
        ds.save_as(buf, enforce_file_format=True)
        out.append(buf.getvalue())
    return out


def phantom_segmenter(ct, work_dir):
    """Remplace TotalSegmentator par les masques exacts du fantôme (rééchantillonnés sur le volume reçu)."""
    import nibabel as nib
    from nibabel.processing import resample_from_to

    from coroseg.phantom import make_phantom

    ct0, cor, aorta, _ = make_phantom(spacing=0.5)
    out = []
    for m in (cor, aorta):
        mi = nib.Nifti1Image(m.astype(np.uint8), ct0.affine)
        out.append(np.asarray(resample_from_to(mi, ct, order=0).dataobj) > 0)
    return out[0], out[1]
