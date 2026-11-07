"""Maillages de surface (marching cubes) et export STL."""

from __future__ import annotations

import numpy as np
from scipy import ndimage
from skimage import measure


def mask_to_mesh(mask: np.ndarray, affine: np.ndarray, smooth_sigma: float = 0.7, step: int = 1):
    """Renvoie (verts_mm, faces) ou None. Les sommets sont en coordonnées monde (mm)."""
    if mask.sum() < 4:
        return None
    idx = np.argwhere(mask)
    lo = np.maximum(idx.min(0) - 3, 0)
    hi = np.minimum(idx.max(0) + 4, mask.shape)
    sub = mask[lo[0]:hi[0], lo[1]:hi[1], lo[2]:hi[2]].astype(np.float32)
    if smooth_sigma > 0:
        sub = ndimage.gaussian_filter(sub, smooth_sigma)
    try:
        verts, faces, _, _ = measure.marching_cubes(sub, level=0.5 if smooth_sigma == 0 else 0.4, step_size=step)
    except (ValueError, RuntimeError):
        return None
    verts = verts + lo
    verts_mm = verts @ affine[:3, :3].T + affine[:3, 3]
    return verts_mm, faces


def write_stl(path, verts: np.ndarray, faces: np.ndarray, name: str = "coroseg") -> None:
    tri = verts[faces]
    nrm = np.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0])
    nrm /= np.linalg.norm(nrm, axis=1, keepdims=True) + 1e-12
    rec = np.zeros(len(faces), dtype=[("n", "<f4", 3), ("v", "<f4", (3, 3)), ("a", "<u2")])
    rec["n"], rec["v"] = nrm, tri
    with open(path, "wb") as f:
        f.write(name.encode()[:80].ljust(80, b"\0"))
        f.write(np.uint32(len(faces)).tobytes())
        f.write(rec.tobytes())
