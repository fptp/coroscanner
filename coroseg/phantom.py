"""Fantôme synthétique de coroscanner (mode démo et tests) — aucune donnée patient."""

from __future__ import annotations

import nibabel as nib
import numpy as np

from .labeling import CD, CX, DIAG, IVA, MARG, TC, BR_CD

# (polyline en mm RAS, rayon début, rayon fin, label attendu)
VESSELS = {
    "TC": ([(-9, -3, 10), (-15, -5, 9.5), (-20, -6, 8)], 2.2, 2.0, TC),
    "IVA": ([(-20, -6, 8), (-24, 8, 2), (-28, 22, -10), (-25, 30, -30), (-20, 32, -45)], 1.9, 0.8, IVA),
    "Diag1": ([(-27, 18, -6), (-40, 22, -14), (-48, 22, -28)], 1.2, 0.6, DIAG),
    "Diag2": ([(-26, 28, -22), (-38, 30, -32), (-42, 28, -42)], 1.0, 0.6, DIAG),
    "Cx": ([(-20, -6, 8), (-32, -14, 4), (-40, -26, -4), (-38, -36, -16)], 1.8, 0.8, CX),
    "Marg1": ([(-36, -20, 0), (-48, -22, -14), (-50, -20, -30)], 1.1, 0.6, MARG),
    "CD": ([(13.4, 11.2, 10), (24, 20, 6), (34, 18, -8), (38, 8, -22), (34, -6, -34), (22, -20, -40)], 2.0, 1.0, CD),
    "BrCD1": ([(37, 12, -16), (40, 26, -28), (36, 32, -40)], 1.0, 0.6, BR_CD),
    # fragment distal de CD interrompu (≈2 mm de « trou ») : doit être rattaché à la CD
    "CD_frag": ([(19.5, -22.5, -40.5), (12, -28, -41), (4, -32, -43)], 0.95, 0.8, CD),
}
STENOSIS = ("IVA", 0.32, 0.55)  # vaisseau, position relative, réduction du rayon


def _resample(poly, step=0.2):
    poly = np.asarray(poly, float)
    seg = np.linalg.norm(np.diff(poly, axis=0), axis=1)
    s = np.concatenate([[0], np.cumsum(seg)])
    t = np.arange(0, s[-1] + 1e-6, step)
    return np.stack([np.interp(t, s, poly[:, k]) for k in range(3)], 1), t / s[-1]


def _tube(shape, affine, pts, radii):
    mask = np.zeros(shape, bool)
    inv = np.linalg.inv(affine)
    sp = np.sqrt((affine[:3, :3] ** 2).sum(0))
    for p, r in zip(pts, radii):
        c = inv[:3, :3] @ p + inv[:3, 3]
        rv = np.ceil(r / sp).astype(int) + 1
        lo = np.maximum(np.floor(c).astype(int) - rv, 0)
        hi = np.minimum(np.floor(c).astype(int) + rv + 2, shape)
        if np.any(hi <= lo):
            continue
        g = np.stack(np.meshgrid(*[np.arange(lo[k], hi[k]) for k in range(3)], indexing="ij"), -1)
        w = g @ affine[:3, :3].T + affine[:3, 3]
        mask[lo[0]:hi[0], lo[1]:hi[1], lo[2]:hi[2]] |= np.linalg.norm(w - p, axis=-1) <= r
    return mask


def _ellipsoid(world, c, r):
    return (((world - np.asarray(c)) / np.asarray(r)) ** 2).sum(-1) <= 1


def make_phantom(spacing: float = 0.5, seed: int = 0):
    """Renvoie (ct nib RAS int16, masque coronaires, masque aorte, dict des polylignes attendues)."""
    rng = np.random.default_rng(seed)
    lo, hi = np.array([-60.0, -50.0, -55.0]), np.array([55.0, 50.0, 45.0])
    shape = tuple(np.round((hi - lo) / spacing).astype(int))
    aff = np.diag([spacing] * 3 + [1.0])
    aff[:3, 3] = lo
    ijk = np.stack(np.meshgrid(*[np.arange(n, dtype=np.float32) for n in shape], indexing="ij"), -1)
    world = ijk * spacing + lo.astype(np.float32)

    vol = np.full(shape, -80.0, np.float32)  # graisse épicardique
    vol[_ellipsoid(world, (-12, 3, -20), (30, 28, 30))] = 90  # myocarde
    vol[_ellipsoid(world, (-12, 3, -20), (17, 14, 20))] = 380  # cavité VG opacifiée
    vol[_ellipsoid(world, (24, 4, -20), (12, 14, 20))] = 340  # VD
    xy = world[..., :2] - np.array([5.0, 0.0], np.float32)
    aorta = (np.linalg.norm(xy, axis=-1) <= 14) & (world[..., 2] >= 4)
    aorta |= _ellipsoid(world, (5, 0, 6), (15, 15, 7))
    vol[aorta] = 430

    cor = np.zeros(shape, bool)
    for name, (poly, r0, r1, _lab) in VESSELS.items():
        pts, f = _resample(poly)
        radii = r0 + (r1 - r0) * f
        if name == STENOSIS[0]:
            radii = radii * (1 - STENOSIS[2] * np.exp(-((f - STENOSIS[1]) / 0.03) ** 2))
        cor |= _tube(shape, aff, pts, radii)
    cor &= ~aorta
    vol[cor] = 420
    # plaque calcifiée en regard de la sténose
    pts, f = _resample(VESSELS["IVA"][0])
    p = pts[np.argmin(abs(f - STENOSIS[1]))] + np.array([1.8, 0, 0])
    vol[_ellipsoid(world, p, (1.0, 1.4, 1.0))] = 900
    vol += rng.normal(0, 22, shape).astype(np.float32)

    ct = nib.Nifti1Image(np.round(vol).astype(np.int16), aff)
    ct.header.set_xyzt_units("mm")
    return ct, cor, aorta, VESSELS
