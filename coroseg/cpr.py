"""Reconstructions curvilignes (CPR redressée) et coupes orthogonales le long d'une ligne centrale."""

from __future__ import annotations

import numpy as np
from scipy import ndimage

from .centerline import Centerline


def rotation_minimizing_frames(points: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Repères à rotation minimale (méthode de double réflexion, Wang et al. 2008).

    Renvoie (T, N, B) : tangentes, normales, binormales, chacune (M,3).
    """
    m = len(points)
    t = np.gradient(points, axis=0)
    t /= np.linalg.norm(t, axis=1, keepdims=True) + 1e-12
    # normale initiale : perpendiculaire à t0, orientée de préférence vers l'antérieur (+y)
    ref = np.array([0.0, 1.0, 0.0]) if abs(t[0][1]) < 0.9 else np.array([1.0, 0.0, 0.0])
    n0 = ref - np.dot(ref, t[0]) * t[0]
    n = np.zeros_like(points)
    n[0] = n0 / np.linalg.norm(n0)
    for i in range(m - 1):
        v1 = points[i + 1] - points[i]
        c1 = v1 @ v1
        if c1 < 1e-12:
            n[i + 1] = n[i]
            continue
        rl = n[i] - (2 / c1) * (v1 @ n[i]) * v1
        tl = t[i] - (2 / c1) * (v1 @ t[i]) * v1
        v2 = t[i + 1] - tl
        c2 = v2 @ v2
        n[i + 1] = rl if c2 < 1e-12 else rl - (2 / c2) * (v2 @ rl) * v2
        n[i + 1] /= np.linalg.norm(n[i + 1]) + 1e-12
    b = np.cross(t, n)
    return t, n, b


def _world_to_vox(affine: np.ndarray, pts: np.ndarray) -> np.ndarray:
    inv = np.linalg.inv(affine)
    return pts @ inv[:3, :3].T + inv[:3, 3]


def _sample(vol: np.ndarray, affine: np.ndarray, pts: np.ndarray, order: int = 1, cval: float = -1024) -> np.ndarray:
    shp = pts.shape[:-1]
    v = _world_to_vox(affine, pts.reshape(-1, 3))
    out = ndimage.map_coordinates(vol, v.T, order=order, mode="constant", cval=cval)
    return out.reshape(shp)


def straightened_cpr(vol, affine, cl: Centerline, angle_deg: float = 0.0, half_width_mm: float = 12.0,
                     pixel_mm: float = 0.3, order: int = 1) -> np.ndarray:
    """CPR redressée : lignes = abscisse curviligne (pas du centerline), colonnes = décalage latéral.

    L'image est rééchantillonnée pour avoir des pixels carrés de `pixel_mm`.
    """
    _, n, b = rotation_minimizing_frames(cl.points)
    a = np.deg2rad(angle_deg)
    d = np.cos(a) * n + np.sin(a) * b
    u = np.arange(-half_width_mm, half_width_mm + 1e-6, pixel_mm)
    s_new = np.arange(0, cl.length, pixel_mm)
    idx = np.interp(s_new, cl.s, np.arange(len(cl.s)))
    i0 = np.floor(idx).astype(int)
    f = (idx - i0)[:, None]
    i1 = np.minimum(i0 + 1, len(cl.s) - 1)
    p = cl.points[i0] * (1 - f) + cl.points[i1] * f
    dd = d[i0] * (1 - f) + d[i1] * f
    dd /= np.linalg.norm(dd, axis=1, keepdims=True)
    grid = p[:, None, :] + u[None, :, None] * dd[:, None, :]
    return _sample(vol, affine, grid, order=order)


def cross_section(vol, affine, cl: Centerline, s_mm: float, half_width_mm: float = 10.0, pixel_mm: float = 0.2,
                  order: int = 1) -> np.ndarray:
    """Coupe orthogonale au vaisseau à l'abscisse s_mm."""
    _, n, b = rotation_minimizing_frames(cl.points)
    i = int(np.clip(np.searchsorted(cl.s, s_mm), 0, len(cl.s) - 1))
    u = np.arange(-half_width_mm, half_width_mm + 1e-6, pixel_mm)
    uu, vv = np.meshgrid(u, u, indexing="ij")
    grid = cl.points[i] + uu[..., None] * n[i] + vv[..., None] * b[i]
    return _sample(vol, affine, grid, order=order)


def window(img: np.ndarray, level: float = 250, width: float = 800) -> np.ndarray:
    lo = level - width / 2
    return (np.clip((img - lo) / width, 0, 1) * 255).astype(np.uint8)
