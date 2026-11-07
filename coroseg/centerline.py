"""Squelettisation, graphe du squelette, arbre des plus courts chemins et lignes centrales lissées."""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from scipy import ndimage
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import dijkstra
from skimage.morphology import skeletonize

# 13 décalages "vers l'avant" : couvrent les 26 voisins sans doublon
_OFFSETS = np.array(
    [(dx, dy, dz) for dx in (-1, 0, 1) for dy in (-1, 0, 1) for dz in (-1, 0, 1) if (dx, dy, dz) > (0, 0, 0)]
)


@dataclass
class SkeletonTree:
    """Arbre (plus courts chemins depuis l'ostium) construit sur les voxels du squelette."""

    ijk: np.ndarray  # (N,3) indices voxel (repère du crop)
    xyz: np.ndarray  # (N,3) coordonnées monde en mm (RAS)
    radius: np.ndarray  # (N,) rayon inscrit local (mm)
    root: int
    parent: np.ndarray  # (N,) -1 pour la racine ou les nœuds non atteints
    dist: np.ndarray  # (N,) distance géodésique à la racine (mm), inf si non atteint
    children: list[list[int]] = field(default_factory=list)
    sub_len: np.ndarray | None = None  # longueur cumulée du sous-arbre (mm)
    sub_w: np.ndarray | None = None  # "volume" du sous-arbre (∑ longueur·r²)

    def edge_len(self, n: int) -> float:
        p = self.parent[n]
        return 0.0 if p < 0 else float(self.dist[n] - self.dist[p])

    def branch_len(self, c: int) -> float:
        """Longueur d'une branche partant de c, segment d'attache compris."""
        return float(self.sub_len[c] + self.edge_len(c))

    def main_path(self, start: int) -> list[int]:
        """Suit à chaque bifurcation l'enfant de plus gros sous-arbre (critère longueur × calibre²)."""
        path = [start]
        n = start
        while self.children[n]:
            n = max(self.children[n], key=lambda c: self.sub_w[c])
            path.append(n)
        return path

    def subtree(self, start: int) -> list[int]:
        out, stack = [], [start]
        while stack:
            n = stack.pop()
            out.append(n)
            stack.extend(self.children[n])
        return out

    def point_at(self, path: list[int], mm: float) -> np.ndarray:
        """Point du chemin situé à `mm` mm (géodésiques) de son début, ou son extrémité."""
        d0 = self.dist[path[0]]
        for n in path:
            if self.dist[n] - d0 >= mm:
                return self.xyz[n]
        return self.xyz[path[-1]]


def skeleton_of(mask: np.ndarray) -> np.ndarray:
    return skeletonize(mask.astype(bool)) > 0


def build_tree(skel: np.ndarray, root_xyz: np.ndarray, affine: np.ndarray, radius_map: np.ndarray) -> SkeletonTree:
    """Construit l'arbre des plus courts chemins sur le squelette, enraciné au voxel le plus proche de root_xyz."""
    ijk = np.argwhere(skel)
    n = len(ijk)
    if n == 0:
        raise ValueError("squelette vide")
    xyz = ijk @ affine[:3, :3].T + affine[:3, 3]
    lut = -np.ones(skel.shape, dtype=np.int64)
    lut[tuple(ijk.T)] = np.arange(n)

    rows, cols, w = [], [], []
    shape = np.array(skel.shape)
    for off in _OFFSETS:
        nb = ijk + off
        ok = np.all((nb >= 0) & (nb < shape), axis=1)
        src = np.nonzero(ok)[0]
        dst = lut[tuple(nb[ok].T)]
        keep = dst >= 0
        src, dst = src[keep], dst[keep]
        d = np.linalg.norm(xyz[dst] - xyz[src], axis=1)
        rows += [src, dst]
        cols += [dst, src]
        w += [d, d]
    rows = np.concatenate(rows) if rows else np.array([], int)
    cols = np.concatenate(cols) if cols else np.array([], int)
    w = np.concatenate(w) if w else np.array([], float)
    g = coo_matrix((w, (rows, cols)), shape=(n, n)).tocsr()

    root = int(np.argmin(np.linalg.norm(xyz - root_xyz, axis=1)))
    dist, pred = dijkstra(g, directed=False, indices=root, return_predecessors=True)
    parent = np.where(pred < 0, -1, pred).astype(np.int64)

    children: list[list[int]] = [[] for _ in range(n)]
    for c in np.nonzero(parent >= 0)[0]:
        children[parent[c]].append(int(c))

    radius = radius_map[tuple(ijk.T)].astype(float)
    tree = SkeletonTree(ijk, xyz, radius, root, parent, dist, children)

    # longueurs / poids des sous-arbres : on remonte des feuilles vers la racine
    sub_len = np.zeros(n)
    sub_w = np.zeros(n)
    reach = np.isfinite(dist)
    for node in np.argsort(-np.where(reach, dist, -1)):
        if not reach[node]:
            continue
        p = parent[node]
        if p >= 0:
            e = dist[node] - dist[p]
            sub_len[p] += sub_len[node] + e
            sub_w[p] += sub_w[node] + e * radius[node] ** 2
    tree.sub_len, tree.sub_w = sub_len, sub_w
    return tree


@dataclass
class Centerline:
    name: str
    label: int
    points: np.ndarray  # (M,3) mm, monde RAS
    radius: np.ndarray  # (M,) mm
    s: np.ndarray  # (M,) abscisse curviligne (mm)

    @property
    def length(self) -> float:
        return float(self.s[-1]) if len(self.s) else 0.0

    def to_dict(self) -> dict:
        return {
            "name": self.name,
            "label": self.label,
            "length_mm": round(self.length, 2),
            "points_mm": np.round(self.points, 3).tolist(),
            "radius_mm": np.round(self.radius, 3).tolist(),
            "s_mm": np.round(self.s, 3).tolist(),
        }

    @staticmethod
    def from_dict(d: dict) -> "Centerline":
        return Centerline(d["name"], d["label"], np.array(d["points_mm"]), np.array(d["radius_mm"]), np.array(d["s_mm"]))


def smooth_centerline(name: str, label: int, xyz: np.ndarray, radius: np.ndarray, step: float = 0.5, sigma_mm: float = 1.5) -> Centerline | None:
    """Rééchantillonne un chemin de voxels à pas constant et le lisse (gaussien)."""
    if len(xyz) < 3:
        return None
    seg = np.linalg.norm(np.diff(xyz, axis=0), axis=1)
    s = np.concatenate([[0], np.cumsum(seg)])
    if s[-1] < 2 * step:
        return None
    s_new = np.arange(0, s[-1] + 1e-6, step)
    pts = np.stack([np.interp(s_new, s, xyz[:, k]) for k in range(3)], axis=1)
    rad = np.interp(s_new, s, radius)
    sig = sigma_mm / step
    pts_s = ndimage.gaussian_filter1d(pts, sig, axis=0, mode="nearest")
    pts_s[0], pts_s[-1] = pts[0], pts[-1]
    rad_s = ndimage.gaussian_filter1d(rad, sig, mode="nearest")
    seg2 = np.linalg.norm(np.diff(pts_s, axis=0), axis=1)
    s2 = np.concatenate([[0], np.cumsum(seg2)])
    return Centerline(name, label, pts_s, rad_s, s2)
