"""Identification des troncs coronaires (TC, IVA, Cx, CD) et de leurs branches à partir d'un masque binaire.

Heuristiques anatomiques (repère RAS : +x = droite du patient, +y = antérieur, +z = supérieur) :
  * un arbre coronaire = composante connexe du masque qui touche l'aorte ; l'ostium = son voxel le plus proche de l'aorte ;
  * arbre gauche / droit : position de l'ostium par rapport au centre de la racine aortique (gauche = x plus petit) ;
  * tronc principal d'un arbre : à chaque bifurcation on suit l'enfant au plus gros sous-arbre (longueur × calibre²) ;
  * TC : de l'ostium gauche à la première bifurcation dont deux enfants sont « significatifs » ;
  * IVA vs Cx : l'IVA part vers l'avant (sillon interventriculaire antérieur), la Cx vers l'arrière (sillon AV gauche) ;
    s'il y a trois branches significatives, celle du milieu est la bissectrice ;
  * toute branche latérale significative d'un tronc reçoit l'étiquette « branche » de ce tronc.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from scipy import ndimage

from .centerline import Centerline, SkeletonTree, build_tree, skeleton_of, smooth_centerline

TC, IVA, CX, CD, DIAG, MARG, BR_CD, RAMUS, UNK = 1, 2, 3, 4, 5, 6, 7, 8, 9


@dataclass
class LabelingResult:
    label_map: np.ndarray  # uint8, même forme que le masque d'entrée
    centerlines: list[Centerline]
    warnings: list[str] = field(default_factory=list)
    ostia_mm: dict[str, list[float]] = field(default_factory=dict)


def _crop_box(mask: np.ndarray, margin_vox: np.ndarray) -> tuple[slice, ...]:
    idx = np.argwhere(mask)
    lo = np.maximum(idx.min(0) - margin_vox, 0)
    hi = np.minimum(idx.max(0) + margin_vox + 1, mask.shape)
    return tuple(slice(int(a), int(b)) for a, b in zip(lo, hi))


def label_coronaries(
    cor_mask: np.ndarray,
    aorta_mask: np.ndarray | None,
    affine: np.ndarray,
    min_branch_mm: float = 10.0,
    touch_mm: float = 6.0,
    reattach_mm: float = 15.0,
    log=print,
) -> LabelingResult:
    warnings: list[str] = []
    spacing = np.sqrt((affine[:3, :3] ** 2).sum(axis=0))
    cor_mask = cor_mask.astype(bool)
    out = np.zeros(cor_mask.shape, np.uint8)
    if not cor_mask.any():
        return LabelingResult(out, [], ["Masque coronaire vide."])

    # ---- travail dans un crop autour des coronaires --------------------------------------
    box = _crop_box(cor_mask, np.ceil(25.0 / spacing).astype(int))
    off = np.array([s.start for s in box])
    aff = affine.copy()
    aff[:3, 3] = affine[:3, 3] + affine[:3, :3] @ off
    cor = cor_mask[box]
    aorta = aorta_mask[box].astype(bool) if aorta_mask is not None else np.zeros_like(cor)

    comps, ncomp = ndimage.label(cor, structure=np.ones((3, 3, 3)))
    sizes = ndimage.sum(cor, comps, index=np.arange(1, ncomp + 1))
    vox_mm3 = float(np.prod(spacing))

    # ---- recherche des ostia ------------------------------------------------------------------
    trees_info = []  # (comp_id, ostium_xyz, dist_to_aorta)
    if aorta.any():
        d_aorta = ndimage.distance_transform_edt(~aorta, sampling=spacing)
        for c in range(1, ncomp + 1):
            if sizes[c - 1] * vox_mm3 < 15:
                continue
            vox = np.argwhere(comps == c)
            d = d_aorta[tuple(vox.T)]
            k = int(np.argmin(d))
            if d[k] <= touch_mm:
                trees_info.append((c, vox[k], float(d[k])))
    else:
        warnings.append("Aorte non disponible : ostia estimés par le point le plus crânial de chaque arbre.")
        for c in (np.argsort(-sizes)[:2] + 1):
            vox = np.argwhere(comps == c)
            trees_info.append((int(c), vox[np.argmax(vox[:, 2])], np.nan))

    if not trees_info:
        warnings.append("Aucune composante coronaire ne touche l'aorte : étiquetage impossible, tout est « non attribué ».")
        out[cor_mask] = UNK
        return LabelingResult(out, [], warnings)

    trees_info.sort(key=lambda t: -sizes[t[0] - 1])
    to_xyz = lambda v: aff[:3, :3] @ np.asarray(v, float) + aff[:3, 3]  # noqa: E731

    # côté de chaque ostium par rapport au centre local de la racine aortique
    def side_score(ost_vox):
        if aorta.any():
            z0 = ost_vox[2]
            dz = int(np.ceil(15 / spacing[2]))
            sl = aorta[:, :, max(z0 - dz, 0): z0 + dz + 1]
            pts = np.argwhere(sl)
            if len(pts):
                pts[:, 2] += max(z0 - dz, 0)
                center = to_xyz(pts.mean(0))
                return float(to_xyz(ost_vox)[0] - center[0])  # <0 : gauche du patient
        return np.nan

    main = trees_info[:2]
    extra = trees_info[2:]
    scores = [side_score(t[1]) for t in main]
    if len(main) == 2:
        order = np.argsort([s if np.isfinite(s) else 0.0 for s in scores]) if any(np.isfinite(scores)) else np.argsort(
            [to_xyz(t[1])[0] for t in main])
        left, right = main[order[0]], main[order[1]]
    else:
        s = scores[0]
        if not np.isfinite(s):
            s = 0.0
        left, right = (main[0], None) if s < 0 else (None, main[0])
        warnings.append(
            f"Un seul arbre coronaire relié à l'aorte (attribué à {'gauche' if left else 'droite'}) : "
            "l'autre est probablement interrompu près de l'ostium."
        )

    # ---- rayon local (EDT corrigé d'un demi-voxel) et squelette ------------------------------------
    edt = ndimage.distance_transform_edt(cor, sampling=spacing)
    radius_map = np.maximum(edt - 0.5 * spacing.min(), 0.5 * spacing.min())
    skel_all = skeleton_of(cor)
    skel_lab = np.zeros(cor.shape, np.uint8)
    centerlines: list[Centerline] = []

    def mk_cl(tree: SkeletonTree, nodes: list[int], name: str, lab: int):
        cl = smooth_centerline(name, lab, tree.xyz[nodes], tree.radius[nodes])
        if cl is not None:
            centerlines.append(cl)

    def paint(tree: SkeletonTree, nodes, lab):
        nodes = np.asarray(nodes, int)
        if len(nodes):
            skel_lab[tuple(tree.ijk[nodes].T)] = lab

    def label_trunk(tree: SkeletonTree, start_path: list[int], lab: int, branch_lab: int, prefix: str, br_prefix: str,
                    lead: list[int] | None = None):
        """Étiquette le tronc (chemin principal) puis ses branches latérales significatives."""
        paint(tree, start_path, lab)
        mk_cl(tree, (lead or []) + start_path, prefix, lab)
        on_path = set(start_path)
        k = 0
        for n in start_path:
            for c in tree.children[n]:
                if c in on_path:
                    continue
                sub = tree.subtree(c)
                if tree.branch_len(c) >= min_branch_mm:
                    k += 1
                    paint(tree, sub, branch_lab)
                    mk_cl(tree, [n] + tree.main_path(c), f"{br_prefix} {k}", branch_lab)
                else:
                    paint(tree, sub, lab)  # petit éperon : rattaché au tronc

    def tree_for(info):
        comp_id, ost_vox, _ = info
        skel = skel_all & (comps == comp_id)
        if skel.sum() < 3:
            return None
        return build_tree(skel, to_xyz(ost_vox), aff, radius_map)

    ostia = {}
    # ---- arbre droit ----------------------------------------------------------------------------
    if right is not None:
        t = tree_for(right)
        if t is not None:
            ostia["droit"] = to_xyz(right[1]).round(2).tolist()
            label_trunk(t, t.main_path(t.root), CD, BR_CD, "CD", "Br. CD")
            paint(t, np.nonzero(~np.isfinite(t.dist))[0], UNK)

    # ---- arbre gauche ---------------------------------------------------------------------------
    if left is not None:
        t = tree_for(left)
        if t is not None:
            ostia["gauche"] = to_xyz(left[1]).round(2).tolist()
            total = t.sub_len[t.root]
            thr = max(min_branch_mm * 1.5, 0.12 * total)
            path = t.main_path(t.root)
            bif, big = None, []
            for i, n in enumerate(path):
                big = [c for c in t.children[n] if t.branch_len(c) >= thr]
                if len(big) >= 2:
                    bif = i
                    break
            if bif is None:
                warnings.append("Bifurcation du tronc commun non identifiée : tronc principal gauche étiqueté IVA.")
                label_trunk(t, path, IVA, DIAG, "IVA", "Br. IVA")
            else:
                lm = path[: bif + 1]
                b = lm[-1]
                lm_len = t.dist[b] - t.dist[t.root]
                if lm_len > 30:
                    warnings.append(f"Tronc commun long ({lm_len:.0f} mm) : vérifier la bifurcation IVA/Cx.")
                paint(t, lm, TC)
                mk_cl(t, lm, "TC", TC)
                on_lm = set(lm)
                for n in lm[:-1]:  # petites branches nées du TC
                    for c in t.children[n]:
                        if c not in on_lm:
                            paint(t, t.subtree(c), TC if t.branch_len(c) < min_branch_mm else UNK)
                big = sorted(big, key=lambda c: -t.sub_w[c])[:3]
                b_xyz = t.xyz[b]
                ant = {c: float(t.point_at([b] + t.main_path(c), 25.0)[1] - b_xyz[1]) for c in big}
                by_ant = sorted(big, key=lambda c: -ant[c])
                lad, lcx = by_ant[0], by_ant[-1]
                for c in t.children[b]:
                    if c in (lad, lcx):
                        continue
                    if len(big) == 3 and c == by_ant[1]:
                        paint(t, t.subtree(c), RAMUS)
                        mk_cl(t, [b] + t.main_path(c), "Bissectrice", RAMUS)
                    else:
                        paint(t, t.subtree(c), DIAG if t.branch_len(c) >= min_branch_mm else TC)
                label_trunk(t, t.main_path(lad), IVA, DIAG, "TC → IVA", "Br. IVA", lead=lm)
                label_trunk(t, t.main_path(lcx), CX, MARG, "TC → Cx", "Br. Cx", lead=lm)
                if abs(ant[lad] - ant[lcx]) < 5:
                    warnings.append("Distinction IVA / Cx peu franche (orientations proches) : à vérifier visuellement.")
            paint(t, np.nonzero(~np.isfinite(t.dist))[0], UNK)

    # composantes reliées à l'aorte au-delà des deux principales (ostium conal séparé, etc.)
    for info in extra:
        s = side_score(info[1])
        lab = BR_CD if np.isfinite(s) and s > 0 else UNK
        skel_lab[skel_all & (comps == info[0])] = lab
        warnings.append(
            f"Ostium supplémentaire ({'droit, probable artère du conus' if lab == BR_CD else 'gauche, IVA/Cx à ostia séparés ?'})."
        )

    # ---- propagation des labels du squelette à tous les voxels de chaque arbre -------------------------
    labelled_comps = [left[0] if left else None, right[0] if right else None] + [e[0] for e in extra]
    in_trees = np.isin(comps, [c for c in labelled_comps if c is not None])
    if skel_lab.any():
        _, idx = ndimage.distance_transform_edt(skel_lab == 0, sampling=spacing, return_indices=True)
        nearest = skel_lab[tuple(idx)]
        local = np.where(in_trees, nearest, 0).astype(np.uint8)
        local[in_trees & (local == 0)] = UNK
    else:
        local = np.where(in_trees, UNK, 0).astype(np.uint8)

    # ---- fragments déconnectés : rattachés au label voisin s'ils sont proches ---------------------------
    rest = cor & ~in_trees
    if rest.any():
        n_frag = 0
        if (local > 0).any():
            d_lab, idx = ndimage.distance_transform_edt(local == 0, sampling=spacing, return_indices=True)
            frag_lab, nf = ndimage.label(rest, structure=np.ones((3, 3, 3)))
            for f in range(1, nf + 1):
                fm = frag_lab == f
                vox = np.argwhere(fm)
                dd = d_lab[tuple(vox.T)]
                k = int(np.argmin(dd))
                if dd[k] <= reattach_mm:
                    src = idx[:, vox[k][0], vox[k][1], vox[k][2]]
                    local[fm] = local[tuple(src)]
                    n_frag += 1
                else:
                    local[fm] = UNK
        else:
            local[rest] = UNK
        if n_frag:
            warnings.append(f"{n_frag} fragment(s) déconnecté(s) rattaché(s) au vaisseau le plus proche (≤ {reattach_mm:.0f} mm).")

    out[box] = local
    order = {TC: 0, IVA: 1, CX: 2, RAMUS: 3, CD: 4, DIAG: 5, MARG: 6, BR_CD: 7}
    centerlines.sort(key=lambda c: (order.get(c.label, 9), c.name))
    return LabelingResult(out, centerlines, warnings, ostia)
