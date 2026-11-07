"""CoroSeg — interface Streamlit.   Lancer :  streamlit run app.py"""

from __future__ import annotations

import tempfile
from pathlib import Path

import numpy as np
import plotly.graph_objects as go
import streamlit as st
from PIL import Image

from coroseg import LABELS
from coroseg.cpr import cross_section, straightened_cpr, window
from coroseg.io import list_dicom_series, load_dicom_series, load_mask_like, load_volume, spacing_of
from coroseg.mesh import mask_to_mesh
from coroseg.pipeline import analyse, narrowing_profile, run
from coroseg.segment import LICENSE_URL, SegmentationError, totalsegmentator_available

st.set_page_config(page_title="CoroSeg", page_icon="🫀", layout="wide")

WL_PRESETS = {"Coronaires (800/250)": (250, 800), "Médiastin (400/40)": (40, 400), "Calcium (1500/450)": (450, 1500)}
LUT = np.zeros((256, 3), np.uint8)
for k, (_, _, rgb) in LABELS.items():
    LUT[k] = rgb
ss = st.session_state


# =============================================================================================
# Utilitaires d'affichage
# =============================================================================================
def to_display(arr2d: np.ndarray, sp_rows: float, sp_cols: float, target_px: float | None = None) -> Image.Image:
    """Image 2D (uint8 ou RGB) → PIL redimensionnée en pixels carrés."""
    im = Image.fromarray(arr2d)
    base = target_px or min(sp_rows, sp_cols)
    w = max(1, int(round(arr2d.shape[1] * sp_cols / base)))
    h = max(1, int(round(arr2d.shape[0] * sp_rows / base)))
    return im.resize((w, h), Image.BILINEAR if arr2d.ndim == 2 else Image.NEAREST)


def slice_view(hu, labels, aorta, sp, orient, idx, level, width, alpha, show_aorta):
    """Coupe en convention radiologique (droite du patient à gauche de l'écran, antérieur/crânial en haut)."""
    if orient == "Axiale":
        g, l, a = hu[:, :, idx], labels[:, :, idx], aorta[:, :, idx]
        tr = lambda x: np.fliplr(np.flipud(x.T))  # noqa: E731
        sr, sc = sp[1], sp[0]
    elif orient == "Coronale":
        g, l, a = hu[:, idx, :], labels[:, idx, :], aorta[:, idx, :]
        tr = lambda x: np.fliplr(np.flipud(x.T))  # noqa: E731
        sr, sc = sp[2], sp[0]
    else:  # sagittale : antérieur à gauche
        g, l, a = hu[idx, :, :], labels[idx, :, :], aorta[idx, :, :]
        tr = lambda x: np.fliplr(np.flipud(x.T))  # noqa: E731
        sr, sc = sp[2], sp[1]
    gray = window(tr(g), level, width)
    rgb = np.repeat(gray[..., None], 3, -1).astype(np.float32)
    lab = tr(l)
    m = lab > 0
    rgb[m] = (1 - alpha) * rgb[m] + alpha * LUT[lab[m]]
    if show_aorta:
        am = tr(a) & ~m
        rgb[am] = (1 - alpha * 0.5) * rgb[am] + alpha * 0.5 * np.array([255, 120, 200])
    return to_display(rgb.astype(np.uint8), sr, sc)


@st.cache_resource(show_spinner=False)
def _series(path: str):
    return list_dicom_series(path)


def build_meshes(res):
    meshes = {}
    for k in LABELS:
        m = res.labels == k
        if m.any():
            meshes[k] = mask_to_mesh(m, res.affine, smooth_sigma=0.6)
    if res.aorta.any() and res.coronary.any():
        sp = spacing_of(res.ct)
        idx = np.argwhere(res.coronary)
        lo = np.maximum(idx.min(0) - (30 / sp).astype(int), 0)
        hi = np.minimum(idx.max(0) + (30 / sp).astype(int), res.aorta.shape)
        crop = np.zeros_like(res.aorta)
        crop[lo[0]:hi[0], lo[1]:hi[1], lo[2]:hi[2]] = res.aorta[lo[0]:hi[0], lo[1]:hi[1], lo[2]:hi[2]]
        meshes["aorta"] = mask_to_mesh(crop, res.affine, smooth_sigma=1.0, step=2)
    return meshes


def set_result(res, source_name):
    ss.res = res
    ss.source_name = source_name
    ss.meshes = None
    ss.export_zip = None


# =============================================================================================
# Barre latérale : chargement et lancement
# =============================================================================================
with st.sidebar:
    st.title("🫀 CoroSeg")
    st.caption("Segmentation et étiquetage des coronaires sur coroscanner. **Outil de recherche — non validé pour le diagnostic.**")

    source = st.radio("Examen", ["Chemin local", "Téléverser un fichier", "Démo (fantôme synthétique)"], key="source")
    ct_loader = None
    src_name = ""
    if source == "Chemin local":
        p = st.text_input("Dossier DICOM, archive .zip ou fichier .nii.gz", placeholder="/chemin/vers/examen")
        if p and Path(p).expanduser().exists():
            path = Path(p).expanduser()
            src_name = path.name
            if path.is_dir():
                with st.spinner("Lecture des séries DICOM…"):
                    series = _series(str(path))
                if not series:
                    st.error("Aucune série DICOM exploitable dans ce dossier.")
                else:
                    i = st.selectbox("Série", range(len(series)), format_func=lambda k: series[k].label,
                                     help="Choisir la reconstruction de la phase la plus calme (souvent 70–75 % ou systolique 40 %), "
                                          "noyau mou, coupes fines.")
                    ct_loader = lambda: load_dicom_series(series[i])  # noqa: E731
            else:
                ct_loader = lambda: load_volume(path)  # noqa: E731
        elif p:
            st.error("Chemin introuvable.")
    elif source == "Téléverser un fichier":
        up = st.file_uploader("Fichier .nii.gz ou .zip de DICOM", type=["gz", "nii", "zip"])
        if up is not None:
            src_name = up.name
            def ct_loader(up=up):
                d = Path(tempfile.mkdtemp(prefix="coroseg_up_"))
                f = d / up.name
                f.write_bytes(up.getbuffer())
                return load_volume(f)

    use_mask = False
    mask_up = aorta_up = None
    if source != "Démo (fantôme synthétique)":
        with st.expander("Segmentation", expanded=True):
            use_mask = st.toggle("Utiliser un masque coronaire existant", help="Saute TotalSegmentator (masque NIfTI binaire aligné sur l'examen).")
            if use_mask:
                mask_up = st.file_uploader("Masque coronaires (.nii.gz)", type=["gz", "nii"], key="mask")
                aorta_up = st.file_uploader("Masque aorte (.nii.gz, recommandé)", type=["gz", "nii"], key="aorta")
            else:
                if not totalsegmentator_available():
                    st.error("TotalSegmentator n'est pas installé : `pip install TotalSegmentator`")
                lic = st.text_input("Licence TotalSegmentator", type="password",
                                    help=f"Nécessaire pour la tâche coronary_arteries (gratuite en académique) : {LICENSE_URL}. "
                                         "Une fois enregistrée elle est mémorisée, le champ peut rester vide ensuite.")
                device = st.selectbox("Calcul", ["gpu", "cpu", "mps"], help="gpu = NVIDIA/CUDA ; mps = Mac Apple Silicon")

    can_run = source == "Démo (fantôme synthétique)" or (ct_loader is not None and (not use_mask or mask_up is not None))
    if st.button("Lancer l'analyse", type="primary", width="stretch", disabled=not can_run):
        with st.status("Analyse en cours…", expanded=True) as status:
            log = st.write
            try:
                if source == "Démo (fantôme synthétique)":
                    from coroseg.phantom import make_phantom

                    log("Génération du fantôme…")
                    ct, cor, aorta, _ = make_phantom()
                    res = analyse(ct, cor, aorta, log=log)
                    src_name = "Fantôme synthétique"
                else:
                    log("Chargement de l'examen…")
                    ct = ct_loader()
                    sp = spacing_of(ct)
                    log(f"Volume {ct.shape} — voxels {sp[0]:.2f} × {sp[1]:.2f} × {sp[2]:.2f} mm")
                    if use_mask:
                        def _tmp(u):
                            f = Path(tempfile.mkdtemp()) / u.name
                            f.write_bytes(u.getbuffer())
                            return f
                        cor = load_mask_like(_tmp(mask_up), ct) > 0
                        aorta = load_mask_like(_tmp(aorta_up), ct) > 0 if aorta_up else None
                        res = analyse(ct, cor, aorta, log=log)
                    else:
                        work = Path(tempfile.mkdtemp(prefix="coroseg_ts_"))
                        res = run(ct, work, license_number=lic, device=device, log=log)
                set_result(res, src_name)
                status.update(label="Analyse terminée", state="complete", expanded=False)
            except SegmentationError as e:
                status.update(label="Échec de la segmentation", state="error")
                st.error(str(e))
            except Exception as e:
                status.update(label="Erreur", state="error")
                st.exception(e)

# =============================================================================================
# Zone principale
# =============================================================================================
if "res" not in ss:
    st.header("Segmentation des artères coronaires")
    st.markdown(
        """
1. **Charger un coroscanner** dans la barre latérale : dossier DICOM (toutes les séries sont listées), archive `.zip` ou NIfTI.
2. **Lancer l'analyse** : TotalSegmentator segmente l'arbre coronaire et l'aorte, puis CoroSeg
   squelettise l'arbre, repère les ostia et identifie **TC, IVA, Cx, CD** et leurs branches.
3. Explorer les **coupes**, la **vue 3D**, les **CPR** et le **profil de calibre**, puis exporter (NIfTI, STL, lignes centrales).

Pas d'examen sous la main ? Choisir **Démo (fantôme synthétique)** pour voir toute la chaîne fonctionner.
"""
    )
    st.stop()

res = ss.res
hu = res.hu
sp = spacing_of(res.ct)

st.subheader(ss.get("source_name") or "Examen")
present = [k for k in LABELS if (res.labels == k).any()]
c = st.columns(4)
c[0].metric("Vaisseaux identifiés", ", ".join(LABELS[k][0] for k in present if k in (1, 2, 3, 4, 8)) or "—")
c[1].metric("Volume de l'arbre", f"{res.coronary.sum() * np.prod(sp) / 1000:.2f} mL")
c[2].metric("Lignes centrales", len(res.centerlines))
c[3].metric("Avertissements", len(res.warnings))
for w in res.warnings:
    st.warning(w, icon="⚠️")

legend = " ".join(
    f"<span style='display:inline-block;padding:2px 8px;margin:2px;border-radius:10px;"
    f"background:rgb{LABELS[k][2]};color:#111;font-size:0.8rem'>{LABELS[k][0]} · {LABELS[k][1]}</span>"
    for k in present
)
st.markdown(legend, unsafe_allow_html=True)

tab2d, tab3d, tabcpr, tabexp = st.tabs(["Coupes", "3D", "CPR & calibre", "Résumé & export"])

# ---------------------------------------------------------------------------------------- Coupes
with tab2d:
    ctl, view = st.columns([1, 3])
    with ctl:
        orient = st.radio("Plan", ["Axiale", "Coronale", "Sagittale"], horizontal=True)
        axis = {"Axiale": 2, "Coronale": 1, "Sagittale": 0}[orient]
        n = hu.shape[axis]
        cidx = np.argwhere(res.coronary)
        default = int(np.median(cidx[:, axis])) if len(cidx) else n // 2
        idx = st.slider("Coupe", 0, n - 1, default, key=f"sl_{orient}")
        preset = st.selectbox("Fenêtre", list(WL_PRESETS))
        level, width = WL_PRESETS[preset]
        alpha = st.slider("Opacité des labels", 0.0, 1.0, 0.55)
        show_ao = st.checkbox("Afficher l'aorte", value=False)
        mip = st.slider("MIP (épaisseur, coupes)", 1, 41, 1, step=2, help="Projection d'intensité maximale autour de la coupe")
    with view:
        if mip > 1:
            h = mip // 2
            sl = [slice(None)] * 3
            sl[axis] = slice(max(idx - h, 0), idx + h + 1)
            g = hu[tuple(sl)].max(axis=axis, keepdims=True)
            lb = res.labels[tuple(sl)].max(axis=axis, keepdims=True)
            ao = res.aorta[tuple(sl)].max(axis=axis, keepdims=True)
            img = slice_view(g, lb, ao, sp, orient, 0, level, width, alpha, show_ao)
        else:
            img = slice_view(hu, res.labels, res.aorta, sp, orient, idx, level, width, alpha, show_ao)
        st.image(img, width="stretch", caption="Convention radiologique : droite du patient à gauche de l'image")

# ---------------------------------------------------------------------------------------- 3D
with tab3d:
    if ss.get("meshes") is None:
        with st.spinner("Calcul des surfaces…"):
            ss.meshes = build_meshes(res)
    o1, o2, o3 = st.columns(3)
    show_ao3 = o1.checkbox("Aorte", value=True)
    show_cl = o2.checkbox("Lignes centrales", value=False)
    show_lab = o3.checkbox("Couleurs par vaisseau", value=True)
    fig = go.Figure()
    for k, m in ss.meshes.items():
        if m is None or (k == "aorta" and not show_ao3):
            continue
        v, f = m
        if k == "aorta":
            col, name, op = "rgb(220,120,180)", "Aorte", 0.25
        else:
            col = f"rgb{LABELS[k][2]}" if show_lab else "rgb(230,80,70)"
            name, op = LABELS[k][1], 1.0
        fig.add_trace(go.Mesh3d(x=v[:, 0], y=v[:, 1], z=v[:, 2], i=f[:, 0], j=f[:, 1], k=f[:, 2], color=col, opacity=op,
                                name=name, showlegend=True, flatshading=False,
                                lighting=dict(ambient=0.45, diffuse=0.8, specular=0.3, roughness=0.5)))
    if show_cl:
        for cl in res.centerlines:
            fig.add_trace(go.Scatter3d(x=cl.points[:, 0], y=cl.points[:, 1], z=cl.points[:, 2], mode="lines",
                                       line=dict(color="white", width=4), name=cl.name, showlegend=False))
    fig.update_layout(
        height=680, margin=dict(l=0, r=0, t=0, b=0), paper_bgcolor="rgba(0,0,0,0)",
        scene=dict(aspectmode="data", xaxis_visible=False, yaxis_visible=False, zaxis_visible=False,
                   camera=dict(eye=dict(x=0.0, y=1.9, z=0.5), up=dict(x=0, y=0, z=1))),
        legend=dict(orientation="h", y=0.02),
    )
    st.plotly_chart(fig, width="stretch")
    st.caption("Vue antérieure par défaut (gauche du patient à droite de l'écran). Faire pivoter à la souris.")

# ---------------------------------------------------------------------------------------- CPR
with tabcpr:
    if not res.centerlines:
        st.info("Aucune ligne centrale disponible.")
    else:
        names = [c.name for c in res.centerlines]
        a, b, c3, d = st.columns([2, 1, 1, 1])
        cl = res.centerlines[names.index(a.selectbox("Vaisseau", names))]
        angle = b.slider("Rotation (°)", 0, 180, 0, step=5)
        hw = c3.slider("Demi-largeur (mm)", 6, 25, 12)
        preset2 = d.selectbox("Fenêtre ", list(WL_PRESETS))
        lv, wd = WL_PRESETS[preset2]
        pos = st.slider("Position le long du vaisseau (mm)", 0.0, float(round(cl.length, 1)), float(round(cl.length / 3, 1)), 0.5)

        px = 0.25
        cpr = window(straightened_cpr(hu, res.affine, cl, angle, hw, px), lv, wd)
        rgb = np.repeat(cpr[..., None], 3, -1)
        r = int(round(pos / px))
        if 0 <= r < len(rgb):
            rgb[max(r - 1, 0): r + 1, :] = [255, 210, 0]
        xs = window(cross_section(hu, res.affine, cl, pos, half_width_mm=8, pixel_mm=0.1), lv, wd)

        k1, k2 = st.columns([1, 2])
        with k1:
            st.image(to_display(rgb, 1, 1).resize((rgb.shape[1] * 3, rgb.shape[0] * 3), Image.BILINEAR),
                     caption=f"CPR redressée — {cl.name} (origine en haut)", width="stretch")
        with k2:
            st.image(Image.fromarray(xs).resize((360, 360), Image.BILINEAR), caption=f"Coupe orthogonale à {pos:.1f} mm (16 × 16 mm)")
            pct = narrowing_profile(cl)
            pf = go.Figure()
            pf.add_trace(go.Scatter(x=cl.s, y=2 * cl.radius, name="Ø du masque (mm)", line=dict(color="#4ea1ff")))
            pf.add_trace(go.Scatter(x=cl.s, y=pct, name="Rétrécissement relatif (%)", yaxis="y2", line=dict(color="#ff7a59", dash="dot")))
            pf.add_vline(x=pos, line_color="#ffd200")
            pf.update_layout(height=280, margin=dict(l=10, r=10, t=10, b=10), xaxis_title="Distance à l'origine (mm)",
                             yaxis=dict(title="Ø (mm)"), yaxis2=dict(title="%", overlaying="y", side="right", range=[0, 100]),
                             legend=dict(orientation="h", y=1.15))
            st.plotly_chart(pf, width="stretch")
            i = int(np.clip(np.searchsorted(cl.s, pos), 0, len(cl.s) - 1))
            st.caption(f"À {pos:.1f} mm : Ø ≈ {2 * cl.radius[i]:.2f} mm"
                       + (f", rétrécissement relatif ≈ {pct[i]:.0f} %" if np.isfinite(pct[i]) else "")
                       + ". Calibre du masque segmenté, indicatif : peu fiable sous ~1,5 mm et en présence de calcifications.")

# ---------------------------------------------------------------------------------------- Export
with tabexp:
    st.dataframe(res.summary_rows(), width="stretch", hide_index=True)
    st.caption("« Rétrécissement max » : réduction du diamètre du masque par rapport à une référence interpolée amont/aval. "
               "Repère de lecture, pas une quantification de sténose.")
    v1, v2 = st.columns(2)
    v1.write("**Volumes (mL)**")
    v1.json(res.volumes_ml())
    v2.write("**Ostia (mm, repère RAS)**")
    v2.json(res.ostia_mm or {})
    if st.button("Préparer l'export"):
        with st.spinner("Écriture des fichiers…"):
            out = Path(tempfile.mkdtemp(prefix="coroseg_out_")) / "coroseg_resultats"
            res.save(out)
            ss.export_zip = res.zip(out).read_bytes()
    if ss.get("export_zip"):
        st.download_button("Télécharger (labels NIfTI, masque, STL, lignes centrales, rapport)", ss.export_zip,
                           file_name="coroseg_resultats.zip", mime="application/zip", type="primary")
