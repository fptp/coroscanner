"""Worker : surveille Orthanc, segmente les coroscanners et renvoie les résultats dans l'étude."""

from __future__ import annotations

import json
import os
import re
import shutil
import time
import traceback
from pathlib import Path

from .config import Config
from .orthanc import Orthanc


def log(*a):
    print(time.strftime("%Y-%m-%d %H:%M:%S"), *a, flush=True)


class State:
    """Petit état persistant : dernier n° de changement lu et études déjà traitées."""

    def __init__(self, path: Path):
        self.path = path
        self.data = {"last_change": None, "done": {}}
        if path.exists():
            try:
                self.data.update(json.loads(path.read_text()))
            except Exception:
                pass

    def save(self):
        self.path.parent.mkdir(parents=True, exist_ok=True)
        tmp = self.path.with_suffix(".tmp")
        tmp.write_text(json.dumps(self.data, indent=1))
        tmp.replace(self.path)


# ----------------------------------------------------------------------------------------------------
def _f(v, default=None):
    try:
        return float(str(v).split("\\")[0])
    except Exception:
        return default


def eligible_series(orthanc: Orthanc, study_id: str, cfg: Config) -> list[dict]:
    """Séries CT candidates de l'étude, la meilleure en premier (coupes les plus fines, puis les plus nombreuses)."""
    inc = re.compile(cfg.include_regex, re.I) if cfg.include_regex else None
    exc = re.compile(cfg.exclude_regex, re.I) if cfg.exclude_regex else None
    out = []
    for s in orthanc.series_of_study(study_id):
        tags = s.get("MainDicomTags", {})
        desc = tags.get("SeriesDescription", "") or ""
        n = len(s.get("Instances", []))
        if tags.get("Modality") != "CT" or n < cfg.min_slices:
            continue
        if exc and exc.search(desc):
            continue
        if inc and not inc.search(desc):
            continue
        first = orthanc.first_instance_tags(s)
        if (first.get("Manufacturer") or "") == "CoroSeg":
            continue
        itype = (first.get("ImageType") or "").upper()
        if "LOCALIZER" in itype:
            continue
        thick = _f(first.get("SliceThickness"), 9.9)
        if thick > cfg.max_slice_thickness:
            continue
        out.append({"id": s["ID"], "description": desc, "n": n, "thickness": thick,
                    "kernel": first.get("ConvolutionKernel", "")})
    out.sort(key=lambda d: (d["thickness"], -d["n"]))
    return out


# ----------------------------------------------------------------------------------------------------
def process_study(orthanc: Orthanc, study_id: str, cfg: Config, series_id: str | None = None,
                  segmenter=None) -> dict:
    """Traite une étude (ou une série précise) et renvoie les IDs Orthanc des instances créées.

    `segmenter(ct_nifti, work_dir)` -> (masque coronaires, masque aorte) permet de remplacer TotalSegmentator (tests).
    """
    from coroseg.pipeline import analyse, run

    from . import dicom_export as dx
    from .device import resolve_device

    t0 = time.time()
    study = orthanc.study(study_id)
    st = study.get("MainDicomTags", {})
    if series_id is None:
        cands = eligible_series(orthanc, study_id, cfg)
        if not cands:
            raise RuntimeError("aucune série CT éligible (coupes fines, assez nombreuses, non exclue par le filtre)")
        best = cands[0]
        series_id = best["id"]
        log(f"  série retenue : « {best['description']} » ({best['n']} coupes, {best['thickness']} mm, {best['kernel']})"
            + (f" parmi {len(cands)} candidates" if len(cands) > 1 else ""))
    work = Path(cfg.work_dir) / study_id / series_id
    if work.exists():
        shutil.rmtree(work)
    dcm_dir = orthanc.download_series(series_id, work / "dicom")
    src = dx.load_source(dcm_dir, log=lambda m: log("  " + m))
    device = resolve_device(cfg.device, log=lambda m: log("  " + m))
    log(f"  volume {src.hu.shape[::-1]} chargé, segmentation ({device})…")

    if segmenter is None:
        res = run(src.ct, work / "_totalseg", license_number=cfg.license_number, device=device,
                  log=lambda m: log("  " + m))
    else:
        cor, aorta = segmenter(src.ct, work)
        res = analyse(src.ct, cor, aorta, log=lambda m: log("  " + m))
    res.save(work / "resultats")
    for w in res.warnings:
        log("  ⚠", w)

    lab_src = dx.labels_on_source_grid(res.labels, res.affine, src)
    objs = []
    if cfg.send_seg:
        objs.append(dx.make_seg(src, lab_src))
    if cfg.fusion != "off":
        objs += dx.make_fusion_series(src, lab_src, cfg.fusion, cfg.fusion_alpha, cfg.window_level, cfg.window_width,
                                      cfg.jpeg, institution=cfg.institution)
    if cfg.send_cpr:
        objs += dx.make_cpr_series(src, res, wl=cfg.window_level, ww=cfg.window_width, jpeg=cfg.jpeg,
                                   institution=cfg.institution)
    if cfg.send_report:
        objs.append(dx.make_report_image(src, res, jpeg=cfg.jpeg, institution=cfg.institution))

    ids, series_out = [], set()
    for ds in objs:
        r = orthanc.upload(dx.to_bytes(ds))
        ids.append(r["ID"])
        series_out.add(r.get("ParentSeries"))
    log(f"  {len(ids)} instance(s) envoyée(s) dans Orthanc ({len(series_out)} série(s))")
    if cfg.forward_modality:
        orthanc.store_to_modality(cfg.forward_modality, sorted(s for s in series_out if s))
        log(f"  résultats transférés vers la modalité {cfg.forward_modality}")
    if not cfg.keep_work:
        shutil.rmtree(work / "dicom", ignore_errors=True)
        shutil.rmtree(work / "_totalseg", ignore_errors=True)
    return {"series_source": series_id, "instances": ids, "series": sorted(s for s in series_out if s),
            "warnings": res.warnings, "seconds": round(time.time() - t0, 1)}


# ----------------------------------------------------------------------------------------------------
class Worker:
    def __init__(self, cfg: Config, segmenter=None):
        self.cfg = cfg
        self.orthanc = Orthanc(cfg.orthanc_url, cfg.orthanc_user, cfg.orthanc_password)
        self.state = State(Path(cfg.work_dir) / "etat_worker.json")
        if segmenter is None and os.environ.get("COROSEG_SEGMENTER") == "fantome":  # tests uniquement
            from .demo import phantom_segmenter

            segmenter = phantom_segmenter
        self.segmenter = segmenter

    def handle(self, study_id: str, force: bool = False) -> None:
        o, cfg = self.orthanc, self.cfg
        labels = o.labels(study_id)
        if not force and (study_id in self.state.data["done"] or cfg.label_done in labels or cfg.label_error in labels):
            return
        try:
            info = o.study(study_id)
        except Exception:
            return  # étude supprimée entre-temps
        tags = info.get("MainDicomTags", {})
        pt = info.get("PatientMainDicomTags", {})
        log(f"Étude {study_id} — {pt.get('PatientName', '?')} — {tags.get('StudyDate', '')} "
            f"« {tags.get('StudyDescription', '')} »")
        desc = tags.get("StudyDescription", "") or ""
        if not force and cfg.study_regex and not re.search(cfg.study_regex, desc, re.I):
            log(f"  description hors filtre COROSEG_STUDY_REGEX ({cfg.study_regex}) : ignorée")
            self.state.data["done"][study_id] = {"statut": "ignorée"}
            self.state.save()
            return
        if not force and self.cfg.trigger == "auto" and not eligible_series(o, study_id, cfg):
            log("  pas de série CT éligible : ignorée")
            self.state.data["done"][study_id] = {"statut": "ignorée"}
            self.state.save()
            return
        try:
            r = process_study(o, study_id, cfg, segmenter=self.segmenter)
            o.add_label(study_id, cfg.label_done)
            o.remove_label(study_id, cfg.label_error)
            self.state.data["done"][study_id] = {"statut": "ok", **{k: r[k] for k in ("series_source", "series", "seconds")}}
            log(f"  terminé en {r['seconds']} s")
        except Exception as e:
            log(f"  ERREUR : {e}")
            traceback.print_exc()
            o.add_label(study_id, cfg.label_error)
            self.state.data["done"][study_id] = {"statut": "erreur", "erreur": str(e)[:500]}
        finally:
            o.remove_label(study_id, cfg.label_todo)
            self.state.save()

    def loop(self, from_start: bool = False) -> None:
        o, cfg = self.orthanc, self.cfg
        while True:
            try:
                ver = o.system().get("Version")
                break
            except Exception as e:
                log(f"Orthanc injoignable ({cfg.orthanc_url}) : {e} — nouvel essai dans 10 s")
                time.sleep(10)
        from .device import describe, resolve_device, set_threads

        set_threads(cfg.threads)
        log(f"Connecté à Orthanc {ver} ({cfg.orthanc_url}), mode « {cfg.trigger} »")
        log(f"Calcul : {describe(resolve_device(cfg.device, log=log))}")
        if self.state.data["last_change"] is None:
            self.state.data["last_change"] = 0 if from_start else o.last_change()
            self.state.save()

        while True:
            try:
                if cfg.trigger == "label":
                    for sid in o.find_studies_with_label(cfg.label_todo):
                        self.handle(sid, force=True)
                else:
                    todo = []
                    while True:
                        ch = o.changes(self.state.data["last_change"], 200)
                        for c in ch.get("Changes", []):
                            if c.get("ChangeType") == "StableStudy" and c["ID"] not in todo:
                                todo.append(c["ID"])
                        self.state.data["last_change"] = ch["Last"]
                        if ch.get("Done", True):
                            break
                    self.state.save()
                    for sid in todo:
                        self.handle(sid)
                    # le label « à traiter » marche aussi en mode auto (relance manuelle depuis Orthanc Explorer 2)
                    for sid in o.find_studies_with_label(cfg.label_todo):
                        self.handle(sid, force=True)
            except KeyboardInterrupt:
                raise
            except Exception as e:
                log(f"Erreur de boucle : {e}")
            time.sleep(cfg.poll_seconds)
