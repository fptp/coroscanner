"""Réception DICOM (mode COROSEG_TRIGGER=dicom) : on pousse le coroscanner à la main vers CoroSeg.

Orthanc (ou le PACS, ou le scanner) envoie l'étude ou la série à l'AET COROSEG (port 11113 par défaut).
Les images sont rangées dans <work>/reception/<StudyInstanceUID>/<SeriesInstanceUID>/. Quand plus rien n'arrive
pour une étude depuis COROSEG_DICOM_SETTLE_S secondes, CoroSeg choisit la meilleure série CT (même règles que le
mode automatique), la segmente et envoie les résultats dans Orthanc par son API REST.
"""

from __future__ import annotations

import re
import shutil
import threading
import time
import traceback
from pathlib import Path

import pydicom

from .config import Config
from .orthanc import Orthanc
from .worker import log, process_dicom_folder

_SAFE = re.compile(r"[^0-9A-Za-z.\-]")


def _safe(uid: str) -> str:
    return _SAFE.sub("_", str(uid or "inconnu"))[:64]


class Receiver:
    """Serveur DICOM (C-STORE + C-ECHO) qui écrit chaque image reçue dans le dossier de réception."""

    def __init__(self, cfg: Config, incoming: Path):
        self.cfg, self.incoming = cfg, incoming
        self.ae = None
        self.server = None

    def start(self) -> None:
        from pynetdicom import AE, AllStoragePresentationContexts, evt
        from pynetdicom.sop_class import Verification

        ae = AE(ae_title=self.cfg.dicom_aet)
        ae.supported_contexts = AllStoragePresentationContexts
        ae.add_supported_context(Verification)
        ae.maximum_pdu_size = 0
        allowed = {a.strip() for a in self.cfg.dicom_allowed_aets.split(",") if a.strip()}
        if allowed:
            ae.require_calling_aet = sorted(allowed)
        self.ae = ae
        self.server = ae.start_server((self.cfg.dicom_host, self.cfg.dicom_port), block=False,
                                      evt_handlers=[(evt.EVT_C_STORE, self._on_store)])
        log(f"Réception DICOM : AET {self.cfg.dicom_aet}, port {self.cfg.dicom_port}"
            + (f", appelants autorisés : {', '.join(sorted(allowed))}" if allowed else ""))

    def _on_store(self, event):
        try:
            ds = event.dataset
            ds.file_meta = event.file_meta
            d = self.incoming / _safe(ds.get("StudyInstanceUID")) / _safe(ds.get("SeriesInstanceUID"))
            d.mkdir(parents=True, exist_ok=True)
            tmp = d / f"{_safe(ds.SOPInstanceUID)}.part"
            ds.save_as(tmp, enforce_file_format=True)
            tmp.replace(tmp.with_suffix(".dcm"))
            return 0x0000
        except Exception as e:
            log(f"Réception : image refusée ({e})")
            return 0xA700  # out of resources

    def stop(self) -> None:
        if self.server is not None:
            self.server.shutdown()


# ----------------------------------------------------------------------------------------------------
def _f(v, default):
    try:
        return float(v)
    except Exception:
        return default


def best_local_series(study_dir: Path, cfg: Config) -> tuple[Path | None, list[str]]:
    """Choisit la série CT à segmenter parmi celles reçues (coupes les plus fines, puis les plus nombreuses)."""
    inc = re.compile(cfg.include_regex, re.I) if cfg.include_regex else None
    exc = re.compile(cfg.exclude_regex, re.I) if cfg.exclude_regex else None
    cands, notes = [], []
    for sd in sorted(p for p in study_dir.iterdir() if p.is_dir()):
        files = list(sd.glob("*.dcm"))
        if not files:
            continue
        h = pydicom.dcmread(files[0], stop_before_pixels=True)
        desc = str(h.get("SeriesDescription", "") or "")
        why = None
        if h.get("Modality") != "CT":
            why = f"modalité {h.get('Modality')}"
        elif str(h.get("Manufacturer", "")) == "CoroSeg":
            why = "résultat CoroSeg"
        elif len(files) < cfg.min_slices:
            why = f"{len(files)} coupes < {cfg.min_slices}"
        elif exc and exc.search(desc):
            why = "exclue par COROSEG_SERIES_EXCLUDE"
        elif inc and not inc.search(desc):
            why = "non retenue par COROSEG_SERIES_INCLUDE"
        elif "LOCALIZER" in "\\".join(h.get("ImageType", [])).upper():
            why = "topogramme"
        else:
            thick = _f(h.get("SliceThickness"), 9.9)
            if thick > cfg.max_slice_thickness:
                why = f"coupes de {thick} mm > {cfg.max_slice_thickness}"
            else:
                cands.append((thick, -len(files), sd, desc, h.get("ConvolutionKernel", "")))
        if why:
            notes.append(f"« {desc} » ignorée : {why}")
    if not cands:
        return None, notes
    thick, n, sd, desc, kernel = sorted(cands, key=lambda c: (c[0], c[1]))[0]
    notes.append(f"série retenue : « {desc} » ({-n} coupes, {thick} mm, {kernel})"
                 + (f" parmi {len(cands)} candidates" if len(cands) > 1 else ""))
    return sd, notes


class Spool:
    """Traite les études reçues, une à la fois, quand plus aucune image n'arrive."""

    def __init__(self, cfg: Config, segmenter=None):
        self.cfg, self.segmenter = cfg, segmenter
        root = Path(cfg.work_dir)
        self.incoming, self.failed = root / "reception", root / "echecs"
        for d in (self.incoming, self.failed):
            d.mkdir(parents=True, exist_ok=True)
        self.orthanc = Orthanc(cfg.orthanc_url, cfg.orthanc_user, cfg.orthanc_password)
        self._stop = threading.Event()

    def ready(self) -> list[Path]:
        now, out = time.time(), []
        for sd in sorted(p for p in self.incoming.iterdir() if p.is_dir()):
            files = list(sd.rglob("*"))
            if not files or any(f.suffix == ".part" for f in files):
                continue
            if now - max(f.stat().st_mtime for f in files) >= self.cfg.dicom_settle_s:
                out.append(sd)
        return out

    def handle(self, study_dir: Path) -> None:
        cfg = self.cfg
        work = Path(cfg.work_dir) / "etudes" / study_dir.name
        if work.exists():
            shutil.rmtree(work)
        work.parent.mkdir(parents=True, exist_ok=True)
        study_dir.replace(work)
        first = next(work.rglob("*.dcm"), None)
        h = pydicom.dcmread(first, stop_before_pixels=True) if first else {}
        log(f"Étude reçue — {h.get('PatientName', '?')} — {h.get('StudyDate', '')} "
            f"« {h.get('StudyDescription', '')} »")
        try:
            sd, notes = best_local_series(work, cfg)
            for n in notes:
                log("  " + n)
            if sd is None:
                raise RuntimeError("aucune série CT éligible dans ce qui a été reçu")
            r = process_dicom_folder(self.orthanc, sd, work / f"_{sd.name}", cfg, self.segmenter)
            if r.get("study"):
                self.orthanc.add_label(r["study"], cfg.label_done)
                self.orthanc.remove_label(r["study"], cfg.label_error)
            log(f"  terminé en {r['seconds']} s")
            if not cfg.keep_work:
                shutil.rmtree(work, ignore_errors=True)
        except Exception as e:
            log(f"  ERREUR : {e}")
            traceback.print_exc()
            target = self.failed / work.name
            shutil.rmtree(target, ignore_errors=True)
            work.replace(target)
            (target / "erreur.txt").write_text(traceback.format_exc())

    def run_forever(self) -> None:
        log(f"File d'attente : {self.incoming} (traitement {self.cfg.dicom_settle_s:g} s après la dernière image)")
        while not self._stop.is_set():
            for sd in self.ready():
                if self._stop.is_set():
                    break
                self.handle(sd)
            self._stop.wait(2)

    def stop(self) -> None:
        self._stop.set()


def serve(cfg: Config, segmenter=None) -> None:
    import os
    import signal

    from .device import describe, resolve_device

    if segmenter is None and os.environ.get("COROSEG_SEGMENTER") == "fantome":  # tests uniquement
        from .demo import phantom_segmenter

        segmenter = phantom_segmenter
    spool = Spool(cfg, segmenter)
    rec = Receiver(cfg, spool.incoming)
    try:
        log(f"Orthanc {spool.orthanc.system()['Version']} joignable pour le retour des résultats ({cfg.orthanc_url})")
    except Exception as e:
        log(f"⚠ Orthanc injoignable pour l'instant ({cfg.orthanc_url}) : {e}")
    log(f"Calcul : {describe(resolve_device(cfg.device, log=log))}")

    def _shutdown(signum, _frame):
        log("Arrêt demandé : fin de l'étude en cours puis arrêt.")
        spool.stop()

    signal.signal(signal.SIGTERM, _shutdown)
    signal.signal(signal.SIGINT, _shutdown)
    rec.start()
    try:
        spool.run_forever()
    finally:
        rec.stop()
        log("Service arrêté.")
