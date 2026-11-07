"""Configuration du pont Orthanc ↔ CoroSeg (variables d'environnement)."""

from __future__ import annotations

import os
from dataclasses import dataclass, field


def load_env_file(path: str | None = None) -> None:
    """Charge /etc/coroseg/coroseg.env (ou COROSEG_ENV_FILE) sans écraser les variables déjà définies."""
    path = path or os.environ.get("COROSEG_ENV_FILE", "/etc/coroseg/coroseg.env")
    try:
        lines = open(path, encoding="utf-8").read().splitlines()
    except OSError:
        return
    for line in lines:
        line = line.strip()
        if line and not line.startswith("#") and "=" in line:
            k, v = line.split("=", 1)
            v = v.strip()
            if len(v) >= 2 and v[0] == v[-1] and v[0] in "'\"":  # guillemets tolérés (comme systemd)
                v = v[1:-1]
            os.environ.setdefault(k.strip(), v)


load_env_file()


def _env(name: str, default: str | None = None) -> str | None:
    v = os.environ.get(name)
    return v if v not in (None, "") else default


def _bool(name: str, default: bool) -> bool:
    v = _env(name)
    return default if v is None else v.strip().lower() in ("1", "true", "yes", "oui", "on")


@dataclass
class Config:
    # --- Orthanc --------------------------------------------------------------------------------
    orthanc_url: str = field(default_factory=lambda: _env("ORTHANC_URL", "http://localhost:8042").rstrip("/"))
    orthanc_user: str | None = field(default_factory=lambda: _env("ORTHANC_USER"))
    orthanc_password: str | None = field(default_factory=lambda: _env("ORTHANC_PASSWORD"))

    # --- déclenchement ----------------------------------------------------------------------------
    # auto  : toute étude stable contenant une série CT éligible est traitée
    # label : seules les études portant le label COROSEG_LABEL_TODO (ex. posé dans Orthanc Explorer 2)
    # dicom : pas de surveillance d'Orthanc ; on pousse l'examen à l'AET COROSEG (envoi manuel)
    trigger: str = field(default_factory=lambda: _env("COROSEG_TRIGGER", "auto"))
    label_todo: str = field(default_factory=lambda: _env("COROSEG_LABEL_TODO", "coroseg"))
    label_done: str = field(default_factory=lambda: _env("COROSEG_LABEL_DONE", "coroseg-ok"))
    label_error: str = field(default_factory=lambda: _env("COROSEG_LABEL_ERROR", "coroseg-erreur"))
    poll_seconds: float = field(default_factory=lambda: float(_env("COROSEG_POLL_S", "10")))
    # dicom : CoroSeg ne surveille pas Orthanc, il reçoit les examens poussés à la main (C-STORE)
    dicom_aet: str = field(default_factory=lambda: _env("COROSEG_DICOM_AET", "COROSEG"))
    dicom_port: int = field(default_factory=lambda: int(_env("COROSEG_DICOM_PORT", "11113")))
    dicom_host: str = field(default_factory=lambda: _env("COROSEG_DICOM_HOST", "0.0.0.0"))
    dicom_allowed_aets: str = field(default_factory=lambda: _env("COROSEG_DICOM_ALLOWED_AETS", ""))
    dicom_settle_s: float = field(default_factory=lambda: float(_env("COROSEG_DICOM_SETTLE_S", "20")))

    # --- choix de la série ---------------------------------------------------------------------------
    min_slices: int = field(default_factory=lambda: int(_env("COROSEG_MIN_SLICES", "80")))
    max_slice_thickness: float = field(default_factory=lambda: float(_env("COROSEG_MAX_THICKNESS_MM", "1.5")))
    include_regex: str = field(default_factory=lambda: _env("COROSEG_SERIES_INCLUDE", ""))
    exclude_regex: str = field(default_factory=lambda: _env(
        "COROSEG_SERIES_EXCLUDE",
        r"score|calc|ca\s*sc|topo|scout|surview|bolus|test|smart|mip|mpr|cpr|vr\b|3d|dose|lung|poumon|coroseg"))
    # Description/étude CT : on ne traite que si elle ressemble à un coroscanner (vide = pas de filtre)
    study_regex: str = field(default_factory=lambda: _env("COROSEG_STUDY_REGEX", ""))

    # --- segmentation ---------------------------------------------------------------------------------
    license_number: str | None = field(default_factory=lambda: _env("TOTALSEG_LICENSE"))
    device: str = field(default_factory=lambda: _env("COROSEG_DEVICE", "auto"))  # auto | cpu | gpu | gpu:N
    threads: int | None = field(default_factory=lambda: int(_env("COROSEG_THREADS", "0")) or None)
    work_dir: str = field(default_factory=lambda: _env("COROSEG_WORK_DIR", "/data/coroseg"))
    keep_work: bool = field(default_factory=lambda: _bool("COROSEG_KEEP_WORK", True))

    # --- sorties renvoyées dans Orthanc -----------------------------------------------------------------
    send_seg: bool = field(default_factory=lambda: _bool("COROSEG_SEND_SEG", True))
    # fusion : all = toutes les coupes, roi = coupes contenant des coronaires (+ marge), off
    fusion: str = field(default_factory=lambda: _env("COROSEG_FUSION", "roi"))
    fusion_alpha: float = field(default_factory=lambda: float(_env("COROSEG_FUSION_ALPHA", "0.55")))
    send_cpr: bool = field(default_factory=lambda: _bool("COROSEG_SEND_CPR", True))
    send_report: bool = field(default_factory=lambda: _bool("COROSEG_SEND_REPORT", True))
    jpeg: bool = field(default_factory=lambda: _bool("COROSEG_JPEG", True))
    window_level: float = field(default_factory=lambda: float(_env("COROSEG_WL", "300")))
    window_width: float = field(default_factory=lambda: float(_env("COROSEG_WW", "900")))
    # renvoi optionnel des résultats vers une modalité DICOM déclarée dans Orthanc (ex. PACS)
    forward_modality: str | None = field(default_factory=lambda: _env("COROSEG_FORWARD_MODALITY"))

    institution: str = field(default_factory=lambda: _env("COROSEG_INSTITUTION", ""))
