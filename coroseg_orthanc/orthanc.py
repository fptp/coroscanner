"""Client REST minimal pour Orthanc."""

from __future__ import annotations

import io
import zipfile
from pathlib import Path

import requests


class Orthanc:
    def __init__(self, url: str, user: str | None = None, password: str | None = None, timeout: float = 300):
        self.url = url.rstrip("/")
        self.s = requests.Session()
        if user:
            self.s.auth = (user, password or "")
        self.timeout = timeout

    # -- bas niveau ------------------------------------------------------------------------------
    def _req(self, method: str, path: str, **kw):
        r = self.s.request(method, f"{self.url}{path}", timeout=self.timeout, **kw)
        r.raise_for_status()
        return r

    def get(self, path: str, **params):
        return self._req("GET", path, params=params or None).json()

    def post(self, path: str, json=None, data=None, headers=None):
        r = self._req("POST", path, json=json, data=data, headers=headers)
        return r.json() if r.content else None

    # -- haut niveau -----------------------------------------------------------------------------
    def system(self) -> dict:
        return self.get("/system")

    def changes(self, since: int | None = None, limit: int = 100) -> dict:
        params = {"limit": limit}
        if since is not None:
            params["since"] = since
        return self.get("/changes", **params)

    def last_change(self) -> int:
        return int(self.get("/changes", last="")["Last"])

    def study(self, study_id: str) -> dict:
        return self.get(f"/studies/{study_id}")

    def series(self, series_id: str) -> dict:
        return self.get(f"/series/{series_id}")

    def series_of_study(self, study_id: str) -> list[dict]:
        return self.get(f"/studies/{study_id}/series")

    def first_instance_tags(self, series: dict) -> dict:
        inst = series["Instances"][0]
        return self.get(f"/instances/{inst}/simplified-tags")

    def download_series(self, series_id: str, dest: Path) -> Path:
        """Télécharge une série en DICOM (fichiers .dcm à plat dans dest)."""
        dest.mkdir(parents=True, exist_ok=True)
        r = self._req("GET", f"/series/{series_id}/archive", stream=True)
        buf = io.BytesIO(r.content)
        with zipfile.ZipFile(buf) as z:
            for i, name in enumerate(n for n in z.namelist() if not n.endswith("/")):
                (dest / f"{i:05d}.dcm").write_bytes(z.read(name))
        return dest

    def upload(self, dicom_bytes: bytes) -> dict:
        return self.post("/instances", data=dicom_bytes, headers={"Content-Type": "application/dicom"})

    def find_studies_with_label(self, label: str) -> list[str]:
        try:
            return self.post("/tools/find", json={"Level": "Study", "Query": {}, "Labels": [label],
                                                  "LabelsConstraint": "All"}) or []
        except requests.HTTPError:
            return []  # Orthanc < 1.12 : pas de labels

    def labels(self, study_id: str) -> list[str]:
        try:
            return self.get(f"/studies/{study_id}/labels")
        except requests.HTTPError:
            return []

    def add_label(self, study_id: str, label: str) -> None:
        try:
            self._req("PUT", f"/studies/{study_id}/labels/{label}", data=b"")
        except requests.HTTPError:
            pass  # Orthanc < 1.12 ou base sans support des labels

    def remove_label(self, study_id: str, label: str) -> None:
        try:
            self._req("DELETE", f"/studies/{study_id}/labels/{label}")
        except requests.HTTPError:
            pass

    def store_to_modality(self, modality: str, resources: list[str]) -> None:
        self.post(f"/modalities/{modality}/store", json={"Resources": resources, "Synchronous": True})
