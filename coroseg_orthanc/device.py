"""Choix du calcul : GPU si disponible, sinon CPU (même logique que SegBrainMRI).

COROSEG_DEVICE = auto (défaut) | cpu | gpu | gpu:N
  auto : GPU si PyTorch voit une carte CUDA, sinon CPU.
  gpu demandé mais indisponible : bascule sur le CPU avec un avertissement.
Le jour où un GPU est installé : pilote NVIDIA + `deploy/install.sh --gpu`, sans toucher au code ni à la config.
"""

from __future__ import annotations


def resolve_device(spec: str | None, log=print) -> str:
    """Renvoie la valeur `device` attendue par TotalSegmentator : 'cpu', 'gpu' ou 'gpu:N'."""
    spec = (spec or "auto").strip().lower().replace("cuda", "gpu")
    if spec not in ("auto", "cpu") and not spec.startswith("gpu"):
        raise ValueError(f"COROSEG_DEVICE inconnu : {spec!r} (attendu : auto, cpu, gpu, gpu:N)")
    if spec == "cpu":
        return "cpu"
    try:
        import torch

        n = torch.cuda.device_count() if torch.cuda.is_available() else 0
    except Exception:
        n = 0
    if n == 0:
        if spec != "auto":
            log(f"⚠ {spec} demandé mais aucun GPU CUDA disponible : calcul sur CPU.")
        return "cpu"
    if spec.startswith("gpu:") and int(spec[4:]) >= n:
        raise ValueError(f"{spec} demandé mais seulement {n} GPU détecté(s).")
    return "gpu" if spec == "auto" else spec


def set_threads(threads: int | None) -> None:
    if threads:
        try:
            import torch
        except ImportError:
            return
        torch.set_num_threads(threads)


def describe(device: str) -> str:
    try:
        import torch

        if device.startswith("gpu"):
            i = int(device[4:]) if ":" in device else 0
            p = torch.cuda.get_device_properties(i)
            return f"GPU {p.name} ({p.total_memory / 1024**3:.1f} Go, CUDA {torch.version.cuda})"
        import os

        return (f"CPU ({os.cpu_count()} cœurs, {torch.get_num_threads()} threads PyTorch) — "
                "compter 5 à 15 min par examen")
    except Exception:
        return device
