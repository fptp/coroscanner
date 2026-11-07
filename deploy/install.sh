#!/usr/bin/env bash
# Installation du worker CoroSeg <-> Orthanc en service, sans Docker (Debian / Ubuntu, en root, depuis le dépôt).
#   ./deploy/install.sh          # PyTorch CPU (par défaut, VM sans GPU)
#   ./deploy/install.sh --gpu    # PyTorch CUDA (pilote NVIDIA déjà installé : nvidia-smi doit marcher)
# Relancer le même script pour mettre à jour après un « git pull ».
set -euo pipefail

PREFIX=/opt/coroseg
DATA=/var/lib/coroseg
ETC=/etc/coroseg
TORCH_INDEX=https://download.pytorch.org/whl/cpu
[[ "${1:-}" == "--gpu" ]] && TORCH_INDEX=https://download.pytorch.org/whl/cu126
REPO_DIR="$(cd "$(dirname "$0")/.." && pwd)"

[[ $EUID -eq 0 ]] || { echo "À lancer en root (ou avec sudo)." >&2; exit 1; }

echo "==> Paquets système"
apt-get update -q
apt-get install -y -q python3 python3-venv python3-pip fonts-dejavu-core

echo "==> Utilisateur de service"
id coroseg &>/dev/null || useradd --system --home-dir "$DATA" --shell /usr/sbin/nologin coroseg
install -d -o coroseg -g coroseg -m 0750 "$DATA" "$DATA/totalseg"
install -d -m 0755 "$PREFIX" "$ETC"

echo "==> Code ($PREFIX)"
rm -rf "$PREFIX/coroseg" "$PREFIX/coroseg_orthanc"
cp -r "$REPO_DIR/coroseg" "$REPO_DIR/coroseg_orthanc" "$REPO_DIR/requirements-orthanc.txt" "$PREFIX/"

echo "==> Environnement Python ($PREFIX/venv) — PyTorch depuis $TORCH_INDEX"
[[ -d "$PREFIX/venv" ]] || python3 -m venv "$PREFIX/venv"
"$PREFIX/venv/bin/pip" install -q --upgrade pip
# torch doit venir de l'index choisi (CPU ou CUDA) : on le réinstalle si on change de mode
"$PREFIX/venv/bin/pip" install -q --force-reinstall torch --index-url "$TORCH_INDEX"
"$PREFIX/venv/bin/pip" install -q -r "$PREFIX/requirements-orthanc.txt"

echo "==> Configuration ($ETC/coroseg.env)"
NEW_CONF=0
if [[ ! -f "$ETC/coroseg.env" ]]; then
  install -m 0640 -g coroseg "$REPO_DIR/deploy/coroseg.env.example" "$ETC/coroseg.env"
  NEW_CONF=1
fi

echo "==> Commande « coroseg » (demo, check, series, process)"
cat > /usr/local/bin/coroseg <<WRAP
#!/bin/sh
cd $PREFIX && exec runuser -u coroseg -- env COROSEG_WORK_DIR=$DATA TOTALSEG_HOME_DIR=$DATA/totalseg \\
  COROSEG_ENV_FILE=$ETC/coroseg.env $PREFIX/venv/bin/python -m coroseg_orthanc "\$@"
WRAP
chmod 0755 /usr/local/bin/coroseg

echo "==> Service systemd"
install -m 0644 "$REPO_DIR/deploy/coroseg-orthanc.service" /etc/systemd/system/coroseg-orthanc.service
systemctl daemon-reload
systemctl enable -q coroseg-orthanc

if [[ $NEW_CONF -eq 1 ]] || grep -q "aca_XXXX" "$ETC/coroseg.env"; then
  echo
  echo "Configuration à compléter : nano $ETC/coroseg.env"
  echo "  (ORTHANC_URL, ORTHANC_USER, ORTHANC_PASSWORD, TOTALSEG_LICENSE)"
  echo "puis : systemctl restart coroseg-orthanc && journalctl -u coroseg-orthanc -f"
  exit 0
fi

systemctl restart coroseg-orthanc

echo "==> Test de connexion avec Orthanc et du calcul"
coroseg check || echo "ATTENTION : Orthanc injoignable, vérifier $ETC/coroseg.env (URL, identifiants) et le pare-feu."

echo "Terminé. Journal du service : journalctl -u coroseg-orthanc -f"
