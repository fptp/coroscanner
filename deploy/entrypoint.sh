#!/bin/sh
set -e
# Enregistre la licence TotalSegmentator (une fois, stockée dans $TOTALSEG_HOME_DIR qui est sur le volume /data)
if [ -n "$TOTALSEG_LICENSE" ]; then
  totalseg_set_license -l "$TOTALSEG_LICENSE" >/dev/null 2>&1 || echo "Avertissement : enregistrement de la licence impossible"
fi
exec python -m coroseg_orthanc "$@"
