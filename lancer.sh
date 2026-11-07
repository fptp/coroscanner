#!/usr/bin/env bash
# Lance l'interface CoroSeg (ouvre http://localhost:8501)
cd "$(dirname "$0")"
exec python -m streamlit run app.py --server.headless false "$@"
