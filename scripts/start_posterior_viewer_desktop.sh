#!/bin/bash
set -euo pipefail

REPO="/Users/ioannism/repos/npe"
HOST="127.0.0.1"
PORT="8876"
LOG_DIR="${REPO}/logs/ui"
LOG_FILE="${LOG_DIR}/npe_posterior_viewer_desktop.log"
UV_BIN="/Users/ioannism/.local/bin/uv"
ICON_FILE="/Users/ioannism/Desktop/NPE Posterior Viewer.app/Contents/Resources/icon.icns"

if [ ! -x "${UV_BIN}" ]; then
  UV_BIN="$(command -v uv)"
fi

mkdir -p "${LOG_DIR}"
cd "${REPO}"

{
  echo
  echo "=== $(date '+%Y-%m-%d %H:%M:%S') launching native NPE posterior viewer ==="
} >>"${LOG_FILE}"

if [ ! -f "${REPO}/viewer-ui/dist/index.html" ]; then
  echo "Building viewer-ui/dist..." >>"${LOG_FILE}"
  npm --prefix "${REPO}/viewer-ui" run build >>"${LOG_FILE}" 2>&1
fi

export NPE_VIEWER_ICON="${ICON_FILE}"

exec "${UV_BIN}" run scripts/npe_posterior_viewer_native.py \
  --host "${HOST}" \
  --port "${PORT}" \
  >>"${LOG_FILE}" 2>&1
