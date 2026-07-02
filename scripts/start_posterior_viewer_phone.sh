#!/bin/bash
set -euo pipefail

REPO="${REPO:-/Users/ioannism/repos/npe}"
PORT="${PORT:-8876}"
LOG_DIR="${REPO}/logs/ui"
LOG_FILE="${LOG_DIR}/npe_posterior_viewer_phone.log"
UV_BIN="${UV_BIN:-/Users/ioannism/.local/bin/uv}"

detect_tailscale_ip() {
  if command -v tailscale >/dev/null 2>&1; then
    tailscale ip -4 2>/dev/null | head -n 1
    return
  fi
  if [ -x /Applications/Tailscale.app/Contents/MacOS/Tailscale ]; then
    /Applications/Tailscale.app/Contents/MacOS/Tailscale ip -4 2>/dev/null | head -n 1
  fi
}

detect_lan_ip() {
  ipconfig getifaddr en0 2>/dev/null || true
}

if [ ! -x "${UV_BIN}" ]; then
  UV_BIN="$(command -v uv)"
fi

TAILSCALE_HOST="$(detect_tailscale_ip || true)"
LAN_HOST="$(detect_lan_ip)"
HOST="${HOST:-${TAILSCALE_HOST:-${LAN_HOST:-127.0.0.1}}}"
PUBLIC_HOST="${PUBLIC_HOST:-${HOST}}"

mkdir -p "${LOG_DIR}"
cd "${REPO}"

{
  echo
  echo "=== $(date '+%Y-%m-%d %H:%M:%S') launching phone-accessible NPE posterior viewer ==="
  echo "host=${HOST}"
  echo "port=${PORT}"
  echo "public_url=http://${PUBLIC_HOST}:${PORT}/"
} >>"${LOG_FILE}"

if [ ! -d "${REPO}/viewer-ui/node_modules" ]; then
  echo "Installing viewer-ui dependencies..." >>"${LOG_FILE}"
  npm --prefix "${REPO}/viewer-ui" ci >>"${LOG_FILE}" 2>&1
fi

echo "Building viewer-ui/dist..." >>"${LOG_FILE}"
npm --prefix "${REPO}/viewer-ui" run build >>"${LOG_FILE}" 2>&1

echo "Open on Pixel: http://${PUBLIC_HOST}:${PORT}/"
echo "Log: ${LOG_FILE}"

exec "${UV_BIN}" run scripts/npe_posterior_viewer.py \
  --host "${HOST}" \
  --port "${PORT}" \
  --strict-port \
  "$@" \
  >>"${LOG_FILE}" 2>&1
