#!/usr/bin/env bash
#
# Publish the SciConBench dashboard to https://sciconbench.cs.princeton.edu
#
#   ./site/publish.sh              regenerate data, then deploy
#   ./site/publish.sh --no-export  deploy the current public/ as-is
#   ./site/publish.sh --dry-run    show what would change, copy nothing
#
# The web root is a group-writable share, so files are pushed with permissions
# the httpd user can actually read (0644 files, 2775 dirs).

set -euo pipefail

SITE_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SRC="${SITE_DIR}/public/"
DEST="${SCICON_WEB_ROOT:-/n/fs/sciconbench/www}"

EXPORT=1
RSYNC_EXTRA=()

for arg in "$@"; do
  case "$arg" in
    --no-export) EXPORT=0 ;;
    --dry-run)   RSYNC_EXTRA+=("--dry-run" "--itemize-changes") ;;
    -h|--help)   sed -n '2,10p' "${BASH_SOURCE[0]}"; exit 0 ;;
    *) echo "unknown option: $arg" >&2; exit 2 ;;
  esac
done

if [[ ! -d "$DEST" ]]; then
  echo "error: web root not found: $DEST" >&2
  exit 1
fi

if [[ $EXPORT -eq 1 ]]; then
  echo "==> regenerating dashboard data"
  python3 "${SITE_DIR}/export_data.py"
fi

if [[ ! -f "${SRC}data/dashboard.json" ]]; then
  echo "error: ${SRC}data/dashboard.json is missing — run export_data.py first" >&2
  exit 1
fi

echo "==> syncing ${SRC} -> ${DEST}"
rsync -rlv --delete \
  --chmod=D2775,F0664 \
  --exclude '.DS_Store' \
  "${RSYNC_EXTRA[@]}" \
  "$SRC" "${DEST}/"

echo "==> live at https://sciconbench.cs.princeton.edu"
