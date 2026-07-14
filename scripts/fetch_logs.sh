#!/usr/bin/env bash
# Pull experiment logs from the remote server, transferring only files
# that are new or changed (rsync compares size+mtime, like a diff).
#
# Usage:
#   scripts/fetch_logs.sh [remote_path] [local_path]
#   scripts/fetch_logs.sh --dry-run          # preview what would be copied
#
# Defaults point at the usual experiment box; override by passing args
# or exporting REMOTE_HOST / REMOTE_PATH / LOCAL_PATH.

set -euo pipefail

REMOTE_HOST="${REMOTE_HOST:-krisna@140.118.2.241}"
REMOTE_PORT="${REMOTE_PORT:-27213}"
REMOTE_PATH="${1:-${REMOTE_PATH:-/media/EXT1_2TB/krisna/rlm/logs}}"
LOCAL_PATH="${2:-${LOCAL_PATH:-/home/krisna/ntust/research/rlm/logs}}"

DRY_RUN=""
for arg in "$@"; do
  [[ "$arg" == "--dry-run" ]] && DRY_RUN="--dry-run"
done

mkdir -p "$LOCAL_PATH"

rsync -avz --itemize-changes --progress $DRY_RUN \
  -e "ssh -p ${REMOTE_PORT}" \
  "${REMOTE_HOST}:${REMOTE_PATH%/}/" \
  "${LOCAL_PATH%/}/"
