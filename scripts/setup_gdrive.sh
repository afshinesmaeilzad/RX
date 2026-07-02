#!/usr/bin/env bash
#
# Mount a Google Drive folder READ-ONLY on a GPU VPS (e.g. Vast.ai) so the
# comparison container can stream the PadChest-GR dataset from /data.
#
# This script contains NO secrets. It expects an rclone remote (default name
# "gdrive") that you configure interactively once. The OAuth token lives only
# in ~/.config/rclone/rclone.conf on the VPS and is never committed to git.
#
# ---------------------------------------------------------------------------
# ONE-TIME SETUP (on the VPS, interactive):
#   rclone config
#     n) New remote
#     name> gdrive
#     Storage> drive            (Google Drive)
#     client_id>                (leave blank or use your own)
#     client_secret>            (leave blank)
#     scope> 1                  (full) or 2 (read-only) -- read-only is fine
#     ... follow the "headless / remote" OAuth flow (rclone authorize on your
#         laptop, paste the token back).
#
# Then run this script:
#   GDRIVE_SUBPATH="PadChest/extracted/BIMCV-Padchest-GR" ./scripts/setup_gdrive.sh
# ---------------------------------------------------------------------------
set -euo pipefail

REMOTE="${GDRIVE_REMOTE:-gdrive}"
SUBPATH="${GDRIVE_SUBPATH:-PadChest/extracted/BIMCV-Padchest-GR}"
MOUNT_POINT="${MOUNT_POINT:-/data}"

if ! command -v rclone >/dev/null 2>&1; then
    echo "rclone not found. Installing..."
    curl https://rclone.org/install.sh | sudo bash
fi

if ! rclone listremotes | grep -q "^${REMOTE}:"; then
    echo "ERROR: rclone remote '${REMOTE}:' is not configured."
    echo "Run 'rclone config' first (see the header of this script)."
    exit 1
fi

sudo mkdir -p "$MOUNT_POINT"
sudo chown "$(id -u):$(id -g)" "$MOUNT_POINT" || true

echo "Mounting ${REMOTE}:${SUBPATH} -> ${MOUNT_POINT} (read-only)..."
rclone mount "${REMOTE}:${SUBPATH}" "$MOUNT_POINT" \
    --read-only \
    --vfs-cache-mode full \
    --vfs-cache-max-size 5G \
    --dir-cache-time 12h \
    --daemon

sleep 3
echo "Mounted. Top-level contents of ${MOUNT_POINT}:"
ls -la "$MOUNT_POINT" | head -20
echo
echo "Now set DATA_DIR_HOST=${MOUNT_POINT} in your .env and run:"
echo "  docker compose run --rm compare"
