#!/usr/bin/env bash
set -euo pipefail
source "$(dirname "$0")/config.sh"
DATA_DIR="${1:-data/plates}"

[ -d "${DATA_DIR}/train" ] || { echo "No dataset at ${DATA_DIR}. Run: python data/generate.py" >&2; exit 1; }
gcloud storage cp -r "${DATA_DIR}/train" "${DATA_DIR}/val" "gs://${BUCKET}/data/"
echo "Uploaded. Files in bucket: $(gcloud storage ls "gs://${BUCKET}/data/train/" | wc -l)"
