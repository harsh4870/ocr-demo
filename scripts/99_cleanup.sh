#!/usr/bin/env bash
set -euo pipefail
source "$(dirname "$0")/config.sh"
read -r -p "Delete Cloud Run '${SERVICE}', bucket gs://${BUCKET} and repo '${REPO}' in ${PROJECT_ID}? [y/N] " ok
[ "${ok}" = "y" ] || exit 0

gcloud run services delete "${SERVICE}" --region="${REGION}" --quiet || true
gcloud storage rm -r "gs://${BUCKET}" || true
gcloud artifacts repositories delete "${REPO}" --location="${REGION}" --quiet || true
gcloud artifacts repositories delete cloud-run-source-deploy --location="${REGION}" --quiet || true
echo "Cleaned up."
