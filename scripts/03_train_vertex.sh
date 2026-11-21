#!/usr/bin/env bash
set -euo pipefail
source "$(dirname "$0")/config.sh"
EPOCHS="${EPOCHS:-15}"
ACCELERATOR="${ACCELERATOR:-NVIDIA_TESLA_T4}"
RUN_ID="run-$(date +%Y%m%d-%H%M%S)"

gcloud builds submit train --tag "${TRAIN_IMAGE}" --region="${REGION}"

if [ "${ACCELERATOR}" = "none" ]; then
  POOL="machine-type=n1-standard-8,replica-count=1,container-image-uri=${TRAIN_IMAGE}"
else
  POOL="machine-type=n1-standard-4,replica-count=1,accelerator-type=${ACCELERATOR},accelerator-count=1,container-image-uri=${TRAIN_IMAGE}"
fi

JOB=$(gcloud ai custom-jobs create \
  --region="${REGION}" \
  --display-name="ocr-${RUN_ID}" \
  --worker-pool-spec="${POOL}" \
  --args="--data=/gcs/${BUCKET}/data,--out=/gcs/${BUCKET}/models/${RUN_ID},--epochs=${EPOCHS}" \
  --format="value(name)")

echo "Job: ${JOB}"
echo "Console: https://console.cloud.google.com/vertex-ai/training/custom-jobs?project=${PROJECT_ID}"
echo "${RUN_ID}" > .last_run
gcloud ai custom-jobs stream-logs "${JOB}" --region="${REGION}" || true
echo "Model artifacts: gs://${BUCKET}/models/${RUN_ID}/"
