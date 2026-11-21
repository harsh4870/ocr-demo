#!/usr/bin/env bash
set -euo pipefail
source "$(dirname "$0")/config.sh"
RUN_ID="${1:-$(cat .last_run 2>/dev/null || true)}"

mkdir -p serve/model
if [ -n "${MODEL_SRC:-}" ]; then
  cp "${MODEL_SRC}/model.onnx" "${MODEL_SRC}/model.json" serve/model/
else
  [ -n "${RUN_ID}" ] || { echo "No run id. Pass one, or set MODEL_SRC=artifacts" >&2; exit 1; }
  gcloud storage cp "gs://${BUCKET}/models/${RUN_ID}/model.onnx" "gs://${BUCKET}/models/${RUN_ID}/model.json" serve/model/
fi
rm -rf serve/model/samples
if [ -d "${DATA_DIR:-data/plates}/val" ]; then cp -r "${DATA_DIR:-data/plates}/val" serve/model/samples
else echo "No ${DATA_DIR:-data/plates}/val found, so the demo button will be disabled" >&2; fi

gcloud run deploy "${SERVICE}" \
  --source serve \
  --region="${REGION}" \
  --allow-unauthenticated \
  --cpu=1 --memory=1Gi \
  --concurrency=8 --min-instances=0 --max-instances=3 \
  --set-env-vars="MODEL_VERSION=${RUN_ID:-local}"

URL=$(gcloud run services describe "${SERVICE}" --region="${REGION}" --format='value(status.url)')
echo "Live at: ${URL}"
curl -s "${URL}/healthz"; echo
