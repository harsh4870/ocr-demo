#!/usr/bin/env bash
set -euo pipefail
source "$(dirname "$0")/config.sh"

gcloud services enable \
  storage.googleapis.com aiplatform.googleapis.com artifactregistry.googleapis.com \
  cloudbuild.googleapis.com run.googleapis.com

gcloud storage buckets describe "gs://${BUCKET}" >/dev/null 2>&1 || \
  gcloud storage buckets create "gs://${BUCKET}" --location="${REGION}" --uniform-bucket-level-access

gcloud artifacts repositories describe "${REPO}" --location="${REGION}" >/dev/null 2>&1 || \
  gcloud artifacts repositories create "${REPO}" --repository-format=docker --location="${REGION}"

echo "Ready: project=${PROJECT_ID} region=${REGION} bucket=gs://${BUCKET} repo=${REPO}"
