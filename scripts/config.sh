export PROJECT_ID="${PROJECT_ID:-$(gcloud config get-value project 2>/dev/null)}"
export REGION="${REGION:-us-central1}"
export BUCKET="${BUCKET:-${PROJECT_ID}-ocr-workshop}"
export REPO="${REPO:-ocr-workshop}"
export TRAIN_IMAGE="${REGION}-docker.pkg.dev/${PROJECT_ID}/${REPO}/trainer:latest"
export SERVICE="${SERVICE:-ocr-api}"

if [ -z "$PROJECT_ID" ]; then
  echo "Set a project first: gcloud config set project <PROJECT_ID>" >&2
  exit 1
fi
