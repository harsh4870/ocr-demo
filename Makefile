PY ?= .venv/bin/python
PYTHON ?= python3.12

.PHONY: setup data train-smoke train export serve serve-model docker-run up down logs

setup:
	$(PYTHON) -m venv .venv && $(PY) -m pip install -r requirements.txt

data:
	$(PY) data/generate.py --out data/plates

train-smoke:
	$(PY) data/generate.py --out data/small --train 2000 --val 200
	$(PY) train/train.py --data data/small --out artifacts-smoke --epochs 2

train:
	$(PY) train/train.py --data data/plates --out artifacts --epochs 15

export:
	cd train && $(abspath $(PY)) export.py --ckpt ../artifacts/model.pt --out ../artifacts

serve:
	cd serve && MODEL_DIR=../artifacts SAMPLES_DIR=../data/plates/val ../.venv/bin/uvicorn app:app --port 8080 --reload

serve-model:
	mkdir -p serve/model && cp artifacts/model.onnx artifacts/model.json serve/model/
	rm -rf serve/model/samples && cp -r data/plates/val serve/model/samples

docker-run: serve-model
	docker build -t ocr-api serve && docker run --rm -p 8080:8080 ocr-api

up: serve-model
	docker compose up -d --build

down:
	docker compose down

logs:
	docker compose logs -f
