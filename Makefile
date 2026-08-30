VENV ?= .venv
PY   ?= $(VENV)/bin/python

.PHONY: help setup external find download clips features train results test pipeline clean-results

help:
	@grep -E '^[a-z-]+:.*?## .*$$' $(MAKEFILE_LIST) | awk 'BEGIN{FS=":.*?## "}{printf "  \033[36m%-12s\033[0m %s\n", $$1, $$2}'

setup: ## create the arm64 venv and install dependencies
	python3.12 -m venv $(VENV)
	$(PY) -m pip install -q --upgrade pip
	$(PY) -m pip install -e ".[dev]"
	@$(PY) -c "import platform,torch;print('arch',platform.machine(),'| torch',torch.__version__,'| mps',torch.backends.mps.is_available())"

external: ## clone the upstream annotation and download-script repos
	mkdir -p $$HOME/datasets/egoexo/external
	test -d $$HOME/datasets/egoexo/external/assembly101-annotations || \
	  git clone https://github.com/assembly-101/assembly101-annotations.git \
	    $$HOME/datasets/egoexo/external/assembly101-annotations
	@echo "NOTE: the annotation CSVs are distributed via the Google Drive link in that repo's README."

find: ## select recordings and write the manifests
	$(PY) scripts/01_find_recordings.py --optimize-exo-size

download: ## fetch one ego + one exo view per selected recording
	$(PY) scripts/02_download.py --yes

clips: ## crop a sample of synchronised ego/exo clips for inspection
	$(PY) scripts/03_crop_clips.py --limit 100

features: ## cache frozen DINOv2 per-frame features
	$(PY) scripts/04_extract_features.py

train: ## sweep configs x label budgets x seeds
	$(PY) scripts/05_train.py

results: ## aggregate into the label-efficiency curve
	$(PY) scripts/06_report.py

ui: ## render the interactive label-efficiency page from the results
	$(PY) scripts/07_build_ui.py

test: ## run the integrity checks
	$(PY) -m pytest -q

pipeline: find download features test train results ui ## everything end to end

clean-results: ## remove sweep outputs
	rm -f results/*.csv results/*.png ui/label_efficiency.html
