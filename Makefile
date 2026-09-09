# Twitter Customer Support Agent - SpotifyCares
#
# The headline path is `make reproduce`: it needs no API keys, makes no network
# calls, and regenerates every table in the README from the committed response
# cache in a couple of minutes.

PY := python
export PYTHONIOENCODING := utf-8

.PHONY: help setup data reproduce live predict judge metrics sheet test verify-cache clean

help:
	@echo "make setup        install dependencies"
	@echo "make data         download corpus, build pairs, index, golden set (~5 min, one off)"
	@echo "make reproduce    regenerate headline results from the committed cache (no keys, offline)"
	@echo "make live         full run against the APIs (needs .env; ~100 min under free-tier limits)"
	@echo "make sheet        build the human labelling sheet"
	@echo "make test         run the test suite"
	@echo "make verify-cache re-run a live sample and diff against the cache"

setup:
	$(PY) -m pip install -r requirements.txt

# One-off data preparation. Not part of `reproduce` because it downloads
# 516MB; the processed artefacts it produces are what the pipeline reads.
data:
	$(PY) src/data/download.py
	$(PY) src/data/brand.py
	$(PY) src/data/threads.py SpotifyCares
	$(PY) src/eval/golden.py
	$(PY) src/agent/retrieve.py

# THE REVIEWER'S PATH. Offline, deterministic, no keys.
# LLM_OFFLINE=1 makes any cache miss a hard error rather than a silent network
# call, so this either reproduces the published numbers exactly or says why not.
reproduce:
	LLM_OFFLINE=1 $(PY) src/run_pipeline.py predict --offline --ignore-roster
	LLM_OFFLINE=1 $(PY) src/run_pipeline.py judge --offline
	$(PY) src/run_pipeline.py metrics
	$(PY) src/eval/report.py

live:
	$(PY) src/run_pipeline.py predict
	$(PY) src/run_pipeline.py judge --with-ceiling
	$(PY) src/run_pipeline.py metrics
	$(PY) src/eval/report.py

predict:
	$(PY) src/run_pipeline.py predict

judge:
	$(PY) src/run_pipeline.py judge

metrics:
	$(PY) src/run_pipeline.py metrics

sheet:
	$(PY) src/eval/build_sheet.py

test:
	$(PY) -m pytest tests/ -q

# Proves the committed cache is real: re-runs N random cached prompts against
# the live APIs and diffs the responses.
verify-cache:
	$(PY) src/eval/verify_cache.py --sample $(or $(SAMPLE),20)

clean:
	rm -rf results/*.parquet results/*.csv results/*.json
