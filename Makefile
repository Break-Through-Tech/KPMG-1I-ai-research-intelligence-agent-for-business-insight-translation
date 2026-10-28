PYTHON ?= ./venv/bin/python

.PHONY: ingest agent test

ingest:
	$(PYTHON) -m scripts.backfill_papers

agent:
	$(PYTHON) -m src.agent

test:
	$(PYTHON) -m pytest tests
