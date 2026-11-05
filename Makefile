PYTHON ?= ./venv/bin/python

.PHONY: ingest agent test

ingest:
	$(PYTHON) -m scripts.backfill_papers

agent:
	$(PYTHON) -m src.agent

ui:
	$(PYTHON) -m streamlit run app.py

test:
	$(PYTHON) -m pytest tests
