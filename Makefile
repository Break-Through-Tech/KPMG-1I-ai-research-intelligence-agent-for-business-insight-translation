PYTHON ?= ./venv/bin/python

.PHONY: ingest agent test agent-interactive benchmark

ingest:
	$(PYTHON) -m scripts.backfill_papers

agent:
	$(PYTHON) -m src.agent

agent-interactive:
	$(PYTHON) -m src.agent --interactive

benchmark:
	$(PYTHON) -m scripts.benchmark_latency
test:
	$(PYTHON) -m pytest tests
