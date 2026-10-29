PYTHON ?= python3

.PHONY: seed-vector-store test_vector_store agent agent-interactive benchmark

seed-vector-store:
	$(PYTHON) -m scripts.seed_vector_store

test-seed-vector-store:
	$(PYTHON) -m tests.test_seed_vector_store

agent:
	$(PYTHON) -m src.agent

agent-interactive:
	$(PYTHON) -m src.agent --interactive

benchmark:
	$(PYTHON) -m scripts.benchmark_latency