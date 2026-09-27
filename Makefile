PYTHON ?= python3

.PHONY: seed-vector-store test_vector_store agent

seed-vector-store:
	$(PYTHON) -m scripts.seed_vector_store

test-seed-vector-store:
	$(PYTHON) -m tests.test_seed_vector_store

agent:
	$(PYTHON) -m src.agent