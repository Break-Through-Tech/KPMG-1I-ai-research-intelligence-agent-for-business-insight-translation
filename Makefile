PYTHON ?= python3

.PHONY: seed-vector-store

seed-vector-store:
	$(PYTHON) -m scripts.seed_vector_store