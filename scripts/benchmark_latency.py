"""Measure cold and warm model/agent latency without changing project data."""

from __future__ import annotations

import argparse
import json
import time
from typing import Any

from src.agent import OLLAMA_KEEP_ALIVE, run_agent
from src.embeddings import get_model


def measure_embedding(query: str) -> dict[str, float]:
    timings: dict[str, float] = {}
    started = time.perf_counter()
    model = get_model(timings)
    timings["first_query_embedding_seconds"] = time.perf_counter() - started

    started = time.perf_counter()
    model.encode(query, show_progress_bar=False)
    timings["warm_query_embedding_seconds"] = time.perf_counter() - started
    return timings


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--query",
        default="How can AI explain medical reports to patients?",
    )
    parser.add_argument(
        "--agent",
        action="store_true",
        help="Also run two agent queries in this process.",
    )
    args = parser.parse_args()

    report: dict[str, Any] = {
        "ollama_keep_alive": OLLAMA_KEEP_ALIVE,
        "embedding": measure_embedding(args.query),
    }

    if args.agent:
        report["agent"] = {
            "first_run": run_agent(args.query)["timings"],
            "second_run": run_agent(args.query)["timings"],
        }

    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
