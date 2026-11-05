"""
End-to-end test of a short conversation through the agent graph.

Runs offline: Qdrant search, the arXiv section fetch, section scoring and the
LLM are all faked, so this checks the follow-up routing, not answer quality.

Run from the repo root:
    python -m pytest tests/test_agent_followup.py -v
"""

import unittest
from types import SimpleNamespace
from unittest.mock import patch

from src import agent
from src.followup import NEW_SEARCH, SAME_PAPERS, FollowUpDecision

CORPUS = [
    {"paper_id": f"2610.0000{i}", "title": f"Paper {i} on hallucination",
     "section_title": "Abstract", "chunk_text": f"Abstract {i}", "chunk_index": 0,
     "score": 0.7 - i / 100}
    for i in range(1, 8)
]


def fake_semantic_search(client, query, top_k=5):
    fake_semantic_search.calls.append((query, top_k))
    return [dict(paper) for paper in CORPUS[:top_k]]


def fake_fetch_sections(arxiv_id):
    fake_fetch_sections.calls.append(arxiv_id)
    return [SimpleNamespace(title="Conclusion", text=f"Conclusion of {arxiv_id}")]


class FakeLLM:
    """Answers with one claim per paper in the context; follow-up decisions are queued."""

    def __init__(self, decisions=()):
        self.decisions = list(decisions)
        self.answer_inputs = []

    def with_structured_output(self, schema, include_raw=False):
        def run(prompt_value):
            text = prompt_value.to_string()
            if schema is FollowUpDecision:
                return {"parsed": self.decisions.pop(0)}
            self.answer_inputs.append(text)
            ids = [p["paper_id"] for p in CORPUS if f"(ID: {p['paper_id']})" in text]
            return {"parsed": agent.LLMAnswer(
                summary="Summary.",
                claims=[agent.Claim(statement="Claim.", paper_id=pid, supporting_quote="q") for pid in ids],
            )}
        return run


class TestAgentFollowUps(unittest.TestCase):
    def setUp(self):
        fake_semantic_search.calls = []
        fake_fetch_sections.calls = []
        patches = [
            patch("src.agent.get_client", return_value=SimpleNamespace(close=lambda: None)),
            patch("src.agent.semantic_search", fake_semantic_search),
            patch("src.agent.fetch_sections", fake_fetch_sections),
            patch("src.agent._score_sections", lambda sections, query: [0.0] * len(sections)),
        ]
        for p in patches:
            p.start()
            self.addCleanup(p.stop)

    def test_conversation(self):
        llm = FakeLLM(decisions=[
            FollowUpDecision(route=NEW_SEARCH, wants_other_papers=True,
                             standalone_query="Other papers on reducing hallucination"),
        ])
        with patch("src.agent.llm", llm):
            # 1. First question: normal search.
            first = agent.run_agent("How can hallucination be reduced?")
            self.assertEqual(first["route"], NEW_SEARCH)
            self.assertEqual(first["document_ids"], ["2610.00001", "2610.00002", "2610.00003"])
            self.assertEqual(len(first["history"]), 1)
            self.assertEqual(len(first["follow_up_suggestions"]), 3)
            self.assertEqual(len(fake_fetch_sections.calls), 3)

            # 2. About one shown paper: no new search or fetch, only that paper in the context.
            second = agent.run_agent("What are the limitations of paper 2?", first["history"])
            self.assertEqual(second["route"], SAME_PAPERS)
            self.assertEqual(len(fake_semantic_search.calls), 1)
            self.assertEqual(len(fake_fetch_sections.calls), 3)
            self.assertEqual(second["document_ids"], ["2610.00002"])
            self.assertIn("Conclusion of 2610.00002", llm.answer_inputs[-1])
            self.assertNotIn("(ID: 2610.00001)", llm.answer_inputs[-1])
            self.assertIn("Paper 2 on hallucination", second["query"])
            # The shown list (and its numbering) carries over.
            self.assertEqual(len(second["history"][-1]["papers"]), 3)

            # 3. Other papers: new search, already-shown papers left out.
            third = agent.run_agent("Are there other papers on this topic?", second["history"])
            self.assertEqual(third["route"], NEW_SEARCH)
            self.assertEqual(third["query"], "Other papers on reducing hallucination")
            self.assertEqual(fake_semantic_search.calls[-1], ("Other papers on reducing hallucination", 6))
            self.assertEqual(third["document_ids"], ["2610.00004", "2610.00005", "2610.00006"])
            self.assertEqual(len(third["history"]), 3)


if __name__ == "__main__":
    unittest.main()
