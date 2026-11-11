"""
End-to-end test of short conversations through the agent graph.

Runs offline: Qdrant search, the arXiv section fetch, section scoring and the
LLMs are all faked, and the graph saves its memory in RAM instead of
chat_memory.sqlite. This checks the follow-up routing, not answer quality.

Run from the repo root:
    python -m pytest tests/test_agent_followup.py -v
"""

import unittest
import uuid
from types import SimpleNamespace
from unittest.mock import patch

from langgraph.checkpoint.memory import MemorySaver

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
    """Stands in for both Ollama models.

    Answers with one claim per paper in the context. Follow-up decisions and
    validation results are taken from queues (validation passes by default).
    """

    def __init__(self, decisions=(), checks=()):
        self.decisions = list(decisions)
        self.checks = list(checks)
        self.answer_inputs = []

    def with_structured_output(self, schema, include_raw=False):
        def run(prompt_value):
            if schema is FollowUpDecision:
                parsed = self.decisions.pop(0)
            elif schema is agent.QuestionCheck:
                parsed = self.checks.pop(0) if self.checks else agent.QuestionCheck(
                    is_about_ai=True, is_valid=True, suggested_query="")
            else:
                text = prompt_value.to_string()
                self.answer_inputs.append(text)
                ids = [p["paper_id"] for p in CORPUS if f"(ID: {p['paper_id']})" in text]
                parsed = agent.LLMAnswer(
                    summary="Summary.",
                    claims=[agent.Claim(statement="Claim.", paper_id=pid, supporting_quote="q")
                            for pid in ids],
                )
            return {"parsed": parsed} if include_raw else parsed
        return run


class TestAgentFollowUps(unittest.TestCase):
    def setUp(self):
        fake_semantic_search.calls = []
        fake_fetch_sections.calls = []
        self.llm = FakeLLM()
        patches = [
            # Fresh in-memory checkpointer, so tests don't touch chat_memory.sqlite.
            patch("src.agent.graph", agent.builder.compile(checkpointer=MemorySaver())),
            patch("src.agent.get_client", return_value=SimpleNamespace(close=lambda: None)),
            patch("src.agent.semantic_search", fake_semantic_search),
            patch("src.agent.fetch_sections", fake_fetch_sections),
            patch("src.agent._score_sections", lambda sections, query: [0.0] * len(sections)),
            patch("src.agent.llm", self.llm),
            patch("src.agent.small_llm", self.llm),
        ]
        for p in patches:
            p.start()
            self.addCleanup(p.stop)
        self.thread_id = str(uuid.uuid4())

    def ask(self, question):
        return agent.run_agent(question, thread_id=self.thread_id)

    def test_conversation(self):
        self.llm.decisions.append(FollowUpDecision(
            route=NEW_SEARCH, wants_other_papers=True,
            standalone_query="Other papers on reducing hallucination",
        ))

        # 1. First question: normal search.
        first = self.ask("How can hallucination be reduced?")
        self.assertEqual(first["route"], NEW_SEARCH)
        self.assertEqual(first["document_ids"], ["2610.00001", "2610.00002", "2610.00003"])
        self.assertEqual(len(first["history"]), 1)
        self.assertEqual(len(first["follow_up_suggestions"]), 3)
        self.assertEqual(len(fake_fetch_sections.calls), 3)

        # 2. About one shown paper: no new search or fetch, only that paper in the context.
        second = self.ask("What are the limitations of paper 2?")
        self.assertEqual(second["route"], SAME_PAPERS)
        self.assertEqual(len(fake_semantic_search.calls), 1)
        self.assertEqual(len(fake_fetch_sections.calls), 3)
        self.assertEqual(second["document_ids"], ["2610.00002"])
        self.assertIn("Conclusion of 2610.00002", self.llm.answer_inputs[-1])
        self.assertNotIn("(ID: 2610.00001)", self.llm.answer_inputs[-1])
        self.assertIn("Paper 2 on hallucination", second["query"])
        # The shown list (and its numbering) carries over.
        self.assertEqual(len(second["history"][-1]["papers"]), 3)

        # 3. Other papers: new search, already-shown papers left out.
        third = self.ask("Are there other papers on this topic?")
        self.assertEqual(third["route"], NEW_SEARCH)
        self.assertEqual(third["query"], "Other papers on reducing hallucination")
        self.assertEqual(fake_semantic_search.calls[-1], ("Other papers on reducing hallucination", 6))
        self.assertEqual(third["document_ids"], ["2610.00004", "2610.00005", "2610.00006"])
        self.assertEqual(len(third["history"]), 3)

        # Shakila's message log keeps every question and answer.
        self.assertEqual(len(third["messages"]), 6)

    def test_threads_are_separate(self):
        self.ask("How can hallucination be reduced?")
        other = agent.run_agent("How can hallucination be reduced?", thread_id=str(uuid.uuid4()))
        self.assertEqual(len(other["history"]), 1)

    def test_rejected_follow_up_does_not_return_previous_papers(self):
        self.ask("How can hallucination be reduced?")
        self.llm.decisions.append(FollowUpDecision(route=NEW_SEARCH, standalone_query="Best pizza?"))
        self.llm.checks.append(agent.QuestionCheck(is_about_ai=False, is_valid=False, suggested_query=""))

        result = self.ask("Best pizza?")
        self.assertFalse(result["is_valid"])
        self.assertEqual(result["retrieved_chunks"], [])
        self.assertEqual(result["citations"], [])
        self.assertEqual(result["follow_up_suggestions"], [])
        # Nothing for "paper 2" to point at now.
        self.assertEqual(result["history"][-1]["papers"], [])

    def test_low_confidence_turn_keeps_no_papers(self):
        with patch.dict(CORPUS[0], score=0.05), patch.dict(CORPUS[1], score=0.05), \
                patch.dict(CORPUS[2], score=0.05):
            result = self.ask("Something the corpus barely covers")
        self.assertEqual(result["confidence"], "low")
        self.assertEqual(result["history"][-1]["papers"], [])
        self.assertEqual(fake_fetch_sections.calls, [])


if __name__ == "__main__":
    unittest.main()
