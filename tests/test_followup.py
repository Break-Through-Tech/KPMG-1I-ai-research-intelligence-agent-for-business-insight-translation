"""
Tests for follow-up question handling (src/followup.py).

Runs offline: the LLM is replaced by a function that returns a fixed decision.

Run from the repo root:
    python -m pytest tests/test_followup.py -v
"""

import unittest

from src.followup import (
    NEW_SEARCH,
    SAME_PAPERS,
    FollowUpDecision,
    add_turn,
    find_paper_references,
    make_turn,
    plan_follow_up,
    suggest_follow_ups,
)

PAPERS = [
    {"paper_id": "2610.33997", "title": "Retrieval Grounding Reduces Hallucination in LLMs",
     "chunk_text": "Abstract one", "score": 0.61},
    {"paper_id": "2610.33904", "title": "Self-Checking Decoding for Factual Generation",
     "chunk_text": "Abstract two", "score": 0.55},
    {"paper_id": "2610.34012", "title": "A Benchmark for Hallucination Detection",
     "chunk_text": "Abstract three", "score": 0.48},
]

HISTORY = [{
    "original_query": "How can hallucination in LLMs be reduced?",
    "query": "How can hallucination in LLMs be reduced?",
    "route": NEW_SEARCH,
    "answer": "Grounding answers in retrieved documents helps.",
    "citations": [{"paper_id": "2610.33904", "supporting_quote": "..."}],
    "papers": PAPERS,
    "paper_sections": {"2610.33997": [{"title": "Conclusion", "text": "..."}]},
}]


class RecordingLLM:
    """Stands in for the LLM: records the call and returns a fixed decision."""

    def __init__(self, decision=None, error=None):
        self.decision = decision
        self.error = error
        self.calls = []

    def __call__(self, inputs):
        self.calls.append(inputs)
        if self.error:
            raise self.error
        return self.decision


class TestFindPaperReferences(unittest.TestCase):
    def test_ordinals(self):
        self.assertEqual(find_paper_references("Explain the second paper", PAPERS), ["2610.33904"])
        self.assertEqual(find_paper_references("what about the last one?", PAPERS), ["2610.34012"])

    def test_numbers(self):
        self.assertEqual(find_paper_references("limitations of paper 3?", PAPERS), ["2610.34012"])
        self.assertEqual(find_paper_references("compare #1 and #2", PAPERS), ["2610.33997", "2610.33904"])
        self.assertEqual(find_paper_references("more on [2]", PAPERS), ["2610.33904"])

    def test_number_out_of_range_is_ignored(self):
        self.assertEqual(find_paper_references("what about paper 7?", PAPERS), [])

    def test_arxiv_id_with_or_without_version(self):
        self.assertEqual(find_paper_references("Summarize 2610.34012v1", PAPERS), ["2610.34012"])
        self.assertEqual(find_paper_references("Summarize 2610.34012", PAPERS), ["2610.34012"])

    def test_title(self):
        question = "What data does self-checking decoding for factual generation use?"
        self.assertEqual(find_paper_references(question, PAPERS), ["2610.33904"])

    def test_these_papers_means_all(self):
        self.assertEqual(
            find_paper_references("Compare these papers", PAPERS),
            [p["paper_id"] for p in PAPERS],
        )

    def test_no_reference(self):
        self.assertEqual(find_paper_references("What about healthcare?", PAPERS), [])
        self.assertEqual(find_paper_references("Give me 3 takeaways", PAPERS), [])


class TestPlanFollowUp(unittest.TestCase):
    def test_first_question_is_a_new_search_without_calling_the_llm(self):
        llm = RecordingLLM()
        plan = plan_follow_up("How can hallucination be reduced?", [], llm)
        self.assertEqual(plan.route, NEW_SEARCH)
        self.assertEqual(plan.standalone_query, "How can hallucination be reduced?")
        self.assertEqual(plan.reason, "no_history")
        self.assertEqual(llm.calls, [])

    def test_explicit_paper_reference_skips_the_llm(self):
        llm = RecordingLLM()
        plan = plan_follow_up("What are the limitations of paper 2?", HISTORY, llm)
        self.assertEqual(plan.route, SAME_PAPERS)
        self.assertEqual(plan.focus_paper_ids, ["2610.33904"])
        self.assertIn("Self-Checking Decoding for Factual Generation", plan.standalone_query)
        self.assertEqual(plan.reason, "explicit_reference")
        self.assertEqual(llm.calls, [])

    def test_these_papers_reuses_all_papers_and_carries_the_topic(self):
        plan = plan_follow_up("How could a business apply these papers?", HISTORY, RecordingLLM())
        self.assertEqual(plan.route, SAME_PAPERS)
        self.assertEqual(plan.focus_paper_ids, [])
        self.assertIn("How can hallucination in LLMs be reduced?", plan.standalone_query)

    def test_asking_for_other_papers_goes_to_the_llm_even_with_a_reference(self):
        llm = RecordingLLM(FollowUpDecision(
            route=NEW_SEARCH,
            standalone_query="Papers similar to self-checking decoding for factual generation",
            wants_other_papers=True,
        ))
        plan = plan_follow_up("Find papers similar to paper 2", HISTORY, llm)
        self.assertEqual(plan.route, NEW_SEARCH)
        self.assertEqual(len(llm.calls), 1)
        self.assertEqual(plan.exclude_paper_ids, [p["paper_id"] for p in PAPERS])

    def test_llm_same_papers_with_rewrite(self):
        llm = RecordingLLM(FollowUpDecision(
            route=SAME_PAPERS,
            standalone_query="Explain in simple terms how the shown papers reduce LLM hallucination",
        ))
        plan = plan_follow_up("Explain that more simply", HISTORY, llm)
        self.assertEqual(plan.route, SAME_PAPERS)
        self.assertEqual(plan.standalone_query, "Explain in simple terms how the shown papers reduce LLM hallucination")
        self.assertEqual(plan.focus_paper_ids, [])
        self.assertEqual(plan.reason, "llm")
        # The prompt got the conversation and the numbered paper list.
        self.assertIn("User: How can hallucination in LLMs be reduced?", llm.calls[0]["history"])
        self.assertIn("[2] paper_id: 2610.33904", llm.calls[0]["papers"])

    def test_llm_paper_ids_are_matched_to_shown_papers(self):
        llm = RecordingLLM(FollowUpDecision(
            route=SAME_PAPERS,
            standalone_query="How was the benchmark built?",
            paper_ids=["2610.34012v1", "9999.99999"],  # versioned, and one made up
        ))
        plan = plan_follow_up("how did they build that benchmark?", HISTORY, llm)
        self.assertEqual(plan.focus_paper_ids, ["2610.34012"])

    def test_llm_new_search_keeps_seen_papers_unless_other_papers_asked(self):
        llm = RecordingLLM(FollowUpDecision(
            route=NEW_SEARCH,
            standalone_query="How can LLM hallucination be reduced in healthcare?",
        ))
        plan = plan_follow_up("What about healthcare?", HISTORY, llm)
        self.assertEqual(plan.route, NEW_SEARCH)
        self.assertEqual(plan.standalone_query, "How can LLM hallucination be reduced in healthcare?")
        self.assertEqual(plan.exclude_paper_ids, [])

    def test_llm_cannot_pick_same_papers_when_none_were_shown(self):
        history = [dict(HISTORY[0], papers=[], paper_sections={})]
        llm = RecordingLLM(FollowUpDecision(route=SAME_PAPERS, standalone_query="Explain more"))
        plan = plan_follow_up("Explain more", history, llm)
        self.assertEqual(plan.route, NEW_SEARCH)

    def test_empty_rewrite_falls_back_to_the_question(self):
        llm = RecordingLLM(FollowUpDecision(route=NEW_SEARCH, standalone_query="  "))
        plan = plan_follow_up("What about diffusion models?", HISTORY, llm)
        self.assertEqual(plan.standalone_query, "What about diffusion models?")

    def test_llm_error_short_follow_up_adds_previous_question(self):
        plan = plan_follow_up("What about healthcare?", HISTORY, RecordingLLM(error=RuntimeError("down")))
        self.assertEqual(plan.route, NEW_SEARCH)
        self.assertEqual(plan.standalone_query, "How can hallucination in LLMs be reduced? What about healthcare?")
        self.assertEqual(plan.reason, "llm_failed")

    def test_llm_unparsed_long_question_is_used_as_is(self):
        question = "What evaluation benchmarks exist for measuring agent tool use reliability?"
        plan = plan_follow_up(question, HISTORY, RecordingLLM(decision=None))
        self.assertEqual(plan.standalone_query, question)

    def test_llm_error_other_papers_still_excludes_seen(self):
        plan = plan_follow_up("Any other papers?", HISTORY, RecordingLLM(error=RuntimeError("down")))
        self.assertEqual(plan.exclude_paper_ids, [p["paper_id"] for p in PAPERS])


class TestTurnsAndSuggestions(unittest.TestCase):
    def test_new_search_turn_keeps_its_papers(self):
        result = {
            "original_query": "q", "query": "q", "route": NEW_SEARCH, "answer": "a",
            "citations": [], "retrieved_chunks": PAPERS[:2], "paper_sections": {},
        }
        turn = make_turn(result, HISTORY)
        self.assertEqual([p["paper_id"] for p in turn["papers"]], ["2610.33997", "2610.33904"])

    def test_same_papers_turn_keeps_the_numbering_the_user_saw(self):
        result = {
            "original_query": "paper 2?", "query": "paper 2?", "route": SAME_PAPERS, "answer": "a",
            "citations": [], "retrieved_chunks": [PAPERS[1]], "paper_sections": {},
        }
        turn = make_turn(result, HISTORY)
        self.assertEqual(turn["papers"], PAPERS)
        self.assertEqual(turn["paper_sections"], HISTORY[0]["paper_sections"])
        # So "paper 3" still means the third paper from the original list.
        plan = plan_follow_up("and paper 3?", [turn], RecordingLLM())
        self.assertEqual(plan.focus_paper_ids, ["2610.34012"])

    def test_unanswered_turn_has_no_papers_to_follow_up_on(self):
        result = {
            "original_query": "pizza?", "query": "pizza?", "route": NEW_SEARCH, "answer": "",
            "citations": [], "retrieved_chunks": PAPERS, "paper_sections": {},
        }
        self.assertEqual(make_turn(result, [])["papers"], [])

    def test_history_is_capped(self):
        history = []
        for i in range(5):
            history = add_turn(history, {"query": str(i)})
        self.assertEqual([t["query"] for t in history], ["2", "3", "4"])

    def test_suggestions_skip_questions_already_asked(self):
        asked = dict(HISTORY[0], original_query="What are the limitations of paper 2?")
        suggestions = suggest_follow_ups(HISTORY + [asked])
        self.assertNotIn("What are the limitations of paper 2?", suggestions)
        self.assertEqual(suggestions[0], "How could a business apply the findings of these papers?")
        self.assertEqual(suggestions[1], "What are the limitations of paper 1?")

    def test_suggestions_point_at_the_most_cited_paper_and_take_known_routes(self):
        suggestions = suggest_follow_ups(HISTORY)
        self.assertEqual(suggestions[0], "What are the limitations of paper 2?")
        self.assertEqual(len(suggestions), 3)

        routes = [plan_follow_up(s, HISTORY, RecordingLLM()).route for s in suggestions[:2]]
        self.assertEqual(routes, [SAME_PAPERS, SAME_PAPERS])
        # The last one asks for other papers, which the LLM rewrites.
        llm = RecordingLLM(FollowUpDecision(
            route=NEW_SEARCH, standalone_query="Other papers on reducing LLM hallucination",
            wants_other_papers=True,
        ))
        plan = plan_follow_up(suggestions[2], HISTORY, llm)
        self.assertEqual(plan.route, NEW_SEARCH)
        self.assertEqual(plan.exclude_paper_ids, [p["paper_id"] for p in PAPERS])

    def test_no_suggestions_without_papers(self):
        self.assertEqual(suggest_follow_ups([{"papers": []}]), [])
        self.assertEqual(suggest_follow_ups([]), [])


if __name__ == "__main__":
    unittest.main()
