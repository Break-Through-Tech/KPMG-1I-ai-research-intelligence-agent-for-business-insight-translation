"""
Follow-up questions: what the agent does with a question asked after an answer.

After every answer, `run_agent` keeps a short history of turns (the question,
the answer, the papers shown and their fetched sections). When the next
question arrives, `plan_follow_up` picks one of two routes:

  same_papers   The question is about the papers already shown, e.g.
                "what are the limitations of the second paper?",
                "explain that in simpler terms", "what does this mean for a bank?".
                The agent reuses those papers and their already-fetched sections:
                no new Qdrant search and no new arXiv fetch.

  new_search    The question needs papers that were not shown, e.g.
                "what about in healthcare?", "are there other approaches?",
                or a new topic. The agent searches Qdrant again. If the user
                asked for other / more papers, the papers already shown are
                left out of the new results.

On both routes the question is rewritten into a standalone question
("what about healthcare?" -> "How can LLM hallucinations be reduced in
healthcare applications?"). The rest of the pipeline (vector search, section
scoring, the answer prompt) only ever sees that standalone question, so it
needs no changes to handle follow-ups.

How the route is picked, in order:
  1. No history                                -> new_search, question unchanged, no LLM call
  2. The question names a shown paper ("paper 2", "the second paper", an arXiv
     ID, a title) or "these papers", and does not ask for other papers
                                               -> same_papers, no LLM call
  3. Otherwise the LLM picks the route and rewrites the question
  4. If the LLM call fails                     -> new_search with the previous
                                                  question + the follow-up

This module only needs pydantic. The LLM is passed in as a function, so the
logic can be tested without Ollama.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Callable, Literal, Optional

from pydantic import BaseModel, Field

SAME_PAPERS = "same_papers"
NEW_SEARCH = "new_search"

# How many past turns run_agent keeps, and how many go into the LLM prompt.
MAX_HISTORY_TURNS = 3
# Past answers are cut to this length in the prompt to keep the call quick.
MAX_ANSWER_CHARS_IN_PROMPT = 600
# If the LLM fails, follow-ups this short get the previous question added.
SHORT_FOLLOW_UP_WORDS = 8


# ---------------------------------------------------------------------------
# What the LLM returns
# ---------------------------------------------------------------------------

class FollowUpDecision(BaseModel):
    route: Literal["same_papers", "new_search"] = Field(
        description=(
            "same_papers if the new message is about the papers or answer already "
            "shown; new_search if it needs papers that were not shown."
        )
    )
    standalone_query: str = Field(
        description=(
            "The new message rewritten as one self-contained question, with words "
            "like 'it', 'that' and 'the second one' replaced by what they refer to."
        )
    )
    paper_ids: list[str] = Field(
        default_factory=list,
        description=(
            "For same_papers: the paper_ids the message is about. Empty if it is "
            "about all of the shown papers."
        ),
    )
    wants_other_papers: bool = Field(
        default=False,
        description="True only if the user asks for other, more or different papers than the ones shown.",
    )


# The {history}, {papers} and {question} placeholders are filled in by
# ChatPromptTemplate in agent.py, so this text must not contain other braces.
FOLLOW_UP_SYSTEM_PROMPT = (
    "You help a research assistant handle follow-up questions about AI research papers.\n"
    "You get the conversation so far, the papers shown in the last answer, and the "
    "user's new message.\n\n"
    "Decide:\n"
    "1. route: 'same_papers' if the new message is about the answer or papers already "
    "shown: it asks to explain, simplify, summarize, compare, go deeper, give "
    "limitations, or give business implications, or it mentions one of the shown "
    "papers. 'new_search' if it needs papers that were not shown: a different topic, "
    "a different industry or application, or other / more / newer papers.\n"
    "2. standalone_query: rewrite the new message as one question that makes sense "
    "without the conversation. Replace words like 'it', 'that', 'this method' or "
    "'the second one' with what they refer to. Keep the user's intent and do not "
    "answer it. If the message already makes sense on its own, copy it.\n"
    "3. paper_ids: for same_papers, the exact paper_ids the message is about. Leave "
    "it empty if the message is about all the shown papers or for new_search.\n"
    "4. wants_other_papers: true only if the user asks for other, more, or "
    "different papers than the ones shown.\n\n"
    "Examples, after a question about reducing LLM hallucination:\n"
    "- 'Explain that in simpler terms' -> same_papers, 'Explain in simple terms how "
    "the shown papers reduce hallucination in large language models'\n"
    "- 'What does this mean for a bank?' -> same_papers, 'What do these methods for "
    "reducing LLM hallucination mean for a bank?'\n"
    "- 'What about in healthcare?' -> new_search, 'How can LLM hallucination be "
    "reduced in healthcare applications?'\n"
    "- 'Are there other approaches?' -> new_search, wants_other_papers true, "
    "'What other approaches reduce hallucination in large language models?'\n"
    "- 'How do diffusion models work?' -> new_search, 'How do diffusion models work?'\n\n"
    "Conversation so far:\n{history}\n\n"
    "Papers shown in the last answer:\n{papers}"
)


# ---------------------------------------------------------------------------
# What plan_follow_up returns
# ---------------------------------------------------------------------------

@dataclass
class FollowUpPlan:
    route: str
    standalone_query: str
    # same_papers: which shown papers to answer from (empty = all of them).
    focus_paper_ids: list[str] = field(default_factory=list)
    # new_search: papers to leave out of the new results.
    exclude_paper_ids: list[str] = field(default_factory=list)
    # Why this route was picked: no_history, explicit_reference, llm,
    # llm_failed. Handy when checking the agent's behaviour in evaluation.
    reason: str = ""


# A function that takes the prompt inputs and returns a FollowUpDecision
# (or None / raises if the model's output could not be parsed).
DecideFn = Callable[[dict], Optional[FollowUpDecision]]


# ---------------------------------------------------------------------------
# Spotting explicit references to the papers that were shown
# ---------------------------------------------------------------------------

_ORDINALS = {
    "first": 1, "1st": 1, "second": 2, "2nd": 2, "third": 3, "3rd": 3,
    "fourth": 4, "4th": 4, "fifth": 5, "5th": 5,
}
_PAPER_NOUN = r"(?:paper|one|study|result|article|source)"
_ORDINAL_REF = re.compile(
    r"\b(first|1st|second|2nd|third|3rd|fourth|4th|fifth|5th|last)\s+" + _PAPER_NOUN + r"\b",
    re.IGNORECASE,
)
# "paper 2", "paper #2", "result 3", "#2", "[2]"
_NUMBER_REF = re.compile(
    r"(?:\b(?:paper|result|source|study|article)\s*#?\s*|#|\[)([1-9])\b",
    re.IGNORECASE,
)
_ARXIV_ID = re.compile(r"\b(\d{4}\.\d{4,5})(?:v\d+)?\b")
_ALL_PAPERS_REF = re.compile(
    r"\b(?:these|those|the above|both|all (?:three|3|the))\s+"
    r"(?:papers|studies|results|articles|sources)\b",
    re.IGNORECASE,
)
# "other papers", "more studies", "similar work", "papers related to" ...
_OTHER_PAPERS = re.compile(
    r"\b(?:other|more|similar|related|different|additional|newer|recent)\s+"
    r"(?:papers?|studies|study|work|research|approaches|methods|techniques|results)\b"
    r"|\b(?:papers?|studies|work|research)\s+(?:similar|related)\s+to\b",
    re.IGNORECASE,
)


def _base_id(paper_id: str) -> str:
    """'2610.33997v1' -> '2610.33997', so IDs match with or without a version."""
    return re.sub(r"v\d+$", "", paper_id.strip())


def _normalize(text: str) -> str:
    return " ".join(text.lower().split())


def asks_for_other_papers(question: str) -> bool:
    return bool(_OTHER_PAPERS.search(question))


def find_paper_references(question: str, papers: list[dict]) -> list[str]:
    """paper_ids of the shown papers that the question explicitly points at.

    `papers` is the list shown to the user, in the order it was shown
    (so "paper 2" / "the second paper" means papers[1]).
    """
    if not papers:
        return []

    found: list[str] = []

    def add(index: int):
        if 0 <= index < len(papers) and papers[index]["paper_id"] not in found:
            found.append(papers[index]["paper_id"])

    for match in _ORDINAL_REF.finditer(question):
        word = match.group(1).lower()
        add(len(papers) - 1 if word == "last" else _ORDINALS[word] - 1)

    for match in _NUMBER_REF.finditer(question):
        add(int(match.group(1)) - 1)

    shown_ids = {_base_id(p["paper_id"]): i for i, p in enumerate(papers)}
    for match in _ARXIV_ID.finditer(question):
        if match.group(1) in shown_ids:
            add(shown_ids[match.group(1)])

    normalized_question = _normalize(question)
    for i, paper in enumerate(papers):
        title = _normalize(paper.get("title") or "")
        if len(title) >= 15 and title in normalized_question:
            add(i)

    if not found and _ALL_PAPERS_REF.search(question):
        found = [p["paper_id"] for p in papers]

    return found


def _with_paper_names(question: str, paper_ids: list[str], papers: list[dict]) -> str:
    """Add the referenced titles so the question makes sense on its own."""
    by_id = {p["paper_id"]: p for p in papers}
    names = "; ".join(f"'{by_id[pid]['title']}' [{pid}]" for pid in paper_ids if pid in by_id)
    return f"{question} (about: {names})" if names else question


# ---------------------------------------------------------------------------
# Formatting the history for the LLM
# ---------------------------------------------------------------------------

def format_history(history: list[dict], max_turns: int = MAX_HISTORY_TURNS) -> str:
    lines = []
    for turn in history[-max_turns:]:
        answer = (turn.get("answer") or "").strip()
        if len(answer) > MAX_ANSWER_CHARS_IN_PROMPT:
            answer = answer[:MAX_ANSWER_CHARS_IN_PROMPT].rstrip() + " ..."
        lines.append(f"User: {turn.get('original_query') or turn.get('query', '')}")
        lines.append(f"Assistant: {answer or '(no answer)'}")
    return "\n".join(lines) or "(none)"


def format_papers(papers: list[dict]) -> str:
    if not papers:
        return "(none)"
    return "\n".join(
        f"[{rank}] paper_id: {p['paper_id']} | title: {p.get('title', '')}"
        for rank, p in enumerate(papers, start=1)
    )


# ---------------------------------------------------------------------------
# The decision
# ---------------------------------------------------------------------------

def seen_paper_ids(history: list[dict]) -> list[str]:
    """Every paper shown so far in the conversation, oldest first."""
    seen: list[str] = []
    for turn in history:
        for paper in turn.get("papers", []):
            if paper["paper_id"] not in seen:
                seen.append(paper["paper_id"])
    return seen


def plan_follow_up(question: str, history: list[dict], decide: Optional[DecideFn] = None) -> FollowUpPlan:
    """Decide how to handle `question` given the earlier turns in `history`."""
    question = question.strip()
    if not history:
        return FollowUpPlan(NEW_SEARCH, question, reason="no_history")

    last = history[-1]
    papers = last.get("papers", [])
    wants_other = asks_for_other_papers(question)

    # Fast path: the user pointed at shown papers directly. No LLM call needed.
    if papers and not wants_other:
        referenced = find_paper_references(question, papers)
        if referenced:
            if len(referenced) < len(papers):
                standalone = _with_paper_names(question, referenced, papers)
                focus = referenced
            else:
                # "these papers" says which papers but not what about, so
                # carry over the previous question for section scoring and
                # for the answer prompt.
                standalone = f"{question} (follow-up to: {last.get('query', '')})"
                focus = []
            return FollowUpPlan(
                SAME_PAPERS, standalone, focus_paper_ids=focus, reason="explicit_reference"
            )

    decision = None
    if decide is not None:
        try:
            decision = decide({
                "history": format_history(history),
                "papers": format_papers(papers),
                "question": question,
            })
        except Exception:
            decision = None

    if decision is None:
        # The model gave nothing usable. A short follow-up like "what about
        # healthcare?" means little on its own, so search with the previous
        # question added. A longer one is most likely a full question already.
        standalone = question
        if len(question.split()) <= SHORT_FOLLOW_UP_WORDS:
            standalone = f"{last.get('query', '')} {question}".strip()
        return FollowUpPlan(
            NEW_SEARCH,
            standalone,
            exclude_paper_ids=seen_paper_ids(history) if wants_other else [],
            reason="llm_failed",
        )

    standalone = decision.standalone_query.strip() or question
    wants_other = wants_other or decision.wants_other_papers

    if decision.route == SAME_PAPERS and papers and not wants_other:
        shown = {_base_id(p["paper_id"]): p["paper_id"] for p in papers}
        focus = []
        for pid in decision.paper_ids:
            real_id = shown.get(_base_id(pid))
            if real_id and real_id not in focus:
                focus.append(real_id)
        if len(focus) == len(papers):
            focus = []
        return FollowUpPlan(SAME_PAPERS, standalone, focus_paper_ids=focus, reason="llm")

    return FollowUpPlan(
        NEW_SEARCH,
        standalone,
        exclude_paper_ids=seen_paper_ids(history) if wants_other else [],
        reason="llm",
    )


# ---------------------------------------------------------------------------
# Remembering a turn, and suggesting what to ask next
# ---------------------------------------------------------------------------

def make_turn(result: dict, history: list[dict]) -> dict:
    """What run_agent keeps about one question-and-answer.

    `papers` is the list shown to the user, so "paper 2" in the next question
    means the same paper the user saw as #2. A same_papers turn keeps the
    previous list even when it only answered about one of them, so the
    numbering doesn't shift under the user.
    """
    answered = bool((result.get("answer") or "").strip())
    if result.get("route") == SAME_PAPERS and history:
        papers = history[-1].get("papers", [])
        sections = history[-1].get("paper_sections", {})
    elif answered:
        papers = [dict(chunk) for chunk in result.get("retrieved_chunks", [])]
        sections = result.get("paper_sections", {})
    else:
        # Nothing was answered (e.g. the question was rejected or retrieval
        # confidence was too low), so there are no papers to follow up on.
        papers, sections = [], {}

    return {
        "original_query": result.get("original_query") or result.get("query", ""),
        "query": result.get("query", ""),
        "route": result.get("route", NEW_SEARCH),
        "answer": result.get("answer", ""),
        "citations": result.get("citations", []),
        "papers": papers,
        "paper_sections": sections,
    }


def add_turn(history: list[dict], turn: dict, max_turns: int = MAX_HISTORY_TURNS) -> list[dict]:
    return (list(history) + [turn])[-max_turns:]


def suggest_follow_ups(history: list[dict], limit: int = 3) -> list[str]:
    """Follow-up questions to offer after the latest answer (history[-1]).

    They are built from templates rather than by the LLM, so they cost no extra
    time and each one takes a known route: the paper questions are same_papers
    (no LLM call) and the last one asks for other papers (new_search, leaving
    out the papers already shown). Questions already asked are skipped.
    """
    if not history or not history[-1].get("papers"):
        return []
    turn = history[-1]
    papers = turn["papers"]
    asked = {_normalize(t.get("original_query", "")) for t in history}

    # Most-cited paper first, then the rest in the order they were shown.
    cited = [c["paper_id"] for c in turn.get("citations", [])]
    ranks = sorted(
        range(1, len(papers) + 1),
        key=lambda rank: (-cited.count(papers[rank - 1]["paper_id"]), rank),
    )

    candidates = [f"What are the limitations of paper {rank}?" for rank in ranks]
    if len(papers) > 1:
        candidates.insert(1, "How could a business apply the findings of these papers?")
    else:
        candidates.insert(1, "How could a business apply the findings of paper 1?")

    suggestions = [q for q in candidates if _normalize(q) not in asked][: limit - 1]
    suggestions.append("Are there other papers on this topic?")
    return suggestions
