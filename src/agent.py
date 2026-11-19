"""Research agent: answer a question from the papers and passages the two searches return.

START -> contextualize_query
  -> same_papers: reuse_evidence -> assemble_context -> generate_answer -> remember_turn -> END
  -> new_search: validate_question
       -> (if valid) retrieve_papers and retrieve_chunks (in parallel)
            -> assemble_context -> generate_answer -> remember_turn -> END
       -> (if invalid) remember_turn -> END

Conversation memory: the same thread_id reuses a short history stored by LangGraph's
SqliteSaver in chat_memory.sqlite. Each turn keeps the question, answer, and the
papers/passages the LLM was given. Follow-ups are planned in src/followup.py.
"""

import sqlite3
import uuid
import re
from collections.abc import Callable
from dataclasses import asdict, replace
from functools import cache, partial
from typing import Any, Optional, TypedDict
from typing import Optional

from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import Runnable
from langchain_ollama import ChatOllama
from langgraph.checkpoint.sqlite import SqliteSaver
from langgraph.graph import END, START, StateGraph
from pydantic import BaseModel, Field
from qdrant_client import QdrantClient
from tokenizers import Tokenizer


from src.followup import (
    FOLLOW_UP_SYSTEM_PROMPT,
    NEW_SEARCH,
    SAME_PAPERS,
    FollowUpDecision,
    MAX_HISTORY_TURNS,
    add_turn,
    plan_follow_up,
    suggest_follow_ups,
)
from src.prompts import answer_prompt
from src.retrieval import ChunkHit, PaperHit, search_chunks, search_papers
from src.vector_store import get_client

LLM_MODEL = "qwen2.5:7b"
VALIDATION_MODEL = "qwen2.5:1.5b"
# Hugging Face copy of the tokenizer qwen2.5:7b uses, so the budget counts the tokens the LLM actually reads.
LLM_TOKENIZER = "Qwen/Qwen2.5-7B-Instruct"
# Ollama's default context is much smaller than what qwen2.5 supports, and anything past it is silently dropped.
LLM_CONTEXT_TOKENS = 32768
# The local LLM reads about 200 tokens a second, so every 1,000 tokens of context adds about 5 s per answer.
CONTEXT_TOKENS = 6000
CHAT_DB = "chat_memory.sqlite"
NO_EVIDENCE_ANSWER = "No relevant papers or passages were found for this question."
NOT_AI_ANSWER = (
    "I can only answer questions about AI research. Try asking about a specific AI topic."
)


class GenerationError(RuntimeError):
    """The LLM failed or returned no answer."""


class QuestionCheck(BaseModel):
    is_about_ai: bool = Field(
        description="True if the question is about AI or machine learning research."
    )
    is_valid: bool = Field(
        description="True if the question is about AI research and has a clear objective."
    )
    suggested_query: str = Field(
        description="If not valid but about AI, a clearer version of the question. Otherwise empty."
    )


validation_prompt = ChatPromptTemplate.from_messages([
    (
        "system",
        "You check questions before they are used to search AI research papers.\n\n"
        "A question is valid if:\n"
        "1. It is about AI research.\n"
        "2. It has a clear objective (it is clear what the user wants to know).\n\n"
        "If the question is about AI but unclear, write a clearer version in suggested_query.\n"
        "If the question is not about AI, leave suggested_query empty.\n\n"
        "Examples:\n"
        "- 'What methods reduce hallucination in LLMs?' -> valid\n"
        "- 'yo why does ai make stuff up' -> not valid, suggest "
        "'What causes hallucination in large language models?'\n"
        "- 'AI stuff' -> not valid, suggest 'What are recent advances in large language models?'\n"
        "- 'What is the best pizza place?' -> not valid, no suggestion",
    ),
    ("human", "{question}"),
])

follow_up_prompt = ChatPromptTemplate.from_messages([
    ("system", FOLLOW_UP_SYSTEM_PROMPT),
    ("human", "New message: {question}"),
])


class AgentState(TypedDict):
    # Standalone question every node works with. For a follow-up this is the rewrite.
    query: str
    original_query: str
    is_valid: bool
    suggested_query: str
    route: str
    focus_paper_ids: list[str]
    exclude_paper_ids: list[str]
    paper_hits: list[dict[str, Any]]  # PaperHit fields
    chunk_hits: list[dict[str, Any]]  # ChunkHit fields
    context: str
    answer: str
    follow_up_suggestions: list[str]
    # Carried across run_agent calls with the same thread_id by the checkpointer.
    # Omitted from each invoke input so prior turns are not wiped.
    history: list[dict[str, Any]]
    citations: list[dict[str, Any]]


def _join(*parts: Optional[str]) -> str:
    return " | ".join(part for part in parts if part)


def _format_paper(hit: dict[str, Any]) -> str:
    header = _join(f"[P{hit['rank']}]", hit["arxiv_id"] and f"arXiv:{hit['arxiv_id']}", hit["title"])
    return f"{header}\nAbstract: {hit['abstract']}" if hit["abstract"] else header


def _format_chunk(hit: dict[str, Any]) -> str:
    header = _join(
        f"[C{hit['rank']}]",
        hit["arxiv_id"] and f"arXiv:{hit['arxiv_id']}",
        hit["title"],
        hit["breadcrumb"] and f"Section: {' > '.join(hit['breadcrumb'])}",
        hit["content_type"] and f"Type: {hit['content_type']}",
    )
    return f"{header}\n{hit['content']}"

def extract_citations(answer: str, chunk_hits: list[dict]) -> list[dict]:
    cited_ranks = {
        int(rank) for rank in re.findall(r"\[C(\d+)\]", answer)
    }

    citations = []

    for hit in chunk_hits:
        if hit["rank"] not in cited_ranks:
            continue

        citations.append({
            "paper_id": hit["arxiv_id"],
            "title": hit["title"],
            "section": " > ".join(hit["breadcrumb"]),
            "url": f"https://arxiv.org/abs/{hit['arxiv_id']}",
            "chunk_id": hit["chunk_id"],
            "citation_label": f"[C{hit['rank']}]",
        })

    return citations


def _papers_section(hits: list[dict[str, Any]]) -> str:
    return "## Papers\n\n" + ("\n\n".join(_format_paper(hit) for hit in hits) or "No papers found.")


def _passages_section(hits: list[dict[str, Any]]) -> str:
    return "## Passages\n\n" + ("\n\n".join(_format_chunk(hit) for hit in hits) or "No passages found.")


def format_context(paper_hits: list[dict[str, Any]], chunk_hits: list[dict[str, Any]]) -> str:
    """Papers and passages as two labelled sections; empty when both searches found nothing."""
    if not paper_hits and not chunk_hits:
        return ""
    return f"{_papers_section(paper_hits)}\n\n{_passages_section(chunk_hits)}"


@cache
def _tokenizer() -> Tokenizer:
    return Tokenizer.from_pretrained(LLM_TOKENIZER)


def count_tokens(text: str) -> int:
    return len(_tokenizer().encode(text).ids)


def _take_while(hits: list[dict[str, Any]], fits: Callable[[list[dict[str, Any]]], bool]) -> list[dict[str, Any]]:
    kept = []
    for hit in hits:
        if not fits(kept + [hit]):
            break
        kept.append(hit)
    return kept


def fit_to_budget(
    paper_hits: list[dict[str, Any]], chunk_hits: list[dict[str, Any]], max_tokens: int
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """The highest-ranked whole papers and passages whose formatted context fits in max_tokens.

    Papers get the share of the budget they take up in the full context; passages fill the rest.
    """
    if not paper_hits and not chunk_hits:
        return [], []
    paper_share = count_tokens(_papers_section(paper_hits)) / count_tokens(format_context(paper_hits, chunk_hits))
    papers = _take_while(paper_hits, lambda kept: count_tokens(_papers_section(kept)) <= paper_share * max_tokens)
    chunks = _take_while(chunk_hits, lambda kept: count_tokens(format_context(papers, kept)) <= max_tokens)
    return papers, chunks


def check_question(question: str) -> Optional[QuestionCheck]:
    """Ask the small model whether the question is usable. None means the check failed."""
    try:
        return (validation_prompt | get_validator_llm().with_structured_output(QuestionCheck)).invoke(
            {"question": question}
        )
    except Exception:
        return None


def decide_follow_up(inputs: dict) -> Optional[FollowUpDecision]:
    structured = get_llm().with_structured_output(FollowUpDecision, include_raw=True)
    return (follow_up_prompt | structured).invoke(inputs)["parsed"]


def contextualize_query(state: AgentState) -> dict:
    """Rewrite a follow-up as a standalone question and pick same_papers vs new_search."""
    plan = plan_follow_up(state["original_query"], state.get("history", []), decide_follow_up)
    return {
        "query": plan.standalone_query,
        "route": plan.route,
        "focus_paper_ids": plan.focus_paper_ids,
        "exclude_paper_ids": plan.exclude_paper_ids,
    }


def route_after_contextualize(state: AgentState):
    # Questions about papers already shown skip validation (short ones like "explain paper 2").
    if state["route"] == SAME_PAPERS:
        return "reuse_evidence"
    return "validate_question"


def reuse_evidence(state: AgentState) -> dict:
    """Answer from the papers/passages already shown instead of searching again."""
    last = state["history"][-1]
    focus = set(state.get("focus_paper_ids") or [])
    papers = list(last.get("paper_hits") or [])
    chunks = list(last.get("chunk_hits") or [])
    if focus:
        papers = [p for p in papers if p["arxiv_id"] in focus]
        chunks = [c for c in chunks if c["arxiv_id"] in focus]
    return {"paper_hits": papers, "chunk_hits": chunks, "is_valid": True}


def validate_question(state: AgentState) -> dict:
    """Same behaviour as main: reject off-topic or unclear questions before search.

    Unclear AI questions get a "Did you mean" message and stop; the clearer wording is
    not substituted automatically. If the checker fails, the question is let through.
    """
    result = check_question(state["query"])

    if result is None:
        return {"is_valid": True, "suggested_query": ""}

    if not result.is_about_ai:
        return {"is_valid": False, "suggested_query": "", "answer": NOT_AI_ANSWER}

    if result.is_valid:
        return {"is_valid": True, "suggested_query": ""}

    if result.suggested_query:
        message = (
            'Your question wasn\'t quite clear. Did you mean: "'
            + result.suggested_query
            + '" Try asking that instead.'
        )
    else:
        message = NOT_AI_ANSWER

    return {
        "is_valid": False,
        "suggested_query": result.suggested_query,
        "answer": message,
    }


def route_after_validation(state: AgentState):
    if state["is_valid"]:
        return "begin_search"
    return "remember_turn"


def begin_search(state: AgentState) -> dict:
    """Gate so both searches start only after the question is validated."""
    return {}


def _drop_excluded_papers(hits: list[PaperHit], exclude: set[str]) -> list[PaperHit]:
    if not exclude:
        return hits
    kept = [hit for hit in hits if hit.arxiv_id not in exclude]
    return [replace(hit, rank=i) for i, hit in enumerate(kept, start=1)]


def _drop_excluded_chunks(hits: list[ChunkHit], exclude: set[str]) -> list[ChunkHit]:
    if not exclude:
        return hits
    kept = [hit for hit in hits if hit.arxiv_id not in exclude]
    return [replace(hit, rank=i) for i, hit in enumerate(kept, start=1)]


def retrieve_papers(state: AgentState, client: QdrantClient) -> dict:
    exclude = set(state.get("exclude_paper_ids") or [])
    # Ask for extras so there are still results after dropping papers already shown.
    hits = search_papers(client, state["query"], top_k=10 + len(exclude))
    return {"paper_hits": [asdict(hit) for hit in _drop_excluded_papers(hits, exclude)[:10]]}


def retrieve_chunks(state: AgentState, client: QdrantClient) -> dict:
    exclude = set(state.get("exclude_paper_ids") or [])
    hits = search_chunks(client, state["query"], top_k=20 + len(exclude))
    return {"chunk_hits": [asdict(hit) for hit in _drop_excluded_chunks(hits, exclude)[:20]]}


def assemble_context(state: AgentState) -> dict:
    return {"context": format_context(*fit_to_budget(state["paper_hits"], state["chunk_hits"], CONTEXT_TOKENS))}


def generate_answer(state: AgentState, llm: Runnable) -> dict:
    if not state["context"]:
        return {"answer": NO_EVIDENCE_ANSWER, "citations": []}

    try:
        message = (answer_prompt | llm).invoke({
            "question": state["query"],
            "context": state["context"],
        })
    except Exception as error:
        raise GenerationError(
            f"The LLM couldn't generate an answer: {error}"
        ) from error

    if not message.content.strip():
        raise GenerationError("The LLM returned an empty answer")

    answer = message.content
    _, chunk_hits = hits_in_context(state)
    citations = extract_citations(answer, chunk_hits)

    return {
        "answer": answer,
        "citations": citations,
    }


def _papers_for_followup(paper_hits: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Numbered list follow-up routing uses for 'paper 2' / suggestions."""
    return [{"paper_id": hit["arxiv_id"], "title": hit["title"]} for hit in paper_hits]


def make_turn(state: AgentState) -> dict[str, Any]:
    """What one question-and-answer keeps for later turns on this thread."""
    history = state.get("history", [])
    if state.get("route") == SAME_PAPERS and history:
        # Keep the previous shown list so "paper 2" numbering does not shift.
        papers = history[-1].get("papers", [])
        paper_hits = history[-1].get("paper_hits", [])
        chunk_hits = history[-1].get("chunk_hits", [])
    elif state["is_valid"] and state.get("context"):
        paper_hits, chunk_hits = hits_in_context(state)
        papers = _papers_for_followup(paper_hits)
    else:
        # Rejected or no evidence: nothing for a later "paper 2" to point at.
        papers, paper_hits, chunk_hits = [], [], []
    return {
        "original_query": state.get("original_query") or state["query"],
        "query": state["query"],
        "route": state.get("route", NEW_SEARCH),
        "answer": state.get("answer", ""),
        "is_valid": state["is_valid"],
        "papers": papers,
        "paper_hits": paper_hits,
        "chunk_hits": chunk_hits,
        "citations": state.get("citations", []),
    }


def remember_turn(state: AgentState) -> dict:
    """Append this turn to the thread history and offer follow-up suggestions."""
    new_history = add_turn(state.get("history", []), make_turn(state), max_turns=MAX_HISTORY_TURNS)
    return {
        "history": new_history,
        "follow_up_suggestions": suggest_follow_ups(new_history),
    }


def get_llm() -> ChatOllama:
    return ChatOllama(model=LLM_MODEL, temperature=0, num_ctx=LLM_CONTEXT_TOKENS)


def get_validator_llm() -> ChatOllama:
    return ChatOllama(model=VALIDATION_MODEL, temperature=0)


_checkpointer: Optional[SqliteSaver] = None


def get_checkpointer() -> SqliteSaver:
    """One SqliteSaver for the process; the DB file is local and gitignored."""
    global _checkpointer
    if _checkpointer is None:
        connection = sqlite3.connect(CHAT_DB, check_same_thread=False)
        _checkpointer = SqliteSaver(connection)
        _checkpointer.setup()
    return _checkpointer


def _initial_state(query: str) -> dict[str, Any]:
    """Per-question fields only. history is left out so the checkpointer keeps prior turns."""
    return {
        "query": query,
        "original_query": query,
        "is_valid": True,
        "suggested_query": "",
        "route": NEW_SEARCH,
        "focus_paper_ids": [],
        "exclude_paper_ids": [],
        "paper_hits": [],
        "chunk_hits": [],
        "context": "",
        "answer": "",
        "follow_up_suggestions": [],
        "citations": [],
    }


def build_graph(client: QdrantClient, llm: Runnable, checkpointer: Optional[SqliteSaver] = None):
    """Both searches share one client: local Qdrant allows only one client per storage folder."""
    graph = StateGraph(AgentState)
    graph.add_node("contextualize_query", contextualize_query)
    graph.add_node("validate_question", validate_question)
    graph.add_node("reuse_evidence", reuse_evidence)
    graph.add_node("begin_search", begin_search)
    graph.add_node("retrieve_papers", partial(retrieve_papers, client=client))
    graph.add_node("retrieve_chunks", partial(retrieve_chunks, client=client))
    graph.add_node("assemble_context", assemble_context)
    graph.add_node("generate_answer", partial(generate_answer, llm=llm))
    graph.add_node("remember_turn", remember_turn)
    graph.add_edge(START, "contextualize_query")
    graph.add_conditional_edges(
        "contextualize_query",
        route_after_contextualize,
        {"reuse_evidence": "reuse_evidence", "validate_question": "validate_question"},
    )
    graph.add_conditional_edges("validate_question", route_after_validation)
    graph.add_edge("reuse_evidence", "assemble_context")
    graph.add_edge("begin_search", "retrieve_papers")
    graph.add_edge("begin_search", "retrieve_chunks")
    graph.add_edge(["retrieve_papers", "retrieve_chunks"], "assemble_context")
    graph.add_edge("assemble_context", "generate_answer")
    graph.add_edge("generate_answer", "remember_turn")
    graph.add_edge("remember_turn", END)
    return graph.compile(checkpointer=checkpointer)


def run_agent(query: str, thread_id: Optional[str] = None) -> AgentState:
    """Answer one question. Pass the same thread_id to ask follow-ups.

    Raises ValueError for a blank query, RetrievalError if Qdrant fails, GenerationError if the LLM fails.
    """
    query = query.strip()
    if not query:
        raise ValueError("The question is blank")
    if thread_id is None:
        thread_id = str(uuid.uuid4())
    client = get_client()
    try:
        result = build_graph(client, get_llm(), get_checkpointer()).invoke(
            _initial_state(query),
            {"configurable": {"thread_id": thread_id}},
        )
    finally:
        client.close()
    result["thread_id"] = thread_id
    return result


def hits_in_context(state: AgentState) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """The papers and passages the LLM was given, out of everything the searches returned."""
    return fit_to_budget(state["paper_hits"], state["chunk_hits"], CONTEXT_TOKENS)


def _print_result(state: AgentState):
    if state["query"] != state["original_query"]:
        print(f"\n(Understood as: {state['query']})")
    if state.get("route") == SAME_PAPERS:
        print("(Answered from papers shown earlier)")
    papers, chunks = hits_in_context(state)
    print(f"\n===== Answer =====\n{state['answer']}")
    print(f"\n===== Papers given to the LLM ({len(papers)} of {len(state['paper_hits'])} retrieved) =====")
    for hit in papers:
        print(f"{hit['rank']:3}. {hit['arxiv_id']}  {hit['title']}")
    print(f"\n===== Passages given to the LLM ({len(chunks)} of {len(state['chunk_hits'])} retrieved) =====")
    for hit in chunks:
        print(f"{hit['rank']:3}. {hit['chunk_id']}  {' > '.join(hit['breadcrumb'])}")
    print(f"\nContext: {count_tokens(state['context']):,} tokens (budget {CONTEXT_TOKENS:,})")
    if state.get("follow_up_suggestions"):
        print("\nYou could ask next:")
        for number, suggestion in enumerate(state["follow_up_suggestions"], start=1):
            print(f"  {number}. {suggestion}")


if __name__ == "__main__":
    # python -m src.agent ["question"]
    # Without a question, asks for one question after another (models and Qdrant load only once).
    import sys

    client = get_client()
    thread_id = str(uuid.uuid4())
    suggestions: list[str] = []
    try:
        agent = build_graph(client, get_llm(), get_checkpointer())
        config = {"configurable": {"thread_id": thread_id}}
        if len(sys.argv) > 1:
            _print_result(agent.invoke(_initial_state(sys.argv[1]), config))
        else:
            while True:
                prompt = "\nResearch question (blank to quit"
                if suggestions:
                    prompt += ", 1-3 for a suggestion, /new for a new thread"
                prompt += "): "
                text = input(prompt).strip()
                if not text:
                    break
                if text == "/new":
                    thread_id = str(uuid.uuid4())
                    config = {"configurable": {"thread_id": thread_id}}
                    suggestions = []
                    print("(Started a new conversation)")
                    continue
                if text.isdigit() and suggestions and 1 <= int(text) <= len(suggestions):
                    text = suggestions[int(text) - 1]
                    print(f"(Asking: {text})")
                state = agent.invoke(_initial_state(text), config)
                _print_result(state)
                suggestions = state.get("follow_up_suggestions") or []
    except (KeyboardInterrupt, EOFError):
        pass
    finally:
        client.close()
