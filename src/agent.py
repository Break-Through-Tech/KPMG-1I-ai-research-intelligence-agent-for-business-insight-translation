import requests
from langgraph.graph import StateGraph, START, END
from typing import TypedDict
from langchain_ollama import ChatOllama
from langchain_core.prompts import ChatPromptTemplate
from src.arxiv_html import fetch_sections
from src.embeddings import get_model
from src.vector_store import get_client, semantic_search
from src.followup import (
    FOLLOW_UP_SYSTEM_PROMPT,
    NEW_SEARCH,
    SAME_PAPERS,
    FollowUpDecision,
    add_turn,
    make_turn,
    plan_follow_up,
    suggest_follow_ups,
)
from pydantic import BaseModel, Field
import re

import sqlite3
import uuid

from typing import Optional

from typing import Annotated
from langchain_core.messages import (
    AnyMessage,
    HumanMessage,
    AIMessage,
)
from langgraph.graph.message import add_messages
from langgraph.checkpoint.sqlite import SqliteSaver

# Ollama's default context is much smaller than what qwen2.5 supports, and
# anything past it is silently dropped.
llm = ChatOllama(model="qwen2.5:7b", temperature=0, num_ctx=32768)
small_llm = ChatOllama(model="qwen2.5:1.5b", temperature=0)

# Keeps the sections of all 3 papers, plus the prompt and answer, inside num_ctx.
MAX_SECTION_CHARS_PER_PAPER = 24_000
# So one long section can't use up a paper's whole budget.
MAX_CHARS_PER_SECTION = MAX_SECTION_CHARS_PER_PAPER // 3

CONCLUSION_TITLE = re.compile(r"conclu", re.IGNORECASE)
# all-MiniLM-L6-v2 only reads about the first 1000 characters of a text.
PIECE_CHARS = 1000
MAX_PIECES_PER_SECTION = 20
HIGH_CONF = 0.40
LOW_CONF = 0.20
#creating the threshold
def assess_confidence(chunks: list[dict]) -> tuple[str, float]:
    top = max((c["score"] for c in chunks), default=0.0)
    if top >= HIGH_CONF:
        return "high", top
    if top >= LOW_CONF:
        return "medium", top
    return "low", top

class AgentState(TypedDict): #adding one field to the agent state for citation 
    # The standalone question every node works with. For a follow-up this is
    # the rewritten question, not what the user typed (see src/followup.py).
    query: str
    original_query: str
    is_valid: bool
    suggested_query: str
    document_ids: list[str]
    retrieved_chunks: list[dict]
    paper_sections: dict[str, list[dict]]
    answer: str
    citations: list[dict]
    confidence: str
    top_score: float

    # Stores conversation messages across turns
    messages: Annotated[list[AnyMessage], add_messages]

    # Follow-up handling (src/followup.py). `history` is saved by the
    # checkpointer with the rest of the state, so it carries over between
    # run_agent calls with the same thread_id. Unlike `messages`, it keeps the
    # papers shown and their fetched sections, which follow-ups reuse.
    history: list[dict]
    route: str
    focus_paper_ids: list[str]
    exclude_paper_ids: list[str]
    follow_up_suggestions: list[str]


class Claim(BaseModel):#output structure 
    statement: str = Field(
        description="One factual claim supported by a retrieved paper."
    )
    paper_id: str = Field(
        description="The exact paper_id of the supporting paper."
    )
    supporting_quote: str = Field(
        description="A short, word-for-word quote from the paper's abstract."
    )

class LLMAnswer(BaseModel):
    summary: str = Field(
        description="A short answer to the user's research question."
    )
    claims: list[Claim] = Field(
        description="The claims supporting the answer, with citations."
    )


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
    ("system",
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
     "- 'What is the best pizza place?' -> not valid, no suggestion"),
    ("human", "{question}"),
])

prompt = ChatPromptTemplate.from_messages([
    ("system",
     "You are a helpful research assistant who answers questions about AI "
     "papers using the excerpts below.\n\n"
     "1. Your goal is to be useful. Read every excerpt and find whatever "
     "helps answer the question: direct answers, related techniques, "
     "ideas that could be adapted, or background that gives useful "
     "context.\n"
     "2. You may connect ideas across excerpts and reason about how they "
     "apply to the question. When a paper does not answer the question "
     "directly, explain how its idea could be relevant or adapted. Say "
     "clearly when something is your inference rather than what the paper "
     "states, for example: 'This paper does not study routing, but its "
     "approach of ... could be applied by ...'.\n"
     "3. Stay grounded. Every claim about what a paper says or found must "
     "match its excerpt. Do not invent results, numbers, or methods.\n"
     "4. Always begin with the most useful information. Never open by "
     "saying the excerpts do not address the question, and do not refuse "
     "to answer.\n"
     "5. Draw on several papers when they have something to offer, and "
     "skip any that have nothing to do with the question.\n"
     "6. Support each claim about a paper with its paper_id and a short, "
     "exact quote from that paper's excerpt. Inferences do not need a "
     "quote, but they should point to the paper they build on.\n"
     "7. If important parts of the question are not covered, finish with "
     "one short sentence saying what is missing.\n\n"
     "Excerpts:\n{context}"),
    ("human", "{question}"),
])

follow_up_prompt = ChatPromptTemplate.from_messages([
    ("system", FOLLOW_UP_SYSTEM_PROMPT),
    ("human", "New message: {question}"),
])

def decide_follow_up(inputs: dict) -> Optional[FollowUpDecision]:
    structured_llm = llm.with_structured_output(FollowUpDecision, include_raw=True)
    return (follow_up_prompt | structured_llm).invoke(inputs)["parsed"]

def contextualize_query(state: AgentState):
    """Rewrite a follow-up as a standalone question and pick its route.

    One step does both: rewriting the question using the conversation, and
    deciding whether to answer from the papers already shown or search again.
    """
    plan = plan_follow_up(state["original_query"], state.get("history", []), decide_follow_up)
    return {
        "query": plan.standalone_query,
        "route": plan.route,
        "focus_paper_ids": plan.focus_paper_ids,
        "exclude_paper_ids": plan.exclude_paper_ids,
    }

def route_after_contextualize(state: AgentState):
    # A question about papers already shown is on topic by definition, so it
    # skips validation (which could reject short ones like "explain paper 2").
    if state["route"] == SAME_PAPERS:
        return "reuse_papers"
    return "validate_question"

def reuse_papers(state: AgentState):
    """Answer from the papers already shown instead of searching again."""
    last = state["history"][-1]
    focus = set(state["focus_paper_ids"])
    papers = [p for p in last["papers"] if not focus or p["paper_id"] in focus]
    paper_ids = {p["paper_id"] for p in papers}
    return {
        "document_ids": [p["paper_id"] for p in papers],
        "retrieved_chunks": papers,
        "paper_sections": {
            pid: sections
            for pid, sections in last["paper_sections"].items()
            if pid in paper_ids
        },
    }

def validate_question(state: AgentState):
    checker = validation_prompt | small_llm.with_structured_output(QuestionCheck)
    try:
        result = checker.invoke({"question": state["query"]})
    except Exception:
        result = None

        # If the check fails, let the question through instead of blocking the user.
    if result is None:
        return {"is_valid": True, "suggested_query": ""}

    if not result.is_about_ai:
        message = "I can only answer questions about AI research. Try asking about a specific AI topic."
        return {
            "is_valid": False,
            "suggested_query": "",
            "answer": message,
            "messages": [AIMessage(content=message)],
        }

    if result.is_valid:
        return {"is_valid": True, "suggested_query": ""}
    
    if result.suggested_query:
        message = "Your question wasn't quite clear. Did you mean: \"" + result.suggested_query + "\" Try asking that instead."
    else:
        message = "I can only answer questions about AI research. Try asking about a specific AI topic."

    return {
        "is_valid": False,
        "suggested_query": result.suggested_query,
        "answer": message,
        "messages": [AIMessage(content=message)],
    }


def route_after_validation(state: AgentState):
    if state["is_valid"]:
        return "retrieve_vectors"
    return "remember_turn"

def retrieve_vectors(state: AgentState):
    top_k = 3
    exclude = set(state.get("exclude_paper_ids") or [])
    client = get_client("./qdrant_data")
    try:
        # Ask for extra results so there are still top_k left after dropping
        # papers the user has already seen.
        results = semantic_search(client, query=state["query"], top_k=top_k + len(exclude))
    finally:
        client.close()
    results = [r for r in results if r["paper_id"] not in exclude][:top_k]
    confidence, top_score = assess_confidence(results)
    return {
        "document_ids": [result["paper_id"] for result in results],
        "retrieved_chunks": results,
        "confidence": confidence,
        "top_score": top_score,
    }

def retrieve_sections(state: AgentState):
    paper_sections = {}
    for arxiv_id in state["document_ids"]:
        try:
            sections = fetch_sections(arxiv_id)
        except requests.RequestException:
            continue
        if sections:
            paper_sections[arxiv_id] = [
                {"title": section.title, "text": section.text} for section in sections
            ]
    return {"paper_sections": paper_sections}

def _embed(texts: list[str]):
    return get_model().encode(texts, normalize_embeddings=True, show_progress_bar=False)

def _score_sections(sections: list[dict], query: str) -> list[float]:
    """How similar each section is to the query, judged by its best-matching piece."""
    pieces, owners = [], []
    for i, section in enumerate(sections):
        text = f"{section['title']}\n{section['text']}"
        for start in range(0, min(len(text), PIECE_CHARS * MAX_PIECES_PER_SECTION), PIECE_CHARS):
            pieces.append(text[start:start + PIECE_CHARS])
            owners.append(i)
    vectors = _embed([query] + pieces)
    scores = [-1.0] * len(sections)
    for owner, score in zip(owners, vectors[1:] @ vectors[0]):
        scores[owner] = max(scores[owner], float(score))
    return scores

def _format_sections(sections: list[dict], query: str, budget: int = MAX_SECTION_CHARS_PER_PAPER) -> str:
    """The conclusion, then the sections most similar to the query, in paper order."""
    if not sections:
        return ""
    scores = _score_sections(sections, query)
    priority = sorted(
        range(len(sections)),
        key=lambda i: (not CONCLUSION_TITLE.search(sections[i]["title"]), -scores[i]),
    )
    chosen = {}
    remaining = budget
    for i in priority:
        if remaining <= 0:
            break
        block = f"{sections[i]['title']}\n{sections[i]['text']}\n"
        chosen[i] = block[:min(MAX_CHARS_PER_SECTION, remaining)]
        remaining -= len(chosen[i])
    return "".join(chosen[i] for i in sorted(chosen))

def generate_answer(state: AgentState):
    context = ""
    for chunk in state["retrieved_chunks"]:
        paper_id = chunk["paper_id"]
        context += f"Title: {chunk['title']} (ID: {paper_id})\n"
        context += f"Abstract: {chunk['chunk_text']}\n"
        context += _format_sections(state["paper_sections"].get(paper_id, []), state["query"])
        context += "\n"

    inputs = {"context": context, "question": state["query"]}

    structured_llm = llm.with_structured_output(LLMAnswer, include_raw=True)
    output = (prompt | structured_llm).invoke(inputs)
    result = output["parsed"]

    if result is None:
        fallback = (prompt | llm).invoke(inputs)
        answer = fallback.content

        return {
            "answer": answer,
            "citations": [],
            "messages": [AIMessage(content=answer)],
        }
    answer = result.summary
    for claim in result.claims:
        answer += f"\n- {claim.statement} [{claim.paper_id}]"

    return {
        "answer": answer,
        "citations": [
            {
                "paper_id": claim.paper_id,
                "supporting_quote": claim.supporting_quote,
            }
            for claim in result.claims
        ],
        "messages": [AIMessage(content=answer)],
    }

def route_after_retrieval(state: AgentState):
    if state["confidence"] == "low":
        return END
    return "retrieve_sections"

def finish_low_confidence(state: AgentState):
    answer = (
        f"The papers found may not match your question very well. "
        f"The highest similarity score was {state['top_score']:.0%}. "
        "Try asking your question differently or using more specific terms."
    )

    return {
        "answer": answer,
        "citations": [],
        "messages": [AIMessage(content=answer)],
    }

def remember_turn(state: AgentState):
    """Add this question and answer to the saved history and suggest follow-ups.

    Every path through the graph ends here, so the history the checkpointer
    saves always includes the latest turn.
    """
    history = state.get("history", [])
    answered_from_papers = state["route"] == SAME_PAPERS or (
        state["is_valid"] and state["confidence"] != "low"
    )
    if not answered_from_papers:
        # The question was rejected or the papers found were a poor match, so
        # there is nothing for "paper 2" to refer to in the next question.
        state = {**state, "retrieved_chunks": [], "paper_sections": {}}
    new_history = add_turn(history, make_turn(state, history))
    return {
        "history": new_history,
        "follow_up_suggestions": suggest_follow_ups(new_history),
    }

builder = StateGraph(AgentState)

builder.add_node("contextualize_query", contextualize_query)
builder.add_node("validate_question", validate_question)
builder.add_node("reuse_papers", reuse_papers)
builder.add_node("retrieve_vectors", retrieve_vectors)
builder.add_node("retrieve_sections", retrieve_sections)
builder.add_node("generate_answer", generate_answer)
builder.add_node("finish_low_confidence", finish_low_confidence)
builder.add_node("remember_turn", remember_turn)

builder.add_edge(START, "contextualize_query")
builder.add_conditional_edges(
    "contextualize_query",
    route_after_contextualize,
    {"reuse_papers": "reuse_papers", "validate_question": "validate_question"},
)
builder.add_conditional_edges(
    "validate_question",
    route_after_validation,
    {"retrieve_vectors": "retrieve_vectors", "remember_turn": "remember_turn"},
)
builder.add_edge("reuse_papers", "generate_answer")
builder.add_conditional_edges(
    "retrieve_vectors",
    route_after_retrieval,
    {"retrieve_sections": "retrieve_sections", END: "finish_low_confidence"},
)
builder.add_edge("retrieve_sections", "generate_answer")
builder.add_edge("generate_answer", "remember_turn")
builder.add_edge("finish_low_confidence", "remember_turn")
builder.add_edge("remember_turn", END)

# Persistent conversation storage
connection = sqlite3.connect(
    "chat_memory.sqlite",
    check_same_thread=False,
)

checkpointer = SqliteSaver(connection)
checkpointer.setup()

graph = builder.compile(checkpointer=checkpointer)

def run_agent(query: str, thread_id: Optional[str] = None) -> AgentState:
    """Answer one question. Use the same thread_id to ask follow-ups.

    The checkpointer keeps `messages` and `history` for each thread_id. Every
    other field is per question, so it is reset here; otherwise the checkpoint
    would hand back the previous question's papers and citations whenever this
    one ends early (e.g. it fails validation).
    """
    if thread_id is None:
        thread_id = str(uuid.uuid4())

    config = {
        "configurable": {
            "thread_id": thread_id
        }
    }
    result = graph.invoke(
        {
            "query": query,
            "original_query": query,
            "messages": [
                HumanMessage(content=query)
            ],
            "is_valid": True,
            "suggested_query": "",
            "route": NEW_SEARCH,
            "focus_paper_ids": [],
            "exclude_paper_ids": [],
            "document_ids": [],
            "retrieved_chunks": [],
            "paper_sections": {},
            "answer": "",
            "citations": [],
            "confidence": "",
            "top_score": 0.0,
            "follow_up_suggestions": [],
        },
        config=config,
    )
    result["thread_id"] = thread_id
    return result

def show(result):
    if result["query"] != result["original_query"]:
        print(f"\n(Understood as: {result['query']})")

    if result["confidence"] == "low":
        print(f"\nWarning: low confidence ({result['top_score']:.0%}). "
              "The papers found may not match your question well.")
        return

    shown = result["history"][-1]["papers"]
    if result["route"] == SAME_PAPERS:
        ranks = [str(rank) for rank, p in enumerate(shown, start=1) if p["paper_id"] in result["document_ids"]]
        print(f"(Answered from paper(s) {', '.join(ranks)} shown earlier)")

    print("\nAnswer:")
    print(result["answer"])
    if result["citations"]:
        print("\nSources:")
    for citation in result["citations"]:
        print(f"- Paper ID: {citation['paper_id']}")
        print(f'  Supporting quote: "{citation["supporting_quote"]}"')

    if result["route"] == NEW_SEARCH and shown:
        print("\nRetrieved papers:")
        for rank, paper in enumerate(shown, start=1):
            preview = paper["chunk_text"][:100].replace("\n", " ")
            print(f"\n[{rank}] {paper['title']}")
            print(f"    Paper ID: {paper['paper_id']}")
            print(f"    Cosine similarity: {paper['score']:.1%}")
            print(f"    Abstract preview: {preview}...")

    if result["follow_up_suggestions"]:
        print("\nYou could ask next:")
        for number, suggestion in enumerate(result["follow_up_suggestions"], start=1):
            print(f"  {number}. {suggestion}")

def run_chat() -> None:
    print("Ask a research question. Then ask a follow-up, or type a suggestion's number.")
    print("Type /new to start a new conversation, or press Enter on an empty line to quit.")
    thread_id = str(uuid.uuid4())
    suggestions: list[str] = []
    while True:
        text = input("\nYou: ").strip()
        if not text:
            return
        if text.lower() == "/new":
            thread_id, suggestions = str(uuid.uuid4()), []
            print("Started a new conversation.")
            continue
        if text.isdigit() and 1 <= int(text) <= len(suggestions):
            text = suggestions[int(text) - 1]
            print(f"You: {text}")
        result = run_agent(text, thread_id=thread_id)
        show(result)
        suggestions = result["follow_up_suggestions"]

if __name__ == "__main__":
    run_chat()
