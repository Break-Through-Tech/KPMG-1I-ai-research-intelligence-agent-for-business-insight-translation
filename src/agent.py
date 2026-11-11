"""Research agent: answer a question from the papers and passages the two searches return.

START -> retrieve_papers and retrieve_chunks (in parallel) -> assemble_context -> generate_answer -> END
"""

from collections.abc import Callable
from dataclasses import asdict
from functools import cache, partial
from typing import Any, TypedDict

from langchain_core.runnables import Runnable
from langchain_ollama import ChatOllama
from langgraph.graph import END, START, StateGraph
from qdrant_client import QdrantClient
from tokenizers import Tokenizer

from src.prompts import answer_prompt
from src.retrieval import search_chunks, search_papers
from src.vector_store import get_client

LLM_MODEL = "qwen2.5:7b"
# Hugging Face copy of the tokenizer qwen2.5:7b uses, so the budget counts the tokens the LLM actually reads.
LLM_TOKENIZER = "Qwen/Qwen2.5-7B-Instruct"
# Ollama's default context is much smaller than what qwen2.5 supports, and anything past it is silently dropped.
LLM_CONTEXT_TOKENS = 32768
# The local LLM reads about 200 tokens a second, so every 1,000 tokens of context adds about 5 s per answer.
CONTEXT_TOKENS = 6000
NO_EVIDENCE_ANSWER = "No relevant papers or passages were found for this question."


class GenerationError(RuntimeError):
    """The LLM failed or returned no answer."""


class AgentState(TypedDict):
    query: str
    paper_hits: list[dict[str, Any]]  # PaperHit fields
    chunk_hits: list[dict[str, Any]]  # ChunkHit fields
    context: str
    answer: str


def _join(*parts: str | None) -> str:
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


def retrieve_papers(state: AgentState, client: QdrantClient) -> dict:
    return {"paper_hits": [asdict(hit) for hit in search_papers(client, state["query"])]}


def retrieve_chunks(state: AgentState, client: QdrantClient) -> dict:
    return {"chunk_hits": [asdict(hit) for hit in search_chunks(client, state["query"])]}


def assemble_context(state: AgentState) -> dict:
    return {"context": format_context(*fit_to_budget(state["paper_hits"], state["chunk_hits"], CONTEXT_TOKENS))}


def generate_answer(state: AgentState, llm: Runnable) -> dict:
    if not state["context"]:
        return {"answer": NO_EVIDENCE_ANSWER}
    try:
        message = (answer_prompt | llm).invoke({"question": state["query"], "context": state["context"]})
    except Exception as error:
        raise GenerationError(f"The LLM couldn't generate an answer: {error}") from error
    if not message.content.strip():
        raise GenerationError("The LLM returned an empty answer")
    return {"answer": message.content}


def get_llm() -> ChatOllama:
    return ChatOllama(model=LLM_MODEL, temperature=0, num_ctx=LLM_CONTEXT_TOKENS)


def build_graph(client: QdrantClient, llm: Runnable):
    """Both searches share one client: local Qdrant allows only one client per storage folder."""
    graph = StateGraph(AgentState)
    graph.add_node("retrieve_papers", partial(retrieve_papers, client=client))
    graph.add_node("retrieve_chunks", partial(retrieve_chunks, client=client))
    graph.add_node("assemble_context", assemble_context)
    graph.add_node("generate_answer", partial(generate_answer, llm=llm))
    graph.add_edge(START, "retrieve_papers")
    graph.add_edge(START, "retrieve_chunks")
    graph.add_edge(["retrieve_papers", "retrieve_chunks"], "assemble_context")
    graph.add_edge("assemble_context", "generate_answer")
    graph.add_edge("generate_answer", END)
    return graph.compile()


def run_agent(query: str) -> AgentState:
    """Raises ValueError for a blank query, RetrievalError if Qdrant fails, GenerationError if the LLM fails."""
    query = query.strip()
    if not query:
        raise ValueError("The question is blank")
    client = get_client()
    try:
        return build_graph(client, get_llm()).invoke({"query": query})
    finally:
        client.close()


def hits_in_context(state: AgentState) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """The papers and passages the LLM was given, out of everything the searches returned."""
    return fit_to_budget(state["paper_hits"], state["chunk_hits"], CONTEXT_TOKENS)


def _print_result(state: AgentState):
    papers, chunks = hits_in_context(state)
    print(f"\n===== Answer =====\n{state['answer']}")
    print(f"\n===== Papers given to the LLM ({len(papers)} of {len(state['paper_hits'])} retrieved) =====")
    for hit in papers:
        print(f"{hit['rank']:3}. {hit['arxiv_id']}  {hit['title']}")
    print(f"\n===== Passages given to the LLM ({len(chunks)} of {len(state['chunk_hits'])} retrieved) =====")
    for hit in chunks:
        print(f"{hit['rank']:3}. {hit['chunk_id']}  {' > '.join(hit['breadcrumb'])}")
    print(f"\nContext: {count_tokens(state['context']):,} tokens (budget {CONTEXT_TOKENS:,})")


if __name__ == "__main__":
    # python -m src.agent ["question"]
    # Without a question, asks for one question after another (models and Qdrant load only once).
    import sys

    client = get_client()
    try:
        agent = build_graph(client, get_llm())
        if len(sys.argv) > 1:
            _print_result(agent.invoke({"query": sys.argv[1]}))
        else:
            while question := input("\nResearch question (blank to quit): ").strip():
                _print_result(agent.invoke({"query": question}))
    except (KeyboardInterrupt, EOFError):
        pass
    finally:
        client.close()
