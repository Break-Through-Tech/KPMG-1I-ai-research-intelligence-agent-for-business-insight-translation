"""Research agent: answer a question from the papers and passages the two searches return.

START -> retrieve_papers and retrieve_chunks (in parallel) -> assemble_context -> generate_answer -> END
"""

from dataclasses import asdict
from functools import partial
from typing import Any, TypedDict

from langchain_core.runnables import Runnable
from langchain_ollama import ChatOllama
from langgraph.graph import END, START, StateGraph
from qdrant_client import QdrantClient

from src.prompts import answer_prompt
from src.retrieval import search_chunks, search_papers
from src.vector_store import get_client

LLM_MODEL = "qwen2.5:7b"
# Ollama's default context is much smaller than what qwen2.5 supports, and anything past it is silently dropped.
LLM_CONTEXT_TOKENS = 32768
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


def format_context(paper_hits: list[dict[str, Any]], chunk_hits: list[dict[str, Any]]) -> str:
    """Papers and passages as two labelled sections; empty when both searches found nothing."""
    if not paper_hits and not chunk_hits:
        return ""
    papers = "\n\n".join(_format_paper(hit) for hit in paper_hits) or "No papers found."
    passages = "\n\n".join(_format_chunk(hit) for hit in chunk_hits) or "No passages found."
    return f"## Papers\n\n{papers}\n\n## Passages\n\n{passages}"


def retrieve_papers(state: AgentState, client: QdrantClient) -> dict:
    return {"paper_hits": [asdict(hit) for hit in search_papers(client, state["query"])]}


def retrieve_chunks(state: AgentState, client: QdrantClient) -> dict:
    return {"chunk_hits": [asdict(hit) for hit in search_chunks(client, state["query"])]}


def assemble_context(state: AgentState) -> dict:
    return {"context": format_context(state["paper_hits"], state["chunk_hits"])}


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


def _print_result(state: AgentState):
    print(f"\n===== Answer =====\n{state['answer']}")
    print(f"\n===== Retrieved papers ({len(state['paper_hits'])}) =====")
    for hit in state["paper_hits"]:
        print(f"{hit['rank']:3}. {hit['arxiv_id']}  {hit['title']}")
    print(f"\n===== Retrieved passages ({len(state['chunk_hits'])}) =====")
    for hit in state["chunk_hits"]:
        print(f"{hit['rank']:3}. {hit['chunk_id']}  {' > '.join(hit['breadcrumb'])}")


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
