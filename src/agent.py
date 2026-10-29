import argparse
import os
from time import perf_counter

import requests
from langgraph.graph import StateGraph, START, END
from typing import TypedDict
from langchain_ollama import ChatOllama
from langchain_core.prompts import ChatPromptTemplate
from src.arxiv_html import fetch_sections
from src.vector_store import get_client, semantic_search
from pydantic import BaseModel, Field

# Ollama's default context is much smaller than what qwen2.5 supports, and
# anything past it is silently dropped.
OLLAMA_KEEP_ALIVE = os.getenv("OLLAMA_KEEP_ALIVE", "10m")


def create_llm():
    return ChatOllama(
        model="qwen2.5:7b",
        temperature=0,
        num_ctx=32768,
        keep_alive=OLLAMA_KEEP_ALIVE,
    )


llm = create_llm()

# Keeps the sections of all 3 papers, plus the prompt and answer, inside num_ctx.
MAX_SECTION_CHARS_PER_PAPER = 24_000

class AgentState(TypedDict): #adding one field to the agent state for citation 
    query: str
    document_ids: list[str]
    retrieved_chunks: list[dict]
    paper_sections: dict[str, list[dict]]
    answer: str
    citations: list[dict]
    timings: dict[str, float]


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

prompt = ChatPromptTemplate.from_messages([
    ("system",
     "You are a research assistant answering questions about AI papers.\n\n"
     "1. Use only the excerpts provided. Do not use outside knowledge.\n"
     "2. First, identify what the question is asking for, such as a solution, "
     "a technique, a comparison, or an evaluation method.\n"
     "3. Examine all the excerpts and select the findings that directly "
     "help answer the question. Do not rely only on the top-ranked paper.\n"
     "4. Explain what the findings suggest in response to the question. "
     "If the question asks how to solve a problem, describe the solutions "
     "supported by the excerpts. Do not mistake methods for evaluating "
     "a solution as the solution itself.\n"
     "5. Support each claim with the correct paper_id and a short, exact "
     "quote from that paper's excerpt.\n"
     "6. If the excerpts do not describe a solution or provide enough "
     "evidence, clearly say what information is missing instead of guessing.\n\n"
     "Excerpts:\n{context}"),
    ("human", "{question}"),
])

def retrieve_vectors(state: AgentState):
    timings = state.get("timings")
    client = get_client("./qdrant_data")
    try:
        results = semantic_search(
            client,
            query=state["query"],
            top_k=3,
            timings=timings,
        )
    finally:
        client.close()
    return {
        "document_ids": [
            result["paper_id"] for result in results
        ],
        "retrieved_chunks": results,
    }

def retrieve_sections(state: AgentState):
    started = perf_counter()
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
    timings = state.get("timings")
    if timings is not None:
        timings["arxiv_section_fetching_seconds"] = perf_counter() - started
    return {"paper_sections": paper_sections}

def format_sections(sections: list[dict], budget: int = MAX_SECTION_CHARS_PER_PAPER) -> str:
    text = ""
    for section in sections:
        text += f"{section['title']}\n{section['text']}\n"
        if len(text) >= budget:
            return text[:budget]
    return text

def generate_answer(state: AgentState):
    timings = state.get("timings")
    context = ""
    for chunk in state["retrieved_chunks"]:
        paper_id = chunk["paper_id"]
        context += f"Title: {chunk['title']} (ID: {paper_id})\n"
        context += f"Abstract: {chunk['chunk_text']}\n"
        context += format_sections(state["paper_sections"].get(paper_id, []))
        context += "\n"

    inputs = {"context": context, "question": state["query"]}

    structured_llm = llm.with_structured_output(LLMAnswer, include_raw=True)
    started = perf_counter()
    output = (prompt | structured_llm).invoke(inputs)
    result = output["parsed"]

    if result is None:
        fallback = (prompt | llm).invoke(inputs)
        if timings is not None:
            timings["ollama_response_generation_seconds"] = perf_counter() - started
        return {"answer": fallback.content, "citations": []}

    if timings is not None:
        timings["ollama_response_generation_seconds"] = perf_counter() - started

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
    }

graph = StateGraph(AgentState)
graph.add_node("retrieve_vectors", retrieve_vectors)
graph.add_node("retrieve_sections", retrieve_sections)
graph.add_node("generate_answer", generate_answer)
graph.add_edge(START, "retrieve_vectors")
graph.add_edge("retrieve_vectors", "retrieve_sections")
graph.add_edge("retrieve_sections", "generate_answer")
graph.add_edge("generate_answer", END)
graph = graph.compile()

def run_agent(query: str) -> AgentState:
    timings = {}
    started = perf_counter()
    result = graph.invoke({
        "query": query,
        "document_ids": [],
        "retrieved_chunks": [],
        "paper_sections": {},
        "answer": "",
        "citations": [],
        "timings": timings,
    })
    result["timings"]["total_execution_seconds"] = perf_counter() - started
    return result


def print_result(result: AgentState) -> None:
    print("\nAnswer:")
    print(result["answer"])
    print("\nTiming:")
    for name, seconds in result["timings"].items():
        print(f"- {name}: {seconds:.3f}s")
    print("\nSources:")
    for citation in result["citations"]:
        print(f"- Paper ID: {citation['paper_id']}")
        print(f'  Supporting quote: "{citation["supporting_quote"]}"')


def run_interactive() -> None:
    while True:
        query = input("Enter your research query (blank to exit): ").strip()
        if not query:
            return
        print_result(run_agent(query))

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--interactive", action="store_true")
    args = parser.parse_args()

    if args.interactive:
        run_interactive()
        raise SystemExit

    query = input("Enter your research query: ").strip()

    result = run_agent(query)

    print_result(result)


    print("\nRetrieved papers:")
    for rank, paper in enumerate(result["retrieved_chunks"], start=1):
        preview = paper["chunk_text"][:100].replace("\n", " ")
        print(f"\nRank: {rank}")
        print(f"Paper ID: {paper['paper_id']}")
        print(f"Title: {paper['title']}")
        print(f"Cosine similarity: {paper['score']:.1%}")
        print(f"Abstract preview: {preview}...")
