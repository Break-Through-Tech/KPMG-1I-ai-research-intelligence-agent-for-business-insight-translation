import requests
from langgraph.graph import StateGraph, START, END
from typing import TypedDict
from langchain_ollama import ChatOllama
from langchain_core.prompts import ChatPromptTemplate
from src.arxiv_html import fetch_sections
from src.vector_store import get_client, semantic_search

# Ollama's default context is much smaller than what qwen2.5 supports, and
# anything past it is silently dropped.
llm = ChatOllama(model="qwen2.5:7b", temperature=0, num_ctx=32768)

# Keeps the sections of all 3 papers, plus the prompt and answer, inside num_ctx.
MAX_SECTION_CHARS_PER_PAPER = 24_000

prompt = ChatPromptTemplate.from_messages([
    ("system",
     "You are a research assistant that answers questions about AI papers. "
     "Use only the paper excerpts below. Cite the paper title for each claim. "
     "If the answer isn't in the excerpts, say you don't know.\n\n"
     "Papers:\n{context}"),
    ("human", "{question}"),
])

class AgentState(TypedDict):
    query: str
    document_ids: list[str]
    retrieved_chunks: list[dict]
    paper_sections: dict[str, list[dict]]
    answer: str

def retrieve_vectors(state: AgentState):
    client = get_client("./qdrant_data")
    try:
        results = semantic_search(client, query=state["query"], top_k=3)
    finally:
        client.close()
    return {
        "document_ids": [
            result["paper_id"] for result in results
        ],
        "retrieved_chunks": results,
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

def format_sections(sections: list[dict], budget: int = MAX_SECTION_CHARS_PER_PAPER) -> str:
    text = ""
    for section in sections:
        text += f"{section['title']}\n{section['text']}\n"
        if len(text) >= budget:
            return text[:budget]
    return text

def generate_answer(state: AgentState):
    context = ""
    for chunk in state["retrieved_chunks"]:
        paper_id = chunk["paper_id"]
        context += f"Title: {chunk['title']} (ID: {paper_id})\n"
        context += f"Abstract: {chunk['chunk_text']}\n"
        context += format_sections(state["paper_sections"].get(paper_id, []))
        context += "\n"

    chain = prompt | llm
    response = chain.invoke({"context": context, "question": state["query"]})
    return {"answer": response.content}

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
    return graph.invoke({
        "query": query,
        "document_ids": [],
        "retrieved_chunks": [],
        "paper_sections": {},
    })

if __name__ == "__main__":
    query = input("Enter your research query: ").strip()

    result = run_agent(query)

    print("\nAnswer:")
    print(result["answer"])

    print("\nRetrieved papers:")
    for rank, paper in enumerate(result["retrieved_chunks"], start=1):
        preview = paper["chunk_text"][:100].replace("\n", " ")
        print(f"\nRank: {rank}")
        print(f"Paper ID: {paper['paper_id']}")
        print(f"Title: {paper['title']}")
        print(f"Cosine similarity: {paper['score']:.1%}")
        print(f"Abstract preview: {preview}...")
