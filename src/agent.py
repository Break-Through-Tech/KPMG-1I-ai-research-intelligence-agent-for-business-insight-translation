from langgraph.graph import StateGraph, START, END
from typing import TypedDict
from langchain_ollama import ChatOllama
from langchain_core.prompts import ChatPromptTemplate
from src.vector_store import get_client, semantic_search
from src.storage.b2_storage import B2Storage

llm = ChatOllama(model="qwen2.5:7b", temperature=0)

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
    object_storage_keys: list[str]
    answer: str

def retrieve_vectors(state: AgentState):
    client = get_client("./qdrant_data")
    try:
        results = semantic_search(
            client,
            query=state["query"],
            top_k=3
        )
    finally:
        client.close()
    return {
        "document_ids": [
            result["paper_id"] for result in results
        ],
        "retrieved_chunks": results,
    }

def retrieve_objects(state: AgentState):
    # b2_storage = B2Storage()
    result = []
    for id in state["document_ids"]:
        # if b2_storage.pdf_exists(id):
        result.append(f"raw/{id}.pdf")
    return {
        "object_storage_keys": result
    }

def generate_answer(state: AgentState):
    context = ""
    for chunk in state["retrieved_chunks"]:
        context += f"Title: {chunk['title']}\n{chunk['chunk_text']}\n\n"

    chain = prompt | llm
    response = chain.invoke({"context": context, "question": state["query"]})
    return {"answer": response.content}

graph = StateGraph(AgentState)
graph.add_node("retrieve_vectors", retrieve_vectors)
graph.add_node("retrieve_objects", retrieve_objects)
graph.add_node("generate_answer", generate_answer)
graph.add_edge(START, "retrieve_vectors")
graph.add_edge("retrieve_vectors", "retrieve_objects")
graph.add_edge("retrieve_objects", "generate_answer")
graph.add_edge("generate_answer", END)
graph = graph.compile()

def run_agent(query: str) -> AgentState:
    return graph.invoke({
        "query": query,
        "document_ids": [],
        "retrieved_chunks": [],
        "object_storage_keys": [],
        "answer": ""
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