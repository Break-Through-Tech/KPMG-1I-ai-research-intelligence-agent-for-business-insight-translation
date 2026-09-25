from langgraph.graph import StateGraph, START, END
from typing import TypedDict
from src.vector_store import get_client, semantic_search

class AgentState(TypedDict):
    query: str
    document_ids: list[str]
    retrieved_chunks: list[dict]

def retrieve(state: AgentState):
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

graph = StateGraph(AgentState)
graph.add_node("retrieve", retrieve)
graph.add_edge(START, "retrieve")
graph.add_edge("retrieve", END)
graph = graph.compile()

result = graph.invoke({
    "query": "How can AI explain medical reports to patients?",
    
})

def run_agent(query: str) -> AgentState:
    return graph.invoke({
        "query": query,
        "document_ids": [],
        "retrieved_chunks": []
    })

if __name__ == "__main__":
    query = input("Enter your research query: ").strip()

    result = run_agent(query)

    print("\nRelevant document IDs:")
    for document_id in result["document_ids"]:
        print(document_id)