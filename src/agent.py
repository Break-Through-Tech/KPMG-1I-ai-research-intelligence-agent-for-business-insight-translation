from langgraph.graph import StateGraph, START, END
from typing import TypedDict

class AgentState(TypedDict):
    query: str
    document_ids: list[str]
    retrieved_chunks: list[dict]

def mock_retrieve(state: AgentState):
    print({"Received query": state['query']})

    return {
        "document_ids": ["1", "2", "3"],
        "retrieved_chunks": ["chunk1", "chunk2", "chunk3"]
    }

graph = StateGraph(AgentState)
graph.add_node("retrieve", mock_retrieve)
graph.add_edge(START, "retrieve")
graph.add_edge("retrieve", END)
graph = graph.compile()

result = graph.invoke({
    "query": "test query",
    "document_ids": [],
    "retrieved_chunks": []
})

print(result)
print(graph.get_graph().draw_mermaid())