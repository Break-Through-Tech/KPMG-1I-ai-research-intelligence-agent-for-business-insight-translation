from langgraph.graph import StateGraph, START, END
from typing import TypedDict
from langchain_ollama import ChatOllama
from langchain_core.prompts import ChatPromptTemplate
from src.vector_store import get_client, semantic_search

llm = ChatOllama(model="qwen2.5:7b", temperature=0)

prompt = ChatPromptTemplate.from_messages([
    ("system",
     "You are a research assistant that answers questions about AI papers. "
     "Use only the paper abstracts below. Cite the paper title for each claim. "
     "If the answer isn't in the abstracts, say you don't know.\n\n"
     "Abstracts:\n{context}"),
    ("human", "{question}"),
])

class AgentState(TypedDict):
    query: str
    retrieved_chunks: list[dict]
    answer: str

def retrieve_vectors(state: AgentState):
    client = get_client("./qdrant_data")
    try:
        results = semantic_search(client, query=state["query"], top_k=3)
    finally:
        client.close()
    return {"retrieved_chunks": results}

def generate_answer(state: AgentState):
    context = ""
    for chunk in state["retrieved_chunks"]:
        context += f"Title: {chunk['title']} (ID: {chunk['paper_id']})\n{chunk['chunk_text']}\n\n"

    chain = prompt | llm
    response = chain.invoke({"context": context, "question": state["query"]})
    return {"answer": response.content}

graph = StateGraph(AgentState)
graph.add_node("retrieve_vectors", retrieve_vectors)
graph.add_node("generate_answer", generate_answer)
graph.add_edge(START, "retrieve_vectors")
graph.add_edge("retrieve_vectors", "generate_answer")
graph.add_edge("generate_answer", END)
graph = graph.compile()

def run_agent(query: str) -> AgentState:
    return graph.invoke({"query": query, "retrieved_chunks": [], "answer": ""})

if __name__ == "__main__":
    query = input("Enter your research query: ").strip()
    result = run_agent(query)

    print("\nAnswer:")
    print(result["answer"])

    print("\nRetrieved papers:")
    for paper in result["retrieved_chunks"]:
        print(f"\nTitle: {paper['title']}")
        print(f"Paper ID: {paper['paper_id']}")
        print(f"Similarity: {paper['score']:.1%}")
        print(f"Abstract: {paper['chunk_text'][:100]}...")