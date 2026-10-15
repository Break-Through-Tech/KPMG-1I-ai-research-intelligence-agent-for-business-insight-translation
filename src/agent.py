from itertools import chain

from langgraph.graph import StateGraph, START, END
from typing import TypedDict
from langchain_ollama import ChatOllama
from langchain_core.prompts import ChatPromptTemplate
from src.vector_store import get_client, semantic_search
from pydantic import BaseModel, Field

class AgentState(TypedDict): #adding one field to the agent state for citation 
    query: str
    document_ids: list[str]
    retrieved_chunks: list[dict]
    answer: str
    citations: list[dict]


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

llm = ChatOllama(model="qwen2.5:7b", temperature=0)


prompt = ChatPromptTemplate.from_messages([
    ("system",
     "You are a research assistant answering questions about AI papers.\n\n"
     "1. Use only the abstracts provided. Do not use outside knowledge.\n"
     "2. First, identify what the question is asking for, such as a solution, "
     "a technique, a comparison, or an evaluation method.\n"
     "3. Examine all the abstracts and select the findings that directly "
     "help answer the question. Do not rely only on the top-ranked paper.\n"
     "4. Explain what the findings suggest in response to the question. "
     "If the question asks how to solve a problem, describe the solutions "
     "supported by the abstracts. Do not mistake methods for evaluating "
     "a solution for the solution itself.\n"
     "5. Support each claim with the correct paper_id and a short, exact "
     "quote from that paper's abstract.\n"
     "6. If the abstracts do not describe a solution or provide enough "
     "evidence, clearly say what information is missing instead of guessing.\n\n"
     "Abstracts:\n{context}"),
    ("human", "{question}"),
])

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


def generate_answer(state: AgentState):
    context = ""
    for chunk in state["retrieved_chunks"]:
        context += f"Title: {chunk['title']} (ID: {chunk['paper_id']})\n{chunk['chunk_text']}\n\n"

    structured_llm = llm.with_structured_output(LLMAnswer)
    chain = prompt | structured_llm

    result = chain.invoke({
        "context": context,
        "question": state["query"]
    })

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
graph.add_node("generate_answer", generate_answer)
graph.add_edge(START, "retrieve_vectors")
graph.add_edge("retrieve_vectors", "generate_answer")
graph.add_edge("generate_answer", END)
graph = graph.compile()

def run_agent(query: str) -> AgentState:
    return graph.invoke({
        "query": query,
        "document_ids": [],
        "retrieved_chunks": [],
        "answer": "",
        "citations": [],
    })

if __name__ == "__main__":
    query = input("Enter your research query: ").strip()

    result = run_agent(query)

    print("\nAnswer:")
    print(result["answer"])
    print("\nSources:")
    for citation in result["citations"]:
        print(f"- Paper ID: {citation['paper_id']}")
        print(f'  Supporting quote: "{citation["supporting_quote"]}"')


    print("\nRetrieved papers:")
    for rank, paper in enumerate(result["retrieved_chunks"], start=1):
        preview = paper["chunk_text"][:100].replace("\n", " ")
        print(f"\nRank: {rank}")
        print(f"Paper ID: {paper['paper_id']}")
        print(f"Title: {paper['title']}")
        print(f"Cosine similarity: {paper['score']:.1%}")
        print(f"Abstract preview: {preview}...")
