import requests
from langgraph.graph import StateGraph, START, END
from typing import TypedDict
from langchain_ollama import ChatOllama
from langchain_core.prompts import ChatPromptTemplate
from src.arxiv_html import fetch_sections
from src.embeddings import get_model
from src.vector_store import get_client, semantic_search
from pydantic import BaseModel, Field
import re

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
    query: str
    is_valid: bool
    suggested_query: str
    document_ids: list[str]
    retrieved_chunks: list[dict]
    paper_sections: dict[str, list[dict]]
    answer: str
    citations: list[dict]
    confidence: str
    top_score: float


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
        return {"is_valid": False, "suggested_query": "", "answer": message}

    if result.is_valid:
        return {"is_valid": True, "suggested_query": ""}
    
    if result.suggested_query:
        message = "Your question wasn't quite clear. Did you mean: \"" + result.suggested_query + "\" Try asking that instead."
    else:
        message = "I can only answer questions about AI research. Try asking about a specific AI topic."

    return {"is_valid": False, "suggested_query": result.suggested_query, "answer": message}


def route_after_validation(state: AgentState):
    if state["is_valid"]:
        return "retrieve_vectors"
    return END

def retrieve_vectors(state: AgentState):
    client = get_client("./qdrant_data")
    try:
        results = semantic_search(client, query=state["query"], top_k=3)
    finally:
        client.close()
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
        return {"answer": fallback.content, "citations": []}

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

def route_after_retrieval(state: AgentState):
    if state["confidence"] == "low":
        return END
    return "retrieve_sections"

graph = StateGraph(AgentState)
graph.add_node("validate_question", validate_question)       
graph.add_node("retrieve_vectors", retrieve_vectors)
graph.add_node("retrieve_sections", retrieve_sections)
graph.add_node("generate_answer", generate_answer)
graph.add_edge(START, "validate_question")
graph.add_conditional_edges("validate_question", route_after_validation)
graph.add_conditional_edges(
    "retrieve_vectors",
    route_after_retrieval,
    {"retrieve_sections": "retrieve_sections", END: END},
)
graph.add_edge("retrieve_sections", "generate_answer")
graph.add_edge("generate_answer", END)
graph = graph.compile()

def run_agent(query: str) -> AgentState:
    return graph.invoke({
        "query": query,
        "is_valid": False,
        "suggested_query": "",
        "document_ids": [],
        "retrieved_chunks": [],
        "paper_sections": {},
        "answer": "",
        "citations": [],
        "confidence": "",
        "top_score": 0.0,
    })

def show(result):
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

if __name__ == "__main__":
    query = input("Enter your research query: ").strip()
    result = run_agent(query)

    if result["confidence"] == "low":
        print(f"\nWarning: low confidence ({result['top_score']:.0%}). "
              "The papers found may not match your question well.")
    else:
        show(result)
