"""
Streamlit UI for the AI research intelligence agent.

Run from the repository root:

    streamlit run app.py

The page calls the same LangGraph pipeline as `python -m src.agent`
(src.agent.run_agent), so answers, citations and timings match the CLI.
"""

import streamlit as st

from src.agent import run_agent

EXAMPLE_QUERIES = [
    "How can AI explain medical reports to patients?",
    "What methods reduce the cost of routing queries between AI models?",
    "How do agents learn reusable task models from computer-use traces?",
]

TIMING_LABELS = {
    "embedding_model_initialization_seconds": "Embedding model initialization",
    "query_embedding_seconds": "Query embedding",
    "qdrant_retrieval_seconds": "Qdrant retrieval",
    "arxiv_section_fetching_seconds": "arXiv section fetching",
    "ollama_response_generation_seconds": "Ollama answer generation",
    "total_execution_seconds": "Total",
}

st.set_page_config(
    page_title="AI Research Intelligence Agent",
    page_icon="🔎",
    layout="wide",
)


def render_timings(timings: dict[str, float]) -> None:
    with st.expander("Timing details"):
        for key, label in TIMING_LABELS.items():
            if key in timings:
                st.metric(label, f"{timings[key]:.2f} s")


def render_retrieved(chunks: list[dict]) -> None:
    st.subheader("Retrieved papers")
    if not chunks:
        st.info("No papers matched this query.")
        return
    for rank, chunk in enumerate(chunks, start=1):
        with st.container(border=True):
            st.markdown(f"**{rank}. {chunk['title']}**")
            st.caption(
                f"Paper ID: `{chunk['paper_id']}` · "
                f"Similarity: {chunk['score']:.1%}"
            )
            preview = chunk["chunk_text"][:300].replace("\n", " ")
            st.write(f"{preview}{'…' if len(chunk['chunk_text']) > 300 else ''}")


def render_citations(citations: list[dict], titles: dict[str, str]) -> None:
    st.subheader("Sources")
    if not citations:
        st.info("The agent did not return any cited quotes for this answer.")
        return
    for citation in citations:
        paper_id = citation["paper_id"]
        with st.container(border=True):
            st.markdown(f"**{titles.get(paper_id, 'Unknown paper')}**")
            st.caption(f"Paper ID: `{paper_id}`")
            st.markdown(f"> {citation['supporting_quote']}")


def main() -> None:
    with st.sidebar:
        st.title("Research Intelligence")
        st.write(
            "Ask a question about the indexed AI research papers. "
            "The agent retrieves the most relevant abstracts, pulls their "
            "full sections when available, and answers with cited quotes."
        )
        st.caption("Requires the local Qdrant store and Ollama (`qwen2.5:7b`).")
        st.divider()
        st.markdown("**Example questions**")
        for example in EXAMPLE_QUERIES:
            if st.button(example, use_container_width=True):
                st.session_state["query"] = example

    st.title("AI Research Intelligence Agent")
    st.write("Turn AI research into business-ready insight, with sources.")

    query = st.text_input(
        "Research question",
        key="query",
        placeholder="e.g. How can AI explain medical reports to patients?",
    )
    ask = st.button("Ask", type="primary", disabled=not query.strip())

    if ask and query.strip():
        with st.spinner("Searching papers and generating an answer…"):
            try:
                result = run_agent(query.strip())
            except Exception as error:  # surface setup problems in the UI
                st.error(
                    "The agent could not complete this request. "
                    "Check that Ollama is running and that `make seed-vector-store` "
                    f"has been run.\n\nDetails: {error}"
                )
                return
        st.session_state["result"] = result

    result = st.session_state.get("result")
    if not result:
        return

    titles = {c["paper_id"]: c["title"] for c in result["retrieved_chunks"]}

    answer_col, sources_col = st.columns([3, 2], gap="large")
    with answer_col:
        st.subheader("Answer")
        st.markdown(result["answer"] or "_No answer was generated._")
        if result.get("timings"):
            render_timings(result["timings"])
    with sources_col:
        render_citations(result["citations"], titles)

    render_retrieved(result["retrieved_chunks"])


main()
