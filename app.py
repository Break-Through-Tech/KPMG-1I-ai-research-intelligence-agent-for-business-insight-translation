"""
Streamlit UI for the AI research intelligence agent.

Run from the repository root:

    streamlit run app.py

The page calls the same LangGraph pipeline as `python -m src.agent`
(src.agent.run_agent), so answers, citations and timings match the CLI.
"""
import uuid
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


def start_new_chat():
    """Create a fresh conversation."""
    st.session_state.thread_id = str(uuid.uuid4())
    st.session_state.chat_history = []


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

    if "thread_id" not in st.session_state:
        start_new_chat()

    with st.sidebar:
        st.title("Research Intelligence")
        st.write(
            "Ask a question about the indexed AI research papers. "
            "The agent retrieves the most relevant abstracts, pulls their "
            "full sections when available, and answers with cited quotes."
        )
        
        st.divider()

        if st.button("＋ New Chat", use_container_width=True):
            start_new_chat()
            st.rerun()

        st.divider()
        st.markdown("**Example questions**")

        for example in EXAMPLE_QUERIES:
            if st.button(example, use_container_width=True):
                st.session_state.pending_query = example
                st.rerun()

    st.title("AI Research Intelligence Agent")
    st.write("Turn AI research into business-ready insight, with sources.")

    # Display the full conversation.
    for message in st.session_state.chat_history:
        display_message(message)

    # Receive a new question.
    query = st.chat_input("Ask a research question or follow up...")

    # Also allow the sidebar example buttons to submit questions.
    if "pending_query" in st.session_state:
        query = st.session_state.pop("pending_query")

    if not query or not query.strip():
        return

    query = query.strip()

    # Display and save the user's message.
    st.session_state.chat_history.append({
        "role": "user",
        "content": query,
    })

    with st.chat_message("user"):
        st.markdown(query)

    # Call the existing LangGraph agent with this conversation's ID.
    with st.chat_message("assistant"):
        with st.spinner("Searching papers and generating an answer..."):

            try:
                result = run_agent(
                    query,
                    thread_id=st.session_state.thread_id,
                )

                answer = result.get(
                    "answer",
                    "No answer was generated.",
                )

                st.markdown(answer)

                if result.get("confidence"):
                    st.caption(
                        f"Retrieval confidence: {result['confidence']} "
                        f"({result.get('top_score', 0):.1%} top similarity)"
                    )

                if result.get("timings"):
                    render_timings(result["timings"])

                titles = {
                    chunk["paper_id"]: chunk["title"]
                    for chunk in result.get("retrieved_chunks", [])
                }

                if result.get("citations") or result.get("retrieved_chunks"):
                    with st.expander("Sources and retrieved papers"):
                        render_citations(
                            result.get("citations", []),
                            titles,
                        )
                        render_retrieved(
                            result.get("retrieved_chunks", [])
                        )

                # Save the assistant's response for display.
                st.session_state.chat_history.append({
                    "role": "assistant",
                    "content": answer,
                    "result": result,
                })

            except Exception as error:
                st.error(
                    "The agent could not complete this request. "
                    f"Details: {error}"
                )

main()