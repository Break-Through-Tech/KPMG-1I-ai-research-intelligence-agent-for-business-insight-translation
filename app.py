"""
Streamlit UI for the AI research intelligence agent.

Run from the repository root:

    streamlit run app.py

The page calls the same LangGraph pipeline as `python -m src.agent`
(src.agent.run_agent), so answers, citations and timings match the CLI.
"""
import uuid
"""Streamlit entrypoint and shared visual shell for Research Intelligence."""

import streamlit as st

st.set_page_config(page_title="AI Research Intelligence Agent", page_icon="🔎", layout="wide")

st.markdown(
    f"""
    <style>
    .stApp {{ background: var(--page-bg); color: var(--text-primary); }}
    .block-container {{ max-width: 1180px; padding: 1.25rem 2rem 2.25rem; }}
    header[data-testid="stHeader"], header[data-testid="stHeader"] [data-testid="stToolbar"] {{ background: transparent !important; }}
    h1, h2, h3 {{ color: var(--text-primary) !important; }}
    p, label {{ color: var(--text-secondary) !important; }}
    footer {{ visibility: hidden; }}

    /* Shared application bar */
    .app-header, .st-key-app-header {{ display: flex; align-items: center; justify-content: space-between; width: calc(100% + 4rem); min-height: 2.45rem; box-sizing: border-box; margin: -0.5rem -2rem 1rem; padding: 0.18rem 2rem; background: var(--surface-strong); border-bottom: 1px solid var(--border); }}
    .app-header-brand {{ display: inline-flex; align-items: center; gap: 0.7rem; color: var(--text-primary); font-size: 0.88rem; font-weight: 700; }}
    .app-header-mark, .sidebar-identity-mark {{ display: inline-flex; align-items: center; justify-content: center; background: var(--accent); color: #fff; font-weight: 800; }}
    .app-header-mark {{ width: 1.8rem; height: 1.8rem; border-radius: 0.32rem; font-size: 0.58rem; }}
    .app-header-placeholder {{ padding: 0.34rem 0.62rem; border: 1px solid var(--border-soft); border-radius: 999px; color: var(--text-muted); font-size: 0.66rem; white-space: nowrap; }}
    .st-key-app-header .stButton > button {{ min-height: 2rem; padding: 0.34rem 0.72rem; background: transparent; border: 1px solid var(--accent); border-radius: 999px; color: var(--text-primary); font-size: 0.68rem; font-weight: 650; }}
    .st-key-app-header .stButton > button:hover {{ background: var(--accent); color: var(--accent-text); }}

    /* Shared sidebar shell */
    [data-testid="stSidebar"] {{ width: 208px; min-width: 208px; max-width: 208px; background: var(--sidebar-bg); border-right: 1px solid var(--border-soft); }}
    [data-testid="stSidebar"] * {{ color: var(--text-secondary); }}
    [data-testid="stSidebar"] .sidebar-identity {{ display: flex; align-items: center; gap: 0.65rem; margin: 0.15rem 0 1rem; }}
    [data-testid="stSidebar"] .sidebar-identity-mark {{ width: 1.8rem; height: 1.8rem; border-radius: 0.3rem; font-size: 0.5rem; }}
    [data-testid="stSidebar"] .sidebar-identity-name {{ color: var(--text-primary); font-size: 0.92rem; font-weight: 700; }}
    [data-testid="stSidebar"] .sidebar-identity-caption {{ display: block; margin-top: 0.1rem; color: var(--text-muted); font-size: 0.67rem; }}
    [data-testid="stSidebar"] .stButton > button {{ width: 100%; min-height: 1.95rem; margin: 0.08rem 0; padding: 0.34rem 0.6rem; background: var(--surface-soft); border: 1px solid transparent; border-radius: 0.4rem; color: var(--text-secondary); font-size: 0.75rem; text-align: left; }}
    [data-testid="stSidebar"] .stButton > button:hover {{ background: color-mix(in srgb, var(--accent) 12%, transparent); border-color: var(--border); color: var(--text-primary); }}
    [data-testid="stSidebar"] .sidebar-nav-label, [data-testid="stSidebar"] .sidebar-section-label {{ margin: 1.05rem 0 0.4rem; color: var(--text-muted); font-size: 0.62rem; font-weight: 800; letter-spacing: 0.12em; text-transform: uppercase; }}
    [data-testid="stSidebar"] .sidebar-history-placeholder {{ padding: 0.65rem 0.2rem; border-top: 1px solid var(--border-soft); border-bottom: 1px solid var(--border-soft); color: var(--text-muted); font-size: 0.72rem; line-height: 1.45; }}
    [data-testid="stSidebar"] [data-testid="stPageLink"] {{ min-height: 1.85rem; padding: 0.3rem 0.45rem; border-radius: 0.4rem; font-size: 0.74rem; }}
    [data-testid="stSidebar"] [data-testid="stPageLink"]:hover {{ background: color-mix(in srgb, var(--accent) 12%, transparent); }}
    [data-testid="stSidebar"] .sidebar-nav-item {{ display: flex; align-items: center; gap: 0.5rem; min-height: 1.85rem; padding: 0.3rem 0.45rem; color: var(--text-muted); font-size: 0.72rem; }}
    [data-testid="stSidebar"] hr {{ margin: 1rem 0; border-color: var(--border-soft); }}
    [data-testid="stSidebar"] [data-testid="stExpander"] {{ border: 1px solid var(--border-soft); border-radius: 0.55rem; background: var(--surface-soft); }}

    /* Research page */
    .home-intro {{ max-width: 670px; margin: 0.5rem auto 0.9rem; text-align: center; }}
    .home-eyebrow {{ margin-bottom: 0.45rem; color: var(--accent); font-size: 0.68rem; font-weight: 800; letter-spacing: 0.14em; }}
    .home-intro h1 {{ margin: 0; color: var(--text-primary); font-size: clamp(1.65rem, 3.3vw, 2.3rem); line-height: 1.1; font-weight: 700; letter-spacing: -0.03em; }}
    .home-intro p {{ max-width: 560px; margin: 0.55rem auto 0; color: var(--text-secondary); font-size: 0.88rem; line-height: 1.45; }}
    .st-key-research-composer {{ max-width: 940px; margin: 0 auto 1.1rem; padding: 0.9rem 1rem 0.55rem; background: var(--surface); border: 1px solid var(--border); border-radius: 11px; box-shadow: 0 8px 24px var(--theme-shadow); }}
    .st-key-research-composer [data-testid="stTextInput"] label {{ display: none; }}
    [data-testid="stTextInput"] input {{ min-height: 3.5rem; padding: 0.8rem 0.95rem; background: var(--input-bg); color: var(--text-primary); border: 1px solid var(--border); border-radius: 8px; }}
    [data-testid="stTextInput"] input::placeholder {{ color: var(--placeholder); opacity: 1; }}
    [data-testid="stTextInput"]:focus-within > div[data-baseweb="input"], [data-testid="stTextInput"]:focus-within [data-baseweb="input"], [data-testid="stTextInput"]:focus-within [data-baseweb="base-input"] {{ border-color: var(--accent) !important; box-shadow: 0 0 0 2px var(--accent) !important; outline: none !important; }}
    .st-key-research-composer .stButton > button {{ width: 2.75rem; height: 2.75rem; min-height: 2.75rem; margin-top: 1.7rem; padding: 0; background: var(--accent); border: 1px solid var(--accent); border-radius: 50%; color: var(--accent-text); font-size: 1.1rem; }}
    .st-key-research-composer .stButton > button:hover {{ background: var(--accent-hover); border-color: var(--accent-hover); }}
    .composer-footer {{ display: flex; gap: 0.45rem; margin-top: 0.55rem; color: var(--text-muted); font-size: 0.68rem; }}
    .composer-footer span {{ padding: 0.22rem 0.45rem; background: var(--surface-soft); border-radius: 999px; }}
    .home-intro, .st-key-research-composer, .st-key-suggested-research, .st-key-discovery-preview, .st-key-kpmg-focus {{ width: 100%; max-width: 940px; margin-left: auto; margin-right: auto; }}
    .st-key-research-composer {{ margin-bottom: 0.9rem; }}
    .st-key-suggested-research {{ margin-bottom: 0.85rem; }}
    .st-key-discovery-preview {{ margin-bottom: 0.85rem; }}
    .st-key-kpmg-focus {{ margin-bottom: 0.4rem; }}
    .suggestions-heading, .discovery-title, .focus-title {{ color: var(--text-primary); font-size: 0.9rem; font-weight: 700; }}
    .suggestions-caption, .discovery-caption, .focus-caption {{ margin: 0.25rem 0 0.45rem; color: var(--text-muted); font-size: 0.7rem; }}
    .st-key-suggested-research .stButton > button {{ min-height: 2.75rem; margin: 0; padding: 0.55rem 0.75rem; background: var(--surface); border: 1px solid var(--border-soft); border-radius: 0; color: var(--text-secondary); font-size: 0.78rem; text-align: left; box-shadow: none; }}
    .st-key-suggested-research .stButton > button:hover {{ background: var(--surface-strong); border-color: var(--accent); color: var(--text-primary); }}
    .discovery-header, .focus-header {{ display: flex; justify-content: space-between; align-items: baseline; gap: 1rem; margin-bottom: 0.4rem; }}
    .discovery-card {{ min-height: 6.3rem; padding: 0.72rem 0.8rem; background: var(--surface); border: 1px solid var(--border-soft); border-radius: 8px; box-shadow: 0 4px 14px var(--theme-shadow); }}
    .discovery-card [data-testid="stPageLink"] {{ padding: 0; color: var(--text-primary); font-size: 0.8rem; font-weight: 650; line-height: 1.35; }}
    .discovery-card [data-testid="stPageLink"]:hover {{ color: var(--accent); }}
    .discovery-card-copy {{ margin-top: 0.35rem; color: var(--text-muted); font-size: 0.7rem; line-height: 1.35; }}
    .focus-chips {{ display: flex; flex-wrap: wrap; gap: 0.4rem; }}
    .focus-chip {{ padding: 0.35rem 0.58rem; background: var(--surface); border: 1px solid var(--border-soft); border-radius: 999px; color: var(--text-secondary); font-size: 0.7rem; }}

    /* Results page */
    .result-shell, .st-key-result-shell {{ max-width: 1080px; margin: 0 auto; }}
    .research-question-context {{ max-width: 1080px; margin: 0 auto 1.15rem; padding: 0.75rem 1rem; background: var(--surface-soft); border-left: 2px solid var(--accent); border-radius: 0 9px 9px 0; }}
    .research-question-context-label, .result-section-label {{ color: var(--accent-soft); font-size: 0.68rem; font-weight: 800; letter-spacing: 0.11em; text-transform: uppercase; }}
    .research-question-context-text {{ margin-top: 0.28rem; color: var(--text-primary); font-size: 0.95rem; line-height: 1.4; }}
    .result-section-label {{ margin: 1.35rem 0 0.6rem; }}
    .executive-insight-card, .st-key-executive-insight {{ padding: 1.35rem 1.5rem; background: var(--surface-strong); border: 1px solid var(--border); border-radius: 14px; box-shadow: 0 8px 22px var(--theme-shadow); }}
    .confidence-panel {{ display: flex; justify-content: space-between; gap: 1rem; margin: 1rem auto 0; padding: 0.8rem 1rem; background: var(--surface-soft); border: 1px solid var(--border-soft); border-radius: 10px; }}
    .confidence-badge {{ color: var(--text-primary); font-size: 0.78rem; font-weight: 750; }}
    .confidence-score, .source-meta {{ color: var(--text-muted); font-size: 0.72rem; }}
    .evidence-card {{ margin: 0.65rem 0; padding: 1rem 1.1rem; background: var(--surface); border: 1px solid var(--border-soft); border-radius: 11px; box-shadow: 0 8px 22px var(--theme-shadow); }}
    .evidence-card blockquote {{ margin: 0.75rem 0 0; padding: 0.65rem 0.85rem; background: var(--surface-soft); border-left: 2px solid var(--accent); color: var(--text-secondary); font-size: 0.88rem; line-height: 1.55; }}
    .evidence-card details {{ margin-top: 0.75rem; padding-top: 0.65rem; border-top: 1px solid var(--border-soft); }}
    .evidence-card summary {{ color: var(--accent-soft); cursor: pointer; font-size: 0.72rem; font-weight: 650; }}
    .evidence-detail {{ margin-top: 0.6rem; padding: 0.65rem 0.75rem; background: var(--surface-soft); border-radius: 8px; color: var(--text-secondary); font-size: 0.78rem; line-height: 1.5; }}
    .more-to-explore {{ margin-top: 1.75rem; padding: 1.05rem 1.1rem 1.15rem; background: var(--surface); border: 1px dashed var(--border); border-radius: 11px; }}
    .more-to-explore-title {{ color: var(--text-primary); font-size: 0.84rem; font-weight: 700; }}
    .more-to-explore-copy, .follow-up-note {{ margin-top: 0.25rem; color: var(--text-muted); font-size: 0.76rem; }}
    .more-to-explore-preview {{ margin-bottom: 0.65rem; color: var(--accent-soft); font-size: 0.64rem; font-weight: 750; letter-spacing: 0.1em; text-transform: uppercase; }}
    .follow-up-suggestion {{ display: flex; justify-content: space-between; gap: 1rem; margin-top: 0.45rem; padding: 0.65rem 0.75rem; background: var(--surface); border: 1px solid var(--border-soft); border-radius: 8px; color: var(--text-secondary); font-size: 0.78rem; }}
    .follow-up-suggestion::after {{ color: var(--accent-soft); content: "→"; }}
    .follow-up-composer {{ margin-top: 0.85rem; padding-top: 0.8rem; border-top: 1px solid var(--border-soft); }}
    .follow-up-composer-label {{ color: var(--text-muted); font-size: 0.72rem; font-weight: 650; }}
    .timing-grid {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(10rem, 1fr)); gap: 0.45rem; padding: 0.35rem 0; }}
    .timing-item {{ padding: 0.55rem 0.65rem; background: var(--surface); border: 1px solid var(--border-soft); border-radius: 8px; }}
    .timing-label {{ color: var(--text-muted); font-size: 0.68rem; }}
    .timing-value {{ margin-top: 0.18rem; color: var(--text-secondary); font-size: 0.82rem; }}
    .st-key-result-shell [data-testid="stExpander"] {{ border: 1px solid var(--border-soft); border-radius: 10px; background: var(--surface-soft); }}

    @media (max-width: 640px) {{
        .block-container {{ padding: 1.5rem 1rem 2rem; }}
        .app-header, .st-key-app-header {{ width: calc(100% + 2rem); margin: -0.5rem -1rem 1.25rem; padding: 0.4rem 1rem; }}
        .app-header-placeholder {{ display: none; }}
        .st-key-research-composer {{ padding: 0.8rem; }}
        .confidence-panel {{ align-items: flex-start; flex-direction: column; gap: 0.35rem; }}
    }}
    </style>
    """,
    unsafe_allow_html=True,
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
research_page = st.Page("views/research.py", title="Research", icon=":material/search:", default=True)
results_page = st.Page("views/results.py", title="Results / Research Session", icon=":material/article:")
latest_page = st.Page("views/latest_ai.py", title="Latest AI", icon=":material/view_list:")
trends_page = st.Page("views/trends.py", title="Trends", icon=":material/trending_up:")
kpmg_focus_page = st.Page("views/kpmg_focus.py", title="KPMG Focus", icon=":material/target:")
paper_workspace_page = st.Page("views/paper_workspace.py", title="Paper Workspace", icon=":material/description:")

navigation = st.navigation(
    {
        "Discover": [research_page, latest_page, trends_page, kpmg_focus_page],
        "Workspace": [results_page, paper_workspace_page],
    },
    position="hidden",
)
navigation.run()
