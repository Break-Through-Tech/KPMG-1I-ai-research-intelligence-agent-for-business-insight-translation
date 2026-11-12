"""Research home page."""

import streamlit as st
from html import escape

from src.agent import run_agent
from views.shared import (
    EXAMPLE_QUERIES,
    PREVIEW_RESULT,
    PREVIEW_RESULTS,
    render_shell,
    select_suggested_question,
)


def main() -> None:
    render_shell()
    st.html(
        """
        <div class="home-intro">
            <div class="home-eyebrow">KPMG AI RESEARCH</div>
            <h1>What would you like to research?</h1>
            <p>Explore AI research and translate evidence into business insight.</p>
        </div>
        """
    )

    with st.container(key="research-composer"):
        query_col, ask_col = st.columns([6, 0.85], gap="small", vertical_alignment="bottom")
        with query_col:
            query = st.text_input(
                "Ask about AI research",
                key="query",
                placeholder="Ask about AI research, trends, risks, or business implications…",
                label_visibility="collapsed",
            )
            st.markdown(
                '<div class="composer-footer"><span>AI research</span><span>Evidence-backed insight</span></div>',
                unsafe_allow_html=True,
            )
        with ask_col:
            ask = st.button("→", type="primary", disabled=not query.strip())

    with st.container(key="suggested-research"):
        st.markdown('<div class="suggestions-heading">Suggested for you</div>', unsafe_allow_html=True)
        st.markdown(
            '<div class="suggestions-caption">Start with a focused question for an evidence-backed business view.</div>',
            unsafe_allow_html=True,
        )
        for index, example in enumerate(EXAMPLE_QUERIES):
            st.button(
                example,
                key=f"suggested-question-{index}",
                use_container_width=True,
                on_click=select_suggested_question,
                args=(example,),
            )

    with st.container(key="discovery-preview"):
        st.markdown(
            """
            <div class="discovery-header"><span class="discovery-title">Explore AI research</span></div>
            """,
            unsafe_allow_html=True,
        )
        columns = st.columns(3, gap="small")
        cards = [
            ("views/latest_ai.py", "Latest AI research", "New research signals will appear here.", ":material/view_list:"),
            ("views/trends.py", "Trending AI topics", "Emerging areas will be surfaced here.", ":material/trending_up:"),
            ("views/kpmg_focus.py", "KPMG-relevant research", "Business-focused findings will appear here.", ":material/target:"),
        ]
        for column, (page, title, copy, icon) in zip(columns, cards):
            with column:
                st.markdown(
                    '<div class="discovery-card">',
                    unsafe_allow_html=True,
                )
                st.page_link(page, label=title, icon=icon)
                st.markdown(
                    f'<div class="discovery-card-copy">{escape(copy)}</div></div>',
                    unsafe_allow_html=True,
                )

    with st.container(key="kpmg-focus"):
        st.markdown(
            """
            <div class="focus-header"><span class="focus-title">KPMG focus areas</span></div>
            <div class="focus-chips">
                <span class="focus-chip">Responsible AI</span><span class="focus-chip">AI Agents</span><span class="focus-chip">AI Governance</span><span class="focus-chip">Cybersecurity</span><span class="focus-chip">Financial Services AI</span><span class="focus-chip">Healthcare / Life Sciences</span><span class="focus-chip">Model Efficiency</span><span class="focus-chip">Workforce Automation</span>
            </div>
            """,
            unsafe_allow_html=True,
        )

    if ask and query.strip():
        if PREVIEW_RESULTS:
            result = PREVIEW_RESULT
        else:
            with st.spinner("Searching papers and generating an answer…"):
                try:
                    result = run_agent(query.strip())
                except Exception as error:
                    st.error(
                        "The agent could not complete this request. Check that Ollama is running and that `make ingest` has been run."
                        f"\n\nDetails: {error}"
                    )
                    return
        st.session_state["result"] = result
        st.switch_page("views/results.py")


main()
