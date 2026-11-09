"""Research session results page."""

import streamlit as st
from html import escape

from views.shared import (
    render_confidence,
    render_evidence,
    render_more_to_explore,
    render_shell,
    render_timings,
)


def main() -> None:
    render_shell()
    result = st.session_state.get("result")
    if not result:
        st.switch_page("views/research.py")
        return

    if st.button("← Back to Research", key="back-to-research"):
        st.switch_page("views/research.py")
    current_query = st.session_state.get("query", "")
    if current_query:
        st.markdown(
            f"""
            <div class="research-question-context">
                <div class="research-question-context-label">Current research question</div>
                <div class="research-question-context-text">{escape(current_query)}</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    retrieved_chunks = result.get("retrieved_chunks", [])
    citations = result.get("citations", [])
    with st.container(key="result-shell"):
        st.markdown('<div class="result-section-label">Executive insight</div>', unsafe_allow_html=True)
        with st.container(key="executive-insight"):
            st.markdown(result.get("answer") or "_No answer was generated._")
        render_confidence(result)
        render_evidence(citations, retrieved_chunks)
        render_more_to_explore(result)
        if result.get("timings"):
            render_timings(result["timings"])


main()
