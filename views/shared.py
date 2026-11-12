"""Shared state, navigation helpers, and result renderers for the Streamlit pages."""

from html import escape

import streamlit as st


THEME_TOKENS = {
    "dark": {
        "page_bg": "#111A24", "sidebar_bg": "#0D1721", "surface": "#182632",
        "surface_strong": "#1A2B39", "surface_soft": "rgba(255, 255, 255, 0.045)",
        "input_bg": "#0E1822", "text_primary": "#F2F6FA", "text_secondary": "#C1CEDA",
        "text_muted": "#8FA2B4", "accent": "#1D9DDD", "accent_hover": "#4AB8E5",
        "accent_soft": "#83CDEC", "accent_text": "#071426",
        "border": "rgba(157, 190, 211, 0.18)", "border_soft": "rgba(157, 190, 211, 0.1)",
        "shadow": "rgba(0, 0, 0, 0.18)", "placeholder": "#93A7B8",
    },
    "light": {
        "page_bg": "#F4F8FB", "sidebar_bg": "#E7F1F8", "surface": "#FFFFFF",
        "surface_strong": "#FFFFFF", "surface_soft": "#EEF5F9", "input_bg": "#FFFFFF",
        "text_primary": "#152A3B", "text_secondary": "#40576C", "text_muted": "#6B8194",
        "accent": "#006FAD", "accent_hover": "#0086C3", "accent_soft": "#0079B8",
        "accent_text": "#FFFFFF", "border": "rgba(43, 91, 125, 0.16)",
        "border_soft": "rgba(43, 91, 125, 0.09)", "shadow": "rgba(30, 74, 104, 0.08)",
        "placeholder": "#6B8194",
    },
}


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

# TEMPORARY local UI preview mode. Set to False to use the real agent path.
PREVIEW_RESULTS = True
PREVIEW_RESULT = {
    "answer": (
        "Demo insight: AI research teams can improve business decision-making by "
        "combining retrieval with clear evidence trails. In practice, this means "
        "pairing a focused research question with a small set of relevant papers, "
        "then translating the findings into operational recommendations.\n\n"
        "- Retrieval quality is the foundation for trustworthy research synthesis.\n"
        "- Evidence-backed summaries make technical findings easier for leaders to act on."
    ),
    "confidence": "high",
    "top_score": 0.78,
    "retrieved_chunks": [
        {
            "score": 0.78,
            "paper_id": "arXiv:2403.01234",
            "title": "Evidence-Grounded Language Models for Enterprise Research",
            "section_title": "Abstract",
            "chunk_text": (
                "We study retrieval-augmented language models that connect generated "
                "answers to source passages, improving traceability and reviewability "
                "for knowledge-intensive business tasks."
            ),
            "chunk_index": 0,
        },
        {
            "score": 0.66,
            "paper_id": "arXiv:2405.06789",
            "title": "From Research Findings to Actionable Organizational Insights",
            "section_title": "Introduction",
            "chunk_text": (
                "Organizations benefit when AI systems summarize technical evidence "
                "in a way that preserves source context and highlights implications "
                "for decisions, processes, and investment priorities."
            ),
            "chunk_index": 0,
        },
        {
            "score": 0.57,
            "paper_id": "arXiv:2407.04567",
            "title": "Evaluating Retrieval Quality in Business Question Answering",
            "section_title": "Results",
            "chunk_text": (
                "Evaluation results indicate that relevance signals and source-level "
                "inspection help users calibrate trust in answers produced from a "
                "large research collection."
            ),
            "chunk_index": 0,
        },
    ],
    "citations": [
        {
            "paper_id": "arXiv:2403.01234",
            "supporting_quote": (
                "Retrieval-augmented language models connect generated answers to "
                "source passages, improving traceability and reviewability."
            ),
        },
        {
            "paper_id": "arXiv:2405.06789",
            "supporting_quote": (
                "AI systems should preserve source context and highlight implications "
                "for decisions, processes, and investment priorities."
            ),
        },
    ],
    "timings": {
        "embedding_model_initialization_seconds": 0.42,
        "query_embedding_seconds": 0.08,
        "qdrant_retrieval_seconds": 0.12,
        "arxiv_section_fetching_seconds": 1.31,
        "ollama_response_generation_seconds": 3.84,
        "total_execution_seconds": 5.77,
    },
}


def select_suggested_question(question: str) -> None:
    st.session_state["query"] = question


def _sync_theme_state() -> None:
    st.session_state["theme"] = (
        "light" if st.session_state["theme_toggle"] else "dark"
    )


def _render_theme_tokens() -> None:
    if "theme_toggle" not in st.session_state:
        st.session_state["theme_toggle"] = st.session_state.get("theme") == "light"

    _sync_theme_state()
    theme = THEME_TOKENS[st.session_state["theme"]]
    st.markdown(
        f"""
        <style>
        .stApp {{
            --page-bg: {theme["page_bg"]};
            --sidebar-bg: {theme["sidebar_bg"]};
            --surface: {theme["surface"]};
            --surface-strong: {theme["surface_strong"]};
            --surface-soft: {theme["surface_soft"]};
            --input-bg: {theme["input_bg"]};
            --text-primary: {theme["text_primary"]};
            --text-secondary: {theme["text_secondary"]};
            --text-muted: {theme["text_muted"]};
            --accent: {theme["accent"]};
            --accent-hover: {theme["accent_hover"]};
            --accent-soft: {theme["accent_soft"]};
            --accent-text: {theme["accent_text"]};
            --border: {theme["border"]};
            --border-soft: {theme["border_soft"]};
            --theme-shadow: {theme["shadow"]};
            --placeholder: {theme["placeholder"]};
        }}
        </style>
        """,
        unsafe_allow_html=True,
    )


def start_new_research() -> None:
    st.session_state["query"] = ""
    st.session_state.pop("result", None)


def render_sidebar() -> None:
    with st.sidebar:
        st.markdown(
            """
            <div class="sidebar-identity">
                <span class="sidebar-identity-mark">KPMG</span>
                <span>
                    <span class="sidebar-identity-name">Research Intelligence</span>
                    <span class="sidebar-identity-caption">AI research workspace</span>
                </span>
            </div>
            """,
            unsafe_allow_html=True,
        )
        if st.button(
            "＋ New Research",
            key="sidebar-new-research",
            use_container_width=True,
        ):
            start_new_research()
            st.switch_page("views/research.py")
        st.markdown('<div class="sidebar-nav-label">Discover</div>', unsafe_allow_html=True)
        st.page_link("views/research.py", label="Research", icon=":material/search:")
        st.page_link("views/latest_ai.py", label="Latest AI", icon=":material/view_list:")
        st.page_link("views/trends.py", label="Trends", icon=":material/trending_up:")
        st.page_link("views/kpmg_focus.py", label="KPMG Focus", icon=":material/target:")
        st.markdown('<div class="sidebar-section-label">Recent</div>', unsafe_allow_html=True)
        st.markdown(
            '<div class="sidebar-history-placeholder">Your recent research sessions will appear here.</div>',
            unsafe_allow_html=True,
        )
        st.markdown('<div class="sidebar-nav-label">Saved</div>', unsafe_allow_html=True)
        st.markdown(
            '<div class="sidebar-nav-item"><span class="sidebar-nav-icon">☆</span>Saved research will appear here</div>',
            unsafe_allow_html=True,
        )
        st.divider()
        with st.expander("System info"):
            st.caption("Requires the local Qdrant store and Ollama (`qwen2.5:7b`).")


def render_top_bar() -> None:
    with st.container(key="app-header"):
        brand_col, model_col, theme_col, new_col = st.columns(
            [2.8, 1, 0.85, 0.9], gap="small", vertical_alignment="center"
        )
        with brand_col:
            st.markdown(
                """
                <div class="app-header-brand">
                    <span class="app-header-mark">KPMG</span>
                    <span>Research Intelligence</span>
                </div>
                """,
                unsafe_allow_html=True,
            )
        with model_col:
            st.markdown(
                '<span class="app-header-placeholder">Model selector</span>',
                unsafe_allow_html=True,
            )
        with theme_col:
            st.toggle(
                "☼ / ☾",
                key="theme_toggle",
                on_change=_sync_theme_state,
                help="Switch between light and dark theme",
            )
        with new_col:
            if st.button("New Research", key="top-new-research"):
                start_new_research()
                st.switch_page("views/research.py")


def render_shell() -> None:
    _render_theme_tokens()
    render_sidebar()
    render_top_bar()


def render_timings(timings: dict[str, float]) -> None:
    with st.expander("Advanced details", expanded=False):
        timing_items = []
        for key, label in TIMING_LABELS.items():
            if key in timings:
                timing_items.append(
                    f"""
                    <div class="timing-item">
                        <div class="timing-label">{escape(label)}</div>
                        <div class="timing-value">{timings[key]:.2f} s</div>
                    </div>
                    """
                )
        st.html(f'<div class="timing-grid">{"".join(timing_items)}</div>')


def render_evidence(citations: list[dict], chunks: list[dict]) -> None:
    st.markdown('<div class="result-section-label">Evidence / sources</div>', unsafe_allow_html=True)
    if not citations and not chunks:
        st.info("The agent did not return any evidence for this answer.")
        return

    citations_by_paper: dict[str, list[dict]] = {}
    chunks_by_paper: dict[str, list[dict]] = {}
    paper_order: list[str] = []
    for chunk in chunks:
        paper_id = str(chunk.get("paper_id", "Unknown"))
        chunks_by_paper.setdefault(paper_id, []).append(chunk)
        if paper_id not in paper_order:
            paper_order.append(paper_id)
    for citation in citations:
        paper_id = str(citation.get("paper_id", "Unknown"))
        citations_by_paper.setdefault(paper_id, []).append(citation)
        if paper_id not in paper_order:
            paper_order.append(paper_id)

    for rank, paper_id in enumerate(paper_order, start=1):
        paper_chunks = chunks_by_paper.get(paper_id, [])
        paper_citations = citations_by_paper.get(paper_id, [])
        first_chunk = paper_chunks[0] if paper_chunks else {}
        title = escape(str(first_chunk.get("title", "Unknown paper")))
        score = first_chunk.get("score")
        score_label = f"Relevance {score:.1%}" if isinstance(score, (int, float)) else "Source relevance unavailable"
        citation_markup = "".join(
            f"<blockquote>{escape(str(citation.get('supporting_quote', '')))}</blockquote>"
            for citation in paper_citations
            if citation.get("supporting_quote")
        )
        detail_markup = []
        for chunk in paper_chunks:
            preview_text = chunk.get("chunk_text", "")
            preview = escape(preview_text[:320].replace("\n", " "))
            if len(preview_text) > 320:
                preview += "…"
            section = chunk.get("section_title")
            section_label = f" · Section: {escape(str(section))}" if section else ""
            chunk_score = chunk.get("score")
            chunk_score_label = f" · Relevance: {chunk_score:.1%}" if isinstance(chunk_score, (int, float)) else ""
            detail_markup.append(
                f"""
                <div class="evidence-detail">
                    <div><strong>Retrieved paper detail</strong>{section_label}{chunk_score_label}</div>
                    <div class="source-meta">Paper ID: {escape(paper_id)}</div>
                    <div>{preview}</div>
                </div>
                """
            )
        if not detail_markup:
            detail_markup.append(
                f"""
                <div class="evidence-detail">
                    <div><strong>Source metadata</strong></div>
                    <div class="source-meta">Paper ID: {escape(paper_id)}</div>
                </div>
                """
            )
        st.markdown(
            f"""
            <article class="evidence-card">
                <div><strong>{rank}. {title}</strong></div>
                <div class="source-meta">{score_label}</div>
                {citation_markup or '<div class="source-meta">No supporting quote returned.</div>'}
                <details><summary>Show retrieved paper detail</summary>{''.join(detail_markup)}</details>
            </article>
            """,
            unsafe_allow_html=True,
        )


def render_confidence(result: dict) -> None:
    confidence = result.get("confidence")
    top_score = result.get("top_score")
    retrieved_chunks = result.get("retrieved_chunks") or []
    if not confidence and (top_score is None or not retrieved_chunks):
        return
    confidence_value = str(confidence).lower() if confidence else ""
    confidence_class = confidence_value if confidence_value in {"high", "medium", "low"} else ""
    confidence_label = f"{confidence_value.title()} confidence" if confidence_class else "Relevance signal"
    score_label = f"Top relevance {top_score:.1%}" if isinstance(top_score, (int, float)) else ""
    st.markdown(
        f"""
        <div class="confidence-panel">
            <span class="confidence-badge confidence-{confidence_class}">{escape(confidence_label)}</span>
            <span class="confidence-score">{escape(score_label)}</span>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_more_to_explore(result: dict) -> None:
    with st.container(key="more-to-explore"):
        st.markdown(
            """
            <section class="more-to-explore">
                <div class="more-to-explore-preview">Temporary UI preview</div>
                <div class="more-to-explore-title">More to explore</div>
                <div class="more-to-explore-copy">Continue from this finding with a focused business question.</div>
                <div class="follow-up-suggestion">What operating-model changes could help apply this insight?</div>
                <div class="follow-up-suggestion">Which implementation risks should leadership monitor first?</div>
                <div class="follow-up-suggestion">How could this inform near-term investment priorities?</div>
                <div class="follow-up-suggestion explore-angle">Explore another angle</div>
                <div class="follow-up-composer"><div class="follow-up-composer-label">Ask a follow-up question</div></div>
            </section>
            """,
            unsafe_allow_html=True,
        )
        st.text_input(
            "Follow-up research question",
            placeholder="Follow-up research will be enabled in a future update.",
            disabled=True,
            label_visibility="collapsed",
        )
        st.markdown(
            '<div class="follow-up-note">Temporary presentation-only controls — no follow-up request is submitted.</div>',
            unsafe_allow_html=True,
        )
