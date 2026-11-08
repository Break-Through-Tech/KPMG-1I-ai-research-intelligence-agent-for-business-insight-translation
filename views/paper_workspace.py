"""Paper Workspace shell for future paper-level research workflows."""

import streamlit as st

from views.shared import render_shell


def main() -> None:
    render_shell()
    st.markdown("# Paper Workspace")
    st.caption("Paper-level review tools are reserved for a future update.")
    st.info(
        "This workspace will support paper metadata, abstracts, PDF preview, evidence highlights, annotations, and sharing once a paper-opening flow is connected."
    )


main()
