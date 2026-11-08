"""Future KPMG Focus discovery page."""

import streamlit as st

from views.shared import render_shell


def main() -> None:
    render_shell()
    st.markdown("# KPMG Focus")
    st.caption("This topic workspace is not yet connected to live research data.")
    st.info("KPMG-relevant AI research will appear here once topic discovery is connected.")


main()
