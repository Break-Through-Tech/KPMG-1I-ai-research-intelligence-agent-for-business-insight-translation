"""Future Latest AI discovery page."""

import streamlit as st

from views.shared import render_shell


def main() -> None:
    render_shell()
    st.markdown("# Latest AI")
    st.caption("This discovery page is not yet connected to live research data.")
    st.info("Latest AI research will appear here once the discovery feed is connected.")


main()
