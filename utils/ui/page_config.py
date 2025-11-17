"""
Page configuration utilities for AIVerse.
Handles page setup and theme application.
"""

import streamlit as st
from utils.theme import Theme


def apply_page_config(title="AI/ML Learning Hub", icon="📊", layout="centered", sidebar_state="collapsed"):
    """
    Apply consistent page configuration.
    
    Args:
        title: Page title
        icon: Page icon (emoji or path)
        layout: "centered" or "wide"
        sidebar_state: "collapsed" or "expanded"
    """
    st.set_page_config(
        page_title=title,
        page_icon=icon,
        layout=layout,
        initial_sidebar_state=sidebar_state
    )


def apply_theme(page_type="page"):
    """
    Apply theme CSS to page.
    
    Args:
        page_type: "page" for explorer pages, "home" for landing page
    """
    if page_type == "home":
        st.markdown(Theme.get_home_css(), unsafe_allow_html=True)
    else:
        st.markdown(Theme.get_page_css(), unsafe_allow_html=True)

