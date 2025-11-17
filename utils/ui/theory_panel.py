"""
Theory panel component for AIVerse.
Displays learning content in tabbed interface.
"""

import streamlit as st


def render_theory_panel(tabs_content):
    """
    Render learning guide with tabbed content.
    
    Args:
        tabs_content: dict with keys ["Definition", "Examples", "ML Usage", "Summary"]
                     Each value should be a function that renders content
    """
    st.subheader("📚 Learning Guide")
    
    tab_names = ["Definition", "Examples", "ML Usage", "Summary"]
    tabs = st.tabs(tab_names)
    
    for tab, name in zip(tabs, tab_names):
        with tab:
            content_func = tabs_content.get(name)
            if callable(content_func):
                content_func()
            elif content_func:
                st.write(content_func)


def render_section_header(text, icon=""):
    """
    Render consistent section headers.
    
    Args:
        text: Header text
        icon: Optional icon prefix
    """
    if icon:
        st.subheader(f"{icon} {text}")
    else:
        st.subheader(text)

