"""
Card components for AIVerse.
Includes module cards and hero sections.
"""

import streamlit as st


def render_module_card(title, description, topics, button_text, page_path, icon="📊"):
    """
    Reusable module card component for home page.
    
    Args:
        title: Card title
        description: Card description
        topics: Topics covered (string)
        button_text: Button label
        page_path: Path to navigate to
        icon: Emoji icon
    """
    st.markdown(f"""
        <div class="module-card">
        <h3>{icon} {title}</h3>
        <p>{description}</p>
        <p><strong>Topics:</strong> {topics}</p>
        </div>
    """, unsafe_allow_html=True)
    
    if st.button(button_text, width='stretch', key=f"btn_{page_path}"):
        st.switch_page(page_path)


def render_hero_section(title, subtitle):
    """
    Render hero section with gradient background.
    
    Args:
        title: Main heading
        subtitle: Subheading text
    """
    st.markdown(f"""
        <div class="hero-section">
            <h1>{title}</h1>
            <p>{subtitle}</p>
        </div>
    """, unsafe_allow_html=True)

