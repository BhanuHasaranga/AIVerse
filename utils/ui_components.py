"""
Reusable UI components for the AI/ML Learning Hub.
Similar to React components - write once, use everywhere.
"""

import streamlit as st
from utils.theme import Theme


def apply_page_config(title="AI/ML Learning Hub", icon="📊", layout="wide", sidebar_state="collapsed"):
    """
    Apply consistent page configuration.
    Similar to: <PageConfig layout="wide" />
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
    Similar to: <ThemeProvider theme={theme}>
    
    Args:
        page_type: "page" for explorer pages, "home" for landing page
    """
    if page_type == "home":
        st.markdown(Theme.get_home_css(), unsafe_allow_html=True)
    else:
        st.markdown(Theme.get_page_css(), unsafe_allow_html=True)


def create_two_column_layout(title, icon=None):
    """
    Create standard 2.5:1 column layout for explorer pages.
    Similar to: <TwoColumnLayout title="Mean Explorer" />
    
    Args:
        title: Page title
        icon: Optional icon (for backward compatibility, not displayed)
    
    Returns:
        tuple: (left_column, right_column)
    """
    st.title(title)
    col1, col2 = st.columns([2.5, 1], gap="large")
    return col1, col2


def render_module_card(title, description, topics, button_text, page_path, icon="📊"):
    """
    Reusable module card component for home page.
    Similar to: <ModuleCard title="Mean Explorer" icon="📊" />
    
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
    
    if st.button(button_text, use_container_width=True, key=f"btn_{page_path}"):
        st.switch_page(page_path)


def render_hero_section(title, subtitle):
    """
    Render hero section with gradient background.
    Similar to: <Hero title="AI/ML Hub" subtitle="Learn interactively" />
    """
    st.markdown(f"""
        <div class="hero-section">
            <h1>{title}</h1>
            <p>{subtitle}</p>
        </div>
    """, unsafe_allow_html=True)


def render_section_header(text, icon=""):
    """Render consistent section headers"""
    if icon:
        st.subheader(f"{icon} {text}")
    else:
        st.subheader(text)


def render_theory_panel(tabs_content):
    """
    Render learning guide with tabbed content.
    Similar to: <TheoryPanel tabs={content} />
    
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


def render_enhanced_sidebar():
    """
    Render enhanced sidebar navigation with sections and styling.
    Similar to: <Sidebar sections={modules} />
    
    Returns:
        str: Selected page name
    """
    # Apply sidebar-specific CSS
    st.markdown("""
        <style>
        [data-testid="stSidebar"] {
            background: linear-gradient(180deg, #667eea 0%, #764ba2 100%);
        }
        [data-testid="stSidebar"] .css-1d391kg, 
        [data-testid="stSidebar"] [data-testid="stMarkdownContainer"] {
            color: white !important;
        }
        [data-testid="stSidebar"] h1, 
        [data-testid="stSidebar"] h2, 
        [data-testid="stSidebar"] h3 {
            color: white !important;
        }
        .sidebar-section {
            margin: 1.5rem 0;
            padding: 0.5rem 0;
            border-bottom: 1px solid rgba(255,255,255,0.2);
        }
        .sidebar-section-title {
            font-size: 0.85rem;
            font-weight: 600;
            text-transform: uppercase;
            letter-spacing: 0.5px;
            color: rgba(255,255,255,0.7) !important;
            margin-bottom: 0.5rem;
        }
        </style>
    """, unsafe_allow_html=True)
    
    # Header
    st.sidebar.markdown("# 🤖 AI/ML Hub")
    st.sidebar.markdown("*Interactive Learning Platform*")
    st.sidebar.divider()
    
    # Navigation sections
    st.sidebar.markdown('<p class="sidebar-section-title">🏠 MAIN</p>', unsafe_allow_html=True)
    home_section = st.sidebar.radio(
        "main_nav",
        ["Home", "About"],
        label_visibility="collapsed"
    )
    
    st.sidebar.markdown('<p class="sidebar-section-title">📊 STATISTICS FOUNDATIONS</p>', unsafe_allow_html=True)
    stats_modules = [
        "Mean Explorer",
        "Median Explorer", 
        "Mode Explorer",
        "Variance Visualizer"
    ]
    stats_section = st.sidebar.radio(
        "stats_nav",
        stats_modules,
        label_visibility="collapsed"
    )
    
    st.sidebar.markdown('<p class="sidebar-section-title">📈 DISTRIBUTIONS & RELATIONSHIPS</p>', unsafe_allow_html=True)
    dist_modules = [
        "Distribution Explorer",
        "Correlation Explorer",
        "Probability Explorer"
    ]
    dist_section = st.sidebar.radio(
        "dist_nav",
        dist_modules,
        label_visibility="collapsed"
    )
    
    # Footer info
    st.sidebar.divider()
    st.sidebar.markdown("### 📚 Progress")
    st.sidebar.progress(0.30, text="Phase 1: 30% Complete")
    st.sidebar.caption("7 modules | 3 phases planned")
    
    # Return selected page (check which section was changed)
    if home_section != "Home":
        return home_section
    
    # Check if user selected from stats or dist sections
    # The most recently selected item will be returned
    # This is a simplified approach - in production you'd track state better
    return stats_section if stats_section else dist_section

