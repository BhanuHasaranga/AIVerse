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


def create_two_column_layout(title, icon=None, module_id=None):
    """
    Create standard 2.5:1 column layout for explorer pages.
    Similar to: <TwoColumnLayout title="Mean Explorer" />
    
    Args:
        title: Page title
        icon: Optional icon (for backward compatibility, not displayed)
        module_id: Optional module ID for progress tracking
    
    Returns:
        tuple: (left_column, right_column)
    """
    from utils.learning_path import get_module_by_id
    
    # Initialize session state
    if 'completed_modules' not in st.session_state:
        st.session_state.completed_modules = set()
    
    # Title and completion tracking
    col_title, col_complete = st.columns([4, 1])
    
    with col_title:
        st.title(title)
    
    with col_complete:
        if module_id:
            module = get_module_by_id(module_id)
            if module:
                is_completed = module_id in st.session_state.completed_modules
                
                if is_completed:
                    st.success("✅ Completed")
                    if st.button("Mark Incomplete", key=f"uncomplete_{module_id}"):
                        st.session_state.completed_modules.discard(module_id)
                        st.rerun()
                else:
                    difficulty_badge = module.get_difficulty_badge()
                    st.info(f"{difficulty_badge} {module.difficulty}")
                    if st.button("Mark Complete", key=f"complete_{module_id}"):
                        st.session_state.completed_modules.add(module_id)
                        st.success("🎉 Great job!")
                        st.rerun()
    
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
    from utils.learning_path import LEARNING_PATH, calculate_total_progress
    
    # Initialize completed modules in session state
    if 'completed_modules' not in st.session_state:
        st.session_state.completed_modules = set()
    
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
        .sidebar-section-title {
            font-size: 0.85rem;
            font-weight: 600;
            text-transform: uppercase;
            letter-spacing: 0.5px;
            color: rgba(255,255,255,0.7) !important;
            margin-bottom: 0.5rem;
        }
        .module-item {
            display: flex;
            align-items: center;
            gap: 8px;
        }
        </style>
    """, unsafe_allow_html=True)
    
    # Header
    st.sidebar.markdown("# 🤖 AIVerse")
    st.sidebar.markdown("*Interactive Learning Platform*")
    st.sidebar.divider()
    
    # Quick access buttons
    col1, col2 = st.sidebar.columns(2)
    with col1:
        if st.button("👋 Welcome", use_container_width=True, key="sidebar_welcome"):
            st.switch_page("pages/welcome.py")
    with col2:
        if st.button("🗺️ Path", use_container_width=True, key="sidebar_path"):
            st.switch_page("pages/learning_path.py")
    
    st.sidebar.divider()
    
    # Navigation sections
    st.sidebar.markdown('<p class="sidebar-section-title">🏠 MAIN</p>', unsafe_allow_html=True)
    home_section = st.sidebar.radio(
        "main_nav",
        ["Home", "About"],
        label_visibility="collapsed"
    )
    
    # Get modules from learning path with status
    phase1_modules = LEARNING_PATH["phase_1"]["modules"]
    
    st.sidebar.markdown('<p class="sidebar-section-title">📊 STATISTICS FOUNDATIONS</p>', unsafe_allow_html=True)
    stats_display = []
    stats_values = []
    for module in phase1_modules[:4]:  # First 4 modules
        is_completed = module.id in st.session_state.completed_modules
        status = "✅" if is_completed else module.get_difficulty_badge()
        display_name = f"{status} {module.title}"
        stats_display.append(display_name)
        stats_values.append(module.title)
    
    stats_idx = st.sidebar.radio(
        "stats_nav",
        range(len(stats_display)),
        format_func=lambda x: stats_display[x],
        label_visibility="collapsed"
    )
    stats_section = stats_values[stats_idx]
    
    st.sidebar.markdown('<p class="sidebar-section-title">📈 DISTRIBUTIONS & RELATIONSHIPS</p>', unsafe_allow_html=True)
    dist_display = []
    dist_values = []
    for module in phase1_modules[4:]:  # Last 3 modules
        is_completed = module.id in st.session_state.completed_modules
        status = "✅" if is_completed else module.get_difficulty_badge()
        display_name = f"{status} {module.title}"
        dist_display.append(display_name)
        dist_values.append(module.title)
    
    dist_idx = st.sidebar.radio(
        "dist_nav",
        range(len(dist_display)),
        format_func=lambda x: dist_display[x],
        label_visibility="collapsed"
    )
    dist_section = dist_values[dist_idx]
    
    # Footer info with real progress
    st.sidebar.divider()
    total_progress = calculate_total_progress(st.session_state.completed_modules)
    st.sidebar.markdown("### 📚 Your Progress")
    st.sidebar.progress(total_progress / 100, text=f"Overall: {total_progress:.0f}%")
    
    completed = len(st.session_state.completed_modules)
    st.sidebar.caption(f"{completed}/7 modules completed")
    
    # Return selected page (check which section was changed)
    if home_section != "Home":
        return home_section
    
    # Return the selected module
    return stats_section if stats_section else dist_section

