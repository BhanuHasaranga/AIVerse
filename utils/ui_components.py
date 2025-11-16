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
    
    # Apply sidebar-specific CSS (hierarchical navigation style)
    st.markdown("""
        <style>
        /* Clean hierarchical sidebar */
        [data-testid="stSidebar"] {
            background-color: #fafbfc;
            border-right: 1px solid #e1e4e8;
            padding-top: 0;
        }
        
        [data-testid="stSidebar"] * {
            font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Helvetica, Arial, sans-serif;
        }
        
        /* Hide radio button labels */
        [data-testid="stSidebar"] .stRadio > label {
            display: none;
        }
        
        /* Navigation items - clean flat style */
        [data-testid="stSidebar"] [role="radiogroup"] label {
            padding: 0.5rem 1rem;
            border-radius: 0;
            font-size: 0.875rem;
            color: #24292e;
            transition: all 0.15s ease;
            cursor: pointer;
            font-weight: 400;
            border-left: 2px solid transparent;
            margin: 0;
        }
        
        [data-testid="stSidebar"] [role="radiogroup"] label:hover {
            background-color: #f6f8fa;
            color: #0969da;
        }
        
        [data-testid="stSidebar"] [role="radiogroup"] label[data-checked="true"] {
            background-color: #ddf4ff;
            color: #0969da;
            font-weight: 500;
            border-left: 2px solid #0969da;
        }
        
        /* Expander styling for sections */
        [data-testid="stSidebar"] .streamlit-expanderHeader {
            font-size: 0.75rem;
            font-weight: 600;
            text-transform: uppercase;
            letter-spacing: 0.5px;
            color: #57606a;
            padding: 0.5rem 1rem;
            border-radius: 0;
            background: transparent;
        }
        
        [data-testid="stSidebar"] .streamlit-expanderHeader:hover {
            background-color: #f6f8fa;
        }
        
        /* Hide the keyboard_arrow_down text */
        [data-testid="stSidebar"] .streamlit-expanderHeader svg {
            display: none;
        }
        
        /* Add custom arrow using CSS */
        [data-testid="stSidebar"] .streamlit-expanderHeader::before {
            content: "▸";
            margin-right: 0.5rem;
            font-size: 0.875rem;
            transition: transform 0.2s ease;
            display: inline-block;
        }
        
        [data-testid="stSidebar"] details[open] > summary.streamlit-expanderHeader::before {
            content: "▾";
        }
        
        /* Divider lines */
        [data-testid="stSidebar"] hr {
            margin: 0.5rem 0;
            border-color: #e1e4e8;
        }
        
        /* Progress section */
        [data-testid="stSidebar"] .stProgress {
            margin: 0.5rem 1rem;
        }
        
        [data-testid="stSidebar"] p {
            color: #57606a;
            font-size: 0.8125rem;
            padding-left: 1rem;
        }
        
        /* Caption text */
        [data-testid="stSidebar"] .stCaptionContainer {
            color: #6e7781 !important;
            padding-left: 1rem;
            font-size: 0.75rem;
        }
        
        /* Buttons - minimal style */
        [data-testid="stSidebar"] button {
            font-size: 0.8125rem;
            border-radius: 6px;
            border: 1px solid #d0d7de;
            background: white;
            color: #24292e;
            font-weight: 500;
            transition: all 0.15s ease;
            padding: 0.375rem 0.75rem;
        }
        
        [data-testid="stSidebar"] button:hover {
            background-color: #f6f8fa;
            border-color: #1f2328;
        }
        
        /* Header styling */
        [data-testid="stSidebar"] h2 {
            padding-left: 1rem !important;
            margin-bottom: 0.25rem !important;
        }
        </style>
    """, unsafe_allow_html=True)
    
    # Header - clean and minimal
    st.sidebar.markdown('<h2 style="color: #24292e; font-size: 1.125rem; font-weight: 600; margin-top: 1rem; margin-bottom: 0.25rem;">AIVerse</h2>', unsafe_allow_html=True)
    st.sidebar.caption("Interactive Learning")
    
    st.sidebar.divider()
    
    # Top-level navigation
    main_nav_options = ["Home", "Welcome", "Learning Path", "About"]
    main_nav_selected = st.sidebar.radio(
        "main_navigation",
        main_nav_options,
        label_visibility="collapsed",
        key="nav_main"
    )
    
    st.sidebar.divider()
    
    # Build navigation with expanders for hierarchy
    phase1_modules = LEARNING_PATH["phase_1"]["modules"]
    
    # STATISTICS FOUNDATIONS (collapsible section)
    with st.sidebar.expander("STATISTICS FOUNDATIONS", expanded=True):
        stats_options = []
        for module in phase1_modules[:4]:
            is_completed = module.id in st.session_state.completed_modules
            if is_completed:
                stats_options.append(f"✓ {module.title}")
            else:
                stats_options.append(module.title)
        
        stats_selected = st.radio(
            "stats_modules",
            stats_options,
            label_visibility="collapsed",
            key="nav_stats"
        )
    
    # DISTRIBUTIONS & PROBABILITY (collapsible section)
    with st.sidebar.expander("DISTRIBUTIONS & PROBABILITY", expanded=True):
        dist_options = []
        for module in phase1_modules[4:]:
            is_completed = module.id in st.session_state.completed_modules
            if is_completed:
                dist_options.append(f"✓ {module.title}")
            else:
                dist_options.append(module.title)
        
        dist_selected = st.radio(
            "dist_modules",
            dist_options,
            label_visibility="collapsed",
            key="nav_dist"
        )
    
    # Determine selected page
    selected = main_nav_selected
    
    # Footer info with real progress
    st.sidebar.divider()
    total_progress = calculate_total_progress(st.session_state.completed_modules)
    
    with st.sidebar.expander("PROGRESS", expanded=False):
        st.progress(total_progress / 100, text=f"{total_progress:.0f}% Complete")
        completed = len(st.session_state.completed_modules)
        st.caption(f"{completed} of 7 modules completed")
    
    # Parse and return clean page name
    if selected in ["Home", "Welcome", "Learning Path", "About"]:
        return selected
    elif selected.startswith("✓ "):
        # Format: "✓ Mean Explorer"
        return selected[2:].strip()
    else:
        # Clean module name
        return selected.strip()

