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
    
    # Apply sidebar-specific CSS (AIVerse style - exact match)
    st.markdown("""
        <style>
        /* AIVerse Sidebar - Clean Educational Design */
        [data-testid="stSidebar"] {
            background-color: #F7F7F7;
            border-right: 1px solid #E5E5E5;
            padding-top: 0;
        }
        
        [data-testid="stSidebar"] * {
            font-family: 'Inter', -apple-system, BlinkMacSystemFont, "Segoe UI", Helvetica, Arial, sans-serif;
        }
        
        /* Hide radio button labels */
        [data-testid="stSidebar"] .stRadio > label {
            display: none;
        }
        
        /* Navigation items with bullet style */
        [data-testid="stSidebar"] [role="radiogroup"] {
            gap: 0;
        }
        
        [data-testid="stSidebar"] [role="radiogroup"] label {
            padding: 0.625rem 1.75rem;
            border-radius: 0;
            font-size: 0.9375rem;
            color: #333333;
            transition: all 0.2s ease;
            cursor: pointer;
            font-weight: 400;
            border-left: 0;
            margin: 0;
            position: relative;
            display: flex;
            align-items: center;
        }
        
        /* Bullet icon - inactive (dark gray circle) */
        [data-testid="stSidebar"] [role="radiogroup"] label::before {
            content: "";
            width: 8px;
            height: 8px;
            border-radius: 50%;
            background-color: #333333;
            margin-right: 0.75rem;
            flex-shrink: 0;
            transition: all 0.2s ease;
        }
        
        /* Bullet icon - active (red filled circle) */
        [data-testid="stSidebar"] [role="radiogroup"] label[data-checked="true"]::before {
            background-color: #EF4444;
            width: 10px;
            height: 10px;
        }
        
        [data-testid="stSidebar"] [role="radiogroup"] label:hover {
            background-color: rgba(0, 0, 0, 0.03);
        }
        
        [data-testid="stSidebar"] [role="radiogroup"] label[data-checked="true"] {
            background-color: transparent;
            color: #333333;
            font-weight: 500;
        }
        
        /* Expander (collapsible section) styling */
        [data-testid="stSidebar"] .streamlit-expanderHeader {
            font-size: 0.6875rem;
            font-weight: 600;
            text-transform: uppercase;
            letter-spacing: 0.8px;
            color: #999999;
            padding: 0.75rem 1.75rem;
            border-radius: 12px;
            background: #EBEBEB;
            margin: 0 1rem;
            margin-bottom: 0.5rem;
        }
        
        [data-testid="stSidebar"] .streamlit-expanderHeader:hover {
            background-color: #E0E0E0;
        }
        
        /* Hide the icon and its text completely */
        [data-testid="stSidebar"] .streamlit-expanderHeader svg,
        [data-testid="stSidebar"] .streamlit-expanderHeader p {
            display: none !important;
        }
        
        /* Hide the icon container */
        [data-testid="stSidebar"] .streamlit-expanderHeader div[class*="icon"] {
            display: none !important;
        }
        
        /* Make header text full width */
        [data-testid="stSidebar"] .streamlit-expanderHeader {
            position: relative;
        }
        
        /* Custom arrow for expander */
        [data-testid="stSidebar"] .streamlit-expanderHeader::after {
            content: "›";
            position: absolute;
            right: 1.5rem;
            font-size: 1.125rem;
            font-weight: 600;
            color: #999999;
            transition: transform 0.2s ease;
        }
        
        [data-testid="stSidebar"] details[open] > summary.streamlit-expanderHeader::after {
            transform: rotate(90deg);
        }
        
        /* Expander content padding */
        [data-testid="stSidebar"] .streamlit-expanderContent {
            padding: 0;
            margin-bottom: 1.5rem;
        }
        
        /* Divider lines - subtle */
        [data-testid="stSidebar"] hr {
            margin: 1.5rem 1.75rem;
            border: none;
            border-top: 1px solid #E0E0E0;
        }
        
        /* Progress section */
        [data-testid="stSidebar"] .stProgress {
            margin: 0.5rem 1.75rem;
        }
        
        [data-testid="stSidebar"] p {
            color: #999999;
            font-size: 0.8125rem;
            padding-left: 1.75rem;
        }
        
        /* Caption text */
        [data-testid="stSidebar"] .stCaptionContainer {
            color: #AAAAAA !important;
            padding-left: 1.75rem;
            font-size: 0.75rem;
        }
        
        /* Buttons - minimal */
        [data-testid="stSidebar"] button {
            font-size: 0.8125rem;
            border-radius: 8px;
            border: 1px solid #E0E0E0;
            background: white;
            color: #333333;
            font-weight: 500;
            transition: all 0.2s ease;
            padding: 0.5rem 1rem;
        }
        
        [data-testid="stSidebar"] button:hover {
            background-color: #FAFAFA;
            border-color: #CCCCCC;
        }
        
        /* Header styling */
        [data-testid="stSidebar"] h2 {
            padding-left: 1.75rem !important;
            margin-bottom: 0.25rem !important;
            color: #333333 !important;
            font-weight: 600 !important;
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

