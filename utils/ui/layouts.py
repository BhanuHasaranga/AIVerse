"""
Layout components for AIVerse pages.
Handles page layout structures.
"""

import streamlit as st
from utils.learning_path import get_module_by_id


def create_two_column_layout(title, icon=None, module_id=None):
    """
    Create standard 2.5:1 column layout for explorer pages.
    
    Args:
        title: Page title
        icon: Optional icon (for backward compatibility, not displayed)
        module_id: Optional module ID for progress tracking
    
    Returns:
        tuple: (left_column, right_column, module_info)
    """
    # Initialize session state
    if 'completed_modules' not in st.session_state:
        st.session_state.completed_modules = set()
    
    # Title and completion tracking
    col_title, col_complete = st.columns([4, 1])
    
    with col_title:
        st.title(title)
    
    # Module info for later use
    module_info = None
    with col_complete:
        if module_id:
            module = get_module_by_id(module_id)
            if module:
                is_completed = module_id in st.session_state.completed_modules
                module_info = {'module': module, 'is_completed': is_completed}
                
                if is_completed:
                    st.success("✅ Completed")
                    if st.button("Mark Incomplete", key=f"uncomplete_{module_id}", use_container_width=True):
                        st.session_state.completed_modules.discard(module_id)
                        st.rerun()
                else:
                    if st.button("Mark Complete", key=f"complete_{module_id}", use_container_width=True):
                        st.session_state.completed_modules.add(module_id)
                        st.success("🎉 Great job!")
                        st.rerun()
    
    col1, col2 = st.columns([2.5, 1], gap="large")
    
    # Add difficulty badge to top of col2 if module exists and not completed
    if module_info and not module_info['is_completed']:
        with col2:
            difficulty_badge = module_info['module'].get_difficulty_badge()
            difficulty_text = module_info['module'].difficulty
            st.markdown(f"""
                <div style='
                    padding: 0.5rem 0.75rem;
                    margin-bottom: 1rem;
                    border-radius: 6px;
                    background-color: rgba(102, 126, 234, 0.1);
                    border-left: 3px solid #667eea;
                    font-size: 0.875rem;
                '>
                    {difficulty_badge} <strong>{difficulty_text}</strong>
                </div>
            """, unsafe_allow_html=True)
    
    return col1, col2

