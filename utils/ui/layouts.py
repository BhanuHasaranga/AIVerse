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
    col_title, col_complete = st.columns([3, 2])
    
    with col_title:
        st.title(title)
    
    with col_complete:
        if module_id:
            module = get_module_by_id(module_id)
            if module:
                is_completed = module_id in st.session_state.completed_modules
                
                if is_completed:
                    st.success("✅ Completed")
                    if st.button("Mark Incomplete", key=f"uncomplete_{module_id}", use_container_width=True):
                        st.session_state.completed_modules.discard(module_id)
                        st.rerun()
                else:
                    # Badge and button on same row
                    difficulty_badge = module.get_difficulty_badge()
                    difficulty_text = module.difficulty
                    
                    col_badge, col_btn, col_spacer = st.columns([1, 3, 1])
                    with col_badge:
                        st.markdown(f"""
                            <div style='
                                padding-top: 0.5rem;
                                font-size: 1.5rem;
                                text-align: center;
                            '>
                                {difficulty_badge}
                            </div>
                        """, unsafe_allow_html=True)
                    with col_btn:
                        if st.button("Mark Complete", key=f"complete_{module_id}"):
                            st.session_state.completed_modules.add(module_id)
                            st.success("🎉 Great job!")
                            st.rerun()
    
    col1, col2 = st.columns([2.5, 1], gap="large")
    
    return col1, col2

