import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import streamlit as st
from utils.ui import apply_page_config, apply_theme
from utils.learning_path import LEARNING_PATH, get_all_modules, calculate_phase_progress, calculate_total_progress
import pandas as pd

# Apply theme
apply_page_config(title="Learning Path", icon="🗺️", sidebar_state="expanded")
apply_theme(page_type="page")

# Render sidebar


# Initialize session state for tracking completed modules
if 'completed_modules' not in st.session_state:
    st.session_state.completed_modules = set()

st.title("🗺️ Your Learning Journey")

st.write("""
Welcome to your personalized AI/ML learning path! This structured curriculum takes you from 
**absolute beginner to advanced practitioner** through hands-on interactive modules.
""")

# Overall progress
total_progress = calculate_total_progress(st.session_state.completed_modules)
col_prog1, col_prog2, col_prog3 = st.columns([2, 1, 1])

with col_prog1:
    st.metric("Overall Progress", f"{total_progress:.0f}%")
    st.progress(total_progress / 100)

with col_prog2:
    completed = len(st.session_state.completed_modules)
    available = len([m for m in get_all_modules() if not m.coming_soon])
    st.metric("Completed", f"{completed}/{available}")

with col_prog3:
    estimated_total = sum(m.estimated_time for m in get_all_modules() if not m.coming_soon)
    st.metric("Total Time", f"{estimated_total} min")

st.divider()

# Learning path by phases
for phase_key, phase_data in LEARNING_PATH.items():
    phase_progress = calculate_phase_progress(phase_key, st.session_state.completed_modules)
    
    # Phase header with status badge
    status_badges = {
        "active": "🟢 Active",
        "coming_soon": "🔵 Coming Soon",
        "planned": "⚪ Planned"
    }
    status = status_badges.get(phase_data["status"], "")
    
    with st.expander(f"**{phase_data['title']}** {status}", expanded=(phase_data["status"] == "active")):
        st.write(phase_data["description"])
        
        if phase_data["status"] == "active":
            st.progress(phase_progress / 100, text=f"{phase_progress:.0f}% Complete")
        
        st.write("")
        
        # Display modules in this phase
        for idx, module in enumerate(phase_data["modules"]):
            # Check if module is unlocked
            prerequisites_met = all(prereq in st.session_state.completed_modules 
                                   for prereq in module.prerequisites)
            is_completed = module.id in st.session_state.completed_modules
            
            # Determine status
            if module.coming_soon:
                status_icon = "🔒"
                status_text = "Coming Soon"
            elif is_completed:
                status_icon = "✅"
                status_text = "Completed"
            elif prerequisites_met:
                status_icon = "🔓"
                status_text = "Available"
            else:
                status_icon = "🔒"
                status_text = "Locked"
            
            # Module card
            col1, col2 = st.columns([3, 1])
            
            with col1:
                difficulty_badge = module.get_difficulty_badge()
                st.markdown(f"""
                **{idx + 1}. {module.icon} {module.title}** {difficulty_badge} {status_icon}
                
                {module.description}
                
                **Topics:** {', '.join(module.topics)} | **Time:** ~{module.estimated_time} min
                """)
                
                # Show prerequisites if any
                if module.prerequisites:
                    prereq_names = []
                    for prereq_id in module.prerequisites:
                        for m in get_all_modules():
                            if m.id == prereq_id:
                                prereq_names.append(m.title)
                                break
                    st.caption(f"Prerequisites: {', '.join(prereq_names)}")
            
            with col2:
                st.write(f"**{status_text}**")
                
                # Action button
                if not module.coming_soon and prerequisites_met and not is_completed:
                    if st.button("Start Learning", key=f"start_{module.id}", use_container_width=True):
                        st.switch_page(module.page_path)
                elif is_completed:
                    if st.button("Review", key=f"review_{module.id}", use_container_width=True):
                        st.switch_page(module.page_path)
                elif not prerequisites_met:
                    st.button("Locked", key=f"locked_{module.id}", disabled=True, use_container_width=True)
            
            if idx < len(phase_data["modules"]) - 1:
                st.write("↓")

st.divider()

# Legend
st.subheader("📖 Legend")

col1, col2, col3 = st.columns(3)

with col1:
    st.write("**Difficulty Levels:**")
    st.write("🟢 Beginner")
    st.write("🟡 Intermediate") 
    st.write("🔴 Advanced")

with col2:
    st.write("**Module Status:**")
    st.write("✅ Completed")
    st.write("🔓 Available")
    st.write("🔒 Locked")

with col3:
    st.write("**Phase Status:**")
    st.write("🟢 Active")
    st.write("🔵 Coming Soon")
    st.write("⚪ Planned")

st.divider()

# Learning tips
st.subheader("💡 Learning Tips")

st.write("""
1. **Follow the sequence** - Each module builds on previous concepts
2. **Hands-on practice** - Use the interactive controls to experiment
3. **Read the theory** - Check the Learning Guide panel on each page
4. **Take your time** - Understanding is more important than speed
5. **Review regularly** - Revisit completed modules to reinforce learning
""")

