import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import streamlit as st
from utils.ui_components import apply_page_config, apply_theme, create_two_column_layout, render_theory_panel
from utils.data_components import render_data_input, display_dataset, display_data_info, display_basic_stats
from utils.chart_components import render_frequency_bar_chart
from collections import Counter
import pandas as pd

# Apply theme
apply_page_config(title="Mode Explorer", icon="👑", sidebar_state="expanded")
apply_theme(page_type="page")

# Render sidebar


# Create layout
col1, col2 = create_two_column_layout("Mode Explorer", module_id="mode")

# LEFT COLUMN
with col1:
    st.subheader("Interactive Mode Explorer")
    
    # Data input with categorical support
    data, data_type = render_data_input(allow_categorical=True, data_size_range=(5, 30, 15))
    
    # Display dataset
    if data:
        display_dataset(data, data_type=data_type)
        display_data_info(data, data_type)
        
        if data_type == "Numeric":
            display_basic_stats(data)
        
        # Calculate mode
        freq_counter = Counter(data)
        max_freq = max(freq_counter.values())
        modes = [val for val, count in freq_counter.items() if count == max_freq]
        
        # Determine modality
        if len(modes) == 1:
            modality = "Unimodal"
            mode_text = f"{modes[0]}"
        elif len(modes) == 2:
            modality = "Bimodal"
            mode_text = f"{modes[0]}, {modes[1]}"
        else:
            modality = "Multimodal"
            mode_text = ", ".join(str(m) for m in modes[:3])
        
        # Display metrics
        col_mode, col_freq, col_modal = st.columns(3)
        col_mode.metric("Mode Value(s)", mode_text)
        col_freq.metric("Frequency", f"{max_freq}x")
        col_modal.metric("Modality", modality)
        
        # Render frequency chart
        freq_df = pd.DataFrame([
            {"Value": str(val), "Frequency": count}
            for val, count in freq_counter.items()
        ]).sort_values("Frequency", ascending=False)
        
        render_frequency_bar_chart(freq_df, highlight_values=modes)
        
        # Step-by-step calculation
        with st.expander("Step-by-Step Calculation", expanded=True):
            st.write("**Step 1: Start with the dataset**")
            if data_type == "Numeric":
                data_str = "[" + ", ".join(str(int(v) if v == int(v) else round(v, 2)) for v in data) + "]"
            else:
                data_str = "[" + ", ".join(f'"{v}"' for v in data) + "]"
            st.code(f"data = {data_str}", language="python")
            
            st.write("**Step 2: Count frequency of each value**")
            freq_code = "{\n"
            for val, count in sorted(freq_counter.items(), key=lambda x: x[1], reverse=True):
                freq_code += f'    "{val}": {count},\n'
            freq_code += "}"
            st.code(freq_code, language="python")
            
            st.write("**Step 3: Find the maximum frequency**")
            st.code(f"max_frequency = {max_freq}", language="python")
            
            st.write("**Step 4: Get value(s) with maximum frequency**")
            if len(modes) == 1:
                st.write(f"Mode = **{modes[0]}** (appears {max_freq} times)")
            else:
                st.write(f"Modes = {modes} ({modality})")

# RIGHT COLUMN
with col2:
    def definition():
        st.write("### Definition")
        st.write("""
        The **mode** is the most frequent value in a dataset.
        
        **Modality Types:**
        - **Unimodal:** One mode
        - **Bimodal:** Two modes
        - **Multimodal:** More than two
        """)
        
        st.write("**Example 1 (Unimodal):**")
        st.write("Data: [2, 3, 3, 4, 4, 4, 5]")
        st.latex(r"\text{Mode} = 4")
        
        st.write("**Example 2 (Bimodal):**")
        st.write("Data: [1, 1, 2, 2, 3]")
        st.latex(r"\text{Modes} = 1, 2")
    
    def examples():
        st.write("### Real-World Examples")
        
        st.write("**Example 1: Clothing Store**")
        st.write("Sizes sold: [M, L, L, M, M, S, M, L, L]")
        st.write("→ Mode = M and L (bimodal)")
        st.write("→ Restock these sizes!")
        
        st.write("**Example 2: Survey Data**")
        st.write("Favorite color: [Red, Red, Blue, Blue, Blue, Green]")
        st.write("→ Mode = Blue")
        st.write("→ Most popular!")
    
    def ml_usage():
        st.write("### Mode in AI/ML")
        
        st.write("**1. Categorical Feature Analysis**")
        st.code("df['Gender'].mode()\n# Output: ['Male']", language="python")
        st.write("→ Detect imbalance")
        
        st.write("**2. Data Imputation**")
        st.code("df['City'].fillna(\n    df['City'].mode()[0]\n)", language="python")
        st.write("→ Fill missing with most common")
        
        st.write("**3. Bias Detection**")
        st.write("Analyze if predictions favor one class")
    
    def summary():
        st.write("### Quick Summary")
        
        data = {
            "Concept": ["Mean", "Median", "Mode"],
            "Best For": ["Numeric", "Numeric (skewed)", "Categorical"]
        }
        st.dataframe(pd.DataFrame(data), use_container_width=True, hide_index=True)
        
        st.write("### When to Use Mode")
        st.write("""
        • **Categorical data** (colors, sizes)
        • **Find what's popular**
        • **Detect imbalance** in datasets
        • **Fill missing data** intelligently
        """)
    
    render_theory_panel({
        "Definition": definition,
        "Examples": examples,
        "ML Usage": ml_usage,
        "Summary": summary
    })
