import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import streamlit as st
from utils.ui import apply_page_config, apply_theme, create_two_column_layout, render_theory_panel
from utils.data_components import render_data_input, display_dataset, display_data_info, display_basic_stats
from utils.chart_components import render_histogram_with_line
import pandas as pd

# Apply theme
apply_page_config(title="Median Explorer", icon="", sidebar_state="expanded")
apply_theme(page_type="page")

# Render sidebar


# Create layout
col1, col2 = create_two_column_layout("Median Explorer", module_id="median")

# LEFT COLUMN
with col1:
    st.subheader("Interactive Median Explorer")
    
    # Data input
    data = render_data_input()
    
    # Sort data for median calculation
    if data:
        data = sorted(data)
        st.session_state['data'] = data
    
    # Display dataset
    if data:
        display_dataset(data)
        display_data_info(data)
        display_basic_stats(data)
        
        # Calculate median
        n = len(data)
        if n % 2 == 0:
            median = (data[n//2 - 1] + data[n//2]) / 2
        else:
            median = data[n//2]
        
        st.metric("Median Value", f"{median:.2f}")
        
        # Render histogram
        render_histogram_with_line(data, median, "Median", "Histogram of Values", "green")
        
        # Step-by-step calculation
        with st.expander("Step-by-Step Calculation", expanded=True):
            st.write("**Step 1: Start with the sorted dataset**")
            data_str = "[" + ", ".join(str(int(v) if v == int(v) else round(v, 2)) for v in data) + "]"
            st.code(f"data = {data_str}", language="python")
            
            st.write("**Step 2: Count the number of values (n)**")
            st.code(f"n = len(data) = {n}", language="python")
            
            st.write("**Step 3: Check if n is odd or even**")
            if n % 2 == 1:
                st.write(f"n = {n} is **odd**")
                st.latex(r"\text{Middle position} = \frac{n + 1}{2} = \frac{" + str(n) + " + 1}{2} = " + str((n + 1) // 2))
                st.code(f"median = data[{n//2}] = {median:.2f}", language="python")
            else:
                st.write(f"n = {n} is **even**")
                st.latex(r"\text{Median} = \frac{\text{data}[" + str(n//2 - 1) + "] + \text{data}[" + str(n//2) + "]}{2}")
                st.code(f"median = (data[{n//2 - 1}] + data[{n//2}]) / 2 = ({data[n//2 - 1]} + {data[n//2]}) / 2 = {median:.2f}", language="python")
            
            st.write("**Step 4: The Median Value**")
            st.latex(fr"\text{{Median}} = {median:.2f}")

# RIGHT COLUMN
with col2:
    def definition():
        st.write("### Definition")
        st.write("""
        The **median** is the middle value of a dataset when all numbers are arranged in order.
        
        It tells us the **central value** that divides the dataset into two equal halves.
        """)
        st.write("**If n is odd:** Middle value")
        st.write("**If n is even:** Average of two middle values")
        
        st.write("**Example 1 (Odd count - n=5):**")
        st.write("Data: [2, 4, 6, 8, 10]")
        st.latex(r"\text{Middle value (3rd) } = 6")
        
        st.write("**Example 2 (Even count - n=4):**")
        st.write("Data: [3, 5, 7, 9]")
        st.latex(r"\text{Median} = \frac{5 + 7}{2} = 6")
    
    def examples():
        st.write("### Real-World Examples")
        
        st.write("**Example 1: House Prices**")
        st.write("Prices: [30M, 35M, 40M, 150M, 200M]")
        st.latex(r"\text{Median} = 40M")
        st.write("→ Even with luxury houses, typical price is **40M**")
        
        st.write("**Example 2: Salaries (Why Median Matters)**")
        st.write("Salaries: [40k, 45k, 50k, 55k, 1,000k]")
        col_a, col_b = st.columns(2)
        with col_a:
            st.latex(r"\text{Mean} = 238k")
            st.write("• Misleading!")
        with col_b:
            st.latex(r"\text{Median} = 50k")
            st.write("• Realistic!")
    
    def ml_usage():
        st.write("### Median in AI/ML")
        
        st.write("**1. Data Cleaning & Imputation**")
        st.write("""
        - Replace missing values with median
        - More robust for noisy data
        - Reduce skewness in distributions
        """)
        
        st.write("**2. Robust Error Metrics**")
        st.latex(r"\text{MedAE} = \text{median}(|y_{pred} - y_{true}|)")
        st.write("→ More robust than MAE for outliers")
        
        st.write("**3. Feature Engineering**")
        st.write("Features grouped and summarized using median for stability")
    
    def summary():
        st.write("### Quick Summary")
        
        data = {
            "Concept": ["Definition", "Purpose", "Advantage"],
            "Description": [
                "Middle value when sorted",
                "Find central tendency",
                "Resists outliers"
            ]
        }
        st.dataframe(pd.DataFrame(data), width='stretch', hide_index=True)
        
        st.write("### Mean vs Median")
        comp = {
            "Aspect": ["Affected by outliers?", "Best for"],
            "Mean": ["• Yes", "Normal distributions"],
            "Median": ["• No", "Skewed distributions"]
        }
        st.dataframe(pd.DataFrame(comp), width='stretch', hide_index=True)
    
    render_theory_panel({
        "Definition": definition,
        "Examples": examples,
        "ML Usage": ml_usage,
        "Summary": summary
    })
