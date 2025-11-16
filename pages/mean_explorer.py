import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import streamlit as st
from utils.ui_components import apply_page_config, apply_theme, create_two_column_layout, render_theory_panel
from utils.data_components import render_data_input, display_dataset, display_data_info, display_basic_stats
from utils.chart_components import render_histogram_with_line
from utils.math_utils import mean
import pandas as pd

# Apply theme
apply_page_config(title="Mean Explorer", icon="📊")
apply_theme(page_type="page")

# Create layout with progress tracking
col1, col2 = create_two_column_layout("Mean Explorer", module_id="mean")

# LEFT COLUMN - Interactive explorer
with col1:
    st.subheader("Interactive Mean Explorer")
    
    # Data input component
    data = render_data_input()
    
    # Display dataset
    if data:
        display_dataset(data)
        display_data_info(data)
        display_basic_stats(data)
        
        # Calculate and display mean
        m = mean(data)
        st.metric("Mean Value", f"{m:.2f}")
        
        # Render histogram with mean line
        render_histogram_with_line(data, m, "Mean", "Histogram of Values", "red")
        
        # Step-by-step calculation
        with st.expander("Step-by-Step Calculation", expanded=True):
            st.write("**Step 1: Start with the dataset**")
            data_str = "[" + ", ".join(str(int(v) if v == int(v) else round(v, 2)) for v in data) + "]"
            st.code(f"data = {data_str}", language="python")
            
            st.write("**Step 2: Calculate the sum of all values**")
            sum_value = sum(data)
            st.latex(r"\text{Sum} = " + " + ".join(str(int(v) if v == int(v) else round(v, 2)) for v in data) + f" = {sum_value}")
            st.code(f"sum(data) = {sum_value}", language="python")
            
            st.write("**Step 3: Count the number of values (n)**")
            n = len(data)
            st.code(f"n = len(data) = {n}", language="python")
            
            st.write("**Step 4: Divide sum by count**")
            st.latex(fr"\text{{Mean}} = \frac{{\text{{Sum}}}}{{n}} = \frac{{{sum_value}}}{{{n}}} = {m:.2f}")
            st.code(f"mean = sum(data) / n = {sum_value} / {n} = {m:.2f}", language="python")
            
            st.write("**Step 5: Using the mean() function**")
            st.code(f"mean(data) = {m:.2f}", language="python")

# RIGHT COLUMN - Theory panel
with col2:
    def definition_content():
        st.write("### Definition")
        st.write("""
        The **mean** is the sum of all numbers divided by how many numbers there are.
        
        It tells us the **central value** of a dataset — where most of the data tends to cluster.
        
        **Formula:**
        """)
        st.latex(r"Mean = \frac{\text{Sum of all values}}{\text{Number of values}}")
        
        st.write("**Simple Example:**")
        st.write("Data: [2, 4, 6, 8, 10]")
        st.latex(r"Mean = \frac{2 + 4 + 6 + 8 + 10}{5} = \frac{30}{5} = 6")
        
        st.write("### Simple Explanation")
        st.write("""
        Imagine you have friends and each brings snacks.
        If you **share all snacks equally**, the amount each person gets is the mean.
        
        It answers: *"If everything were balanced equally, what would the middle value be?"*
        """)
    
    def examples_content():
        st.write("### Real-World Examples")
        
        st.write("**Example 1: Salary Analysis**")
        st.write("Company salaries: [30k, 40k, 50k, 80k, 100k]")
        st.latex(r"Mean = \frac{30 + 40 + 50 + 80 + 100}{5} = 60k")
        st.write("→ Average employee earns ~**60k**")
        
        st.write("**Example 2: Daily Temperature**")
        st.write("7 days: [29, 30, 28, 31, 30, 29, 30]°C")
        st.latex(r"Mean = \frac{29+30+28+31+30+29+30}{7} ≈ 29.6°C")
        st.write("→ Average daily temperature: **29.6°C**")
    
    def ml_usage_content():
        st.write("### Mean in AI & Machine Learning")
        
        st.write("**1. Data Normalization**")
        st.write("Before training, we center data around 0:")
        st.latex(r"x_{normalized} = x - mean(x)")
        st.write("→ Helps models converge faster")
        
        st.write("**2. Loss Functions (MSE)**")
        st.write("Most ML models minimize mean error:")
        st.latex(r"MSE = \frac{1}{n}\sum(y_{pred} - y_{true})^2")
        st.write("→ Used in regression to measure prediction accuracy")
        
        st.write("**3. Batch Normalization**")
        st.write("""
        Neural networks normalize batches using mean & variance.
        → Stabilizes training, improves convergence
        """)
    
    def summary_content():
        st.write("### Quick Summary")
        
        summary_data = {
            "Concept": ["Definition", "Purpose", "Formula", "Why Important", "AI Usage"],
            "Description": [
                "Central average of data",
                "Summarize with one number",
                "(Sum) / (Count)",
                "Understand trends & build models",
                "Normalization, loss functions"
            ]
        }
        
        summary_df = pd.DataFrame(summary_data)
        st.dataframe(summary_df, use_container_width=True, hide_index=True)
        
        st.write("### Why the Mean Matters")
        st.write("""
        • **Single representative number** for your dataset
        • **Compares groups** or trends
        • **Foundation for** variance, std deviation, z-scores
        • **Essential in** data science & ML pipelines
        • **Balances** the dataset perfectly (like a seesaw)
        """)
    
    render_theory_panel({
        "Definition": definition_content,
        "Examples": examples_content,
        "ML Usage": ml_usage_content,
        "Summary": summary_content
    })
