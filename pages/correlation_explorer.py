import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import streamlit as st
from utils.ui_components import apply_page_config, apply_theme, create_two_column_layout, render_theory_panel
from utils.chart_components import render_scatter_with_regression
import pandas as pd
import numpy as np

# Apply theme
apply_page_config(title="Correlation Explorer", icon="", sidebar_state="expanded")
apply_theme(page_type="page")

# Render sidebar


# Create layout
col1, col2 = create_two_column_layout("Correlation & Covariance Explorer", module_id="correlation")

# LEFT COLUMN
with col1:
    st.subheader("Interactive Correlation Explorer")
    
    # Data input selection
    input_method = st.radio("Choose data input method:", 
                           ["Generate Correlated Data", "Upload CSV", "Enter Manually"], 
                           horizontal=True)
    
    if input_method == "Generate Correlated Data":
        data_size = st.slider("Select dataset size:", min_value=10, max_value=100, value=30)
        corr_type = st.radio("Relationship:", 
                            ["Positive (↗)", "Negative (↘)", "No Correlation (Random)"], 
                            horizontal=True)
        
        if st.button("Generate Data"):
            x = np.random.uniform(0, 100, data_size)
            
            if corr_type == "Positive (↗)":
                y = x + np.random.normal(0, 5, data_size)
            elif corr_type == "Negative (↘)":
                y = 100 - x + np.random.normal(0, 5, data_size)
            else:
                y = np.random.uniform(0, 100, data_size)
            
            data = pd.DataFrame({"X": x, "Y": y})
            st.session_state['data'] = data
        
        data = st.session_state.get('data', pd.DataFrame())
    
    elif input_method == "Upload CSV":
        uploaded_file = st.file_uploader("Upload CSV (2+ numeric columns):", type=['csv'])
        if uploaded_file:
            try:
                df = pd.read_csv(uploaded_file)
                numeric_cols = df.select_dtypes(include=[np.number]).columns[:2]
                if len(numeric_cols) >= 2:
                    data = df[numeric_cols].copy()
                    data.columns = ["X", "Y"]
                    st.session_state['data'] = data
                    st.success("• Data loaded!")
                else:
                    st.error("• Need 2+ numeric columns")
                    data = pd.DataFrame()
            except Exception as e:
                st.error(f"• Error: {e}")
                data = pd.DataFrame()
        else:
            data = st.session_state.get('data', pd.DataFrame())
    
    else:  # Manual
        st.write("**Enter X, Y pairs (comma-separated, one per line):**")
        st.write("Example: 2, 40")
        
        manual_input = st.text_area(
            "Enter X, Y pairs:",
            placeholder="2, 40\n4, 50\n6, 60",
            height=150
        )
        
        if manual_input:
            try:
                lines = [line.strip() for line in manual_input.split('\n') if line.strip()]
                x_vals, y_vals = [], []
                for line in lines:
                    parts = line.split(',')
                    if len(parts) == 2:
                        x_vals.append(float(parts[0].strip()))
                        y_vals.append(float(parts[1].strip()))
                
                if len(x_vals) >= 3:
                    data = pd.DataFrame({"X": x_vals, "Y": y_vals})
                    st.session_state['data'] = data
                    st.success(f"• Loaded {len(x_vals)} points")
                else:
                    st.error("• Need 3+ data points")
                    data = pd.DataFrame()
            except:
                st.error("• Invalid format")
                data = pd.DataFrame()
        else:
            data = st.session_state.get('data', pd.DataFrame())
    
    # Analyze and display
    if not data.empty and len(data) >= 3:
        st.write(f"**Data points:** {len(data)}")
        
        with st.expander(f"Dataset ({len(data)} pairs)", expanded=False):
            st.dataframe(data, use_container_width=True)
        
        # Calculate statistics
        x_mean, y_mean = data['X'].mean(), data['Y'].mean()
        x_std, y_std = data['X'].std(), data['Y'].std()
        covariance = ((data['X'] - x_mean) * (data['Y'] - y_mean)).mean()
        correlation = covariance / (x_std * y_std)
        
        # Display metrics
        col_cov, col_corr, col_str = st.columns(3)
        col_cov.metric("Covariance", f"{covariance:.2f}")
        col_corr.metric("Correlation (r)", f"{correlation:.4f}")
        
        # Strength interpretation
        if abs(correlation) >= 0.9:
            strength, color = "Very Strong", ""
        elif abs(correlation) >= 0.7:
            strength, color = "Strong", ""
        elif abs(correlation) >= 0.5:
            strength, color = "Moderate", ""
        elif abs(correlation) >= 0.3:
            strength, color = "Weak", ""
        else:
            strength, color = "Very Weak", ""
        
        col_str.metric("Strength", f"{color} {strength}")
        
        # Additional stats
        col1a, col2a, col3a, col4a = st.columns(4)
        col1a.metric("X Mean", f"{x_mean:.2f}")
        col2a.metric("Y Mean", f"{y_mean:.2f}")
        col3a.metric("X Std Dev", f"{x_std:.2f}")
        col4a.metric("Y Std Dev", f"{y_std:.2f}")
        
        # Render scatter plot
        render_scatter_with_regression(data, correlation=correlation)
        
        # Step-by-step calculation
        with st.expander("Step-by-Step Calculation", expanded=True):
            st.write("**Step 1: Calculate means**")
            st.latex(fr"\bar{{x}} = {x_mean:.2f}, \quad \bar{{y}} = {y_mean:.2f}")
            
            st.write("**Step 2: Calculate covariance**")
            st.latex(fr"\text{{Cov}}(X,Y) = \frac{{\sum(x_i - \bar{{x}})(y_i - \bar{{y}})}}{{n}} = {covariance:.2f}")
            
            st.write("**Step 3: Calculate standard deviations**")
            st.latex(fr"\sigma_X = {x_std:.2f}, \quad \sigma_Y = {y_std:.2f}")
            
            st.write("**Step 4: Calculate correlation**")
            st.latex(fr"r = \frac{{\text{{Cov}}(X,Y)}}{{\sigma_X \times \sigma_Y}} = {correlation:.4f}")
            
            st.write("**Step 5: Interpretation**")
            if correlation > 0.7:
                st.write(f"• **Strong positive relationship** (r = {correlation:.2f})")
            elif correlation > 0.3:
                st.write(f"• **Moderate positive relationship** (r = {correlation:.2f})")
            elif correlation > -0.3:
                st.write(f"• **Weak/No relationship** (r = {correlation:.2f})")
            elif correlation > -0.7:
                st.write(f"• **Moderate negative relationship** (r = {correlation:.2f})")
            else:
                st.write(f"• **Strong negative relationship** (r = {correlation:.2f})")

# RIGHT COLUMN
with col2:
    def definition():
        st.write("### Definition")
        st.write("**Covariance** measures how two variables change together.")
        st.latex(r"\text{Cov}(X,Y) = \frac{1}{n}\sum(x_i - \bar{x})(y_i - \bar{y})")
        
        st.write("**Correlation** is normalized covariance (-1 to +1).")
        st.latex(r"r = \frac{\text{Cov}(X,Y)}{\sigma_X \sigma_Y}")
        
        st.write("""
        - **+1** = Perfect positive
        - **0** = No relationship
        - **-1** = Perfect negative
        """)
    
    def examples():
        st.write("### Real-World Examples")
        
        st.write("**Example 1: Study Hours vs Scores**")
        st.latex(r"r \approx +0.85")
        st.write("→ More study → Higher scores • ")
        
        st.write("**Example 2: Temperature vs Ice Cream Sales**")
        st.latex(r"r \approx +0.95")
        st.write("→ Hotter → More sales • ")
        
        st.write("**Example 3: Rainfall vs Sunscreen**")
        st.latex(r"r \approx -0.70")
        st.write("→ More rain → Less sunscreen • ")
    
    def ml_usage():
        st.write("### In AI/ML")
        
        st.write("**1. Feature Selection**")
        st.write("High correlation = redundant features")
        
        st.write("**2. Dimensionality Reduction**")
        st.write("PCA uses covariance matrix")
        
        st.write("**3. Anomaly Detection**")
        st.write("Low correlation = potential anomaly")
    
    def summary():
        st.write("### Quick Summary")
        
        data = {
            "Concept": ["Covariance", "Correlation"],
            "Meaning": ["Direction", "Strength"],
            "Range": ["-∞ to +∞", "-1 to +1"]
        }
        st.dataframe(pd.DataFrame(data), use_container_width=True, hide_index=True)
        
        st.write("### Strength Guide")
        st.write("""
        |r| ≥ 0.9 → Very Strong 0.7 ≤ |r| < 0.9 → Strong 0.5 ≤ |r| < 0.7 → Moderate 0.3 ≤ |r| < 0.5 → Weak |r| < 0.3 → Very Weak """)
    
    render_theory_panel({
        "Definition": definition,
        "Examples": examples,
        "ML Usage": ml_usage,
        "Summary": summary
    })
