import sys
from pathlib import Path

# Add parent directory to Python path
sys.path.insert(0, str(Path(__file__).parent.parent))

import streamlit as st
from utils.data_utils import generate_random_data
import plotly.express as px
import plotly.graph_objects as go
import pandas as pd
import numpy as np
import math

# Set page layout to wide mode with custom margins
st.set_page_config(layout="wide", initial_sidebar_state="collapsed")

# Custom CSS to reduce margins and padding
st.markdown("""
    <style>
    .main {
        max-width: 100%;
        padding-left: 1rem;
        padding-right: 1rem;
    }
    [data-testid="stMetricContainer"] {
        background-color: rgba(28, 131, 225, 0.1);
        padding: 10px;
        border-radius: 5px;
    }
    </style>
    """, unsafe_allow_html=True)

st.title("Variance & Standard Deviation Explorer")

# Create two columns with increased gap: 2.5:1 ratio (main content : theory)
col1, col2 = st.columns([2.5, 1], gap="large")

# LEFT COLUMN (2.5/3.5 width) - Interactive chart and controls
with col1:
    st.subheader("Interactive Variance Explorer")
    
    # Data input method selection
    input_method = st.radio("Choose data input method:", ["Generate Random", "Upload CSV", "Enter Manually"], horizontal=True)
    
    if input_method == "Generate Random":
        # Dataset controls for random data
        data_size = st.slider("Select dataset size:", min_value=5, max_value=50, value=20)
        
        # Spread control
        st.write("**Adjust data spread:**")
        spread_type = st.radio("Data characteristics:", ["Tight (Low Variance)", "Medium (Moderate Variance)", "Wide (High Variance)"], horizontal=True)
        
        if st.button("Generate Random Data"):
            if spread_type == "Tight (Low Variance)":
                mean_val = 50
                std_dev = 3
            elif spread_type == "Medium (Moderate Variance)":
                mean_val = 50
                std_dev = 10
            else:  # Wide
                mean_val = 50
                std_dev = 20
            
            # Generate normally distributed data
            data = list(np.random.normal(mean_val, std_dev, data_size))
            st.session_state['data'] = data
        
        data = st.session_state.get('data', [])
    
    elif input_method == "Upload CSV":
        # CSV upload
        uploaded_file = st.file_uploader("Upload CSV file (single column with numbers):", type=['csv'])
        if uploaded_file is not None:
            try:
                df = pd.read_csv(uploaded_file)
                data = df.iloc[:, 0].tolist()
                st.session_state['data'] = data
                st.success("• Data loaded successfully!")
            except Exception as e:
                st.error(f"• Error reading file: {e}")
            
            data = st.session_state.get('data', [])
        else:
            data = st.session_state.get('data', [])
    
    else:  # Enter Manually
        # Manual input
        st.write("**Enter numbers separated by commas:**")
        manual_input = st.text_input(
            "Example: 10, 12, 14, 16, 18",
            placeholder="Enter your data here..."
        )
        
        if manual_input:
            try:
                # Parse the input
                data = [float(x.strip()) for x in manual_input.split(',')]
                st.session_state['data'] = data
                st.success(f"• Loaded {len(data)} values")
            except ValueError:
                st.error("• Invalid input. Please enter numbers separated by commas.")
                data = []
        else:
            data = st.session_state.get('data', [])
    
    # Display the dataset in computer science format
    if data:
        # Format data as [value1, value2, value3, ...]
        data_str = "[" + ", ".join(str(int(v) if v == int(v) else round(v, 2)) for v in data) + "]"
        
        # Use expander for large datasets to save space
        if len(data) > 15:
            with st.expander(f"Dataset ({len(data)} values)", expanded=False):
                st.code(data_str, language="python")
        else:
            st.code(data_str, language="python")
        
        st.write(f"**Data points:** {len(data)}")
        
        # Calculate statistics
        mean_val = np.mean(data)
        variance = np.var(data)
        std_dev = np.std(data)
        
        # Display key metrics
        col_mean, col_var, col_std = st.columns(3)
        col_mean.metric("Mean (μ)", f"{mean_val:.2f}")
        col_var.metric("Variance (σ²)", f"{variance:.2f}")
        col_std.metric("Std Dev (σ)", f"{std_dev:.2f}")
        
        # Additional statistics
        col_min, col_max, col_range = st.columns(3)
        col_min.metric("Min Value", f"{min(data):.2f}")
        col_max.metric("Max Value", f"{max(data):.2f}")
        col_range.metric("Range", f"{max(data) - min(data):.2f}")

    # Plot histogram with normal distribution overlay
    if data:
        df = pd.DataFrame({"Values": data})
        fig = px.histogram(df, x="Values", nbins=max(len(data)//3, 5), title="Distribution of Values",
                          labels={"Values": "Values", "count": "Frequency"})
        
        # Add mean line
        fig.add_vline(x=mean_val, line_dash="dash", line_color="red", 
                     annotation_text=f"Mean ({mean_val:.2f})", annotation_position="top right")
        
        # Add standard deviation bands
        fig.add_vline(x=mean_val - std_dev, line_dash="dot", line_color="orange", 
                     annotation_text=f"μ - σ", annotation_position="top left")
        fig.add_vline(x=mean_val + std_dev, line_dash="dot", line_color="orange",
                     annotation_text=f"μ + σ", annotation_position="top left")
        
        st.plotly_chart(fig, use_container_width=True)
        
        # Visualize the 68-95-99.7 rule
        st.write("### Standard Deviation Rule (68-95-99.7)")
        
        within_1sigma = sum(1 for x in data if mean_val - std_dev <= x <= mean_val + std_dev) / len(data) * 100
        within_2sigma = sum(1 for x in data if mean_val - 2*std_dev <= x <= mean_val + 2*std_dev) / len(data) * 100
        within_3sigma = sum(1 for x in data if mean_val - 3*std_dev <= x <= mean_val + 3*std_dev) / len(data) * 100
        
        col_1sig, col_2sig, col_3sig = st.columns(3)
        col_1sig.metric("Within ±1σ", f"{within_1sigma:.1f}%", f"(Expected: ~68%)")
        col_2sig.metric("Within ±2σ", f"{within_2sigma:.1f}%", f"(Expected: ~95%)")
        col_3sig.metric("Within ±3σ", f"{within_3sigma:.1f}%", f"(Expected: ~99.7%)")
        
        # Step-by-step calculation (after chart)
        with st.expander("� Step-by-Step Calculation", expanded=True):
            # Step 1: Show the data
            st.write("**Step 1: Start with the dataset**")
            st.code(f"data = {data_str}", language="python")
            
            # Step 2: Calculate mean
            st.write("**Step 2: Calculate the mean (μ)**")
            st.latex(fr"\mu = \frac{{\sum x_i}}{{n}} = \frac{{{sum(data):.2f}}}{{{len(data)}}} = {mean_val:.2f}")
            st.code(f"mean = sum(data) / n = {mean_val:.2f}", language="python")
            
            # Step 3: Calculate differences from mean
            st.write("**Step 3: Calculate differences from mean (xᵢ - μ)**")
            differences = [f"{x - mean_val:.2f}" for x in data[:5]]
            if len(data) > 5:
                differences.append("...")
            st.write("Differences: [" + ", ".join(differences) + "]")
            
            # Step 4: Square the differences
            st.write("**Step 4: Square the differences**")
            squared_diffs = [f"{(x - mean_val)**2:.2f}" for x in data[:5]]
            if len(data) > 5:
                squared_diffs.append("...")
            st.write("Squared: [" + ", ".join(squared_diffs) + "]")
            
            # Step 5: Calculate variance
            st.write("**Step 5: Calculate variance (average of squared differences)**")
            st.latex(fr"\sigma^2 = \frac{{\sum(x_i - \mu)^2}}{{n}} = {variance:.2f}")
            st.code(f"variance = sum((x - mean)² for x in data) / n = {variance:.2f}", language="python")
            
            # Step 6: Calculate standard deviation
            st.write("**Step 6: Calculate standard deviation (√variance)**")
            st.latex(fr"\sigma = \sqrt{{\sigma^2}} = \sqrt{{{variance:.2f}}} = {std_dev:.2f}")
            st.code(f"std_dev = sqrt(variance) = {std_dev:.2f}", language="python")
            
            # Step 7: Interpretation
            st.write("**Step 7: Interpretation**")
            st.write(f"📌 Most values are about **{std_dev:.2f}** units away from the mean ({mean_val:.2f})")
            st.write(f"📌 ~68% of data falls within [{mean_val - std_dev:.2f}, {mean_val + std_dev:.2f}]")

# RIGHT COLUMN (1/3.5 width) - Theory notes
with col2:
    st.subheader("📚 Learning Guide")
    
    # Tab-based navigation for different content sections
    tab1, tab2, tab3, tab4 = st.tabs(["Definition", "Examples", "ML Usage", "Summary"])
    
    with tab1:
        st.write("### Definition")
        st.write("""
        Both **variance** and **standard deviation** measure how spread out data is from the mean.
        """)
        
        st.write("**Variance (σ²):**")
        st.write("Average of squared differences from mean")
        st.latex(r"\sigma^2 = \frac{1}{n}\sum(x_i - \mu)^2")
        
        st.write("**Standard Deviation (σ):**")
        st.write("Square root of variance (easier to interpret)")
        st.latex(r"\sigma = \sqrt{\sigma^2}")
        
        st.write("### Simple Explanation")
        st.write("""
        **Spread = how inconsistent your data is.**
        
        Tight data → Low variance → Consistent results
        
        Spread data → High variance → Scattered results
        
        Standard deviation is the same idea, but in original units.
        """)
    
    with tab2:
        st.write("### Real-World Examples")
        
        st.write("**Example 1: Test Scores**")
        st.write("""
        Class A: [65, 67, 70, 72, 73]
        → Low variance → Consistent performance
        
        Class B: [40, 55, 70, 85, 100]
        → High variance → Wildly different scores
        """)
        
        st.write("**Example 2: Stock Returns**")
        st.write("""
        Stock A: avg 10%, variance 2
        → Steady performer • Stock B: avg 10%, variance 50
        → Risky, unpredictable • """)
        
        st.write("**Example 3: Manufacturing**")
        st.write("""
        Low variance → Quality consistent • High variance → Defects likely • """)
    
    with tab3:
        st.write("### Variance in AI/ML")
        
        st.write("**1. Model Variance (Bias-Variance Tradeoff)**")
        st.write("""
        - High variance → Overfits (memorizes)
        - Low variance → Underfits (too simple)
        - Goal: Balance! 🎯
        """)
        
        st.write("**2. Loss Functions**")
        st.write("""
        MSE squares errors (= variance):
        """)
        st.latex(r"MSE = \frac{1}{n}\sum(y_{pred} - y_{true})^2")
        
        st.write("**3. Feature Scaling (Z-score)**")
        st.write("""
        Standardize with std dev:
        """)
        st.latex(r"Z = \frac{x - \mu}{\sigma}")
        st.write("→ Helps ML models converge faster")
        
        st.write("**4. Anomaly Detection**")
        st.write("""
        Detect outliers using 3σ rule:
        Values beyond ±3σ = anomalies • """)
    
    with tab4:
        st.write("### Quick Summary")
        
        summary_data = {
            "Concept": ["Variance (σ²)", "Std Dev (σ)", "Low Variance", "High Variance"],
            "Meaning": [
                "Avg squared differences",
                "√Variance (easier units)",
                "Tight, consistent data",
                "Wide, scattered data"
            ],
            "In AI": [
                "Detect overfitting",
                "Feature scaling",
                "Good generalization",
                "Model overfits"
            ]
        }
        
        summary_df = pd.DataFrame(summary_data)
        st.dataframe(summary_df, use_container_width=True, hide_index=True)
        
        st.write("### Why It Matters")
        st.write("""
        • Understand data stability
        • Prevent overfitting
        • Scale features properly
        • Detect anomalies
        • Build reliable ML models
        """)
        
        st.write("### 68-95-99.7 Rule")
        st.write("""
        For normal distributions:
        
        68% within ±1σ
        95% within ±2σ
        99.7% within ±3σ
        
        Used in anomaly detection! 🎯
        """)
