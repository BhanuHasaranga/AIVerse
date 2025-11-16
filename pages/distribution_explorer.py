import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import streamlit as st
from utils.ui_components import apply_page_config, apply_theme, create_two_column_layout, render_theory_panel
import pandas as pd
import numpy as np
from scipy import stats
import plotly.express as px

# Apply theme
apply_page_config(title="Distribution Explorer", icon="🔔")
apply_theme(page_type="page")

# Create layout
col1, col2 = create_two_column_layout("Distribution Explorer")

# LEFT COLUMN
with col1:
    st.subheader("Interactive Distribution Visualizer")
    
    # Distribution type selection
    dist_type = st.radio("Choose distribution type:", 
                         ["Normal (Bell Curve)", "Uniform (Flat)", "Right-Skewed", "Left-Skewed"], 
                         horizontal=True)
    
    # Sample size control
    sample_size = st.slider("Sample size:", min_value=100, max_value=10000, value=2000, step=100)
    
    # Generate data based on distribution type
    if dist_type == "Normal (Bell Curve)":
        data = np.random.normal(loc=50, scale=15, size=sample_size)
        dist_name = "Normal Distribution"
        description = "Symmetrical bell curve - most values cluster around mean"
    elif dist_type == "Uniform (Flat)":
        data = np.random.uniform(low=0, high=100, size=sample_size)
        dist_name = "Uniform Distribution"
        description = "Flat shape - equal probability across range"
    elif dist_type == "Right-Skewed":
        data = np.random.exponential(scale=20, size=sample_size) + 10
        dist_name = "Right-Skewed Distribution"
        description = "Tail extends to right - few very large values"
    else:  # Left-Skewed
        data = 100 - np.random.exponential(scale=20, size=sample_size)
        dist_name = "Left-Skewed Distribution"
        description = "Tail extends to left - few very small values"
    
    # Calculate statistics
    mean_val = np.mean(data)
    median_val = np.median(data)
    mode_val = stats.mode(data, keepdims=True).mode[0]
    std_dev = np.std(data)
    skewness = stats.skew(data)
    kurtosis_val = stats.kurtosis(data)
    
    st.write(f"### {dist_name}")
    st.write(description)
    
    # Display key statistics
    col_mean, col_median, col_mode, col_std = st.columns(4)
    col_mean.metric("Mean (μ)", f"{mean_val:.2f}")
    col_median.metric("Median", f"{median_val:.2f}")
    col_mode.metric("Mode", f"{mode_val:.2f}")
    col_std.metric("Std Dev (σ)", f"{std_dev:.2f}")
    
    col_skew, col_kurt, col_min, col_max = st.columns(4)
    col_skew.metric("Skewness", f"{skewness:.3f}")
    col_kurt.metric("Kurtosis", f"{kurtosis_val:.3f}")
    col_min.metric("Min", f"{np.min(data):.2f}")
    col_max.metric("Max", f"{np.max(data):.2f}")
    
    # Create histogram with statistics
    fig = px.histogram(pd.DataFrame({"Values": data}), x="Values", nbins=50,
                      title=f"{dist_name} (n={sample_size})",
                      labels={"Values": "Values", "count": "Frequency"})
    
    # Add vertical lines for mean, median
    fig.add_vline(x=mean_val, line_dash="dash", line_color="red",
                 annotation_text=f"Mean: {mean_val:.2f}", annotation_position="top right")
    fig.add_vline(x=median_val, line_dash="dash", line_color="green",
                 annotation_text=f"Median: {median_val:.2f}", annotation_position="top left")
    
    # Add shaded regions for standard deviation (for normal distribution)
    if dist_type == "Normal (Bell Curve)":
        fig.add_vrect(x0=mean_val - std_dev, x1=mean_val + std_dev,
                     fillcolor="blue", opacity=0.1, line_width=0,
                     annotation_text="±1σ (68%)", annotation_position="top")
    
    st.plotly_chart(fig, use_container_width=True)
    
    # Skewness & Kurtosis explanation
    st.write("### Distribution Characteristics")
    
    col_skew_exp, col_kurt_exp = st.columns(2)
    
    with col_skew_exp:
        st.write("**Skewness:**")
        if skewness > 0.5:
            st.write(f"**Right-Skewed** ({skewness:.3f})")
            st.write("Long tail to the right - few very large values")
        elif skewness < -0.5:
            st.write(f"**Left-Skewed** ({skewness:.3f})")
            st.write("Long tail to the left - few very small values")
        else:
            st.write(f"**Approximately Symmetric** ({skewness:.3f})")
            st.write("Balanced distribution - mean ≈ median")
    
    with col_kurt_exp:
        st.write("**Kurtosis:**")
        if kurtosis_val > 0:
            st.write(f"**Leptokurtic** ({kurtosis_val:.3f})")
            st.write("Sharp peak, heavy tails - outliers more likely")
        elif kurtosis_val < 0:
            st.write(f"**Platykurtic** ({kurtosis_val:.3f})")
            st.write("Flat peak, light tails - fewer outliers")
        else:
            st.write(f"**Mesokurtic** ({kurtosis_val:.3f})")
            st.write("Normal level of peak and tails")
    
    # Step-by-step interpretation
    with st.expander("Distribution Analysis", expanded=True):
        st.write("**Step 1: Identify Distribution Type**")
        st.write(f"Current: {dist_name}")
        
        st.write("**Step 2: Check Central Tendency**")
        st.write(f"Mean: {mean_val:.2f}")
        st.write(f"Median: {median_val:.2f}")
        if abs(mean_val - median_val) < 1:
            st.write("• Mean ≈ Median → Symmetric distribution")
        elif mean_val > median_val:
            st.write("• Mean > Median → Right-skewed distribution")
        else:
            st.write("• Mean < Median → Left-skewed distribution")
        
        st.write("**Step 3: Calculate Spread**")
        st.latex(fr"\sigma = {std_dev:.2f}")
        
        if dist_type == "Normal (Bell Curve)":
            within_1sigma = sum(1 for x in data if mean_val - std_dev <= x <= mean_val + std_dev) / len(data) * 100
            within_2sigma = sum(1 for x in data if mean_val - 2*std_dev <= x <= mean_val + 2*std_dev) / len(data) * 100
            within_3sigma = sum(1 for x in data if mean_val - 3*std_dev <= x <= mean_val + 3*std_dev) / len(data) * 100
            
            st.write("**68-95-99.7 Rule:**")
            col_1s, col_2s, col_3s = st.columns(3)
            col_1s.metric("±1σ", f"{within_1sigma:.1f}%", "(Expected: 68%)")
            col_2s.metric("±2σ", f"{within_2sigma:.1f}%", "(Expected: 95%)")
            col_3s.metric("±3σ", f"{within_3sigma:.1f}%", "(Expected: 99.7%)")
        
        st.write("**Step 4: AI/ML Implications**")
        if dist_type == "Normal (Bell Curve)":
            st.write("• Good for linear regression, neural networks")
            st.write("• Assume normality in hypothesis testing")
            st.write("• Use standard normalization")
        elif dist_type == "Uniform (Flat)":
            st.write("• Good for random initialization of weights")
            st.write("• No extreme outliers to worry about")
            st.write("• Some algorithms assume normality - may need transformation")
        else:  # Skewed
            st.write("• Consider log/sqrt transformation before modeling")
            st.write("• Outliers may heavily influence model")
            st.write("• Use robust methods (median, quantiles)")

# RIGHT COLUMN
with col2:
    def definition():
        st.write("### Distribution Types")
        
        st.write("**1. Normal Distribution**")
        st.write("""
        Bell-shaped, symmetrical
        
        Mean = Median = Mode
        
        68% within ±1σ
        """)
        
        st.write("**2. Uniform Distribution**")
        st.write("""
        Flat, equal probability
        
        All values equally likely
        
        No skewness
        """)
        
        st.write("**3. Right-Skewed**")
        st.write("""
        Tail extends right
        
        Mean > Median > Mode
        
        Few extreme large values
        """)
        
        st.write("**4. Left-Skewed**")
        st.write("""
        Tail extends left
        
        Mean < Median < Mode
        
        Few extreme small values
        """)
    
    def examples():
        st.write("### Real-World Examples")
        
        st.write("**Normal Distribution:**")
        st.write("""
        - Human heights
        - IQ scores
        - Exam scores
        - Measurement errors
        """)
        
        st.write("**Uniform Distribution:**")
        st.write("""
        - Dice rolls (1-6)
        - Random numbers
        - Fair lottery
        """)
        
        st.write("**Right-Skewed:**")
        st.write("""
        - Income distribution
        - Website visits
        - Product prices
        """)
        
        st.write("**Left-Skewed:**")
        st.write("""
        - Age at retirement
        - Reaction times
        - Lifetime in reliability
        """)
    
    def ml_usage():
        st.write("### Impact on ML")
        
        st.write("**Normal Data:**")
        st.write("""
        • Most algorithms work well
        • Use directly
        • Standard scaling safe
        """)
        
        st.write("**Uniform Data:**")
        st.write("""
        • Good for initialization
        • May need normalization
        • No outlier issues
        """)
        
        st.write("**Skewed Data:**")
        st.write("""
        • Transform first!
        - Log transform
        - Box-Cox transform
        - Binning/bucketing
        • Outliers problematic
        """)
    
    def summary():
        st.write("### Quick Summary")
        
        summary_data = {
            "Type": ["Normal", "Uniform", "Right-Skew", "Left-Skew"],
            "Shape": ["Bell", "Flat", "Tail →", "← Tail"],
            "Mean=Median": ["Yes", "Yes", "No", "No"],
            "AI Action": ["Use directly", "Transform", "Log transform", "Log transform"]
        }
        
        summary_df = pd.DataFrame(summary_data)
        st.dataframe(summary_df, use_container_width=True, hide_index=True)
        
        st.write("### Key Takeaway")
        st.write("""
        **Before building any ML model:**
        
        1. Plot your data
        2. Check distribution type
        3. Transform if needed
        4. Apply appropriate scaling
        5. Train with confidence!
        """)
    
    render_theory_panel({
        "Definition": definition,
        "Examples": examples,
        "ML Usage": ml_usage,
        "Summary": summary
    })
