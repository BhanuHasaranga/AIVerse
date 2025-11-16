import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import streamlit as st
from utils.data_utils import generate_random_data
import plotly.express as px
import plotly.graph_objects as go
import pandas as pd
import numpy as np

st.set_page_config(layout="wide", initial_sidebar_state="collapsed")

st.markdown("""
    <style>
    .main { max-width: 100%; padding-left: 1rem; padding-right: 1rem; }
    [data-testid="stMetricContainer"] { background-color: rgba(28, 131, 225, 0.1); padding: 10px; border-radius: 5px; }
    </style>
    """, unsafe_allow_html=True)

st.title("🔔 Distribution Visualizer")

col1, col2 = st.columns([3, 1])

with col1:
    st.subheader("📊 Probability Distributions")
    
    dist_type = st.selectbox("Select distribution type:", ["Normal", "Uniform", "Exponential"])
    sample_size = st.slider("Sample size:", min_value=100, max_value=10000, value=5000)
    
    # Generate samples
    if dist_type == "Normal":
        samples = np.random.normal(loc=50, scale=15, size=sample_size)
    elif dist_type == "Uniform":
        samples = np.random.uniform(low=0, high=100, size=sample_size)
    else:  # Exponential
        samples = np.random.exponential(scale=20, size=sample_size)
    
    st.write(f"**Generated {sample_size} samples from {dist_type} distribution**")
    
    # Plot
    df = pd.DataFrame({"Values": samples})
    fig = px.histogram(df, x="Values", nbins=50, title=f"{dist_type} Distribution", 
                       labels={"Values": "Values", "count": "Frequency"})
    fig.update_layout(showlegend=False)
    st.plotly_chart(fig, use_container_width=True)
    
    # Stats
    col_mean, col_std, col_skew = st.columns(3)
    col_mean.metric("Mean", f"{np.mean(samples):.2f}")
    col_std.metric("Std Dev", f"{np.std(samples):.2f}")
    col_skew.metric("Min/Max", f"{np.min(samples):.2f} / {np.max(samples):.2f}")

with col2:
    st.subheader("📚 Theory")
    with st.expander("📖 Distributions?", expanded=True):
        st.write("""
        **Distribution** describes how values are spread.
        
        **Normal (Bell Curve):**
        Most values near center, symmetric
        
        **Uniform:**
        All values equally likely
        
        **Exponential:**
        Rapid decrease, long tail
        
        **Why useful?**
        - Foundation of statistics
        - ML algorithms assume distributions
        - Data understanding
        """)
