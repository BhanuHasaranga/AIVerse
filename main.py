import sys
from pathlib import Path

# Add project root to Python path
sys.path.insert(0, str(Path(__file__).parent))

import streamlit as st
from utils.ui_components import (
    apply_page_config, 
    apply_theme, 
    render_hero_section, 
    render_module_card
)

# Apply theme
apply_page_config(
    title="AI/ML Learning Hub",
    icon="🤖",
    sidebar_state="expanded"
)
apply_theme(page_type="home")

# HOME PAGE
if True:
    render_hero_section(
        "🤖 AI/ML Learning Hub",
        "Master fundamental ML & Statistics concepts through interactive visualizations"
    )
    
    st.write("""
    Welcome to your **interactive learning platform**! Here you'll explore core AI/ML concepts through 
    real-time visualizations and hands-on simulations.
    """)
    
    st.subheader("📚 Week 1: Statistics Foundations")
    
    col1, col2 = st.columns(2)
    
    with col1:
        render_module_card(
            title="Mean Explorer",
            description="Understand averages and central tendency through interactive histograms. Adjust dataset size and see how the mean changes in real-time.",
            topics="Average, Sum, Central Tendency",
            button_text="📊 Start Mean Explorer",
            page_path="pages/mean_explorer.py",
            icon="📊"
        )
    
    with col2:
        render_module_card(
            title="Median Explorer",
            description="Learn about the median value - the middle point of your data. Perfect for understanding data distribution.",
            topics="Median, Percentiles, Sorted Data",
            button_text="📈 Start Median Explorer",
            page_path="pages/median_explorer.py",
            icon="📈"
        )
    
    col3, col4 = st.columns(2)
    
    with col3:
        render_module_card(
            title="Variance Visualizer",
            description="Discover how spread out your data is. Variance is crucial for understanding data variability in ML.",
            topics="Variance, Standard Deviation, Spread",
            button_text="📉 Start Variance Visualizer",
            page_path="pages/variance_visualizer.py",
            icon="📉"
        )
    
    with col4:
        render_module_card(
            title="Mode Explorer",
            description="Find the most frequent value in your dataset. Essential for categorical data analysis.",
            topics="Mode, Frequency, Modality",
            button_text="👑 Start Mode Explorer",
            page_path="pages/mode_explorer.py",
            icon="👑"
        )
    
    col5, col6 = st.columns(2)
    
    with col5:
        render_module_card(
            title="Distribution Explorer",
            description="Explore probability distributions with advanced statistics including skewness, kurtosis, and the 68-95-99.7 rule.",
            topics="Normal, Uniform, Skewed Distributions, Kurtosis",
            button_text="🔔 Start Distribution Explorer",
            page_path="pages/distribution_explorer.py",
            icon="🔔"
        )
    
    with col6:
        render_module_card(
            title="Correlation Explorer",
            description="Analyze relationships between variables with correlation and covariance.",
            topics="Correlation, Covariance, Scatter Plots",
            button_text="🔗 Start Correlation Explorer",
            page_path="pages/correlation_explorer.py",
            icon="🔗"
        )
    
    # Add Probability Explorer
    st.write("")
    col7, col_spacer = st.columns([1, 1])
    
    with col7:
        render_module_card(
            title="Probability Explorer",
            description="Master probability fundamentals, conditional probability, Bayes' theorem, and random variables with interactive simulations.",
            topics="Probability, Bayes, Conditional, Random Variables",
            button_text="🎲 Start Probability Explorer",
            page_path="pages/probability_explorer.py",
            icon="🎲"
        )
    
    st.divider()
    
    st.subheader("🎯 How to Use This Platform")
    st.write("""
    1. **Select a module** from the sidebar
    2. **Read the theory** on the right panel to understand concepts
    3. **Interact with controls** - use sliders and buttons to experiment
    4. **Observe visualizations** - see how data changes in real-time
    5. **Learn by doing** - change parameters and predict outcomes
    """)

