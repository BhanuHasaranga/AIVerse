import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import streamlit as st
from utils.ui_components import apply_page_config, apply_theme, render_enhanced_sidebar

# Apply theme
apply_page_config(title="Welcome", icon="👋", sidebar_state="expanded")
apply_theme(page_type="home")

# Render sidebar
render_enhanced_sidebar()

# Custom CSS for welcome page
st.markdown("""
<style>
.welcome-hero {
    background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
    padding: 4rem 2rem;
    border-radius: 15px;
    text-align: center;
    color: white;
    margin-bottom: 2rem;
}
.welcome-hero h1 {
    font-size: 3.5rem;
    margin-bottom: 1rem;
}
.welcome-hero p {
    font-size: 1.3rem;
    opacity: 0.95;
}
.feature-box {
    background: #f0f2f6;
    padding: 2rem;
    border-radius: 10px;
    border-left: 5px solid #667eea;
    margin-bottom: 1.5rem;
}
.feature-box h3 {
    color: #667eea;
    margin-top: 0;
}
.step-card {
    background: white;
    padding: 1.5rem;
    border-radius: 10px;
    box-shadow: 0 2px 8px rgba(0,0,0,0.1);
    margin-bottom: 1rem;
    border-top: 4px solid #667eea;
}
</style>
""", unsafe_allow_html=True)

# Hero section
st.markdown("""
<div class="welcome-hero">
    <h1>👋 Welcome to AIVerse</h1>
    <p>Your Interactive Journey to Master AI & Machine Learning</p>
    <p style="font-size: 1rem; margin-top: 1rem;">From Zero to AI Engineer • Learn by Doing • Visualize Every Concept</p>
</div>
""", unsafe_allow_html=True)

# Introduction
col1, col2 = st.columns([2, 1])

with col1:
    st.subheader("🎯 What You'll Learn")
    
    st.markdown("""
    <div class="feature-box">
    <h3>📊 Phase 1: Statistics Foundations</h3>
    Master the mathematical building blocks of AI:
    <ul>
    <li>Central Tendency (Mean, Median, Mode)</li>
    <li>Data Spread (Variance, Standard Deviation)</li>
    <li>Probability Distributions</li>
    <li>Correlation & Relationships</li>
    </ul>
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown("""
    <div class="feature-box">
    <h3>🤖 Phase 2: Machine Learning Fundamentals</h3>
    Build your first ML models:
    <ul>
    <li>Linear & Logistic Regression</li>
    <li>Gradient Descent & Optimization</li>
    <li>Model Evaluation & Validation</li>
    <li>Feature Engineering</li>
    </ul>
    <small>🔵 Coming Soon</small>
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown("""
    <div class="feature-box">
    <h3>🧠 Phase 3: Deep Learning & Advanced Topics</h3>
    Master neural networks:
    <ul>
    <li>Neural Network Architectures</li>
    <li>CNNs for Computer Vision</li>
    <li>RNNs & LSTMs for Sequences</li>
    <li>Transformers & Attention</li>
    </ul>
    <small>⚪ Planned</small>
    </div>
    """, unsafe_allow_html=True)

with col2:
    st.subheader("✨ Why AIVerse?")
    
    st.write("""
    **Interactive Visualizations**
    See concepts come alive with real-time charts and animations
    
    **Learn by Doing**
    Manipulate parameters and watch outcomes change instantly
    
    **Beginner Friendly**
    No prior experience needed - start from scratch
    
    **Structured Path**
    Follow a proven curriculum from basics to advanced
    
    **Self-Paced**
    Learn at your own speed, review anytime
    
    **Free & Open**
    Completely free educational resource
    """)

st.divider()

# How it works
st.subheader("🚀 How It Works")

col1, col2, col3 = st.columns(3)

with col1:
    st.markdown("""
    <div class="step-card">
    <h3 style="color: #667eea;">1️⃣ Explore</h3>
    <p>Choose a module from the structured learning path</p>
    </div>
    """, unsafe_allow_html=True)

with col2:
    st.markdown("""
    <div class="step-card">
    <h3 style="color: #667eea;">2️⃣ Interact</h3>
    <p>Play with controls, generate data, and experiment</p>
    </div>
    """, unsafe_allow_html=True)

with col3:
    st.markdown("""
    <div class="step-card">
    <h3 style="color: #667eea;">3️⃣ Understand</h3>
    <p>Read theory, see visualizations, master concepts</p>
    </div>
    """, unsafe_allow_html=True)

st.divider()

# Call to action
st.subheader("🎓 Ready to Start Learning?")

col1, col2, col3 = st.columns([1, 2, 1])

with col2:
    st.write("Choose your starting point:")
    
    col_btn1, col_btn2 = st.columns(2)
    
    with col_btn1:
        if st.button("📚 View Learning Path", use_container_width=True, type="primary"):
            st.switch_page("pages/learning_path.py")
    
    with col_btn2:
        if st.button("🚀 Jump to Home", use_container_width=True):
            st.switch_page("main.py")

st.divider()

# Statistics
st.subheader("📊 Platform Stats")

col1, col2, col3, col4 = st.columns(4)

col1.metric("Interactive Modules", "7", delta="Phase 1 Complete")
col2.metric("Total Learning Time", "~2.5 hours", delta="Active content")
col3.metric("Difficulty Levels", "3", delta="Beginner to Advanced")
col4.metric("Future Modules", "10+", delta="In development")

st.divider()

# Testimonial / Quote section
st.info("""
**💡 Learning Philosophy**

"The best way to learn AI is not by reading equations, but by *seeing* them in action. 
AIVerse transforms abstract concepts into interactive experiences, making machine learning 
accessible to everyone."
""")

# Footer
st.write("")
st.caption("Built with ❤️ for learners worldwide | Open source educational project")

