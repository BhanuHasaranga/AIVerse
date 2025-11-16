"""
AIVerse - Interactive AI/ML Learning Platform
Main application entry point with navigation
"""
import streamlit as st
from pathlib import Path

# Page configuration
st.set_page_config(
    page_title="AIVerse - Interactive AI/ML Learning",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better UI
st.markdown("""
    <style>
    .main-header {
        font-size: 3rem;
        font-weight: bold;
        text-align: center;
        color: #1E88E5;
        margin-bottom: 1rem;
    }
    .sub-header {
        font-size: 1.2rem;
        text-align: center;
        color: #666;
        margin-bottom: 2rem;
    }
    .module-card {
        padding: 1.5rem;
        border-radius: 10px;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        margin-bottom: 1rem;
    }
    </style>
""", unsafe_allow_html=True)

# Main header
st.markdown('<h1 class="main-header">🤖 AIVerse</h1>', unsafe_allow_html=True)
st.markdown('<p class="sub-header">Interactive AI/ML Learning Hub with Hands-on Visualizations</p>', unsafe_allow_html=True)

# Sidebar navigation
st.sidebar.title("📚 Navigation")
st.sidebar.markdown("---")

# Module selection
module = st.sidebar.radio(
    "Choose a Module",
    [
        "🏠 Home",
        "📊 Statistics Explorer",
        "📈 Linear Regression",
        "🎯 K-Means Clustering",
        "🌳 Decision Trees",
        "🧠 Neural Networks Basics",
        "📖 About"
    ]
)

# Module routing
if module == "🏠 Home":
    st.markdown("### Welcome to AIVerse! 🎉")
    st.write("""
    AIVerse is your interactive companion for learning AI and Machine Learning concepts through 
    hands-on visualizations and simulations.
    """)
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("#### 📊 Statistics")
        st.write("""
        - Explore mean, median, mode
        - Understand variance and standard deviation
        - Visualize correlation
        - Step-by-step calculations
        """)
        
        st.markdown("#### 🤖 Machine Learning")
        st.write("""
        - Linear Regression with visualizations
        - K-Means Clustering explorer
        - Decision Tree interactive builder
        - Neural Network fundamentals
        """)
    
    with col2:
        st.markdown("#### 🎯 Features")
        st.write("""
        - **Interactive**: Real-time parameter adjustments
        - **Visual**: Beautiful charts and graphs
        - **Educational**: Step-by-step explanations
        - **Beginner-friendly**: No prior knowledge needed
        """)
        
        st.markdown("#### 🚀 Coming Soon")
        st.write("""
        - Deep Learning modules
        - Computer Vision applications
        - Natural Language Processing
        - Advanced neural architectures
        """)
    
    st.markdown("---")
    st.info("👈 Select a module from the sidebar to get started!")

elif module == "📊 Statistics Explorer":
    from modules import statistics_explorer
    statistics_explorer.main()

elif module == "📈 Linear Regression":
    from modules import linear_regression
    linear_regression.main()

elif module == "🎯 K-Means Clustering":
    from modules import kmeans_clustering
    kmeans_clustering.main()

elif module == "🌳 Decision Trees":
    from modules import decision_trees
    decision_trees.main()

elif module == "🧠 Neural Networks Basics":
    from modules import neural_networks
    neural_networks.main()

elif module == "📖 About":
    st.markdown("### About AIVerse")
    st.write("""
    **AIVerse** is an interactive AI/ML learning platform designed to make complex concepts 
    accessible through hands-on visualizations and simulations.
    
    #### 🎓 Learning Philosophy
    - Learn by doing
    - Visualize concepts in real-time
    - Step-by-step explanations
    - Professional code structure
    
    #### 🛠️ Built With
    - Streamlit for interactive web apps
    - NumPy & Pandas for data processing
    - Scikit-learn for ML algorithms
    - Matplotlib & Plotly for visualizations
    
    #### 📧 Contact
    Contributions and feedback are welcome!
    """)

# Footer
st.sidebar.markdown("---")
st.sidebar.markdown("### 💡 Tips")
st.sidebar.info("Adjust parameters and see results in real-time!")
