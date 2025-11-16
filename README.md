# 🤖 AIVerse

Interactive AI/ML learning platform with hands-on visualizations and simulations. Master statistics, machine learning, and AI concepts through beautiful, beginner-friendly explorers. Built with Streamlit & Python.

[![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)](https://www.python.org/)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.28+-red.svg)](https://streamlit.io/)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

## 🌟 Features

### 📊 Statistics Explorer
- **Central Tendency**: Interactive exploration of mean, median, and mode
- **Dispersion Measures**: Variance, standard deviation, and range with step-by-step calculations
- **Correlation Analysis**: Real-time correlation visualization with scatter plots
- **Probability Distributions**: Normal, uniform, exponential, and binomial distributions
- **Custom Dataset Analysis**: Upload your own CSV files for statistical analysis

### 📈 Linear Regression
- **Interactive Demo**: Adjust parameters and see regression lines in real-time
- **Real Dataset Analysis**: Work with synthetic datasets similar to California Housing
- **Metrics Visualization**: R² score, RMSE, MSE with residual plots
- **Theory Section**: Complete mathematical foundation and formulas

### 🎯 K-Means Clustering
- **Interactive Clustering**: Real-time K-Means visualization with multiple dataset types
- **Step-by-Step Algorithm**: Watch clustering happen iteration by iteration
- **Silhouette Analysis**: Evaluate clustering quality automatically
- **Multiple Dataset Types**: Blobs, moons, circles, and random patterns

### 🌳 Decision Trees
- **Interactive Classification**: Build decision trees with adjustable parameters
- **Decision Boundary Visualization**: See how trees partition feature space
- **Tree Structure Visualization**: Full tree diagrams with split conditions
- **Feature Importance**: Understand which features matter most

### 🧠 Neural Networks Basics
- **Interactive Demo**: Train neural networks with custom architectures
- **Architecture Explorer**: Design and visualize network structures
- **Activation Functions**: Compare ReLU, sigmoid, and tanh
- **Training Visualization**: Watch loss curves and decision boundaries evolve

## 🚀 Getting Started

### Prerequisites
- Python 3.8 or higher
- pip package manager

### Installation

1. Clone the repository:
```bash
git clone https://github.com/BhanuHasaranga/AIVerse.git
cd AIVerse
```

2. Install required packages:
```bash
pip install -r requirements.txt
```

3. Run the application:
```bash
streamlit run app.py
```

4. Open your browser and navigate to `http://localhost:8501`

## 📚 Usage

### Basic Navigation
1. Launch the app using `streamlit run app.py`
2. Use the sidebar to select different modules
3. Adjust parameters using sliders and inputs
4. Watch visualizations update in real-time
5. Expand "Step-by-Step" sections for detailed explanations

### Example: Statistics Explorer
```python
# The app handles everything interactively!
# Just enter your data in the text area:
# Example: 12, 15, 18, 20, 22, 25, 28, 30, 32, 35

# The app will automatically calculate:
# - Mean, Median, Mode
# - Variance, Standard Deviation
# - Show visualizations with histograms and box plots
```

### Example: Training a Neural Network
```python
# 1. Select "Neural Networks Basics" from sidebar
# 2. Choose dataset type (e.g., "Moons")
# 3. Configure architecture (layers and neurons)
# 4. Set learning rate and iterations
# 5. Click "Train Network"
# 6. Watch decision boundary and loss curve
```

## 🎓 Learning Path

We recommend following this order for beginners:

1. **Statistics Explorer** - Build foundation in statistics
2. **Linear Regression** - Understand supervised learning basics
3. **K-Means Clustering** - Learn unsupervised learning
4. **Decision Trees** - Explore tree-based models
5. **Neural Networks Basics** - Introduction to deep learning

## 🛠️ Built With

- **[Streamlit](https://streamlit.io/)** - Interactive web app framework
- **[NumPy](https://numpy.org/)** - Numerical computing
- **[Pandas](https://pandas.pydata.org/)** - Data manipulation
- **[Scikit-learn](https://scikit-learn.org/)** - Machine learning algorithms
- **[Matplotlib](https://matplotlib.org/)** - Data visualization
- **[Seaborn](https://seaborn.pydata.org/)** - Statistical visualizations
- **[Plotly](https://plotly.com/)** - Interactive plots
- **[SciPy](https://scipy.org/)** - Scientific computing

## 📖 Project Structure

```
AIVerse/
├── app.py                          # Main application entry point
├── requirements.txt                # Python dependencies
├── README.md                       # Project documentation
├── LICENSE                         # MIT License
└── modules/                        # Learning modules
    ├── __init__.py
    ├── statistics_explorer.py      # Statistics module
    ├── linear_regression.py        # Linear regression module
    ├── kmeans_clustering.py        # K-Means clustering module
    ├── decision_trees.py           # Decision trees module
    └── neural_networks.py          # Neural networks module
```

## 🔮 Roadmap

### Coming Soon
- [ ] Deep Learning modules
  - Convolutional Neural Networks (CNNs)
  - Recurrent Neural Networks (RNNs)
  - Transfer Learning
- [ ] Computer Vision applications
  - Image classification
  - Object detection
  - Image segmentation
- [ ] Natural Language Processing
  - Text classification
  - Sentiment analysis
  - Word embeddings
- [ ] Advanced Topics
  - Reinforcement Learning
  - Generative Adversarial Networks (GANs)
  - Transformers and Attention mechanisms

## 🤝 Contributing

Contributions are welcome! Here's how you can help:

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

### Ideas for Contributions
- Add new AI/ML modules
- Improve visualizations
- Add more datasets
- Enhance documentation
- Fix bugs
- Optimize performance

## 📝 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- Inspired by the need for interactive AI/ML education
- Built for learners at all levels
- Special thanks to the open-source community

## 📧 Contact

For questions, suggestions, or feedback:
- Open an issue on GitHub
- Contribute to the project
- Share your learning experience!

## ⭐ Show Your Support

If you find AIVerse helpful, please consider:
- Starring the repository ⭐
- Sharing with others who want to learn AI/ML
- Contributing new modules and features

---

**Happy Learning!** 🎉 Explore, experiment, and master AI/ML with AIVerse!
