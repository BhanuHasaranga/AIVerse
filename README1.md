<<<<<<< HEAD
# 🤖 AI/ML Learning Hub

An interactive educational web application for learning fundamental statistics and machine learning concepts through hands-on visualizations.

![Python](https://img.shields.io/badge/Python-3.13-blue)
![Streamlit](https://img.shields.io/badge/Streamlit-1.51-red)
![License](https://img.shields.io/badge/License-MIT-green)

## 📖 Overview

**AI/ML Learning Hub** is a Streamlit-based platform designed to teach core statistics and ML concepts through interactive visualizations. Each module includes theory, real-world examples, and ML applications.

## ✨ Features

### 📊 Interactive Learning Modules

- **Mean Explorer** - Understand averages and central tendency
- **Median Explorer** - Learn the middle value (robust to outliers)
- **Mode Explorer** - Find most frequent values
- **Variance Visualizer** - Explore data spread and standard deviation
- **Distribution Visualizer** - Probability distributions
- **Correlation Explorer** - Analyze relationships between variables

### 🎨 Modern UI/UX

- Responsive design with gradient hero sections
- Side-by-side layout: visualization + theory
- Tabbed learning guides (Definition, Examples, ML Usage, Summary)
- Real-time interactive charts with Plotly

### 📥 Flexible Data Input

1. **Generate Random Data** - Quick experimentation
2. **Upload CSV** - Use your own datasets
3. **Manual Entry** - Type values directly

## 🚀 Quick Start

### Prerequisites

- Python 3.13+
- pip package manager

### Installation

1. Clone the repository:
```bash
git clone <repository-url>
cd "AI ML Learning"
```

2. Create and activate virtual environment:
```bash
python -m venv venv

# Windows
venv\Scripts\activate

# macOS/Linux
source venv/bin/activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

### Run the Application

```bash
streamlit run main.py
```

The app will open in your browser at `http://localhost:8501`

## 📁 Project Structure

```
AI ML Learning/
├── main.py                      # Landing page & navigation hub
├── pages/                       # Streamlit multi-page modules
│   ├── mean_explorer.py         # Mean calculation & visualization
│   ├── median_explorer.py       # Median calculation
│   ├── mode_explorer.py         # Mode finding
│   ├── variance_visualizer.py  # Variance & std deviation
│   ├── distribution_visualizer.py  # Probability distributions
│   └── correlation_explorer.py # Correlation & covariance
├── utils/                       # Shared utilities
│   ├── __init__.py
│   ├── math_utils.py           # Mathematical functions
│   └── data_utils.py           # Data generation utilities
├── assets/                      # Static files (images, icons)
├── requirements.txt             # Python dependencies
├── .gitignore                   # Git ignore rules
└── README.md                    # This file
```

## 🎓 Learning Path

### Week 1: Statistics Foundations (Current)
- ✅ Mean - Central tendency
- ✅ Median - Middle value
- ✅ Mode - Most frequent value
- ✅ Variance - Data spread
- ✅ Distribution - Probability patterns

### Week 2: Relationships & Correlation
- ✅ Covariance & Correlation
- 🔜 Scatter plots
- 🔜 Linear relationships

### Week 3: Regression (Coming Soon)
- Linear Regression
- Model fitting
- Predictions

### Week 4+: Advanced ML (Roadmap)
- Classification algorithms
- Clustering
- Neural Networks

## 🛠️ Technologies

- **Streamlit** - Web framework
- **Plotly** - Interactive visualizations
- **NumPy** - Numerical computing
- **Pandas** - Data manipulation
- **Python 3.13** - Core language

## 📚 Educational Components

Each module includes:
- ✅ **Theory** - Clear definitions and formulas
- ✅ **Examples** - Real-world applications
- ✅ **ML Context** - How it's used in AI/ML
- ✅ **Interactive Visualizations** - Learn by doing
- ✅ **Step-by-Step Calculations** - Understand the math

## 🤝 Contributing

Contributions welcome! Feel free to:
- Report bugs
- Suggest new features
- Add new learning modules
- Improve documentation

## 📝 License

MIT License - feel free to use for educational purposes

## 🎯 Goals

- Make ML/Statistics accessible to beginners
- Provide hands-on interactive learning
- Bridge theory with practical applications
- Build intuition through visualization

## 💡 Tips for Learning

1. Start with **Mean Explorer** to understand basics
2. Experiment with different datasets
3. Read the theory tabs for each module
4. Try manual input to test specific scenarios
5. Compare Mean, Median, Mode on same dataset

## 📧 Support

For questions or feedback, open an issue on GitHub.

---

**Happy Learning! 🚀📊**

Built with ❤️ for the AI/ML learning community

=======
# AIVerse
Interactive AI/ML learning platform with hands-on visualizations and simulations. Master statistics, machine learning, and AI concepts through beautiful, beginner-friendly explorers. Built with Streamlit &amp; Python.
>>>>>>> cda480b8f7e0eb61b329cddf75809439a6308fa0
