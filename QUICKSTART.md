# AIVerse Quick Start Guide

Welcome to AIVerse! This guide will help you get started with the interactive AI/ML learning platform.

## Installation

1. Clone the repository:
```bash
git clone https://github.com/BhanuHasaranga/AIVerse.git
cd AIVerse
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Run the application:
```bash
streamlit run app.py
```

4. Open your browser to `http://localhost:8501`

## Navigation

Use the sidebar to navigate between different modules:
- 🏠 **Home** - Overview and introduction
- 📊 **Statistics Explorer** - Learn statistics interactively
- 📈 **Linear Regression** - Understand regression analysis
- 🎯 **K-Means Clustering** - Explore clustering algorithms
- 🌳 **Decision Trees** - Build decision tree models
- 🧠 **Neural Networks Basics** - Introduction to neural networks
- 📖 **About** - Learn more about AIVerse

## Module Guides

### Statistics Explorer
Learn fundamental statistical concepts:
- **Central Tendency**: Mean, median, mode with visualizations
- **Dispersion**: Variance, standard deviation, range
- **Correlation**: Scatter plots and correlation coefficients
- **Distributions**: Normal, uniform, exponential distributions
- **Custom Dataset**: Upload and analyze your own CSV files

**Tips:**
- Enter your own data in the text boxes
- Click "Step-by-Step Calculations" to see the math
- Adjust parameters to see real-time updates

### Linear Regression
Understand how linear regression works:
- **Interactive Demo**: Generate data with adjustable slope, intercept, and noise
- **Real Dataset**: Work with synthetic housing data
- **Metrics**: R² score, RMSE, residual plots

**Tips:**
- Try different noise levels to see how it affects the fit
- Compare training vs test accuracy
- Examine the residual plot for model quality

### K-Means Clustering
Visualize clustering algorithms:
- **Interactive Demo**: Multiple dataset types (blobs, moons, circles)
- **Step-by-Step**: Watch K-Means algorithm in action
- **Metrics**: Inertia and silhouette scores

**Tips:**
- Try different K values to find optimal clusters
- Use the step-by-step view to understand the algorithm
- Check silhouette score for cluster quality

### Decision Trees
Build classification trees:
- **Interactive Demo**: Adjustable tree parameters
- **Tree Visualization**: See the complete tree structure
- **Feature Importance**: Understand which features matter

**Tips:**
- Adjust max_depth to control overfitting
- View the tree structure to understand decisions
- Check confusion matrix for accuracy

### Neural Networks Basics
Introduction to neural networks:
- **Introduction**: Learn the basics
- **Interactive Demo**: Train networks with custom architectures
- **Architecture Explorer**: Design your own networks
- **Theory**: Mathematical foundations

**Tips:**
- Start with simple architectures
- Watch the loss curve to monitor training
- Try different activation functions

## Learning Path

Recommended order for beginners:
1. Statistics Explorer → Build foundation
2. Linear Regression → Supervised learning basics
3. K-Means Clustering → Unsupervised learning
4. Decision Trees → Tree-based models
5. Neural Networks Basics → Deep learning introduction

## Troubleshooting

**App won't start:**
- Make sure all dependencies are installed: `pip install -r requirements.txt`
- Check Python version (3.8+ required)

**Visualizations not showing:**
- Wait a few seconds for plots to render
- Check browser console for errors

**Module errors:**
- Ensure you're running from the project root directory
- Verify all module files are present in the `modules/` directory

## Tips for Best Experience

1. **Adjust Parameters**: Don't be afraid to experiment with sliders and inputs
2. **Read Explanations**: Expand the "Step-by-Step" sections for detailed math
3. **Try Different Data**: Upload your own datasets in Statistics Explorer
4. **Take Your Time**: Each module has multiple tabs to explore
5. **Learn by Doing**: Adjust parameters and observe the results

## Contributing

Want to add more AI simulations or learning tools?
1. Fork the repository
2. Create a new module in `modules/`
3. Add your module to the navigation in `app.py`
4. Submit a pull request

## Need Help?

- Check the README.md for detailed documentation
- Open an issue on GitHub for bugs or suggestions
- Share your learning experience!

---

Happy Learning! 🎉
