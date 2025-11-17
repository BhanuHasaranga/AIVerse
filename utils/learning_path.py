"""
Learning path configuration with difficulty levels and prerequisites.
Defines the structured learning journey from beginner to advanced.
"""

class LearningModule:
    """Represents a single learning module with metadata"""
    
    def __init__(self, id, title, icon, description, difficulty, topics, 
                 estimated_time, prerequisites=None, page_path=None, coming_soon=False):
        self.id = id
        self.title = title
        self.icon = icon
        self.description = description
        self.difficulty = difficulty  # "Beginner", "Intermediate", "Advanced"
        self.topics = topics
        self.estimated_time = estimated_time  # in minutes
        self.prerequisites = prerequisites or []
        self.page_path = page_path
        self.coming_soon = coming_soon
    
    def get_difficulty_badge(self):
        """Return colored badge for difficulty level"""
        badges = {
            "Beginner": "🟢",
            "Intermediate": "🟡", 
            "Advanced": "🔴"
        }
        return badges.get(self.difficulty, "⚪")


# Define the complete learning path
LEARNING_PATH = {
    "phase_1": {
        "title": "Phase 1: Statistics Foundations",
        "description": "Master the fundamentals of statistics and data analysis",
        "status": "active",
        "modules": [
            # Week 1: Central Tendency
            LearningModule(
                id="mean",
                title="Mean Explorer",
                icon="📊",
                description="Learn what averages are and how to calculate them. Perfect starting point!",
                difficulty="Beginner",
                topics=["Average", "Sum", "Central Tendency"],
                estimated_time=15,
                prerequisites=[],
                page_path="pages/mean_explorer.py"
            ),
            LearningModule(
                id="median",
                title="Median Explorer",
                icon="📈",
                description="Understand the middle value and why it matters for skewed data.",
                difficulty="Beginner",
                topics=["Median", "Percentiles", "Outliers"],
                estimated_time=15,
                prerequisites=["mean"],
                page_path="pages/median_explorer.py"
            ),
            LearningModule(
                id="mode",
                title="Mode Explorer",
                icon="👑",
                description="Find the most frequent values in both numbers and categories.",
                difficulty="Beginner",
                topics=["Frequency", "Categorical Data"],
                estimated_time=12,
                prerequisites=["mean"],
                page_path="pages/mode_explorer.py"
            ),
            
            # Week 2: Spread & Variability
            LearningModule(
                id="variance",
                title="Variance Visualizer",
                icon="📉",
                description="Discover how spread out your data is - crucial for ML!",
                difficulty="Intermediate",
                topics=["Variance", "Standard Deviation", "Spread"],
                estimated_time=20,
                prerequisites=["mean", "median"],
                page_path="pages/variance_visualizer.py"
            ),
            LearningModule(
                id="distribution",
                title="Distribution Explorer",
                icon="🔔",
                description="Explore probability distributions with skewness and kurtosis analysis.",
                difficulty="Intermediate",
                topics=["Normal", "Uniform", "Skewness", "Kurtosis"],
                estimated_time=25,
                prerequisites=["variance"],
                page_path="pages/distribution_explorer.py"
            ),
            
            # Week 3: Relationships
            LearningModule(
                id="correlation",
                title="Correlation Explorer",
                icon="🔗",
                description="Analyze relationships between variables with correlation and covariance.",
                difficulty="Intermediate",
                topics=["Correlation", "Covariance", "Scatter Plots"],
                estimated_time=20,
                prerequisites=["variance"],
                page_path="pages/correlation_explorer.py"
            ),
            
            # Week 4: Probability
            LearningModule(
                id="probability",
                title="Probability Explorer",
                icon="🎲",
                description="Master probability basics, Bayes' theorem, and random variables.",
                difficulty="Intermediate",
                topics=["Probability", "Bayes", "Random Variables"],
                estimated_time=30,
                prerequisites=["distribution"],
                page_path="pages/probability_explorer.py"
            ),
        ]
    },
    
    "phase_2": {
        "title": "Phase 2: Linear Algebra for ML",
        "description": "Master the mathematical foundations essential for machine learning",
        "status": "coming_soon",
        "modules": [
            # Vectors
            LearningModule(
                id="vectors",
                title="Vectors & Operations",
                icon="📍",
                description="Learn about vectors, operations, dot products, and norms.",
                difficulty="Intermediate",
                topics=["Vectors", "Dot Product", "Norms", "Projections"],
                estimated_time=30,
                prerequisites=["probability"],
                coming_soon=True
            ),
            # Matrices
            LearningModule(
                id="matrices",
                title="Matrices & Multiplication",
                icon="🔢",
                description="Understand matrices, matrix operations, and multiplication.",
                difficulty="Intermediate",
                topics=["Matrices", "Matrix Multiplication", "Transpose", "Identity"],
                estimated_time=35,
                prerequisites=["vectors"],
                coming_soon=True
            ),
            # Determinants
            LearningModule(
                id="determinants",
                title="Determinants & Inverse",
                icon="🔄",
                description="Explore determinants, matrix inverse, and solving systems.",
                difficulty="Advanced",
                topics=["Determinants", "Inverse Matrices", "Linear Systems"],
                estimated_time=40,
                prerequisites=["matrices"],
                coming_soon=True
            ),
            # Eigenvalues
            LearningModule(
                id="eigenvalues",
                title="Eigenvalues & Eigenvectors",
                icon="⚡",
                description="Master eigenvalues, eigenvectors, and their ML applications.",
                difficulty="Advanced",
                topics=["Eigenvalues", "Eigenvectors", "Diagonalization", "PCA"],
                estimated_time=45,
                prerequisites=["determinants"],
                coming_soon=True
            ),
        ]
    },
    
    "phase_3": {
        "title": "Phase 3: Machine Learning Fundamentals",
        "description": "Build your first ML models with supervised learning",
        "status": "coming_soon",
        "modules": [
            LearningModule(
                id="linear_regression",
                title="Linear Regression",
                icon="📐",
                description="Predict continuous values by fitting lines to data.",
                difficulty="Intermediate",
                topics=["Regression", "Prediction", "Loss Functions"],
                estimated_time=35,
                prerequisites=["eigenvalues", "correlation"],
                coming_soon=True
            ),
            LearningModule(
                id="gradient_descent",
                title="Gradient Descent",
                icon="⛰️",
                description="Understand how ML models learn through optimization.",
                difficulty="Advanced",
                topics=["Optimization", "Learning Rate", "Convergence"],
                estimated_time=40,
                prerequisites=["linear_regression"],
                coming_soon=True
            ),
            LearningModule(
                id="logistic_regression",
                title="Logistic Regression",
                icon="🎯",
                description="Classify data into categories using probability.",
                difficulty="Intermediate",
                topics=["Classification", "Sigmoid", "Binary Outcomes"],
                estimated_time=30,
                prerequisites=["linear_regression", "probability"],
                coming_soon=True
            ),
            LearningModule(
                id="model_evaluation",
                title="Model Evaluation",
                icon="📊",
                description="Measure model performance with metrics and validation.",
                difficulty="Intermediate",
                topics=["Accuracy", "Precision", "Recall", "F1-Score"],
                estimated_time=25,
                prerequisites=["logistic_regression"],
                coming_soon=True
            ),
        ]
    },
    
    "phase_4": {
        "title": "Phase 4: Advanced ML & Deep Learning",
        "description": "Master complex algorithms and neural networks",
        "status": "planned",
        "modules": [
            LearningModule(
                id="decision_trees",
                title="Decision Trees",
                icon="🌳",
                description="Build interpretable models with tree-based algorithms.",
                difficulty="Intermediate",
                topics=["Trees", "Entropy", "Information Gain"],
                estimated_time=30,
                prerequisites=["model_evaluation"],
                coming_soon=True
            ),
            LearningModule(
                id="random_forest",
                title="Random Forest",
                icon="🌲",
                description="Ensemble learning with multiple decision trees.",
                difficulty="Advanced",
                topics=["Ensemble", "Bagging", "Feature Importance"],
                estimated_time=35,
                prerequisites=["decision_trees"],
                coming_soon=True
            ),
            LearningModule(
                id="neural_networks",
                title="Neural Networks",
                icon="🧠",
                description="Introduction to deep learning and neural architectures.",
                difficulty="Advanced",
                topics=["Neurons", "Layers", "Activation Functions"],
                estimated_time=45,
                prerequisites=["gradient_descent"],
                coming_soon=True
            ),
            LearningModule(
                id="cnn",
                title="Convolutional Networks",
                icon="🖼️",
                description="Computer vision with CNNs - understand image recognition.",
                difficulty="Advanced",
                topics=["Convolution", "Pooling", "Image Classification"],
                estimated_time=50,
                prerequisites=["neural_networks"],
                coming_soon=True
            ),
            LearningModule(
                id="rnn",
                title="Recurrent Networks",
                icon="🔄",
                description="Sequence modeling with RNNs and LSTMs.",
                difficulty="Advanced",
                topics=["Sequences", "LSTM", "Time Series"],
                estimated_time=50,
                prerequisites=["neural_networks"],
                coming_soon=True
            ),
        ]
    }
}


def get_all_modules():
    """Get flat list of all modules across all phases"""
    modules = []
    for phase in LEARNING_PATH.values():
        modules.extend(phase["modules"])
    return modules


def get_module_by_id(module_id):
    """Get a specific module by its ID"""
    for module in get_all_modules():
        if module.id == module_id:
            return module
    return None


def get_next_recommended_module(completed_modules):
    """
    Get the next recommended module based on completed modules.
    Returns the first unlocked module that hasn't been completed.
    """
    all_modules = get_all_modules()
    
    for module in all_modules:
        # Skip if already completed or coming soon
        if module.id in completed_modules or module.coming_soon:
            continue
        
        # Check if all prerequisites are met
        prerequisites_met = all(prereq in completed_modules for prereq in module.prerequisites)
        
        if prerequisites_met:
            return module
    
    return None


def get_available_modules(completed_modules):
    """Get all modules that are currently unlocked and available"""
    available = []
    
    for module in get_all_modules():
        if module.coming_soon:
            continue
        
        prerequisites_met = all(prereq in completed_modules for prereq in module.prerequisites)
        
        if prerequisites_met:
            available.append(module)
    
    return available


def calculate_phase_progress(phase_key, completed_modules):
    """Calculate completion percentage for a phase"""
    phase = LEARNING_PATH[phase_key]
    total_modules = len([m for m in phase["modules"] if not m.coming_soon])
    
    if total_modules == 0:
        return 0
    
    completed_count = sum(1 for m in phase["modules"] 
                         if m.id in completed_modules and not m.coming_soon)
    
    return (completed_count / total_modules) * 100


def calculate_total_progress(completed_modules):
    """Calculate overall completion percentage"""
    all_modules = [m for m in get_all_modules() if not m.coming_soon]
    
    if not all_modules:
        return 0
    
    completed_count = sum(1 for m in all_modules if m.id in completed_modules)
    return (completed_count / len(all_modules)) * 100

