"""
Decision Trees Module
Interactive visualization of decision tree algorithm for classification
"""
import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.datasets import make_classification, make_moons, make_circles
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
import seaborn as sns

def main():
    st.title("🌳 Decision Trees Explorer")
    st.write("Learn decision trees through interactive visualizations")
    
    tab1, tab2, tab3 = st.tabs(["Interactive Demo", "Model Interpretation", "Theory"])
    
    with tab1:
        st.header("Interactive Decision Tree")
        
        col1, col2 = st.columns([1, 2])
        
        with col1:
            st.subheader("Data Generation")
            
            dataset_type = st.selectbox(
                "Dataset type",
                ["Simple Classification", "Moons", "Circles"]
            )
            
            n_samples = st.slider("Number of samples", 100, 500, 200)
            
            if dataset_type == "Simple Classification":
                n_features = 2
                n_classes = st.slider("Number of classes", 2, 4, 2)
                class_sep = st.slider("Class separation", 0.5, 3.0, 1.0, 0.1)
                
                X, y = make_classification(
                    n_samples=n_samples,
                    n_features=n_features,
                    n_redundant=0,
                    n_informative=2,
                    n_classes=n_classes,
                    n_clusters_per_class=1,
                    class_sep=class_sep,
                    random_state=42
                )
            elif dataset_type == "Moons":
                noise = st.slider("Noise level", 0.0, 0.3, 0.15, 0.01)
                X, y = make_moons(n_samples=n_samples, noise=noise, random_state=42)
            else:  # Circles
                noise = st.slider("Noise level", 0.0, 0.3, 0.15, 0.01)
                X, y = make_circles(n_samples=n_samples, noise=noise, factor=0.5, random_state=42)
            
            st.subheader("Tree Parameters")
            max_depth = st.slider("Max depth", 1, 10, 3)
            min_samples_split = st.slider("Min samples to split", 2, 20, 2)
            min_samples_leaf = st.slider("Min samples per leaf", 1, 20, 1)
            
            # Split data
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=0.3, random_state=42
            )
            
            # Train decision tree
            dt = DecisionTreeClassifier(
                max_depth=max_depth,
                min_samples_split=min_samples_split,
                min_samples_leaf=min_samples_leaf,
                random_state=42
            )
            dt.fit(X_train, y_train)
            
            # Predictions
            y_train_pred = dt.predict(X_train)
            y_test_pred = dt.predict(X_test)
            
            # Metrics
            train_acc = accuracy_score(y_train, y_train_pred)
            test_acc = accuracy_score(y_test, y_test_pred)
            
            st.metric("Training Accuracy", f"{train_acc:.3f}")
            st.metric("Test Accuracy", f"{test_acc:.3f}")
            st.metric("Tree Depth", dt.get_depth())
            st.metric("Number of Leaves", dt.get_n_leaves())
            
            if train_acc > test_acc + 0.1:
                st.warning("Possible overfitting! Try reducing max_depth")
            elif test_acc > 0.9:
                st.success("Excellent performance!")
        
        with col2:
            st.subheader("Decision Boundary")
            
            # Create mesh for decision boundary
            h = 0.02
            x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
            y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
            xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                                np.arange(y_min, y_max, h))
            
            Z = dt.predict(np.c_[xx.ravel(), yy.ravel()])
            Z = Z.reshape(xx.shape)
            
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
            
            # Decision boundary plot
            ax1.contourf(xx, yy, Z, alpha=0.3, cmap='viridis')
            scatter = ax1.scatter(X_train[:, 0], X_train[:, 1], c=y_train, 
                                 cmap='viridis', edgecolors='black', s=50, alpha=0.7,
                                 label='Training data')
            ax1.scatter(X_test[:, 0], X_test[:, 1], c=y_test, 
                       cmap='viridis', edgecolors='red', s=50, alpha=0.7,
                       linewidths=2, label='Test data')
            ax1.set_xlabel('Feature 1')
            ax1.set_ylabel('Feature 2')
            ax1.set_title(f'Decision Boundary (Depth={dt.get_depth()})')
            ax1.legend()
            ax1.grid(True, alpha=0.3)
            
            # Confusion matrix
            cm = confusion_matrix(y_test, y_test_pred)
            sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax2, cbar=False)
            ax2.set_xlabel('Predicted')
            ax2.set_ylabel('Actual')
            ax2.set_title('Confusion Matrix (Test Set)')
            
            plt.tight_layout()
            st.pyplot(fig)
            
            with st.expander("📊 Classification Report"):
                report = classification_report(y_test, y_test_pred)
                st.text(report)
    
    with tab2:
        st.header("Model Interpretation")
        
        st.write("Visualize the decision tree structure")
        
        # Tree visualization controls
        col1, col2 = st.columns(2)
        with col1:
            feature_names = [f'Feature {i}' for i in range(X.shape[1])]
            show_feature_names = st.checkbox("Show feature names", value=True)
        with col2:
            show_class_names = st.checkbox("Show class names", value=True)
        
        # Plot the tree
        fig, ax = plt.subplots(figsize=(20, 10))
        
        class_names = [f'Class {i}' for i in range(len(np.unique(y)))]
        
        plot_tree(
            dt,
            ax=ax,
            feature_names=feature_names if show_feature_names else None,
            class_names=class_names if show_class_names else None,
            filled=True,
            rounded=True,
            fontsize=10
        )
        
        st.pyplot(fig)
        
        st.write("""
        ### Understanding the Tree Visualization
        
        Each node shows:
        - **Condition**: The splitting criterion (e.g., Feature 0 <= 0.5)
        - **Gini**: Impurity measure (0 = pure, 0.5 = maximum impurity for binary)
        - **Samples**: Number of training samples at this node
        - **Value**: Number of samples per class
        - **Class**: Predicted class for samples at this node
        
        **Colors:**
        - Darker colors indicate higher purity (more samples of one class)
        - Different colors represent different predicted classes
        """)
        
        # Feature importance
        if hasattr(dt, 'feature_importances_'):
            st.subheader("Feature Importance")
            
            importances = dt.feature_importances_
            indices = np.argsort(importances)[::-1]
            
            fig, ax = plt.subplots(figsize=(10, 6))
            ax.bar(range(len(importances)), importances[indices], color='skyblue', edgecolor='black')
            ax.set_xlabel('Feature Index')
            ax.set_ylabel('Importance')
            ax.set_title('Feature Importance')
            ax.set_xticks(range(len(importances)))
            ax.set_xticklabels([feature_names[i] for i in indices])
            ax.grid(True, alpha=0.3, axis='y')
            
            st.pyplot(fig)
            
            st.write("""
            Feature importance shows which features are most useful for making predictions.
            Higher values indicate more important features.
            """)
    
    with tab3:
        st.header("Decision Trees Theory")
        
        st.write("""
        ### What are Decision Trees?
        
        Decision trees are a supervised learning algorithm used for both classification 
        and regression. They make decisions by recursively splitting data based on 
        feature values.
        
        ### How Decision Trees Work
        """)
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.write("""
            **Tree Structure:**
            - **Root Node**: Top of the tree (all data)
            - **Internal Nodes**: Decision points (splits)
            - **Leaf Nodes**: Final predictions
            - **Branches**: Outcomes of decisions
            
            **Building Process:**
            1. Start with all data at root
            2. Find best feature to split on
            3. Create child nodes
            4. Recursively repeat for each child
            5. Stop at max depth or min samples
            """)
        
        with col2:
            st.write("""
            **Splitting Criteria:**
            - **Gini Impurity**: Probability of incorrect classification
            - **Entropy**: Measure of randomness/disorder
            - **Information Gain**: Reduction in entropy
            
            **Hyperparameters:**
            - max_depth: Maximum tree depth
            - min_samples_split: Min samples to split
            - min_samples_leaf: Min samples per leaf
            - max_features: Features to consider
            """)
        
        st.write("\n### Mathematical Formulation")
        
        st.write("**Gini Impurity:**")
        st.latex(r"Gini = 1 - \sum_{i=1}^{C} p_i^2")
        
        st.write("**Entropy:**")
        st.latex(r"Entropy = -\sum_{i=1}^{C} p_i \log_2(p_i)")
        
        st.write("**Information Gain:**")
        st.latex(r"IG = Entropy(parent) - \sum_{j} \frac{|S_j|}{|S|} Entropy(S_j)")
        
        st.write("""
        Where:
        - C: number of classes
        - p_i: proportion of class i
        - S: parent node samples
        - S_j: child node j samples
        """)
        
        st.write("\n### Advantages and Limitations")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.write("""
            **Advantages:**
            - Easy to understand and interpret
            - Requires little data preprocessing
            - Can handle numerical and categorical data
            - Non-parametric (no assumptions)
            - Feature importance available
            - Handles non-linear relationships
            """)
        
        with col2:
            st.write("""
            **Limitations:**
            - Prone to overfitting
            - Unstable (small data changes affect tree)
            - Biased toward features with more levels
            - Can create complex trees
            - Not optimal for some problems
            - Can be outperformed by ensembles
            """)
        
        st.write("\n### Preventing Overfitting")
        
        st.write("""
        **Pre-pruning (Early Stopping):**
        - Limit max_depth
        - Set min_samples_split
        - Set min_samples_leaf
        - Limit max_features
        
        **Post-pruning:**
        - Build full tree
        - Remove branches that don't improve validation performance
        - Use cost-complexity pruning (ccp_alpha)
        
        **Ensemble Methods:**
        - Random Forests: Multiple trees with bagging
        - Gradient Boosting: Sequential tree building
        - AdaBoost: Weighted ensemble
        """)

if __name__ == "__main__":
    main()
