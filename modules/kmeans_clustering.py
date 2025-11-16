"""
K-Means Clustering Module
Interactive visualization of K-Means clustering algorithm
"""
import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.datasets import make_blobs, make_moons, make_circles
from sklearn.metrics import silhouette_score

def main():
    st.title("🎯 K-Means Clustering Explorer")
    st.write("Learn K-Means clustering through interactive visualizations")
    
    tab1, tab2, tab3 = st.tabs(["Interactive Demo", "Step-by-Step", "Theory"])
    
    with tab1:
        st.header("Interactive K-Means Clustering")
        
        col1, col2 = st.columns([1, 2])
        
        with col1:
            st.subheader("Data Generation")
            
            dataset_type = st.selectbox(
                "Dataset type",
                ["Blobs (Clusters)", "Moons", "Circles", "Random"]
            )
            
            n_samples = st.slider("Number of samples", 50, 500, 200)
            
            if dataset_type == "Blobs (Clusters)":
                n_true_clusters = st.slider("True number of clusters", 2, 8, 3)
                cluster_std = st.slider("Cluster spread", 0.1, 3.0, 1.0, 0.1)
                
                X, y_true = make_blobs(
                    n_samples=n_samples,
                    n_features=2,
                    centers=n_true_clusters,
                    cluster_std=cluster_std,
                    random_state=42
                )
            elif dataset_type == "Moons":
                noise = st.slider("Noise level", 0.0, 0.3, 0.1, 0.01)
                X, y_true = make_moons(n_samples=n_samples, noise=noise, random_state=42)
            elif dataset_type == "Circles":
                noise = st.slider("Noise level", 0.0, 0.3, 0.1, 0.01)
                X, y_true = make_circles(n_samples=n_samples, noise=noise, random_state=42)
            else:
                X = np.random.randn(n_samples, 2)
                y_true = None
            
            st.subheader("K-Means Parameters")
            n_clusters = st.slider("Number of clusters (K)", 2, 10, 3)
            max_iter = st.slider("Max iterations", 10, 500, 300, 10)
            n_init = st.slider("Number of initializations", 1, 20, 10)
            
            # Fit K-Means
            kmeans = KMeans(
                n_clusters=n_clusters,
                max_iter=max_iter,
                n_init=n_init,
                random_state=42
            )
            y_pred = kmeans.fit_predict(X)
            
            # Calculate metrics
            inertia = kmeans.inertia_
            silhouette = silhouette_score(X, y_pred)
            
            st.metric("Inertia", f"{inertia:.2f}")
            st.metric("Silhouette Score", f"{silhouette:.3f}")
            st.metric("Iterations", kmeans.n_iter_)
            
            if silhouette > 0.7:
                st.success("Excellent clustering!")
            elif silhouette > 0.5:
                st.info("Good clustering")
            elif silhouette > 0.3:
                st.warning("Fair clustering")
            else:
                st.error("Poor clustering - try different K")
        
        with col2:
            st.subheader("Clustering Visualization")
            
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
            
            # Original data or true labels
            if y_true is not None:
                scatter1 = ax1.scatter(X[:, 0], X[:, 1], c=y_true, cmap='viridis', 
                                      alpha=0.6, s=50, edgecolors='black')
                ax1.set_title(f'True Labels (if applicable)')
            else:
                scatter1 = ax1.scatter(X[:, 0], X[:, 1], alpha=0.6, s=50, 
                                      edgecolors='black', c='gray')
                ax1.set_title('Original Data')
            
            ax1.set_xlabel('Feature 1')
            ax1.set_ylabel('Feature 2')
            ax1.grid(True, alpha=0.3)
            
            # K-Means clustering
            scatter2 = ax2.scatter(X[:, 0], X[:, 1], c=y_pred, cmap='viridis', 
                                  alpha=0.6, s=50, edgecolors='black')
            
            # Plot centroids
            centroids = kmeans.cluster_centers_
            ax2.scatter(centroids[:, 0], centroids[:, 1], 
                       c='red', marker='X', s=300, linewidths=3,
                       edgecolors='black', label='Centroids', zorder=10)
            
            ax2.set_xlabel('Feature 1')
            ax2.set_ylabel('Feature 2')
            ax2.set_title(f'K-Means Clustering (K={n_clusters})')
            ax2.legend()
            ax2.grid(True, alpha=0.3)
            
            plt.tight_layout()
            st.pyplot(fig)
            
            with st.expander("📝 Understanding the Metrics"):
                st.write("""
                **Inertia:**
                - Sum of squared distances to nearest cluster center
                - Lower is better
                - Decreases as K increases
                
                **Silhouette Score:**
                - Measures how similar points are to their own cluster vs other clusters
                - Range: -1 to 1
                - Higher is better
                - > 0.7: Excellent
                - 0.5-0.7: Good
                - 0.3-0.5: Fair
                - < 0.3: Poor
                
                **Choosing K:**
                - Use elbow method (plot inertia vs K)
                - Use silhouette score
                - Consider domain knowledge
                """)
    
    with tab2:
        st.header("Step-by-Step K-Means Algorithm")
        
        st.write("Watch K-Means clustering happen step by step")
        
        col1, col2 = st.columns([1, 2])
        
        with col1:
            st.subheader("Setup")
            
            n_samples_demo = st.slider("Number of samples", 20, 100, 50, key="demo_samples")
            n_clusters_demo = st.slider("Number of clusters", 2, 5, 3, key="demo_clusters")
            
            # Generate simple data
            np.random.seed(42)
            X_demo, _ = make_blobs(
                n_samples=n_samples_demo,
                n_features=2,
                centers=n_clusters_demo,
                cluster_std=1.0,
                random_state=42
            )
            
            iteration = st.slider("Iteration", 0, 10, 0)
            
            st.write("""
            **Algorithm Steps:**
            1. Initialize K centroids randomly
            2. Assign each point to nearest centroid
            3. Update centroids to mean of assigned points
            4. Repeat steps 2-3 until convergence
            """)
        
        with col2:
            st.subheader(f"Iteration {iteration}")
            
            # Manual K-Means implementation for visualization
            if iteration == 0:
                # Random initialization
                np.random.seed(42)
                centroids = X_demo[np.random.choice(len(X_demo), n_clusters_demo, replace=False)]
                labels = np.zeros(len(X_demo))
            else:
                # Run K-Means for 'iteration' steps
                kmeans_demo = KMeans(n_clusters=n_clusters_demo, max_iter=iteration, 
                                    n_init=1, random_state=42)
                labels = kmeans_demo.fit_predict(X_demo)
                centroids = kmeans_demo.cluster_centers_
            
            fig, ax = plt.subplots(figsize=(10, 8))
            
            # Plot points
            scatter = ax.scatter(X_demo[:, 0], X_demo[:, 1], c=labels, 
                               cmap='viridis', alpha=0.6, s=100, edgecolors='black')
            
            # Plot centroids
            ax.scatter(centroids[:, 0], centroids[:, 1], 
                      c='red', marker='X', s=400, linewidths=3,
                      edgecolors='black', label='Centroids', zorder=10)
            
            # Draw lines from points to centroids
            if iteration > 0:
                for i, point in enumerate(X_demo):
                    centroid = centroids[int(labels[i])]
                    ax.plot([point[0], centroid[0]], [point[1], centroid[1]], 
                           'k-', alpha=0.1, linewidth=0.5)
            
            ax.set_xlabel('Feature 1')
            ax.set_ylabel('Feature 2')
            ax.set_title(f'K-Means at Iteration {iteration}')
            ax.legend()
            ax.grid(True, alpha=0.3)
            
            st.pyplot(fig)
            
            if iteration == 0:
                st.info("Iteration 0: Random initialization of centroids")
            else:
                st.info(f"Iteration {iteration}: Points assigned to nearest centroids, centroids updated")
    
    with tab3:
        st.header("K-Means Clustering Theory")
        
        st.write("""
        ### What is K-Means Clustering?
        
        K-Means is an unsupervised learning algorithm that groups similar data points 
        into K clusters. It's one of the most popular clustering algorithms.
        
        ### Algorithm Steps
        """)
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.write("""
            **1. Initialization:**
            - Choose K (number of clusters)
            - Randomly initialize K centroids
            
            **2. Assignment:**
            - Assign each point to nearest centroid
            - Use Euclidean distance
            
            **3. Update:**
            - Recalculate centroids
            - Centroid = mean of assigned points
            """)
        
        with col2:
            st.write("""
            **4. Convergence:**
            - Repeat assignment and update
            - Stop when centroids don't change
            - Or reach max iterations
            
            **Time Complexity:**
            - O(n * K * i * d)
            - n: samples, K: clusters
            - i: iterations, d: features
            """)
        
        st.write("\n### Mathematical Formulation")
        
        st.write("**Objective Function:**")
        st.latex(r"J = \sum_{i=1}^{n} \sum_{k=1}^{K} w_{ik} ||x_i - \mu_k||^2")
        
        st.write("""
        Where:
        - w_ik = 1 if point i belongs to cluster k, 0 otherwise
        - μ_k = centroid of cluster k
        - ||x_i - μ_k|| = Euclidean distance
        """)
        
        st.write("\n**Centroid Update:**")
        st.latex(r"\mu_k = \frac{1}{|C_k|} \sum_{x_i \in C_k} x_i")
        
        st.write("\n### Advantages and Limitations")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.write("""
            **Advantages:**
            - Simple and easy to implement
            - Fast and efficient
            - Scales well to large datasets
            - Works well with spherical clusters
            """)
        
        with col2:
            st.write("""
            **Limitations:**
            - Must specify K beforehand
            - Sensitive to initialization
            - Assumes spherical clusters
            - Sensitive to outliers
            """)
        
        st.write("\n### Choosing the Right K")
        
        st.write("""
        **Elbow Method:**
        - Plot inertia vs K
        - Look for "elbow" where improvement slows
        
        **Silhouette Method:**
        - Calculate silhouette score for different K
        - Choose K with highest score
        
        **Domain Knowledge:**
        - Sometimes K is known from problem context
        """)

if __name__ == "__main__":
    main()
