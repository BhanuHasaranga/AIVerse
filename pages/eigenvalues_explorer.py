import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import streamlit as st
from utils.ui import apply_page_config, apply_theme, create_two_column_layout, render_theory_panel
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from scipy.linalg import eig

# Apply theme
apply_page_config(title="Eigenvalues Explorer", icon="⚡", sidebar_state="expanded")
apply_theme(page_type="page")

# Create layout
col1, col2 = create_two_column_layout("Eigenvalues & Eigenvectors", module_id="eigenvalues")

# LEFT COLUMN
with col1:
    st.subheader("Interactive Eigenvalues Explorer")
    
    # Tab selection
    eig_tab1, eig_tab2, eig_tab3, eig_tab4 = st.tabs([
        "Eigenvalues & Eigenvectors",
        "Transformation Invariance",
        "Diagonalization",
        "PCA Demo"
    ])
    
    with eig_tab1:
        st.write("### Eigenvalue Calculator")
        
        st.write("**Enter 2×2 Matrix A:**")
        col1, col2 = st.columns(2)
        with col1:
            a11 = st.number_input("a₁₁", value=4.0, step=0.1, key="eig_a11")
            a21 = st.number_input("a₂₁", value=2.0, step=0.1, key="eig_a21")
        with col2:
            a12 = st.number_input("a₁₂", value=1.0, step=0.1, key="eig_a12")
            a22 = st.number_input("a₂₂", value=3.0, step=0.1, key="eig_a22")
        
        A = np.array([[a11, a12], [a21, a22]])
        
        st.write("**Matrix A:**")
        st.dataframe(pd.DataFrame(A), use_container_width=True, hide_index=True)
        
        if st.button("Calculate Eigenvalues & Eigenvectors"):
            try:
                eigenvalues, eigenvectors = eig(A)
                
                st.write("### Results")
                
                for i, (eigval, eigvec) in enumerate(zip(eigenvalues, eigenvectors.T)):
                    st.write(f"**Eigenvalue λ{i+1} = {eigval:.3f}**")
                    st.write(f"**Eigenvector v{i+1}:**")
                    st.write(f"v{i+1} = ({eigvec[0]:.3f}, {eigvec[1]:.3f})")
                    
                    # Verify: Av = λv
                    Av = np.dot(A, eigvec)
                    lambda_v = eigval * eigvec
                    st.write(f"**Verification: Av{i+1} = λ{i+1}v{i+1}**")
                    st.write(f"Av{i+1} = ({Av[0]:.3f}, {Av[1]:.3f})")
                    st.write(f"λ{i+1}v{i+1} = ({lambda_v[0]:.3f}, {lambda_v[1]:.3f})")
                    if np.allclose(Av, lambda_v):
                        st.success("✅ Verified!")
                    st.divider()
                
                # Characteristic equation
                with st.expander("Characteristic Equation"):
                    det_A = np.linalg.det(A)
                    trace_A = np.trace(A)
                    st.write("**For 2×2 matrix:**")
                    st.latex(r"\det(A - \lambda I) = 0")
                    st.latex(r"\lambda^2 - \text{tr}(A)\lambda + \det(A) = 0")
                    st.write(f"λ² - {trace_A:.2f}λ + {det_A:.2f} = 0")
                    st.write(f"**Solutions:** λ₁ = {eigenvalues[0]:.3f}, λ₂ = {eigenvalues[1]:.3f}")
            
            except Exception as e:
                st.error(f"Error calculating eigenvalues: {e}")
    
    with eig_tab2:
        st.write("### Transformation Invariance")
        
        st.write("**Eigenvectors don't change direction under transformation!**")
        
        st.write("**Enter 2×2 Matrix A:**")
        col1, col2 = st.columns(2)
        with col1:
            a11 = st.number_input("a₁₁", value=3.0, step=0.1, key="trans_a11")
            a21 = st.number_input("a₂₁", value=1.0, step=0.1, key="trans_a21")
        with col2:
            a12 = st.number_input("a₁₂", value=1.0, step=0.1, key="trans_a12")
            a22 = st.number_input("a₂₂", value=3.0, step=0.1, key="trans_a22")
        
        A = np.array([[a11, a12], [a21, a22]])
        
        # Calculate eigenvalues/eigenvectors
        try:
            eigenvalues, eigenvectors = eig(A)
            
            # Visualize
            fig = go.Figure()
            
            # Original eigenvectors
            for i, (eigval, eigvec) in enumerate(zip(eigenvalues, eigenvectors.T)):
                # Normalize for visualization
                eigvec_norm = eigvec / np.linalg.norm(eigvec) * 2
                
                # Original
                fig.add_trace(go.Scatter(
                    x=[0, eigvec_norm[0]], y=[0, eigvec_norm[1]],
                    mode='lines+markers',
                    name=f'v{i+1} (original)',
                    line=dict(color=['#667eea', '#764ba2'][i], width=3)
                ))
                
                # Transformed
                transformed = np.dot(A, eigvec_norm)
                fig.add_trace(go.Scatter(
                    x=[0, transformed[0]], y=[0, transformed[1]],
                    mode='lines+markers',
                    name=f'Av{i+1} (transformed)',
                    line=dict(color=['#667eea', '#764ba2'][i], width=3, dash='dash')
                ))
            
            fig.update_layout(
                title="Eigenvectors: Direction Preserved Under Transformation",
                xaxis_title="X", yaxis_title="Y",
                xaxis=dict(range=[-5, 5], zeroline=True),
                yaxis=dict(range=[-5, 5], zeroline=True),
                height=400
            )
            st.plotly_chart(fig, use_container_width=True)
            
            st.write("**Key Insight:**")
            st.write("Eigenvectors only get scaled by their eigenvalues, direction stays the same!")
        
        except Exception as e:
            st.error(f"Error: {e}")
    
    with eig_tab3:
        st.write("### Diagonalization")
        
        st.write("**A = PDP⁻¹ where D is diagonal**")
        
        st.write("**Enter 2×2 Matrix A:**")
        col1, col2 = st.columns(2)
        with col1:
            a11 = st.number_input("a₁₁", value=4.0, step=0.1, key="diag_a11")
            a21 = st.number_input("a₂₁", value=2.0, step=0.1, key="diag_a21")
        with col2:
            a12 = st.number_input("a₁₂", value=1.0, step=0.1, key="diag_a12")
            a22 = st.number_input("a₂₂", value=3.0, step=0.1, key="diag_a22")
        
        A = np.array([[a11, a12], [a21, a22]])
        
        if st.button("Diagonalize"):
            try:
                eigenvalues, eigenvectors = eig(A)
                
                # P = matrix of eigenvectors
                P = eigenvectors
                
                # D = diagonal matrix of eigenvalues
                D = np.diag(eigenvalues)
                
                # P inverse
                P_inv = np.linalg.inv(P)
                
                st.write("**Matrix A:**")
                st.dataframe(pd.DataFrame(A), use_container_width=True, hide_index=True)
                
                st.write("**P (Eigenvectors):**")
                st.dataframe(pd.DataFrame(P), use_container_width=True, hide_index=True)
                
                st.write("**D (Eigenvalues on diagonal):**")
                st.dataframe(pd.DataFrame(D), use_container_width=True, hide_index=True)
                
                st.write("**P⁻¹:**")
                st.dataframe(pd.DataFrame(P_inv), use_container_width=True, hide_index=True)
                
                # Verify: A = PDP⁻¹
                reconstructed = np.dot(np.dot(P, D), P_inv)
                st.write("**Verification: PDP⁻¹ = A**")
                st.dataframe(pd.DataFrame(reconstructed), use_container_width=True, hide_index=True)
                
                if np.allclose(reconstructed, A):
                    st.success("✅ Diagonalization verified!")
            
            except Exception as e:
                st.error(f"Error: {e}")
    
    with eig_tab4:
        st.write("### PCA (Principal Component Analysis) Demo")
        
        st.write("**PCA finds directions of maximum variance using eigenvectors!**")
        
        # Generate sample data
        np.random.seed(42)
        n_points = st.slider("Number of points:", 20, 100, 50)
        
        # Create correlated data
        mean = [0, 0]
        cov = [[3, 2], [2, 3]]
        data = np.random.multivariate_normal(mean, cov, n_points)
        
        # Center the data
        data_centered = data - np.mean(data, axis=0)
        
        # Covariance matrix
        cov_matrix = np.cov(data_centered.T)
        
        st.write("**Covariance Matrix:**")
        st.dataframe(pd.DataFrame(cov_matrix), use_container_width=True, hide_index=True)
        
        # Calculate eigenvalues/eigenvectors
        eigenvalues, eigenvectors = eig(cov_matrix)
        
        # Sort by eigenvalue (descending)
        idx = eigenvalues.argsort()[::-1]
        eigenvalues = eigenvalues[idx]
        eigenvectors = eigenvectors[:, idx]
        
        st.write("### Principal Components")
        for i, (eigval, eigvec) in enumerate(zip(eigenvalues, eigenvectors.T)):
            variance_explained = eigval / np.sum(eigenvalues) * 100
            st.write(f"**PC{i+1}:** Eigenvalue = {eigval:.3f}, Variance = {variance_explained:.1f}%")
            st.write(f"Direction: ({eigvec[0]:.3f}, {eigvec[1]:.3f})")
        
        # Visualization
        fig = go.Figure()
        
        # Data points
        fig.add_trace(go.Scatter(
            x=data_centered[:, 0], y=data_centered[:, 1],
            mode='markers',
            name='Data Points',
            marker=dict(color='#667eea', size=5)
        ))
        
        # Principal components (scaled by eigenvalue)
        for i, (eigval, eigvec) in enumerate(zip(eigenvalues, eigenvectors.T)):
            scale = np.sqrt(eigval) * 2
            fig.add_trace(go.Scatter(
                x=[0, eigvec[0]*scale], y=[0, eigvec[1]*scale],
                mode='lines+markers',
                name=f'PC{i+1}',
                line=dict(color=['#ef4444', '#764ba2'][i], width=4)
            ))
        
        fig.update_layout(
            title="PCA: Principal Components (Eigenvectors)",
            xaxis_title="X", yaxis_title="Y",
            xaxis=dict(zeroline=True),
            yaxis=dict(zeroline=True),
            height=500
        )
        st.plotly_chart(fig, use_container_width=True)
        
        st.write("**Key Insight:**")
        st.write("PC1 (red) = direction of maximum variance")
        st.write("PC2 (purple) = direction of second most variance")

# RIGHT COLUMN
with col2:
    def definition():
        st.write("### Eigenvalues & Eigenvectors")
        st.write("""
        For matrix A, if:
        - **Av = λv**
        - Then **λ** is eigenvalue
        - And **v** is eigenvector
        """)
        
        st.write("### Characteristic Equation")
        st.latex(r"\det(A - \lambda I) = 0")
        st.write("**Solves for eigenvalues λ**")
        
        st.write("### For 2×2 Matrix")
        st.latex(r"\lambda^2 - \text{tr}(A)\lambda + \det(A) = 0")
        
        st.write("### Diagonalization")
        st.latex(r"A = PDP^{-1}")
        st.write("**P** = eigenvectors, **D** = eigenvalues")
        
        st.write("### PCA")
        st.write("""
        - Covariance matrix eigenvectors
        - Principal components
        - Dimension reduction
        """)
    
    def examples():
        st.write("### Real-World Examples")
        
        st.write("**Vibrations:**")
        st.write("""
        - Natural frequencies
        - Mode shapes
        - Engineering structures
        """)
        
        st.write("**Google PageRank:**")
        st.write("""
        - Web page importance
        - Largest eigenvector
        - Ranking algorithm
        """)
        
        st.write("**Image Processing:**")
        st.write("""
        - Face recognition
        - Eigenfaces
        - Feature extraction
        """)
    
    def ml_usage():
        st.write("### ML Applications")
        
        st.write("**1. PCA (Dimensionality Reduction)**")
        st.write("""
        - Find principal components
        - Reduce feature dimensions
        - Preserve maximum variance
        - Used in: preprocessing, visualization
        """)
        
        st.write("**2. Spectral Clustering**")
        st.write("""
        - Graph Laplacian eigenvalues
        - Community detection
        - Image segmentation
        """)
        
        st.write("**3. Matrix Factorization**")
        st.write("""
        - SVD (Singular Value Decomposition)
        - Recommender systems
        - Latent factor models
        """)
        
        st.write("**4. Optimization**")
        st.write("""
        - Hessian eigenvalues
        - Convergence analysis
        - Learning rate selection
        """)
    
    def summary():
        st.write("### Key Takeaways")
        
        summary_data = {
            "Property": ["Av = λv", "A = PDP⁻¹", "PCA", "Variance"],
            "Meaning": ["Scaling", "Diagonalize", "Components", "Max direction"],
            "ML Use": ["Transform", "Decompose", "Reduce dims", "Feature extract"]
        }
        
        summary_df = pd.DataFrame(summary_data)
        st.dataframe(summary_df, use_container_width=True, hide_index=True)
        
        st.write("### Next Steps")
        st.write("""
        1. Master eigenvalue calculation
        2. Understand PCA
        3. Ready for **ML Fundamentals**!
        """)
    
    render_theory_panel({
        "Definition": definition,
        "Examples": examples,
        "ML Usage": ml_usage,
        "Summary": summary
    })

