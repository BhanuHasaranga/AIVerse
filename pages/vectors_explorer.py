import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import streamlit as st
from utils.ui import apply_page_config, apply_theme, create_two_column_layout, render_theory_panel
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px

# Apply theme
apply_page_config(title="Vectors Explorer", icon="📍", sidebar_state="expanded")
apply_theme(page_type="page")

# Create layout
col1, col2 = create_two_column_layout("Vectors & Operations", module_id="vectors")

# LEFT COLUMN
with col1:
    st.subheader("Interactive Vector Explorer")
    
    # Tab selection
    vec_tab1, vec_tab2, vec_tab3, vec_tab4 = st.tabs([
        "Vector Basics",
        "Operations",
        "Dot Product",
        "Norms & Projections"
    ])
    
    with vec_tab1:
        st.write("### Vector Basics")
        
        dim = st.radio("Dimension:", [2, 3], horizontal=True)
        
        if dim == 2:
            col_x1, col_y1 = st.columns(2)
            with col_x1:
                v1_x = st.number_input("Vector v₁: x", value=3.0, step=0.1, key="v1_x_2d")
            with col_y1:
                v1_y = st.number_input("Vector v₁: y", value=4.0, step=0.1, key="v1_y_2d")
            
            v1 = np.array([v1_x, v1_y])
            
            # 2D Visualization
            fig = go.Figure()
            fig.add_trace(go.Scatter(
                x=[0, v1[0]], y=[0, v1[1]],
                mode='lines+markers',
                name='v₁',
                line=dict(color='#667eea', width=3),
                marker=dict(size=10)
            ))
            fig.add_trace(go.Scatter(
                x=[0], y=[0],
                mode='markers',
                name='Origin',
                marker=dict(size=8, color='red')
            ))
            fig.update_layout(
                title="2D Vector Visualization",
                xaxis_title="X",
                yaxis_title="Y",
                xaxis=dict(range=[-10, 10], zeroline=True, zerolinecolor='gray'),
                yaxis=dict(range=[-10, 10], zeroline=True, zerolinecolor='gray'),
                showlegend=True,
                height=400
            )
            st.plotly_chart(fig, use_container_width=True)
            
            st.write(f"**Vector v₁:** ({v1[0]:.2f}, {v1[1]:.2f})")
            magnitude = np.linalg.norm(v1)
            st.write(f"**Magnitude (||v₁||):** {magnitude:.2f}")
            angle = np.degrees(np.arctan2(v1[1], v1[0]))
            st.write(f"**Angle (θ):** {angle:.2f}°")
        
        else:  # 3D
            col_x1, col_y1, col_z1 = st.columns(3)
            with col_x1:
                v1_x = st.number_input("Vector v₁: x", value=2.0, step=0.1, key="v1_x_3d")
            with col_y1:
                v1_y = st.number_input("Vector v₁: y", value=3.0, step=0.1, key="v1_y_3d")
            with col_z1:
                v1_z = st.number_input("Vector v₁: z", value=4.0, step=0.1, key="v1_z_3d")
            
            v1 = np.array([v1_x, v1_y, v1_z])
            
            # 3D Visualization
            fig = go.Figure(data=go.Scatter3d(
                x=[0, v1[0]], y=[0, v1[1]], z=[0, v1[2]],
                mode='lines+markers',
                name='v₁',
                line=dict(color='#667eea', width=5),
                marker=dict(size=5)
            ))
            fig.update_layout(
                title="3D Vector Visualization",
                scene=dict(
                    xaxis_title="X",
                    yaxis_title="Y",
                    zaxis_title="Z",
                    xaxis=dict(range=[-5, 5]),
                    yaxis=dict(range=[-5, 5]),
                    zaxis=dict(range=[-5, 5])
                ),
                height=500
            )
            st.plotly_chart(fig, use_container_width=True)
            
            st.write(f"**Vector v₁:** ({v1[0]:.2f}, {v1[1]:.2f}, {v1[2]:.2f})")
            magnitude = np.linalg.norm(v1)
            st.write(f"**Magnitude (||v₁||):** {magnitude:.2f}")
    
    with vec_tab2:
        st.write("### Vector Operations")
        
        dim = st.radio("Dimension:", [2, 3], horizontal=True, key="ops_dim")
        
        if dim == 2:
            col1, col2 = st.columns(2)
            with col1:
                st.write("**Vector v₁:**")
                v1_x = st.number_input("x₁", value=3.0, step=0.1, key="op_v1_x")
                v1_y = st.number_input("y₁", value=4.0, step=0.1, key="op_v1_y")
                v1 = np.array([v1_x, v1_y])
            
            with col2:
                st.write("**Vector v₂:**")
                v2_x = st.number_input("x₂", value=1.0, step=0.1, key="op_v2_x")
                v2_y = st.number_input("y₂", value=2.0, step=0.1, key="op_v2_y")
                v2 = np.array([v2_x, v2_y])
            
            # Operations
            op = st.radio("Operation:", ["Addition", "Subtraction", "Scalar Multiplication"], horizontal=True)
            
            if op == "Scalar Multiplication":
                scalar = st.number_input("Scalar (k):", value=2.0, step=0.1)
                result = scalar * v1
                st.write(f"**k × v₁ = {scalar} × ({v1[0]}, {v1[1]}) = ({result[0]:.2f}, {result[1]:.2f})**")
            elif op == "Addition":
                result = v1 + v2
                st.write(f"**v₁ + v₂ = ({v1[0]}, {v1[1]}) + ({v2[0]}, {v2[1]}) = ({result[0]:.2f}, {result[1]:.2f})**")
            else:  # Subtraction
                result = v1 - v2
                st.write(f"**v₁ - v₂ = ({v1[0]}, {v1[1]}) - ({v2[0]}, {v2[1]}) = ({result[0]:.2f}, {result[1]:.2f})**")
            
            # Visualization
            fig = go.Figure()
            fig.add_trace(go.Scatter(x=[0, v1[0]], y=[0, v1[1]], mode='lines+markers', name='v₁', line=dict(color='#667eea', width=3)))
            fig.add_trace(go.Scatter(x=[0, v2[0]], y=[0, v2[1]], mode='lines+markers', name='v₂', line=dict(color='#764ba2', width=3)))
            
            if op != "Scalar Multiplication":
                fig.add_trace(go.Scatter(x=[0, result[0]], y=[0, result[1]], mode='lines+markers', name='Result', line=dict(color='#ef4444', width=3, dash='dash')))
            else:
                fig.add_trace(go.Scatter(x=[0, result[0]], y=[0, result[1]], mode='lines+markers', name=f'{scalar}×v₁', line=dict(color='#ef4444', width=3, dash='dash')))
            
            fig.update_layout(
                title=f"Vector {op}",
                xaxis_title="X", yaxis_title="Y",
                xaxis=dict(range=[-10, 10], zeroline=True),
                yaxis=dict(range=[-10, 10], zeroline=True),
                height=400
            )
            st.plotly_chart(fig, use_container_width=True)
    
    with vec_tab3:
        st.write("### Dot Product")
        
        col1, col2 = st.columns(2)
        with col1:
            st.write("**Vector v₁:**")
            v1_x = st.number_input("x₁", value=3.0, step=0.1, key="dot_v1_x")
            v1_y = st.number_input("y₁", value=4.0, step=0.1, key="dot_v1_y")
            v1 = np.array([v1_x, v1_y])
        
        with col2:
            st.write("**Vector v₂:**")
            v2_x = st.number_input("x₂", value=1.0, step=0.1, key="dot_v2_x")
            v2_y = st.number_input("y₂", value=2.0, step=0.1, key="dot_v2_y")
            v2 = np.array([v2_x, v2_y])
        
        dot_product = np.dot(v1, v2)
        st.write(f"**Dot Product: v₁ · v₂ = {dot_product:.2f}**")
        
        # Step-by-step calculation
        with st.expander("Step-by-Step Calculation"):
            st.write(f"v₁ · v₂ = x₁x₂ + y₁y₂")
            st.write(f"v₁ · v₂ = ({v1[0]})×({v2[0]}) + ({v1[1]})×({v2[1]})")
            st.write(f"v₁ · v₂ = {v1[0]*v2[0]:.2f} + {v1[1]*v2[1]:.2f} = {dot_product:.2f}")
        
        # Geometric interpretation
        angle = np.arccos(np.clip(dot_product / (np.linalg.norm(v1) * np.linalg.norm(v2)), -1, 1))
        angle_deg = np.degrees(angle)
        st.write(f"**Angle between vectors (θ):** {angle_deg:.2f}°")
        st.write(f"**Geometric formula:** v₁ · v₂ = ||v₁|| ||v₂|| cos(θ) = {np.linalg.norm(v1):.2f} × {np.linalg.norm(v2):.2f} × cos({angle_deg:.2f}°) = {dot_product:.2f}")
        
        # Visualization
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=[0, v1[0]], y=[0, v1[1]], mode='lines+markers', name='v₁', line=dict(color='#667eea', width=3)))
        fig.add_trace(go.Scatter(x=[0, v2[0]], y=[0, v2[1]], mode='lines+markers', name='v₂', line=dict(color='#764ba2', width=3)))
        fig.update_layout(
            title="Dot Product Visualization",
            xaxis_title="X", yaxis_title="Y",
            xaxis=dict(range=[-10, 10], zeroline=True),
            yaxis=dict(range=[-10, 10], zeroline=True),
            height=400
        )
        st.plotly_chart(fig, use_container_width=True)
    
    with vec_tab4:
        st.write("### Norms & Projections")
        
        col1, col2 = st.columns(2)
        with col1:
            st.write("**Vector v₁:**")
            v1_x = st.number_input("x₁", value=3.0, step=0.1, key="norm_v1_x")
            v1_y = st.number_input("y₁", value=4.0, step=0.1, key="norm_v1_y")
            v1 = np.array([v1_x, v1_y])
        
        with col2:
            st.write("**Vector v₂ (for projection):**")
            v2_x = st.number_input("x₂", value=1.0, step=0.1, key="norm_v2_x")
            v2_y = st.number_input("y₂", value=0.0, step=0.1, key="norm_v2_y")
            v2 = np.array([v2_x, v2_y])
        
        # Norms
        st.write("### Vector Norms")
        l2_norm = np.linalg.norm(v1)
        l1_norm = np.sum(np.abs(v1))
        inf_norm = np.max(np.abs(v1))
        
        col_n1, col_n2, col_n3 = st.columns(3)
        col_n1.metric("L₂ Norm (Euclidean)", f"{l2_norm:.2f}")
        col_n2.metric("L₁ Norm (Manhattan)", f"{l1_norm:.2f}")
        col_n3.metric("L∞ Norm (Max)", f"{inf_norm:.2f}")
        
        st.write(f"**L₂:** ||v₁||₂ = √(x² + y²) = √({v1[0]²:.2f} + {v1[1]²:.2f}) = {l2_norm:.2f}")
        st.write(f"**L₁:** ||v₁||₁ = |x| + |y| = |{v1[0]}| + |{v1[1]}| = {l1_norm:.2f}")
        st.write(f"**L∞:** ||v₁||∞ = max(|x|, |y|) = max(|{v1[0]}|, |{v1[1]}|) = {inf_norm:.2f}")
        
        # Projection
        st.write("### Vector Projection")
        if np.linalg.norm(v2) > 0:
            projection = (np.dot(v1, v2) / np.dot(v2, v2)) * v2
            st.write(f"**Projection of v₁ onto v₂:**")
            st.write(f"proj_v₂(v₁) = ((v₁·v₂)/(v₂·v₂)) × v₂ = ({np.dot(v1, v2):.2f}/{np.dot(v2, v2):.2f}) × ({v2[0]}, {v2[1]}) = ({projection[0]:.2f}, {projection[1]:.2f})")
            
            # Visualization
            fig = go.Figure()
            fig.add_trace(go.Scatter(x=[0, v1[0]], y=[0, v1[1]], mode='lines+markers', name='v₁', line=dict(color='#667eea', width=3)))
            fig.add_trace(go.Scatter(x=[0, v2[0]], y=[0, v2[1]], mode='lines+markers', name='v₂', line=dict(color='#764ba2', width=3)))
            fig.add_trace(go.Scatter(x=[0, projection[0]], y=[0, projection[1]], mode='lines+markers', name='Projection', line=dict(color='#ef4444', width=3, dash='dash')))
            fig.add_trace(go.Scatter(x=[projection[0], v1[0]], y=[projection[1], v1[1]], mode='lines', name='Perpendicular', line=dict(color='gray', width=1, dash='dot')))
            fig.update_layout(
                title="Vector Projection",
                xaxis_title="X", yaxis_title="Y",
                xaxis=dict(range=[-10, 10], zeroline=True),
                yaxis=dict(range=[-10, 10], zeroline=True),
                height=400
            )
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.warning("v₂ must be non-zero for projection")

# RIGHT COLUMN
with col2:
    def definition():
        st.write("### What are Vectors?")
        st.write("""
        A **vector** is a mathematical object with:
        - **Magnitude** (length)
        - **Direction**
        
        In ML, vectors represent:
        - Features (data points)
        - Weights (model parameters)
        - Embeddings (word/document vectors)
        """)
        
        st.write("### Dot Product")
        st.latex(r"v_1 \cdot v_2 = x_1 x_2 + y_1 y_2")
        st.write("**Geometric:**")
        st.latex(r"v_1 \cdot v_2 = ||v_1|| ||v_2|| \cos(\theta)")
        
        st.write("### Vector Norms")
        st.write("**L₂ (Euclidean):**")
        st.latex(r"||v||_2 = \sqrt{x^2 + y^2}")
        st.write("**L₁ (Manhattan):**")
        st.latex(r"||v||_1 = |x| + |y|")
        st.write("**L∞ (Max):**")
        st.latex(r"||v||_\infty = \max(|x|, |y|)")
        
        st.write("### Projection")
        st.latex(r"\text{proj}_{v_2}(v_1) = \frac{v_1 \cdot v_2}{v_2 \cdot v_2} v_2")
    
    def examples():
        st.write("### Real-World Examples")
        
        st.write("**Feature Vectors:**")
        st.write("""
        In ML, each data point is a vector:
        - Image: [pixel₁, pixel₂, ..., pixelₙ]
        - Text: [word₁_count, word₂_count, ...]
        - User: [age, income, clicks, ...]
        """)
        
        st.write("**Similarity (Dot Product):**")
        st.write("""
        Cosine similarity uses dot product:
        - Recommender systems
        - Search engines
        - Document matching
        """)
        
        st.write("**Distance (Norms):**")
        st.write("""
        L₂ norm = Euclidean distance
        - KNN algorithm
        - Clustering (K-means)
        - Anomaly detection
        """)
    
    def ml_usage():
        st.write("### ML Applications")
        
        st.write("**1. Feature Representation**")
        st.write("""
        Every data point = vector
        - Images → pixel vectors
        - Text → word embeddings
        - Users → feature vectors
        """)
        
        st.write("**2. Similarity & Distance**")
        st.write("""
        - Dot product → cosine similarity
        - L₂ norm → Euclidean distance
        - Used in: KNN, clustering, search
        """)
        
        st.write("**3. Optimization**")
        st.write("""
        - Gradient descent uses vector operations
        - Model weights are vectors
        - Updates: w = w - α∇L
        """)
        
        st.write("**4. Neural Networks**")
        st.write("""
        - Inputs/outputs are vectors
        - Matrix-vector multiplication
        - Activation functions operate on vectors
        """)
    
    def summary():
        st.write("### Key Takeaways")
        
        summary_data = {
            "Concept": ["Vector", "Dot Product", "Norms", "Projection"],
            "Formula": ["v = (x, y)", "v₁·v₂ = x₁x₂ + y₁y₂", "||v|| = √(x²+y²)", "proj = (v₁·v₂)/(v₂·v₂) × v₂"],
            "ML Use": ["Features", "Similarity", "Distance", "Dimensionality reduction"]
        }
        
        summary_df = pd.DataFrame(summary_data)
        st.dataframe(summary_df, use_container_width=True, hide_index=True)
        
        st.write("### Next Steps")
        st.write("""
        1. Master vector operations
        2. Understand dot product geometrically
        3. Learn different norms
        4. Move to **Matrices** next!
        """)
    
    render_theory_panel({
        "Definition": definition,
        "Examples": examples,
        "ML Usage": ml_usage,
        "Summary": summary
    })

