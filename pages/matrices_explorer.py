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
apply_page_config(title="Matrices Explorer", icon="🔢", sidebar_state="expanded")
apply_theme(page_type="page")

# Create layout
col1, col2 = create_two_column_layout("Matrices & Multiplication", module_id="matrices")

# LEFT COLUMN
with col1:
    st.subheader("Interactive Matrix Explorer")
    
    # Tab selection
    mat_tab1, mat_tab2, mat_tab3, mat_tab4 = st.tabs([
        "Matrix Basics",
        "Operations",
        "Multiplication",
        "Transformations"
    ])
    
    with mat_tab1:
        st.write("### Matrix Basics")
        
        size = st.selectbox("Matrix Size:", ["2×2", "3×3"], index=0)
        
        if size == "2×2":
            st.write("**Enter Matrix A:**")
            col1, col2 = st.columns(2)
            with col1:
                a11 = st.number_input("a₁₁", value=1.0, step=0.1, key="a11")
                a21 = st.number_input("a₂₁", value=2.0, step=0.1, key="a21")
            with col2:
                a12 = st.number_input("a₁₂", value=3.0, step=0.1, key="a12")
                a22 = st.number_input("a₂₂", value=4.0, step=0.1, key="a22")
            
            A = np.array([[a11, a12], [a21, a22]])
            
            st.write("**Matrix A:**")
            st.dataframe(pd.DataFrame(A), use_container_width=True, hide_index=True)
            
            # Special matrices
            st.write("### Special Matrices")
            col_id, col_zero, col_trans = st.columns(3)
            
            with col_id:
                I = np.eye(2)
                st.write("**Identity:**")
                st.dataframe(pd.DataFrame(I), use_container_width=True, hide_index=True)
            
            with col_zero:
                Z = np.zeros((2, 2))
                st.write("**Zero:**")
                st.dataframe(pd.DataFrame(Z), use_container_width=True, hide_index=True)
            
            with col_trans:
                A_T = A.T
                st.write("**Aᵀ (Transpose):**")
                st.dataframe(pd.DataFrame(A_T), use_container_width=True, hide_index=True)
        
        else:  # 3x3
            st.write("**Enter Matrix A:**")
            col1, col2, col3 = st.columns(3)
            with col1:
                a11 = st.number_input("a₁₁", value=1.0, step=0.1, key="a11_3")
                a21 = st.number_input("a₂₁", value=2.0, step=0.1, key="a21_3")
                a31 = st.number_input("a₃₁", value=3.0, step=0.1, key="a31_3")
            with col2:
                a12 = st.number_input("a₁₂", value=4.0, step=0.1, key="a12_3")
                a22 = st.number_input("a₂₂", value=5.0, step=0.1, key="a22_3")
                a32 = st.number_input("a₃₂", value=6.0, step=0.1, key="a32_3")
            with col3:
                a13 = st.number_input("a₁₃", value=7.0, step=0.1, key="a13_3")
                a23 = st.number_input("a₂₃", value=8.0, step=0.1, key="a23_3")
                a33 = st.number_input("a₃₃", value=9.0, step=0.1, key="a33_3")
            
            A = np.array([[a11, a12, a13], [a21, a22, a23], [a31, a32, a33]])
            
            st.write("**Matrix A:**")
            st.dataframe(pd.DataFrame(A), use_container_width=True, hide_index=True)
    
    with mat_tab2:
        st.write("### Matrix Operations")
        
        size = st.selectbox("Matrix Size:", ["2×2", "3×3"], index=0, key="ops_size")
        
        if size == "2×2":
            st.write("**Matrix A:**")
            col1, col2 = st.columns(2)
            with col1:
                a11 = st.number_input("a₁₁", value=1.0, step=0.1, key="op_a11")
                a21 = st.number_input("a₂₁", value=2.0, step=0.1, key="op_a21")
            with col2:
                a12 = st.number_input("a₁₂", value=3.0, step=0.1, key="op_a12")
                a22 = st.number_input("a₂₂", value=4.0, step=0.1, key="op_a22")
            A = np.array([[a11, a12], [a21, a22]])
            
            st.write("**Matrix B:**")
            col1, col2 = st.columns(2)
            with col1:
                b11 = st.number_input("b₁₁", value=5.0, step=0.1, key="op_b11")
                b21 = st.number_input("b₂₁", value=6.0, step=0.1, key="op_b21")
            with col2:
                b12 = st.number_input("b₁₂", value=7.0, step=0.1, key="op_b12")
                b22 = st.number_input("b₂₂", value=8.0, step=0.1, key="op_b22")
            B = np.array([[b11, b12], [b21, b22]])
            
            op = st.radio("Operation:", ["Addition", "Subtraction", "Scalar Multiplication"], horizontal=True)
            
            if op == "Scalar Multiplication":
                scalar = st.number_input("Scalar (k):", value=2.0, step=0.1)
                result = scalar * A
                st.write(f"**k × A = {scalar} × A:**")
            elif op == "Addition":
                result = A + B
                st.write("**A + B:**")
            else:  # Subtraction
                result = A - B
                st.write("**A - B:**")
            
            st.dataframe(pd.DataFrame(result), use_container_width=True, hide_index=True)
    
    with mat_tab3:
        st.write("### Matrix Multiplication")
        
        st.write("**Matrix A (m×n):**")
        m = st.number_input("Rows (m):", min_value=1, max_value=3, value=2, step=1)
        n = st.number_input("Columns (n):", min_value=1, max_value=3, value=2, step=1)
        
        st.write("Enter values for A:")
        A_vals = []
        for i in range(int(m)):
            cols = st.columns(int(n))
            row = []
            for j, col in enumerate(cols):
                val = col.number_input(f"a{i+1}{j+1}", value=float(i*n + j + 1), step=0.1, key=f"mat_a_{i}_{j}")
                row.append(val)
            A_vals.append(row)
        A = np.array(A_vals)
        
        st.write("**Matrix B (n×p):**")
        p = st.number_input("Columns (p):", min_value=1, max_value=3, value=2, step=1)
        
        st.write("Enter values for B:")
        B_vals = []
        for i in range(int(n)):
            cols = st.columns(int(p))
            row = []
            for j, col in enumerate(cols):
                val = col.number_input(f"b{i+1}{j+1}", value=float(i*p + j + 1), step=0.1, key=f"mat_b_{i}_{j}")
                row.append(val)
            B_vals.append(row)
        B = np.array(B_vals)
        
        if st.button("Calculate A × B"):
            result = np.dot(A, B)
            
            st.write("**Result (A × B):**")
            st.dataframe(pd.DataFrame(result), use_container_width=True, hide_index=True)
            
            # Step-by-step
            with st.expander("Step-by-Step Calculation"):
                st.write(f"**A is {m}×{n}, B is {n}×{p}, Result is {m}×{p}**")
                st.write("")
                for i in range(int(m)):
                    for j in range(int(p)):
                        steps = " + ".join([f"({A[i][k]})×({B[k][j]})" for k in range(int(n))])
                        st.write(f"**c{i+1}{j+1}** = {steps} = {result[i][j]:.2f}")
    
    with mat_tab4:
        st.write("### Matrix Transformations")
        
        transform_type = st.selectbox("Transformation:", ["Rotation", "Scaling", "Shear", "Reflection"])
        
        # 2D vector to transform
        st.write("**Vector to Transform:**")
        col1, col2 = st.columns(2)
        with col1:
            v_x = st.number_input("x", value=1.0, step=0.1, key="trans_vx")
        with col2:
            v_y = st.number_input("y", value=1.0, step=0.1, key="trans_vy")
        v = np.array([v_x, v_y])
        
        if transform_type == "Rotation":
            angle = st.slider("Angle (degrees):", -180, 180, 45)
            angle_rad = np.radians(angle)
            T = np.array([
                [np.cos(angle_rad), -np.sin(angle_rad)],
                [np.sin(angle_rad), np.cos(angle_rad)]
            ])
            st.write(f"**Rotation Matrix (θ = {angle}°):**")
        
        elif transform_type == "Scaling":
            scale_x = st.slider("Scale X:", 0.1, 3.0, 2.0, 0.1)
            scale_y = st.slider("Scale Y:", 0.1, 3.0, 1.5, 0.1)
            T = np.array([[scale_x, 0], [0, scale_y]])
            st.write(f"**Scaling Matrix:**")
        
        elif transform_type == "Shear":
            shear_x = st.slider("Shear X:", -2.0, 2.0, 1.0, 0.1)
            T = np.array([[1, shear_x], [0, 1]])
            st.write(f"**Shear Matrix:**")
        
        else:  # Reflection
            axis = st.radio("Reflect across:", ["X-axis", "Y-axis", "Line y=x"], horizontal=True)
            if axis == "X-axis":
                T = np.array([[1, 0], [0, -1]])
            elif axis == "Y-axis":
                T = np.array([[-1, 0], [0, 1]])
            else:
                T = np.array([[0, 1], [1, 0]])
            st.write(f"**Reflection Matrix ({axis}):**")
        
        st.dataframe(pd.DataFrame(T), use_container_width=True, hide_index=True)
        
        # Apply transformation
        v_transformed = np.dot(T, v)
        
        st.write(f"**Original vector:** ({v[0]:.2f}, {v[1]:.2f})")
        st.write(f"**Transformed vector:** ({v_transformed[0]:.2f}, {v_transformed[1]:.2f})")
        
        # Visualization
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=[0, v[0]], y=[0, v[1]],
            mode='lines+markers',
            name='Original',
            line=dict(color='#667eea', width=3)
        ))
        fig.add_trace(go.Scatter(
            x=[0, v_transformed[0]], y=[0, v_transformed[1]],
            mode='lines+markers',
            name='Transformed',
            line=dict(color='#ef4444', width=3, dash='dash')
        ))
        fig.update_layout(
            title=f"{transform_type} Transformation",
            xaxis_title="X", yaxis_title="Y",
            xaxis=dict(range=[-5, 5], zeroline=True),
            yaxis=dict(range=[-5, 5], zeroline=True),
            height=400
        )
        st.plotly_chart(fig, use_container_width=True)

# RIGHT COLUMN
with col2:
    def definition():
        st.write("### What are Matrices?")
        st.write("""
        A **matrix** is a rectangular array of numbers:
        - **Rows** (horizontal)
        - **Columns** (vertical)
        - **Size:** m × n (m rows, n columns)
        """)
        
        st.write("### Matrix Operations")
        st.write("**Addition/Subtraction:**")
        st.latex(r"(A \pm B)_{ij} = A_{ij} \pm B_{ij}")
        
        st.write("**Scalar Multiplication:**")
        st.latex(r"(kA)_{ij} = k \cdot A_{ij}")
        
        st.write("### Matrix Multiplication")
        st.latex(r"(AB)_{ij} = \sum_{k=1}^{n} A_{ik} B_{kj}")
        st.write("**Rule:** A (m×n) × B (n×p) = C (m×p)")
        
        st.write("### Transpose")
        st.latex(r"(A^T)_{ij} = A_{ji}")
        st.write("Rows become columns, columns become rows")
    
    def examples():
        st.write("### Real-World Examples")
        
        st.write("**Data Tables:**")
        st.write("""
        - Each row = data point
        - Each column = feature
        - Dataset = matrix
        """)
        
        st.write("**Image Processing:**")
        st.write("""
        - Image = matrix of pixels
        - Filters = matrix operations
        - Convolution = matrix multiplication
        """)
        
        st.write("**Transformations:**")
        st.write("""
        - Rotation, scaling, translation
        - Computer graphics
        - Data preprocessing
        """)
    
    def ml_usage():
        st.write("### ML Applications")
        
        st.write("**1. Data Representation**")
        st.write("""
        - X: n×m matrix (n samples, m features)
        - Each row = one data point
        - Each column = one feature
        """)
        
        st.write("**2. Linear Transformations**")
        st.write("""
        - Neural network layers: y = Wx + b
        - W = weight matrix
        - Matrix multiplication = forward pass
        """)
        
        st.write("**3. Feature Engineering**")
        st.write("""
        - PCA: dimension reduction
        - Transformations: scaling, rotation
        - Normalization matrices
        """)
        
        st.write("**4. Optimization**")
        st.write("""
        - Gradient matrices
        - Hessian matrix (2nd derivatives)
        - Batch operations
        """)
    
    def summary():
        st.write("### Key Takeaways")
        
        summary_data = {
            "Operation": ["Addition", "Multiplication", "Transpose", "Transform"],
            "Rule": ["Element-wise", "Row×Column", "Flip rows/cols", "T × v"],
            "ML Use": ["Data prep", "Neural nets", "Feature transform", "Graphics"]
        }
        
        summary_df = pd.DataFrame(summary_data)
        st.dataframe(summary_df, use_container_width=True, hide_index=True)
        
        st.write("### Next Steps")
        st.write("""
        1. Master matrix multiplication
        2. Understand transformations
        3. Learn **Determinants** next!
        """)
    
    render_theory_panel({
        "Definition": definition,
        "Examples": examples,
        "ML Usage": ml_usage,
        "Summary": summary
    })

