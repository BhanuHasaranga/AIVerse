import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import streamlit as st
from utils.ui import apply_page_config, apply_theme, create_two_column_layout, render_theory_panel
import numpy as np
import pandas as pd
import plotly.graph_objects as go

# Apply theme
apply_page_config(title="Determinants Explorer", icon="🔄", sidebar_state="expanded")
apply_theme(page_type="page")

# Create layout
col1, col2 = create_two_column_layout("Determinants & Inverse", module_id="determinants")

# LEFT COLUMN
with col1:
    st.subheader("Interactive Determinants Explorer")
    
    # Tab selection
    det_tab1, det_tab2, det_tab3, det_tab4 = st.tabs([
        "Determinant Calculator",
        "Geometric Meaning",
        "Matrix Inverse",
        "Linear Systems"
    ])
    
    with det_tab1:
        st.write("### Determinant Calculator")
        
        size = st.selectbox("Matrix Size:", ["2×2", "3×3"], index=0)
        
        if size == "2×2":
            st.write("**Enter Matrix A:**")
            col1, col2 = st.columns(2)
            with col1:
                a11 = st.number_input("a₁₁", value=1.0, step=0.1, key="det_a11")
                a21 = st.number_input("a₂₁", value=2.0, step=0.1, key="det_a21")
            with col2:
                a12 = st.number_input("a₁₂", value=3.0, step=0.1, key="det_a12")
                a22 = st.number_input("a₂₂", value=4.0, step=0.1, key="det_a22")
            
            A = np.array([[a11, a12], [a21, a22]])
            
            st.write("**Matrix A:**")
            st.dataframe(pd.DataFrame(A), use_container_width=True, hide_index=True)
            
            det = np.linalg.det(A)
            st.metric("Determinant", f"{det:.2f}")
            
            # Step-by-step
            with st.expander("Step-by-Step Calculation"):
                st.write("**For 2×2 matrix:**")
                st.latex(r"\det(A) = a_{11}a_{22} - a_{12}a_{21}")
                st.write(f"det(A) = ({a11})×({a22}) - ({a12})×({a21})")
                st.write(f"det(A) = {a11*a22:.2f} - {a12*a21:.2f} = {det:.2f}")
        
        else:  # 3x3
            st.write("**Enter Matrix A:**")
            col1, col2, col3 = st.columns(3)
            with col1:
                a11 = st.number_input("a₁₁", value=1.0, step=0.1, key="det_a11_3")
                a21 = st.number_input("a₂₁", value=2.0, step=0.1, key="det_a21_3")
                a31 = st.number_input("a₃₁", value=3.0, step=0.1, key="det_a31_3")
            with col2:
                a12 = st.number_input("a₁₂", value=4.0, step=0.1, key="det_a12_3")
                a22 = st.number_input("a₂₂", value=5.0, step=0.1, key="det_a22_3")
                a32 = st.number_input("a₃₂", value=6.0, step=0.1, key="det_a32_3")
            with col3:
                a13 = st.number_input("a₁₃", value=7.0, step=0.1, key="det_a13_3")
                a23 = st.number_input("a₂₃", value=8.0, step=0.1, key="det_a23_3")
                a33 = st.number_input("a₃₃", value=9.0, step=0.1, key="det_a33_3")
            
            A = np.array([[a11, a12, a13], [a21, a22, a23], [a31, a32, a33]])
            
            st.write("**Matrix A:**")
            st.dataframe(pd.DataFrame(A), use_container_width=True, hide_index=True)
            
            det = np.linalg.det(A)
            st.metric("Determinant", f"{det:.2f}")
            
            # Step-by-step (Sarrus rule)
            with st.expander("Step-by-Step Calculation (Sarrus Rule)"):
                st.write("**For 3×3 matrix:**")
                st.write("det(A) = a₁₁(a₂₂a₃₃ - a₂₃a₃₂) - a₁₂(a₂₁a₃₃ - a₂₃a₃₁) + a₁₃(a₂₁a₃₂ - a₂₂a₃₁)")
                term1 = a11 * (a22*a33 - a23*a32)
                term2 = a12 * (a21*a33 - a23*a31)
                term3 = a13 * (a21*a32 - a22*a31)
                st.write(f"det(A) = {a11}×({a22}×{a33} - {a23}×{a32}) - {a12}×({a21}×{a33} - {a23}×{a31}) + {a13}×({a21}×{a32} - {a22}×{a31})")
                st.write(f"det(A) = {term1:.2f} - {term2:.2f} + {term3:.2f} = {det:.2f}")
        
        # Properties
        st.write("### Properties")
        if abs(det) < 1e-10:
            st.warning("⚠️ Determinant is 0 - Matrix is singular (not invertible)")
        else:
            st.success("✅ Determinant is non-zero - Matrix is invertible")
    
    with det_tab2:
        st.write("### Geometric Interpretation")
        
        st.write("**2D: Determinant = Area of Parallelogram**")
        
        col1, col2 = st.columns(2)
        with col1:
            v1_x = st.number_input("Vector v₁: x", value=3.0, step=0.1, key="geo_v1_x")
            v1_y = st.number_input("Vector v₁: y", value=1.0, step=0.1, key="geo_v1_y")
        with col2:
            v2_x = st.number_input("Vector v₂: x", value=1.0, step=0.1, key="geo_v2_x")
            v2_y = st.number_input("Vector v₂: y", value=3.0, step=0.1, key="geo_v2_y")
        
        # Matrix with columns as vectors
        A = np.array([[v1_x, v2_x], [v1_y, v2_y]])
        det = np.linalg.det(A)
        area = abs(det)
        
        st.metric("Area of Parallelogram", f"{area:.2f}")
        st.write(f"**Determinant:** {det:.2f}")
        
        # Visualization
        fig = go.Figure()
        # Parallelogram
        parallelogram_x = [0, v1_x, v1_x + v2_x, v2_x, 0]
        parallelogram_y = [0, v1_y, v1_y + v2_y, v2_y, 0]
        fig.add_trace(go.Scatter(
            x=parallelogram_x, y=parallelogram_y,
            fill='toself', fillcolor='rgba(102, 126, 234, 0.3)',
            line=dict(color='#667eea', width=2),
            name='Parallelogram'
        ))
        # Vectors
        fig.add_trace(go.Scatter(
            x=[0, v1_x], y=[0, v1_y],
            mode='lines+markers',
            name='v₁',
            line=dict(color='#ef4444', width=3)
        ))
        fig.add_trace(go.Scatter(
            x=[0, v2_x], y=[0, v2_y],
            mode='lines+markers',
            name='v₂',
            line=dict(color='#764ba2', width=3)
        ))
        fig.update_layout(
            title="Determinant = Area of Parallelogram",
            xaxis_title="X", yaxis_title="Y",
            xaxis=dict(range=[-5, 10], zeroline=True),
            yaxis=dict(range=[-5, 10], zeroline=True),
            height=400
        )
        st.plotly_chart(fig, use_container_width=True)
    
    with det_tab3:
        st.write("### Matrix Inverse")
        
        size = st.selectbox("Matrix Size:", ["2×2", "3×3"], index=0, key="inv_size")
        
        if size == "2×2":
            st.write("**Enter Matrix A:**")
            col1, col2 = st.columns(2)
            with col1:
                a11 = st.number_input("a₁₁", value=1.0, step=0.1, key="inv_a11")
                a21 = st.number_input("a₂₁", value=2.0, step=0.1, key="inv_a21")
            with col2:
                a12 = st.number_input("a₁₂", value=3.0, step=0.1, key="inv_a12")
                a22 = st.number_input("a₂₂", value=4.0, step=0.1, key="inv_a22")
            
            A = np.array([[a11, a12], [a21, a22]])
            
            st.write("**Matrix A:**")
            st.dataframe(pd.DataFrame(A), use_container_width=True, hide_index=True)
            
            det = np.linalg.det(A)
            
            if abs(det) < 1e-10:
                st.error("❌ Matrix is singular (determinant = 0). Inverse does not exist.")
            else:
                A_inv = np.linalg.inv(A)
                st.write("**A⁻¹ (Inverse):**")
                st.dataframe(pd.DataFrame(A_inv), use_container_width=True, hide_index=True)
                
                # Verify
                I = np.dot(A, A_inv)
                st.write("**Verification: A × A⁻¹ = I (Identity):**")
                st.dataframe(pd.DataFrame(I), use_container_width=True, hide_index=True)
                
                with st.expander("Formula for 2×2 Inverse"):
                    st.write("**For 2×2 matrix:**")
                    st.latex(r"A^{-1} = \frac{1}{\det(A)} \begin{bmatrix} a_{22} & -a_{12} \\ -a_{21} & a_{11} \end{bmatrix}")
                    st.write(f"A⁻¹ = (1/{det:.2f}) × [[{a22}, {-a12}], [{-a21}, {a11}]]")
    
    with det_tab4:
        st.write("### Linear System Solver (Ax = b)")
        
        st.write("**System: Ax = b**")
        
        st.write("**Matrix A (2×2):**")
        col1, col2 = st.columns(2)
        with col1:
            a11 = st.number_input("a₁₁", value=2.0, step=0.1, key="sys_a11")
            a21 = st.number_input("a₂₁", value=1.0, step=0.1, key="sys_a21")
        with col2:
            a12 = st.number_input("a₁₂", value=1.0, step=0.1, key="sys_a12")
            a22 = st.number_input("a₂₂", value=3.0, step=0.1, key="sys_a22")
        
        A = np.array([[a11, a12], [a21, a22]])
        
        st.write("**Vector b:**")
        col1, col2 = st.columns(2)
        with col1:
            b1 = st.number_input("b₁", value=5.0, step=0.1, key="sys_b1")
        with col2:
            b2 = st.number_input("b₂", value=7.0, step=0.1, key="sys_b2")
        
        b = np.array([b1, b2])
        
        st.write("**System:**")
        st.latex(fr"\begin{{bmatrix}} {a11} & {a12} \\ {a21} & {a22} \end{{bmatrix}} \begin{{bmatrix}} x_1 \\ x_2 \end{{bmatrix}} = \begin{{bmatrix}} {b1} \\ {b2} \end{{bmatrix}}")
        
        det = np.linalg.det(A)
        
        if abs(det) < 1e-10:
            st.error("❌ System has no unique solution (determinant = 0)")
        else:
            x = np.linalg.solve(A, b)
            st.write("**Solution x:**")
            st.write(f"x₁ = {x[0]:.2f}")
            st.write(f"x₂ = {x[1]:.2f}")
            
            # Verify
            st.write("**Verification: Ax = b**")
            Ax = np.dot(A, x)
            st.write(f"Ax = [{Ax[0]:.2f}, {Ax[1]:.2f}]")
            st.write(f"b = [{b[0]:.2f}, {b[1]:.2f}]")
            if np.allclose(Ax, b):
                st.success("✅ Solution verified!")

# RIGHT COLUMN
with col2:
    def definition():
        st.write("### What is a Determinant?")
        st.write("""
        The **determinant** is a scalar value:
        - Computed from square matrix
        - Measures "scaling factor"
        - Tells if matrix is invertible
        """)
        
        st.write("### 2×2 Determinant")
        st.latex(r"\det(A) = a_{11}a_{22} - a_{12}a_{21}")
        
        st.write("### 3×3 Determinant")
        st.latex(r"\det(A) = a_{11}(a_{22}a_{33} - a_{23}a_{32}) - a_{12}(a_{21}a_{33} - a_{23}a_{31}) + a_{13}(a_{21}a_{32} - a_{22}a_{31})")
        
        st.write("### Matrix Inverse")
        st.latex(r"A^{-1} = \frac{1}{\det(A)} \text{adj}(A)")
        st.write("**Exists only if det(A) ≠ 0**")
        
        st.write("### Linear Systems")
        st.latex(r"Ax = b \Rightarrow x = A^{-1}b")
        st.write("**Unique solution if det(A) ≠ 0**")
    
    def examples():
        st.write("### Real-World Examples")
        
        st.write("**Area/Volume:**")
        st.write("""
        - 2D: Area of parallelogram
        - 3D: Volume of parallelepiped
        - Scaling factor of transformation
        """)
        
        st.write("**System Solving:**")
        st.write("""
        - Circuit analysis
        - Structural engineering
        - Economic models
        """)
        
        st.write("**Invertibility:**")
        st.write("""
        - det = 0 → singular (no inverse)
        - det ≠ 0 → invertible
        - Used in matrix operations
        """)
    
    def ml_usage():
        st.write("### ML Applications")
        
        st.write("**1. Feature Independence**")
        st.write("""
        - det = 0 → features are linearly dependent
        - Multicollinearity detection
        - Feature selection
        """)
        
        st.write("**2. Change of Variables**")
        st.write("""
        - Jacobian determinant
        - Coordinate transformations
        - Probability distributions
        """)
        
        st.write("**3. Optimization**")
        st.write("""
        - Hessian determinant
        - Critical point classification
        - Convexity checking
        """)
        
        st.write("**4. Linear Systems**")
        st.write("""
        - Solving Ax = b
        - Normal equations
        - Least squares solutions
        """)
    
    def summary():
        st.write("### Key Takeaways")
        
        summary_data = {
            "Size": ["2×2", "3×3", "n×n"],
            "Meaning": ["Area", "Volume", "Scaling"],
            "Zero det": ["No inverse", "Dependent rows", "No unique solution"]
        }
        
        summary_df = pd.DataFrame(summary_data)
        st.dataframe(summary_df, use_container_width=True, hide_index=True)
        
        st.write("### Next Steps")
        st.write("""
        1. Understand geometric meaning
        2. Master inverse calculation
        3. Learn **Eigenvalues** next!
        """)
    
    render_theory_panel({
        "Definition": definition,
        "Examples": examples,
        "ML Usage": ml_usage,
        "Summary": summary
    })

