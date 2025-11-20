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
            st.dataframe(pd.DataFrame(A), width='stretch', hide_index=True)
            
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
            st.dataframe(pd.DataFrame(A), width='stretch', hide_index=True)
            
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
        st.plotly_chart(fig, width='stretch')
    
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
            st.dataframe(pd.DataFrame(A), width='stretch', hide_index=True)
            
            det = np.linalg.det(A)
            
            if abs(det) < 1e-10:
                st.error("❌ Matrix is singular (determinant = 0). Inverse does not exist.")
            else:
                A_inv = np.linalg.inv(A)
                st.write("**A⁻¹ (Inverse):**")
                st.dataframe(pd.DataFrame(A_inv), width='stretch', hide_index=True)
                
                # Verify
                I = np.dot(A, A_inv)
                st.write("**Verification: A × A⁻¹ = I (Identity):**")
                st.dataframe(pd.DataFrame(I), width='stretch', hide_index=True)
                
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
        st.write("### Definition")
        st.write("""
        The **determinant** is a scalar value computed from a square matrix that:
        - Measures the "scaling factor" of linear transformations
        - Determines if a matrix is invertible
        - Represents area (2D) or volume (3D) in geometric terms
        """)
        
        st.write("### 2×2 Determinant Formula")
        st.latex(r"\det(A) = a_{11}a_{22} - a_{12}a_{21}")
        st.write("**Example:**")
        st.latex(r"\det\begin{bmatrix} 1 & 3 \\ 2 & 4 \end{bmatrix} = 1 \times 4 - 3 \times 2 = -2")
        
        st.write("### 3×3 Determinant (Sarrus Rule)")
        st.latex(r"\det(A) = a_{11}(a_{22}a_{33} - a_{23}a_{32}) - a_{12}(a_{21}a_{33} - a_{23}a_{31}) + a_{13}(a_{21}a_{32} - a_{22}a_{31})")
        
        st.write("### Geometric Interpretation")
        st.write("""
        - **2D:** |det(A)| = area of parallelogram formed by column vectors
        - **3D:** |det(A)| = volume of parallelepiped formed by column vectors
        - **Sign:** Negative = orientation flip, Positive = same orientation
        """)
        
        st.write("### Matrix Inverse")
        st.latex(r"A^{-1} = \frac{1}{\det(A)} \text{adj}(A)")
        st.write("**Critical:** Inverse exists **only if det(A) ≠ 0**")
        st.write("If det(A) = 0, matrix is **singular** (not invertible)")
        
        st.write("### Linear Systems")
        st.latex(r"Ax = b \Rightarrow x = A^{-1}b")
        st.write("**Cramer's Rule:** Unique solution exists if det(A) ≠ 0")
    
    def examples():
        st.write("### Real-World Examples")
        
        st.write("**Example 1: Area Calculation**")
        st.write("Parallelogram with sides (3, 1) and (1, 3):")
        st.latex(r"A = \begin{bmatrix} 3 & 1 \\ 1 & 3 \end{bmatrix}")
        st.latex(r"\det(A) = 3 \times 3 - 1 \times 1 = 8")
        st.write("→ Area of parallelogram = **8 square units**")
        
        st.write("**Example 2: System of Equations**")
        st.write("Solve: 2x + y = 5, x + 3y = 7")
        st.latex(r"A = \begin{bmatrix} 2 & 1 \\ 1 & 3 \end{bmatrix}, \quad \det(A) = 5")
        st.write("Since det(A) = 5 ≠ 0, system has unique solution")
        st.latex(r"x = \frac{\det\begin{bmatrix} 5 & 1 \\ 7 & 3 \end{bmatrix}}{5} = \frac{8}{5}")
        
        st.write("**Example 3: Transformation Scaling**")
        st.write("Matrix A scales area by factor of |det(A)|")
        st.write("If det(A) = 4, transformation doubles area (2×2 = 4)")
        
        st.write("**Example 4: Singular Matrix**")
        st.write("Matrix with dependent rows/columns:")
        st.latex(r"A = \begin{bmatrix} 2 & 4 \\ 1 & 2 \end{bmatrix}, \quad \det(A) = 0")
        st.write("→ No inverse, system may have no solution or infinite solutions")
    
    def ml_usage():
        st.write("### In AI/ML")
        
        st.write("**1. Feature Independence & Multicollinearity**")
        st.write("""
        - **det = 0** → Features are linearly dependent
        - **Problem:** Redundant features, unstable models
        - **Solution:** Remove dependent features or use regularization
        - **Detection:** Check if det(XᵀX) ≈ 0 in normal equations
        """)
        st.code("""
# Check feature independence
cov_matrix = np.cov(X.T)
if abs(np.linalg.det(cov_matrix)) < 1e-10:
    print("Warning: Linearly dependent features!")
        """)
        
        st.write("**2. Change of Variables (Jacobian)**")
        st.write("""
        - **Jacobian determinant** for coordinate transformations
        - Used in: Normalizing flows, GANs, variational inference
        - Preserves probability mass during transformations
        """)
        st.latex(r"p_y(y) = p_x(x) \left|\det\left(\frac{\partial y}{\partial x}\right)\right|")
        
        st.write("**3. Optimization & Convexity**")
        st.write("""
        - **Hessian determinant** (2nd derivatives)
        - **det(H) > 0** → Local minimum/maximum
        - **det(H) = 0** → Saddle point
        - Used in: Newton's method, convexity checking
        """)
        st.code("""
# Check if critical point is minimum
H = compute_hessian(loss_function, x)
if det(H) > 0 and H[0,0] > 0:
    print("Local minimum found!")
        """)
        
        st.write("**4. Linear Systems & Normal Equations**")
        st.write("""
        - Solving **Ax = b** in least squares
        - **Normal equations:** (XᵀX)β = Xᵀy
        - **Unique solution** if det(XᵀX) ≠ 0
        - Used in: Linear regression, ridge regression
        """)
        st.latex(r"\beta = (X^T X)^{-1} X^T y")
        st.write("**Note:** If det(XᵀX) = 0, use regularization (ridge)")
        
        st.write("**5. Matrix Decompositions**")
        st.write("""
        - **LU decomposition:** det(A) = det(L) × det(U)
        - **QR decomposition:** Used in least squares
        - **Cholesky:** For positive definite matrices
        """)
    
    def summary():
        st.write("### Quick Summary")
        
        summary_data = {
            "Property": ["Definition", "2×2 Formula", "Geometric", "Invertibility", "Systems"],
            "Description": [
                "Scaling factor scalar",
                "ad - bc",
                "Area/Volume",
                "det ≠ 0 → invertible",
                "det ≠ 0 → unique solution"
            ],
            "ML Use": [
                "Feature independence",
                "Calculations",
                "Transformations",
                "Matrix operations",
                "Linear regression"
            ]
        }
        
        summary_df = pd.DataFrame(summary_data)
        st.dataframe(summary_df, width='stretch', hide_index=True)
        
        st.write("### Determinant Rules")
        st.write("""
        - **det(AB) = det(A) × det(B)**
        - **det(Aᵀ) = det(A)**
        - **det(kA) = kⁿ det(A)** (n = matrix size)
        - **det(A⁻¹) = 1/det(A)**
        - **det(I) = 1** (identity matrix)
        """)
        
        st.write("### When det = 0?")
        st.write("""
        - Rows/columns are linearly dependent
        - Matrix is singular (no inverse)
        - System Ax = b has no unique solution
        - Transformation collapses space (area/volume = 0)
        """)
        
        st.write("### Next Steps")
        st.write("""
        1. Master 2×2 and 3×3 calculations
        2. Understand geometric meaning
        3. Learn matrix inverse computation
        4. Move to **Eigenvalues & Eigenvectors** next!
        """)
    
    render_theory_panel({
        "Definition": definition,
        "Examples": examples,
        "ML Usage": ml_usage,
        "Summary": summary
    })

