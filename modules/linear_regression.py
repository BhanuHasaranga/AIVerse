"""
Linear Regression Module
Interactive visualization of linear regression with step-by-step explanations
"""
import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import train_test_split

def main():
    st.title("📈 Linear Regression Explorer")
    st.write("Learn linear regression through interactive visualizations")
    
    tab1, tab2, tab3 = st.tabs(["Interactive Demo", "Real Dataset", "Theory"])
    
    with tab1:
        st.header("Interactive Linear Regression")
        
        col1, col2 = st.columns([1, 2])
        
        with col1:
            st.subheader("Data Generation")
            
            n_points = st.slider("Number of points", 10, 200, 50)
            slope = st.slider("True slope", -5.0, 5.0, 2.0, 0.1)
            intercept = st.slider("True intercept", -10.0, 10.0, 1.0, 0.5)
            noise = st.slider("Noise level", 0.0, 5.0, 1.0, 0.1)
            
            # Generate data
            np.random.seed(42)
            X = np.random.uniform(-5, 5, n_points)
            y_true = slope * X + intercept
            y = y_true + noise * np.random.randn(n_points)
            
            # Fit model
            X_reshaped = X.reshape(-1, 1)
            model = LinearRegression()
            model.fit(X_reshaped, y)
            
            # Predictions
            y_pred = model.predict(X_reshaped)
            
            # Metrics
            mse = mean_squared_error(y, y_pred)
            rmse = np.sqrt(mse)
            r2 = r2_score(y, y_pred)
            
            st.write("**True Model:**")
            st.write(f"y = {slope}x + {intercept}")
            
            st.write("\n**Fitted Model:**")
            st.write(f"y = {model.coef_[0]:.3f}x + {model.intercept_:.3f}")
            
            st.metric("R² Score", f"{r2:.4f}")
            st.metric("RMSE", f"{rmse:.4f}")
            st.metric("MSE", f"{mse:.4f}")
        
        with col2:
            st.subheader("Visualization")
            
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
            
            # Scatter plot with regression line
            ax1.scatter(X, y, alpha=0.6, s=50, label='Data points')
            ax1.plot(X, y_true, 'g--', linewidth=2, label=f'True: y={slope}x+{intercept}')
            ax1.plot(X, y_pred, 'r-', linewidth=2, label=f'Fitted: y={model.coef_[0]:.2f}x+{model.intercept_:.2f}')
            ax1.set_xlabel('X')
            ax1.set_ylabel('y')
            ax1.set_title('Linear Regression Fit')
            ax1.legend()
            ax1.grid(True, alpha=0.3)
            
            # Residuals plot
            residuals = y - y_pred
            ax2.scatter(y_pred, residuals, alpha=0.6, s=50)
            ax2.axhline(y=0, color='r', linestyle='--', linewidth=2)
            ax2.set_xlabel('Predicted Values')
            ax2.set_ylabel('Residuals')
            ax2.set_title('Residual Plot')
            ax2.grid(True, alpha=0.3)
            
            plt.tight_layout()
            st.pyplot(fig)
            
            with st.expander("📝 Understanding the Results"):
                st.write("""
                **R² Score (Coefficient of Determination):**
                - Measures how well the model fits the data
                - Range: 0 to 1 (higher is better)
                - R² = 1 means perfect fit
                - R² = 0 means model is no better than predicting the mean
                
                **RMSE (Root Mean Squared Error):**
                - Average prediction error in the same units as y
                - Lower values indicate better fit
                
                **Residuals:**
                - Difference between actual and predicted values
                - Should be randomly scattered around zero for a good model
                """)
                
                st.write("\n**Linear Regression Formula:**")
                st.latex(r"y = \beta_0 + \beta_1 x + \epsilon")
                st.write("Where:")
                st.write("- y: dependent variable")
                st.write("- x: independent variable")
                st.write("- β₀: intercept")
                st.write("- β₁: slope")
                st.write("- ε: error term")
    
    with tab2:
        st.header("Real Dataset Analysis")
        
        # Sample datasets
        dataset_choice = st.selectbox(
            "Choose a dataset",
            ["California Housing (sample)", "Boston Housing (synthetic)", "Custom Dataset"]
        )
        
        if dataset_choice == "California Housing (sample)":
            st.info("Using a synthetic California Housing dataset sample")
            
            # Create synthetic data similar to California Housing
            np.random.seed(42)
            n_samples = 200
            
            median_income = np.random.uniform(1, 15, n_samples)
            house_age = np.random.uniform(1, 50, n_samples)
            avg_rooms = np.random.uniform(3, 10, n_samples)
            
            # Price roughly correlates with income and rooms, inversely with age
            price = (median_income * 3 + avg_rooms * 1.5 - house_age * 0.1 + 
                    np.random.randn(n_samples) * 2)
            
            df = pd.DataFrame({
                'Median_Income': median_income,
                'House_Age': house_age,
                'Avg_Rooms': avg_rooms,
                'Price': price
            })
            
            st.write("**Dataset Preview:**")
            st.dataframe(df.head(10))
            
            # Feature selection
            feature = st.selectbox("Select feature (X)", 
                                  ['Median_Income', 'House_Age', 'Avg_Rooms'])
            target = 'Price'
            
            X = df[[feature]].values
            y = df[target].values
            
            # Split data
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=0.2, random_state=42
            )
            
            # Train model
            model = LinearRegression()
            model.fit(X_train, y_train)
            
            # Predictions
            y_train_pred = model.predict(X_train)
            y_test_pred = model.predict(X_test)
            
            # Metrics
            train_r2 = r2_score(y_train, y_train_pred)
            test_r2 = r2_score(y_test, y_test_pred)
            train_rmse = np.sqrt(mean_squared_error(y_train, y_train_pred))
            test_rmse = np.sqrt(mean_squared_error(y_test, y_test_pred))
            
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Training R²", f"{train_r2:.4f}")
                st.metric("Training RMSE", f"{train_rmse:.4f}")
            with col2:
                st.metric("Test R²", f"{test_r2:.4f}")
                st.metric("Test RMSE", f"{test_rmse:.4f}")
            with col3:
                st.metric("Slope", f"{model.coef_[0]:.4f}")
                st.metric("Intercept", f"{model.intercept_:.4f}")
            
            # Visualization
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
            
            # Training data
            ax1.scatter(X_train, y_train, alpha=0.6, s=50, label='Training data')
            ax1.scatter(X_test, y_test, alpha=0.6, s=50, color='red', label='Test data')
            
            # Regression line
            X_range = np.linspace(X.min(), X.max(), 100).reshape(-1, 1)
            y_range = model.predict(X_range)
            ax1.plot(X_range, y_range, 'g-', linewidth=2, label='Regression line')
            
            ax1.set_xlabel(feature)
            ax1.set_ylabel(target)
            ax1.set_title('Linear Regression on Real Data')
            ax1.legend()
            ax1.grid(True, alpha=0.3)
            
            # Actual vs Predicted
            ax2.scatter(y_test, y_test_pred, alpha=0.6, s=50)
            ax2.plot([y.min(), y.max()], [y.min(), y.max()], 'r--', linewidth=2, label='Perfect prediction')
            ax2.set_xlabel('Actual Price')
            ax2.set_ylabel('Predicted Price')
            ax2.set_title('Actual vs Predicted (Test Set)')
            ax2.legend()
            ax2.grid(True, alpha=0.3)
            
            plt.tight_layout()
            st.pyplot(fig)
        
        elif dataset_choice == "Custom Dataset":
            uploaded_file = st.file_uploader("Upload CSV file", type=['csv'])
            
            if uploaded_file is not None:
                try:
                    df = pd.read_csv(uploaded_file)
                    st.write("**Data Preview:**")
                    st.dataframe(df.head())
                    
                    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
                    
                    if len(numeric_cols) >= 2:
                        col1, col2 = st.columns(2)
                        with col1:
                            feature = st.selectbox("Select feature (X)", numeric_cols)
                        with col2:
                            target = st.selectbox("Select target (y)", 
                                                [col for col in numeric_cols if col != feature])
                        
                        # Train model and show results...
                        st.info("Model training on custom dataset - implement full analysis here")
                    else:
                        st.warning("Dataset needs at least 2 numeric columns")
                        
                except Exception as e:
                    st.error(f"Error loading file: {e}")
            else:
                st.info("Upload a CSV file to analyze your own data")
    
    with tab3:
        st.header("Linear Regression Theory")
        
        st.write("""
        ### What is Linear Regression?
        
        Linear regression is a statistical method for modeling the relationship between 
        a dependent variable (y) and one or more independent variables (X).
        
        ### Key Concepts
        """)
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.write("""
            **Simple Linear Regression:**
            - One independent variable
            - Formula: y = β₀ + β₁x + ε
            - Finds the best-fitting line
            
            **Multiple Linear Regression:**
            - Multiple independent variables
            - Formula: y = β₀ + β₁x₁ + β₂x₂ + ... + ε
            """)
        
        with col2:
            st.write("""
            **Model Training:**
            1. Calculate slope and intercept
            2. Minimize sum of squared errors
            3. Use ordinary least squares (OLS)
            
            **Assumptions:**
            - Linear relationship
            - Independence of errors
            - Homoscedasticity
            - Normal distribution of errors
            """)
        
        st.write("\n### Mathematical Formulation")
        
        st.write("**Ordinary Least Squares (OLS):**")
        st.latex(r"\beta_1 = \frac{\sum (x_i - \bar{x})(y_i - \bar{y})}{\sum (x_i - \bar{x})^2}")
        st.latex(r"\beta_0 = \bar{y} - \beta_1 \bar{x}")
        
        st.write("\n**Mean Squared Error (MSE):**")
        st.latex(r"MSE = \frac{1}{n} \sum_{i=1}^{n} (y_i - \hat{y}_i)^2")
        
        st.write("\n**R² Score:**")
        st.latex(r"R^2 = 1 - \frac{\sum (y_i - \hat{y}_i)^2}{\sum (y_i - \bar{y})^2}")

if __name__ == "__main__":
    main()
