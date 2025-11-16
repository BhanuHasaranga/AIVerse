"""
Statistics Explorer Module
Interactive tool for exploring statistical concepts with step-by-step calculations
"""
import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats

def main():
    st.title("📊 Statistics Explorer")
    st.write("Learn statistics through interactive visualizations and step-by-step calculations")
    
    # Tabs for different statistical concepts
    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "Central Tendency", 
        "Dispersion", 
        "Correlation", 
        "Distributions",
        "Custom Dataset"
    ])
    
    with tab1:
        st.header("Central Tendency Measures")
        st.write("Explore mean, median, and mode")
        
        col1, col2 = st.columns([1, 2])
        
        with col1:
            st.subheader("Input Data")
            data_input = st.text_area(
                "Enter numbers (comma-separated)",
                value="12, 15, 18, 20, 22, 25, 28, 30, 32, 35",
                height=100
            )
            
            try:
                data = np.array([float(x.strip()) for x in data_input.split(',')])
                
                # Calculate statistics
                mean_val = np.mean(data)
                median_val = np.median(data)
                mode_result = stats.mode(data, keepdims=True)
                mode_val = mode_result.mode[0] if len(mode_result.mode) > 0 else "No mode"
                
                st.metric("Count", len(data))
                st.metric("Mean", f"{mean_val:.2f}")
                st.metric("Median", f"{median_val:.2f}")
                st.metric("Mode", f"{mode_val}")
                
            except Exception as e:
                st.error(f"Error parsing data: {e}")
                return
        
        with col2:
            st.subheader("Visualization")
            
            # Create visualization
            fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8))
            
            # Histogram with measures
            ax1.hist(data, bins=15, alpha=0.7, color='skyblue', edgecolor='black')
            ax1.axvline(mean_val, color='red', linestyle='--', linewidth=2, label=f'Mean: {mean_val:.2f}')
            ax1.axvline(median_val, color='green', linestyle='--', linewidth=2, label=f'Median: {median_val:.2f}')
            ax1.set_xlabel('Value')
            ax1.set_ylabel('Frequency')
            ax1.set_title('Distribution with Central Tendency Measures')
            ax1.legend()
            ax1.grid(True, alpha=0.3)
            
            # Box plot
            ax2.boxplot(data, vert=False, widths=0.5)
            ax2.axvline(mean_val, color='red', linestyle='--', linewidth=2, label='Mean')
            ax2.axvline(median_val, color='green', linestyle='--', linewidth=2, label='Median')
            ax2.set_xlabel('Value')
            ax2.set_title('Box Plot')
            ax2.legend()
            ax2.grid(True, alpha=0.3)
            
            plt.tight_layout()
            st.pyplot(fig)
            
            # Step-by-step calculation
            with st.expander("📝 Step-by-Step Calculations"):
                st.write("**Mean Calculation:**")
                st.latex(r"\text{Mean} = \frac{\sum x_i}{n}")
                st.write(f"Mean = ({' + '.join(map(str, data))}) / {len(data)}")
                st.write(f"Mean = {sum(data)} / {len(data)} = {mean_val:.2f}")
                
                st.write("\n**Median Calculation:**")
                sorted_data = np.sort(data)
                st.write(f"Sorted data: {sorted_data}")
                if len(data) % 2 == 0:
                    st.write(f"Median = ({sorted_data[len(data)//2-1]} + {sorted_data[len(data)//2]}) / 2 = {median_val:.2f}")
                else:
                    st.write(f"Median = {sorted_data[len(data)//2]} = {median_val:.2f}")
    
    with tab2:
        st.header("Measures of Dispersion")
        st.write("Understand variance, standard deviation, and range")
        
        col1, col2 = st.columns([1, 2])
        
        with col1:
            st.subheader("Input Data")
            data_input2 = st.text_area(
                "Enter numbers",
                value="10, 20, 30, 40, 50",
                height=100,
                key="dispersion_data"
            )
            
            try:
                data2 = np.array([float(x.strip()) for x in data_input2.split(',')])
                
                # Calculate dispersion measures
                variance = np.var(data2, ddof=1)
                std_dev = np.std(data2, ddof=1)
                range_val = np.max(data2) - np.min(data2)
                
                st.metric("Variance", f"{variance:.2f}")
                st.metric("Std Deviation", f"{std_dev:.2f}")
                st.metric("Range", f"{range_val:.2f}")
                st.metric("Min", f"{np.min(data2):.2f}")
                st.metric("Max", f"{np.max(data2):.2f}")
                
            except Exception as e:
                st.error(f"Error parsing data: {e}")
                return
        
        with col2:
            st.subheader("Visualization")
            
            fig, ax = plt.subplots(figsize=(10, 6))
            
            mean_val2 = np.mean(data2)
            x_range = np.linspace(np.min(data2) - 10, np.max(data2) + 10, 100)
            
            # Plot data points
            ax.scatter(data2, [0] * len(data2), s=100, c='blue', alpha=0.6, label='Data Points')
            
            # Show mean and std dev
            ax.axvline(mean_val2, color='red', linestyle='--', linewidth=2, label=f'Mean: {mean_val2:.2f}')
            ax.axvspan(mean_val2 - std_dev, mean_val2 + std_dev, alpha=0.2, color='green', 
                      label=f'±1 Std Dev: {std_dev:.2f}')
            
            ax.set_xlabel('Value')
            ax.set_title('Data Distribution with Standard Deviation')
            ax.legend()
            ax.grid(True, alpha=0.3)
            
            st.pyplot(fig)
            
            # Step-by-step calculation
            with st.expander("📝 Step-by-Step Calculations"):
                st.write("**Variance Calculation:**")
                st.latex(r"\text{Variance} = \frac{\sum (x_i - \bar{x})^2}{n-1}")
                st.write(f"Mean (x̄) = {mean_val2:.2f}")
                st.write("\nDeviations from mean:")
                for i, val in enumerate(data2):
                    st.write(f"({val} - {mean_val2:.2f})² = {(val - mean_val2)**2:.2f}")
                st.write(f"\nVariance = {variance:.2f}")
                st.write(f"Standard Deviation = √{variance:.2f} = {std_dev:.2f}")
    
    with tab3:
        st.header("Correlation Analysis")
        st.write("Explore relationships between variables")
        
        col1, col2 = st.columns([1, 2])
        
        with col1:
            st.subheader("Generate Data")
            n_points = st.slider("Number of points", 10, 100, 50)
            correlation = st.slider("Correlation strength", -1.0, 1.0, 0.7, 0.1)
            noise = st.slider("Noise level", 0.0, 2.0, 0.5, 0.1)
            
            # Generate correlated data
            np.random.seed(42)
            x = np.random.randn(n_points)
            y = correlation * x + noise * np.random.randn(n_points)
            
            # Calculate correlation
            corr_coef = np.corrcoef(x, y)[0, 1]
            
            st.metric("Correlation Coefficient", f"{corr_coef:.3f}")
            
            if abs(corr_coef) > 0.7:
                st.success("Strong correlation")
            elif abs(corr_coef) > 0.4:
                st.info("Moderate correlation")
            else:
                st.warning("Weak correlation")
        
        with col2:
            st.subheader("Scatter Plot")
            
            fig, ax = plt.subplots(figsize=(10, 6))
            
            # Scatter plot
            ax.scatter(x, y, alpha=0.6, s=50)
            
            # Add regression line
            z = np.polyfit(x, y, 1)
            p = np.poly1d(z)
            ax.plot(x, p(x), "r--", linewidth=2, label=f'Best fit line')
            
            ax.set_xlabel('X Variable')
            ax.set_ylabel('Y Variable')
            ax.set_title(f'Correlation: {corr_coef:.3f}')
            ax.legend()
            ax.grid(True, alpha=0.3)
            
            st.pyplot(fig)
            
            with st.expander("📝 Understanding Correlation"):
                st.write("""
                **Correlation Coefficient (r):**
                - r = +1: Perfect positive correlation
                - r = 0: No correlation
                - r = -1: Perfect negative correlation
                
                **Formula:**
                """)
                st.latex(r"r = \frac{\sum (x_i - \bar{x})(y_i - \bar{y})}{\sqrt{\sum (x_i - \bar{x})^2 \sum (y_i - \bar{y})^2}}")
    
    with tab4:
        st.header("Probability Distributions")
        st.write("Explore common statistical distributions")
        
        dist_type = st.selectbox(
            "Select Distribution",
            ["Normal (Gaussian)", "Uniform", "Exponential", "Binomial"]
        )
        
        if dist_type == "Normal (Gaussian)":
            col1, col2 = st.columns(2)
            with col1:
                mean = st.slider("Mean (μ)", -10.0, 10.0, 0.0, 0.5)
                std = st.slider("Standard Deviation (σ)", 0.1, 5.0, 1.0, 0.1)
            
            x = np.linspace(mean - 4*std, mean + 4*std, 1000)
            y = stats.norm.pdf(x, mean, std)
            
            fig, ax = plt.subplots(figsize=(10, 6))
            ax.plot(x, y, linewidth=2)
            ax.fill_between(x, y, alpha=0.3)
            ax.axvline(mean, color='red', linestyle='--', label=f'Mean: {mean}')
            ax.set_xlabel('Value')
            ax.set_ylabel('Probability Density')
            ax.set_title(f'Normal Distribution (μ={mean}, σ={std})')
            ax.legend()
            ax.grid(True, alpha=0.3)
            st.pyplot(fig)
            
        elif dist_type == "Uniform":
            col1, col2 = st.columns(2)
            with col1:
                a = st.slider("Minimum (a)", -10.0, 0.0, -5.0, 0.5)
                b = st.slider("Maximum (b)", 0.0, 10.0, 5.0, 0.5)
            
            x = np.linspace(a - 2, b + 2, 1000)
            y = stats.uniform.pdf(x, a, b - a)
            
            fig, ax = plt.subplots(figsize=(10, 6))
            ax.plot(x, y, linewidth=2)
            ax.fill_between(x, y, alpha=0.3)
            ax.set_xlabel('Value')
            ax.set_ylabel('Probability Density')
            ax.set_title(f'Uniform Distribution (a={a}, b={b})')
            ax.grid(True, alpha=0.3)
            st.pyplot(fig)
    
    with tab5:
        st.header("Upload Your Own Dataset")
        st.write("Analyze your own data")
        
        uploaded_file = st.file_uploader("Upload CSV file", type=['csv'])
        
        if uploaded_file is not None:
            try:
                df = pd.read_csv(uploaded_file)
                st.write("**Data Preview:**")
                st.dataframe(df.head())
                
                # Select numeric columns
                numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
                
                if numeric_cols:
                    selected_col = st.selectbox("Select column to analyze", numeric_cols)
                    
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        st.write("**Statistics:**")
                        st.write(df[selected_col].describe())
                    
                    with col2:
                        fig, ax = plt.subplots(figsize=(8, 6))
                        ax.hist(df[selected_col].dropna(), bins=20, alpha=0.7, edgecolor='black')
                        ax.set_xlabel(selected_col)
                        ax.set_ylabel('Frequency')
                        ax.set_title(f'Distribution of {selected_col}')
                        ax.grid(True, alpha=0.3)
                        st.pyplot(fig)
                else:
                    st.warning("No numeric columns found in the dataset")
                    
            except Exception as e:
                st.error(f"Error loading file: {e}")
        else:
            st.info("Upload a CSV file to analyze your own data")

if __name__ == "__main__":
    main()
