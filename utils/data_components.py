"""
Reusable data input and display components.
Similar to form/input components in React.
"""

import streamlit as st
import pandas as pd
from utils.data_utils import generate_random_data


def render_data_input(
    input_types=["Generate Random", "Upload CSV", "Enter Manually"],
    data_size_range=(5, 20, 10),
    allow_categorical=False
):
    """
    Reusable data input component with multiple input methods.
    Similar to: <DataInput types={["random", "csv", "manual"]} />
    
    Args:
        input_types: List of input method names to show
        data_size_range: (min, max, default) for slider
        allow_categorical: If True, support categorical data
    
    Returns:
        data: List of values
        data_type: "Numeric" or "Categorical" (if allow_categorical=True)
    """
    input_method = st.radio("Choose data input method:", input_types, horizontal=True)
    
    data = []
    data_type = "Numeric"
    
    if input_method == "Generate Random":
        if allow_categorical:
            st.write("**Choose data type:**")
            data_type = st.radio("Numeric or Categorical?", ["Numeric", "Categorical"], horizontal=True)
        
        data_size = st.slider(
            "Select dataset size:",
            min_value=data_size_range[0],
            max_value=data_size_range[1],
            value=data_size_range[2]
        )
        
        if st.button("Generate Random Data"):
            if allow_categorical and data_type == "Categorical":
                import random
                categories = ["Red", "Blue", "Green", "Yellow", "Purple"]
                data = [random.choice(categories) for _ in range(data_size)]
            else:
                data = generate_random_data(data_size)
                if input_method == "Generate Random" and "median" in str(st.session_state.get("page_type", "")):
                    data = sorted(data)
            
            st.session_state['data'] = data
            if allow_categorical:
                st.session_state['data_type'] = data_type
        
        data = st.session_state.get('data', [])
        if allow_categorical:
            data_type = st.session_state.get('data_type', 'Numeric')
    
    elif input_method == "Upload CSV":
        uploaded_file = st.file_uploader("Upload CSV file (single column):", type=['csv'])
        if uploaded_file is not None:
            try:
                df = pd.read_csv(uploaded_file)
                data = df.iloc[:, 0].tolist()
                st.session_state['data'] = data
                
                # Auto-detect data type
                if allow_categorical:
                    try:
                        [float(x) for x in data]
                        data_type = "Numeric"
                    except:
                        data_type = "Categorical"
                    st.session_state['data_type'] = data_type
                
                st.success("✅ Data loaded successfully!")
            except Exception as e:
                st.error(f"❌ Error reading file: {e}")
        
        data = st.session_state.get('data', [])
        if allow_categorical:
            data_type = st.session_state.get('data_type', 'Numeric')
    
    else:  # Enter Manually
        st.write("**Enter values separated by commas:**")
        if allow_categorical:
            st.write("For numeric: `10, 20, 30`")
            st.write("For categorical: `Red, Blue, Green`")
        
        manual_input = st.text_input(
            "Example: 10, 20, 30, 40, 50",
            placeholder="Enter your data here..."
        )
        
        if manual_input:
            try:
                raw_data = [x.strip() for x in manual_input.split(',')]
                
                # Try to convert to numeric
                try:
                    data = [float(x) for x in raw_data]
                    data_type = "Numeric"
                except:
                    if allow_categorical:
                        data = raw_data
                        data_type = "Categorical"
                    else:
                        st.error("❌ Invalid input. Please enter numbers.")
                        data = []
                
                if data:
                    st.session_state['data'] = data
                    if allow_categorical:
                        st.session_state['data_type'] = data_type
                        st.success(f"✅ Loaded {len(data)} values ({data_type})")
                    else:
                        st.success(f"✅ Loaded {len(data)} values")
            except ValueError:
                st.error("❌ Invalid input.")
                data = []
        else:
            data = st.session_state.get('data', [])
            if allow_categorical:
                data_type = st.session_state.get('data_type', 'Numeric')
    
    if allow_categorical:
        return data, data_type
    return data


def display_dataset(data, max_inline=15, data_type="Numeric"):
    """
    Display dataset in code block format.
    Similar to: <DataDisplay data={data} />
    
    Args:
        data: List of values to display
        max_inline: Max items to show inline (else use expander)
        data_type: "Numeric" or "Categorical"
    """
    if not data:
        return
    
    # Format data string
    if data_type == "Numeric":
        data_str = "[" + ", ".join(
            str(int(v) if v == int(v) else round(v, 2)) for v in data
        ) + "]"
    else:
        data_str = "[" + ", ".join(f'"{v}"' for v in data) + "]"
    
    # Show inline or in expander
    if len(data) > max_inline:
        with st.expander(f"📋 Dataset ({len(data)} values)", expanded=False):
            st.code(data_str, language="python")
    else:
        st.code(data_str, language="python")


def display_data_info(data, data_type="Numeric"):
    """Display data point count and type"""
    if data_type == "Numeric":
        st.write(f"**Data points:** {len(data)}")
    else:
        st.write(f"**Data points:** {len(data)} | **Type:** {data_type}")


def display_basic_stats(data):
    """
    Display min, max, range metrics for numeric data.
    Similar to: <BasicStats data={data} />
    """
    if not data:
        return
    
    col_min, col_max, col_range = st.columns(3)
    col_min.metric("Min Value", f"{min(data):.2f}")
    col_max.metric("Max Value", f"{max(data):.2f}")
    col_range.metric("Range", f"{max(data) - min(data):.2f}")


def display_step_by_step_calculation(steps):
    """
    Render step-by-step calculation in expander.
    Similar to: <CalculationSteps steps={steps} />
    
    Args:
        steps: List of dicts with {title, content, code?, latex?}
    """
    with st.expander("📐 Step-by-Step Calculation", expanded=True):
        for i, step in enumerate(steps, 1):
            st.write(f"**Step {i}: {step['title']}**")
            
            if 'content' in step:
                st.write(step['content'])
            
            if 'code' in step:
                st.code(step['code'], language="python")
            
            if 'latex' in step:
                st.latex(step['latex'])

