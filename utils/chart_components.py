"""
Reusable chart/visualization components.
Wrappers around Plotly for consistent chart styling.
"""

import plotly.express as px
import plotly.graph_objects as go
import pandas as pd
import numpy as np
import streamlit as st


def render_histogram_with_line(data, line_value, line_label, title="Histogram of Values", color="red"):
    """
    Render histogram with vertical line marker.
    Similar to: <Histogram data={data} marker={mean} />
    
    Args:
        data: List of values
        line_value: X position for vertical line
        line_label: Label for the line
        title: Chart title
        color: Line color
    """
    df = pd.DataFrame({"Values": data})
    fig = px.histogram(df, x="Values", nbins=max(len(data)//2, 5), title=title)
    fig.add_vline(
        x=line_value,
        line_dash="dash",
        line_color=color,
        annotation_text=line_label,
        annotation_position="top right"
    )
    st.plotly_chart(fig, use_container_width=True)


def render_frequency_bar_chart(freq_data, highlight_values=None, title="Frequency Distribution"):
    """
    Render frequency bar chart with optional highlighting.
    Similar to: <FrequencyChart data={freqData} highlight={modes} />
    
    Args:
        freq_data: DataFrame with "Value" and "Frequency" columns
        highlight_values: List of values to highlight
        title: Chart title
    """
    fig = px.bar(
        freq_data,
        x="Value",
        y="Frequency",
        title=title,
        labels={"Value": "Values", "Frequency": "Occurrence Count"}
    )
    
    # Highlight specific bars
    if highlight_values:
        highlight_values_str = [str(v) for v in highlight_values]
        colors = ["#1f77b4" if str(val) in highlight_values_str else "#d3d3d3"
                 for val in freq_data["Value"]]
        fig.update_traces(marker_color=colors)
    
    st.plotly_chart(fig, use_container_width=True)


def render_scatter_with_regression(data, x_col="X", y_col="Y", correlation=None):
    """
    Render scatter plot with regression line.
    Similar to: <ScatterPlot data={data} showRegression={true} />
    
    Args:
        data: DataFrame with X and Y columns
        x_col: Name of X column
        y_col: Name of Y column
        correlation: Correlation coefficient to show in title
    """
    title = "Scatter Plot: Relationship between Variables"
    if correlation is not None:
        title = f"Scatter Plot (r = {correlation:.3f})"
    
    fig = px.scatter(
        data,
        x=x_col,
        y=y_col,
        title=title,
        labels={x_col: f"Variable {x_col}", y_col: f"Variable {y_col}"}
    )
    
    # Add regression line
    z = np.polyfit(data[x_col], data[y_col], 1)
    p = np.poly1d(z)
    x_line = np.linspace(data[x_col].min(), data[x_col].max(), 100)
    y_line = p(x_line)
    
    fig.add_trace(go.Scatter(
        x=x_line,
        y=y_line,
        mode='lines',
        name='Trend Line',
        line=dict(color='red', dash='dash')
    ))
    
    fig.update_layout(hovermode='closest')
    st.plotly_chart(fig, use_container_width=True)


def render_distribution_chart(data, dist_type="histogram", title="Distribution"):
    """
    Render distribution visualization.
    Similar to: <DistributionChart data={data} type="histogram" />
    
    Args:
        data: List or DataFrame
        dist_type: "histogram", "density", or "box"
        title: Chart title
    """
    if isinstance(data, list):
        df = pd.DataFrame({"Values": data})
    else:
        df = data
    
    if dist_type == "histogram":
        fig = px.histogram(df, x="Values", title=title)
    elif dist_type == "density":
        fig = px.density_contour(df, x="Values", title=title)
    elif dist_type == "box":
        fig = px.box(df, y="Values", title=title)
    else:
        fig = px.histogram(df, x="Values", title=title)
    
    st.plotly_chart(fig, use_container_width=True)


def render_comparison_chart(values_dict, chart_type="bar", title="Comparison"):
    """
    Render comparison chart for multiple values.
    Similar to: <ComparisonChart data={comparison} />
    
    Args:
        values_dict: Dict of {label: value}
        chart_type: "bar" or "metric"
        title: Chart title
    """
    if chart_type == "metric":
        cols = st.columns(len(values_dict))
        for col, (label, value) in zip(cols, values_dict.items()):
            col.metric(label, f"{value:.2f}")
    else:
        df = pd.DataFrame([
            {"Metric": k, "Value": v}
            for k, v in values_dict.items()
        ])
        fig = px.bar(df, x="Metric", y="Value", title=title)
        st.plotly_chart(fig, use_container_width=True)

