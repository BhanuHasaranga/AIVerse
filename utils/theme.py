"""
Centralized theme configuration for the AI/ML Learning Hub.
Similar to styled-components or CSS-in-JS in React.
"""

class Theme:
    """Theme configuration with color palette and styling constants"""
    
    # Color Palette
    PRIMARY_COLOR = "#667eea"
    SECONDARY_COLOR = "#764ba2"
    BACKGROUND_LIGHT = "#f0f2f6"
    METRIC_BG = "rgba(28, 131, 225, 0.1)"
    TEXT_DARK = "#555"
    BORDER_COLOR = "#667eea"
    
    # Gradient
    GRADIENT = "linear-gradient(135deg, #667eea 0%, #764ba2 100%)"
    
    # Spacing
    PADDING_CARD = "1.5rem"
    PADDING_PAGE = "0.5rem"
    BORDER_RADIUS = "8px"
    BORDER_RADIUS_LARGE = "10px"
    
    @staticmethod
    def get_page_css():
        """CSS for individual explorer pages"""
        return f"""
        <style>
        /* Show Streamlit default navigation */
        [data-testid="stSidebarNav"] {{
            display: block;
        }}
        
        .main {{
            max-width: 100%;
            padding-top: 0.5rem;
            padding-left: 0.5rem;
            padding-right: 0.5rem;
        }}
        .block-container {{
            padding-top: 1.5rem !important;
            padding-left: 1rem !important;
            padding-right: 1rem !important;
            max-width: 100% !important;
        }}
        h1 {{
            margin-top: 0 !important;
            padding-top: 0 !important;
        }}
        [data-testid="stMetricContainer"] {{
            background-color: {Theme.METRIC_BG};
            padding: 10px;
            border-radius: 5px;
        }}
        </style>
        """
    
    @staticmethod
    def get_home_css():
        """CSS for home/landing page"""
        return f"""
        <style>
        /* Show Streamlit default navigation */
        [data-testid="stSidebarNav"] {{
            display: block;
        }}
        
        .main {{
            padding-top: 1rem;
            padding-left: 0.5rem;
            padding-right: 0.5rem;
        }}
        .block-container {{
            padding-left: 1rem !important;
            padding-right: 1rem !important;
            max-width: 100% !important;
        }}
        .hero-section {{
            background: {Theme.GRADIENT};
            padding: 3rem;
            border-radius: {Theme.BORDER_RADIUS_LARGE};
            color: white;
            margin-bottom: 2rem;
        }}
        .hero-section h1 {{
            font-size: 3rem;
            margin: 0;
        }}
        .hero-section p {{
            font-size: 1.2rem;
            margin: 0.5rem 0 0 0;
        }}
        .module-card {{
            background-color: {Theme.BACKGROUND_LIGHT};
            padding: {Theme.PADDING_CARD};
            border-radius: {Theme.BORDER_RADIUS};
            margin-bottom: 1rem;
            border-left: 4px solid {Theme.BORDER_COLOR};
        }}
        .module-card h3 {{
            margin-top: 0;
            color: {Theme.PRIMARY_COLOR};
        }}
        .module-card p {{
            margin: 0.5rem 0;
            color: {Theme.TEXT_DARK};
        }}
        </style>
        """

