"""
UI Components Package for AIVerse.
Organized, reusable UI components.
"""

from .page_config import apply_page_config, apply_theme
from .layouts import create_two_column_layout
from .cards import render_module_card, render_hero_section
from .theory_panel import render_theory_panel

__all__ = [
    'apply_page_config',
    'apply_theme',
    'create_two_column_layout',
    'render_module_card',
    'render_hero_section',
    'render_theory_panel',
]

