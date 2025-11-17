# UI Components Package

Modular, reusable UI components for AIVerse platform.

## Structure

```
utils/ui/
├── __init__.py          # Package exports
├── page_config.py       # Page setup & theme application
├── layouts.py           # Page layout components
├── cards.py             # Card & hero components
└── theory_panel.py      # Learning guide components
```

## Components

### Page Configuration (`page_config.py`)
- `apply_page_config()` - Set page title, icon, layout
- `apply_theme()` - Apply CSS theme (home/page)

### Layouts (`layouts.py`)
- `create_two_column_layout()` - Standard 2.5:1 layout with completion tracking

### Cards (`cards.py`)
- `render_module_card()` - Interactive module cards for home page
- `render_hero_section()` - Gradient hero banner

### Theory Panel (`theory_panel.py`)
- `render_theory_panel()` - Tabbed learning guide
- `render_section_header()` - Consistent section headers

## Usage

### In main.py
```python
from utils.ui import (
    apply_page_config, 
    apply_theme,
    render_hero_section,
    render_module_card
)
```

### In explorer pages
```python
from utils.ui import (
    apply_page_config,
    apply_theme,
    create_two_column_layout,
    render_theory_panel
)
```

## Benefits

✅ **Separation of Concerns** - Each component has its own file
✅ **Easy to Maintain** - Find and update specific components quickly
✅ **Reusable** - Import only what you need
✅ **Scalable** - Add new components without cluttering
✅ **Clean Imports** - Single `from utils.ui import ...`

