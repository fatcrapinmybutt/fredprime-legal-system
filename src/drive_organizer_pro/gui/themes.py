"""
MBP LLC themed styles for DriveOrganizerPro.

© 2026 MBP LLC. All rights reserved.
"""

# MBP Dark Theme Colors
MBP_DARK_THEME = {
    "bg_primary": "#1e1e1e",
    "bg_secondary": "#2d2d2d",
    "bg_tertiary": "#3a3a3a",
    "accent_green": "#00ff00",
    "accent_gold": "#ffd700",
    "text_primary": "#ffffff",
    "text_secondary": "#888888",
    "error_red": "#ff0000",
    "warning_orange": "#ffaa00",
    "success_green": "#00cc00",
    "button_hover": "#505050",
    "border": "#555555"
}

# Tkinter style configuration
def get_tk_styles():
    """Get tkinter style configuration dictionary."""
    return {
        'window': {
            'bg': MBP_DARK_THEME['bg_primary']
        },
        'frame': {
            'bg': MBP_DARK_THEME['bg_primary']
        },
        'label': {
            'bg': MBP_DARK_THEME['bg_primary'],
            'fg': MBP_DARK_THEME['text_primary'],
            'font': ('Segoe UI', 10)
        },
        'label_secondary': {
            'bg': MBP_DARK_THEME['bg_primary'],
            'fg': MBP_DARK_THEME['text_secondary'],
            'font': ('Segoe UI', 9)
        },
        'label_header': {
            'bg': MBP_DARK_THEME['bg_primary'],
            'fg': MBP_DARK_THEME['accent_gold'],
            'font': ('Segoe UI', 14, 'bold')
        },
        'button': {
            'bg': MBP_DARK_THEME['bg_tertiary'],
            'fg': MBP_DARK_THEME['text_primary'],
            'activebackground': MBP_DARK_THEME['button_hover'],
            'activeforeground': MBP_DARK_THEME['text_primary'],
            'relief': 'flat',
            'padx': 20,
            'pady': 10,
            'font': ('Segoe UI', 10, 'bold'),
            'cursor': 'hand2'
        },
        'button_primary': {
            'bg': MBP_DARK_THEME['accent_green'],
            'fg': MBP_DARK_THEME['bg_primary'],
            'activebackground': MBP_DARK_THEME['success_green'],
            'activeforeground': MBP_DARK_THEME['bg_primary'],
            'relief': 'flat',
            'padx': 20,
            'pady': 10,
            'font': ('Segoe UI', 11, 'bold'),
            'cursor': 'hand2'
        },
        'checkbutton': {
            'bg': MBP_DARK_THEME['bg_primary'],
            'fg': MBP_DARK_THEME['text_primary'],
            'selectcolor': MBP_DARK_THEME['bg_secondary'],
            'activebackground': MBP_DARK_THEME['bg_primary'],
            'activeforeground': MBP_DARK_THEME['accent_green'],
            'font': ('Segoe UI', 10)
        },
        'entry': {
            'bg': MBP_DARK_THEME['bg_secondary'],
            'fg': MBP_DARK_THEME['text_primary'],
            'insertbackground': MBP_DARK_THEME['text_primary'],
            'relief': 'flat',
            'font': ('Segoe UI', 10)
        },
        'text': {
            'bg': MBP_DARK_THEME['bg_secondary'],
            'fg': MBP_DARK_THEME['text_primary'],
            'insertbackground': MBP_DARK_THEME['text_primary'],
            'relief': 'flat',
            'font': ('Consolas', 9)
        },
        'progressbar': {
            'background': MBP_DARK_THEME['accent_green'],
            'troughcolor': MBP_DARK_THEME['bg_secondary'],
            'borderwidth': 0,
            'relief': 'flat'
        }
    }

def apply_theme_to_widget(widget, style_name):
    """
    Apply theme styles to a widget.

    Args:
        widget: Tkinter widget
        style_name: Style name from get_tk_styles()
    """
    styles = get_tk_styles()
    if style_name in styles:
        widget.config(**styles[style_name])
