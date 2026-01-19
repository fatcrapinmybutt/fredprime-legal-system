"""
Splash screen with MBP LLC pig logo.

© 2026 MBP LLC. All rights reserved.
"""

import tkinter as tk
from pathlib import Path
from .themes import MBP_DARK_THEME


class SplashScreen:
    """Startup splash screen with pig logo."""

    def __init__(self, duration_ms: int = 3000):
        """
        Initialize splash screen.

        Args:
            duration_ms: Duration to show splash in milliseconds
        """
        self.duration_ms = duration_ms
        self.root = tk.Tk()
        self.root.title("DriveOrganizerPro")
        self.root.overrideredirect(True)  # Remove window decorations

        # Get screen dimensions
        screen_width = self.root.winfo_screenwidth()
        screen_height = self.root.winfo_screenheight()

        # Window size
        window_width = 800
        window_height = 600

        # Center window
        x = (screen_width - window_width) // 2
        y = (screen_height - window_height) // 2

        self.root.geometry(f"{window_width}x{window_height}+{x}+{y}")
        self.root.configure(bg=MBP_DARK_THEME['bg_primary'])

        # Load and display pig logo
        self._create_widgets()

    def _create_widgets(self):
        """Create splash screen widgets."""
        # Main frame
        frame = tk.Frame(self.root, bg=MBP_DARK_THEME['bg_primary'])
        frame.pack(fill=tk.BOTH, expand=True, padx=20, pady=20)

        # Try to load pig logo
        logo_path = Path(__file__).parent.parent.parent.parent / 'assets' / 'branding' / 'pig_logo_splash.txt'

        if logo_path.exists():
            try:
                with open(logo_path, 'r', encoding='utf-8') as f:
                    logo_text = f.read()

                logo_label = tk.Label(
                    frame,
                    text=logo_text,
                    bg=MBP_DARK_THEME['bg_primary'],
                    fg=MBP_DARK_THEME['accent_gold'],
                    font=('Courier New', 8),
                    justify=tk.LEFT
                )
                logo_label.pack(pady=20)

            except Exception:
                # Fallback to simple text
                self._create_fallback_logo(frame)
        else:
            self._create_fallback_logo(frame)

        # Version info
        version_label = tk.Label(
            frame,
            text="Version 1.0.0 - Level 9999 Edition",
            bg=MBP_DARK_THEME['bg_primary'],
            fg=MBP_DARK_THEME['text_secondary'],
            font=('Segoe UI', 10)
        )
        version_label.pack(pady=10)

        # Loading message
        loading_label = tk.Label(
            frame,
            text="Initializing Maximum Business Performance...",
            bg=MBP_DARK_THEME['bg_primary'],
            fg=MBP_DARK_THEME['accent_green'],
            font=('Segoe UI', 11, 'bold')
        )
        loading_label.pack(pady=20)

    def _create_fallback_logo(self, parent):
        """Create fallback logo if file not found."""
        fallback_text = """
    🐷 MBP LLC
    DriveOrganizerPro
    Level 9999 Edition

    Maximum Business Performance
    "Powered by Pork™"
        """

        logo_label = tk.Label(
            parent,
            text=fallback_text,
            bg=MBP_DARK_THEME['bg_primary'],
            fg=MBP_DARK_THEME['accent_gold'],
            font=('Segoe UI', 16, 'bold'),
            justify=tk.CENTER
        )
        logo_label.pack(pady=20)

    def show(self):
        """Show splash screen."""
        self.root.update()
        self.root.after(self.duration_ms, self.root.destroy)
        self.root.mainloop()

    def destroy(self):
        """Close splash screen."""
        self.root.destroy()
