"""
Reusable GUI components for DriveOrganizerPro.

© 2026 MBP LLC. All rights reserved.
"""

import tkinter as tk
from tkinter import ttk
from .themes import MBP_DARK_THEME, apply_theme_to_widget


class ThemedButton(tk.Button):
    """Themed button widget."""

    def __init__(self, parent, primary=False, **kwargs):
        """
        Initialize themed button.

        Args:
            parent: Parent widget
            primary: If True, use primary button style
            **kwargs: Additional button arguments
        """
        super().__init__(parent, **kwargs)
        style = 'button_primary' if primary else 'button'
        apply_theme_to_widget(self, style)


class ThemedLabel(tk.Label):
    """Themed label widget."""

    def __init__(self, parent, header=False, secondary=False, **kwargs):
        """
        Initialize themed label.

        Args:
            parent: Parent widget
            header: If True, use header style
            secondary: If True, use secondary text style
            **kwargs: Additional label arguments
        """
        super().__init__(parent, **kwargs)

        if header:
            apply_theme_to_widget(self, 'label_header')
        elif secondary:
            apply_theme_to_widget(self, 'label_secondary')
        else:
            apply_theme_to_widget(self, 'label')


class ThemedFrame(tk.Frame):
    """Themed frame widget."""

    def __init__(self, parent, **kwargs):
        """Initialize themed frame."""
        super().__init__(parent, **kwargs)
        apply_theme_to_widget(self, 'frame')


class ThemedCheckbutton(tk.Checkbutton):
    """Themed checkbutton widget."""

    def __init__(self, parent, **kwargs):
        """Initialize themed checkbutton."""
        super().__init__(parent, **kwargs)
        apply_theme_to_widget(self, 'checkbutton')


class StatusBar(ThemedFrame):
    """Status bar widget."""

    def __init__(self, parent):
        """Initialize status bar."""
        super().__init__(parent)
        self.pack(side=tk.BOTTOM, fill=tk.X, padx=5, pady=5)

        self.status_label = ThemedLabel(self, text="Ready", anchor=tk.W, secondary=True)
        self.status_label.pack(side=tk.LEFT, fill=tk.X, expand=True)

    def set_status(self, text: str):
        """Update status text."""
        self.status_label.config(text=text)


class ProgressPanel(ThemedFrame):
    """Progress display panel."""

    def __init__(self, parent):
        """Initialize progress panel."""
        super().__init__(parent)

        # Progress bar
        self.progress_var = tk.DoubleVar()
        self.progress_bar = ttk.Progressbar(
            self,
            variable=self.progress_var,
            maximum=100,
            mode='determinate'
        )
        self.progress_bar.pack(fill=tk.X, padx=5, pady=5)

        # Progress label
        self.progress_label = ThemedLabel(self, text="0 / 0 files processed", secondary=True)
        self.progress_label.pack(pady=2)

        # Status label
        self.status_label = ThemedLabel(self, text="", secondary=True)
        self.status_label.pack(pady=2)

    def update_progress(self, current: int, total: int, status: str = ""):
        """
        Update progress display.

        Args:
            current: Current progress value
            total: Total value
            status: Status message
        """
        if total > 0:
            percentage = (current / total) * 100
            self.progress_var.set(percentage)

        self.progress_label.config(text=f"{current} / {total} files processed")

        if status:
            self.status_label.config(text=status)

    def reset(self):
        """Reset progress display."""
        self.progress_var.set(0)
        self.progress_label.config(text="0 / 0 files processed")
        self.status_label.config(text="")


class LogDisplay(tk.Frame):
    """Log display widget with scrolling."""

    def __init__(self, parent):
        """Initialize log display."""
        super().__init__(parent, bg=MBP_DARK_THEME['bg_primary'])

        # Create text widget with scrollbar
        self.text = tk.Text(
            self,
            height=10,
            wrap=tk.WORD,
            bg=MBP_DARK_THEME['bg_secondary'],
            fg=MBP_DARK_THEME['text_primary'],
            font=('Consolas', 9),
            relief='flat'
        )

        scrollbar = tk.Scrollbar(self, command=self.text.yview)
        self.text.config(yscrollcommand=scrollbar.set)

        self.text.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)

        # Configure tags for colored text
        self.text.tag_config('INFO', foreground=MBP_DARK_THEME['accent_green'])
        self.text.tag_config('WARNING', foreground=MBP_DARK_THEME['warning_orange'])
        self.text.tag_config('ERROR', foreground=MBP_DARK_THEME['error_red'])
        self.text.tag_config('SUCCESS', foreground=MBP_DARK_THEME['success_green'])

    def append(self, text: str, tag: str = None):
        """
        Append text to log.

        Args:
            text: Text to append
            tag: Optional tag for coloring
        """
        self.text.insert(tk.END, text + "\n", tag)
        self.text.see(tk.END)

    def clear(self):
        """Clear log display."""
        self.text.delete(1.0, tk.END)


class DriveSelector(ThemedFrame):
    """Drive selection widget."""

    def __init__(self, parent):
        """Initialize drive selector."""
        super().__init__(parent)

        self.drive_vars = {}
        self.checkbuttons = {}

        ThemedLabel(self, text="Select Drives to Organize:", header=False).pack(anchor=tk.W, pady=5)

        # Common drive letters
        drives = ['E:', 'F:', 'H:', 'I:', 'J:']

        for drive in drives:
            var = tk.BooleanVar(value=True)
            self.drive_vars[drive] = var

            cb = ThemedCheckbutton(
                self,
                text=f"{drive} Drive",
                variable=var
            )
            cb.pack(anchor=tk.W, padx=20, pady=2)
            self.checkbuttons[drive] = cb

    def get_selected_drives(self):
        """Get list of selected drives."""
        return [drive for drive, var in self.drive_vars.items() if var.get()]
