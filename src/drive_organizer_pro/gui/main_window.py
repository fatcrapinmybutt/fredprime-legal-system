"""
Main GUI window for DriveOrganizerPro.

© 2026 MBP LLC. All rights reserved.
"""

import tkinter as tk
from tkinter import filedialog, messagebox
from pathlib import Path
import threading

from ..core.organizer_engine import OrganizerEngine
from .components import (
    ThemedButton, ThemedLabel, ThemedFrame, ThemedCheckbutton,
    StatusBar, ProgressPanel, LogDisplay, DriveSelector
)
from .themes import MBP_DARK_THEME
from .splash_screen import SplashScreen


class MainWindow:
    """Main application window."""

    def __init__(self):
        """Initialize main window."""
        self.root = tk.Tk()
        self.root.title("🐷 MBP LLC DriveOrganizerPro - Level 9999 Edition")
        self.root.geometry("900x700")
        self.root.configure(bg=MBP_DARK_THEME['bg_primary'])

        # Initialize organizer engine
        self.engine = OrganizerEngine()
        self.is_running = False

        # Options
        self.dry_run_var = tk.BooleanVar(value=True)
        self.remove_empty_var = tk.BooleanVar(value=True)
        self.handle_duplicates_var = tk.BooleanVar(value=True)
        self.create_sub_buckets_var = tk.BooleanVar(value=True)

        # Create widgets
        self._create_widgets()

        # Center window
        self._center_window()

    def _create_widgets(self):
        """Create all widgets."""
        # Main container
        main_container = ThemedFrame(self.root)
        main_container.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)

        # Header
        self._create_header(main_container)

        # Drive selector
        self.drive_selector = DriveSelector(main_container)
        self.drive_selector.pack(fill=tk.X, pady=10)

        # Source path selection
        self._create_path_selector(main_container)

        # Options
        self._create_options(main_container)

        # Action buttons
        self._create_buttons(main_container)

        # Progress panel
        self.progress_panel = ProgressPanel(main_container)
        self.progress_panel.pack(fill=tk.X, pady=10)

        # Log display
        log_frame = ThemedFrame(main_container)
        log_frame.pack(fill=tk.BOTH, expand=True, pady=10)

        ThemedLabel(log_frame, text="Log Output:", header=False).pack(anchor=tk.W)
        self.log_display = LogDisplay(log_frame)
        self.log_display.pack(fill=tk.BOTH, expand=True, pady=5)

        # Status bar
        self.status_bar = StatusBar(self.root)

        # Initial log message
        self.log("🐷 DriveOrganizerPro initialized - Maximum Business Performance ready!", "SUCCESS")
        self.log("Powered by Pork™", "INFO")

    def _create_header(self, parent):
        """Create header section."""
        header_frame = ThemedFrame(parent)
        header_frame.pack(fill=tk.X, pady=10)

        # Title
        title_label = ThemedLabel(
            header_frame,
            text="🐷 DriveOrganizerPro",
            header=True
        )
        title_label.pack()

        # Tagline
        tagline_label = ThemedLabel(
            header_frame,
            text="Level 9999 Edition - Maximum Business Performance",
            secondary=True
        )
        tagline_label.pack()

    def _create_path_selector(self, parent):
        """Create path selection section."""
        path_frame = ThemedFrame(parent)
        path_frame.pack(fill=tk.X, pady=10)

        ThemedLabel(path_frame, text="Source Directory:", header=False).pack(anchor=tk.W)

        entry_frame = ThemedFrame(path_frame)
        entry_frame.pack(fill=tk.X, pady=5)

        self.source_path_var = tk.StringVar()
        self.source_entry = tk.Entry(
            entry_frame,
            textvariable=self.source_path_var,
            bg=MBP_DARK_THEME['bg_secondary'],
            fg=MBP_DARK_THEME['text_primary'],
            font=('Segoe UI', 10),
            relief='flat'
        )
        self.source_entry.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=(0, 5))

        browse_btn = ThemedButton(entry_frame, text="Browse...", command=self._browse_directory)
        browse_btn.pack(side=tk.RIGHT)

    def _create_options(self, parent):
        """Create options section."""
        options_frame = ThemedFrame(parent)
        options_frame.pack(fill=tk.X, pady=10)

        ThemedLabel(options_frame, text="Options:", header=False).pack(anchor=tk.W)

        # Dry run
        dry_run_cb = ThemedCheckbutton(
            options_frame,
            text="DRY RUN (Preview Only - RECOMMENDED FIRST TIME)",
            variable=self.dry_run_var
        )
        dry_run_cb.pack(anchor=tk.W, padx=20, pady=2)

        # Remove empty directories
        remove_empty_cb = ThemedCheckbutton(
            options_frame,
            text="Remove empty directories after organization",
            variable=self.remove_empty_var
        )
        remove_empty_cb.pack(anchor=tk.W, padx=20, pady=2)

        # Handle duplicates
        duplicates_cb = ThemedCheckbutton(
            options_frame,
            text="Detect and quarantine duplicate files",
            variable=self.handle_duplicates_var
        )
        duplicates_cb.pack(anchor=tk.W, padx=20, pady=2)

        # Create sub-buckets
        sub_buckets_cb = ThemedCheckbutton(
            options_frame,
            text="Create sub-buckets (Meek1-4, LitigationOS, etc.)",
            variable=self.create_sub_buckets_var
        )
        sub_buckets_cb.pack(anchor=tk.W, padx=20, pady=2)

    def _create_buttons(self, parent):
        """Create action buttons."""
        button_frame = ThemedFrame(parent)
        button_frame.pack(fill=tk.X, pady=10)

        # Organize button
        self.organize_btn = ThemedButton(
            button_frame,
            text="🚀 ORGANIZE DRIVES",
            command=self._start_organization,
            primary=True
        )
        self.organize_btn.pack(side=tk.LEFT, padx=5)

        # Revert button
        self.revert_btn = ThemedButton(
            button_frame,
            text="⚠️ REVERT CHANGES",
            command=self._revert_changes
        )
        self.revert_btn.pack(side=tk.LEFT, padx=5)

        # Clear log button
        clear_log_btn = ThemedButton(
            button_frame,
            text="Clear Log",
            command=self.log_display.clear
        )
        clear_log_btn.pack(side=tk.RIGHT, padx=5)

    def _browse_directory(self):
        """Open directory browser."""
        directory = filedialog.askdirectory(title="Select Directory to Organize")
        if directory:
            self.source_path_var.set(directory)

    def _start_organization(self):
        """Start organization process."""
        if self.is_running:
            messagebox.showwarning("Already Running", "Organization is already in progress!")
            return

        source_path = self.source_path_var.get()
        if not source_path:
            messagebox.showerror("No Source", "Please select a source directory!")
            return

        source_path = Path(source_path)
        if not source_path.exists():
            messagebox.showerror("Invalid Path", "Selected directory does not exist!")
            return

        # Confirm action
        if not self.dry_run_var.get():
            result = messagebox.askyesno(
                "Confirm Organization",
                "This will move files on your drive. Continue?",
                icon='warning'
            )
            if not result:
                return

        # Disable buttons
        self.organize_btn.config(state=tk.DISABLED)
        self.is_running = True

        # Clear previous logs
        self.log_display.clear()
        self.progress_panel.reset()

        # Run in thread
        thread = threading.Thread(target=self._run_organization, args=(source_path,))
        thread.daemon = True
        thread.start()

    def _run_organization(self, source_path: Path):
        """Run organization in background thread."""
        try:
            mode = "DRY RUN" if self.dry_run_var.get() else "LIVE"
            self.log(f"Starting organization ({mode})...", "INFO")
            self.log(f"Source: {source_path}", "INFO")

            stats = self.engine.organize_drive(
                source_path=source_path,
                dry_run=self.dry_run_var.get(),
                remove_empty=self.remove_empty_var.get(),
                handle_duplicates=self.handle_duplicates_var.get(),
                create_sub_buckets=self.create_sub_buckets_var.get(),
                progress_callback=self._update_progress
            )

            # Show results
            self.log("", "INFO")
            self.log("═" * 60, "SUCCESS")
            self.log("ORGANIZATION COMPLETE!", "SUCCESS")
            self.log("═" * 60, "SUCCESS")
            self.log(f"Files processed: {stats['files_processed']}", "INFO")
            self.log(f"Files moved: {stats['files_moved']}", "INFO")
            self.log(f"Files skipped: {stats['files_skipped']}", "INFO")
            self.log(f"Duplicates found: {stats['duplicates_found']}", "INFO")
            self.log(f"Errors: {stats['errors']}", "ERROR" if stats['errors'] > 0 else "INFO")

            self.root.after(0, self.status_bar.set_status, "Organization complete!")

            if not self.dry_run_var.get():
                messagebox.showinfo("Success", "Organization complete! Files have been organized.")

        except Exception as e:
            self.log(f"ERROR: {e}", "ERROR")
            self.root.after(0, lambda: messagebox.showerror("Error", f"Organization failed: {e}"))

        finally:
            # Re-enable buttons
            self.is_running = False
            self.root.after(0, lambda: self.organize_btn.config(state=tk.NORMAL))

    def _update_progress(self, current: int, total: int, status: str):
        """Update progress from background thread."""
        self.root.after(0, self.progress_panel.update_progress, current, total, status)
        self.root.after(0, self.status_bar.set_status, status)

    def _revert_changes(self):
        """Revert last organization."""
        if self.is_running:
            messagebox.showwarning("Running", "Please wait for current operation to finish!")
            return

        result = messagebox.askyesno(
            "Confirm Revert",
            "This will attempt to revert the last organization. Continue?",
            icon='warning'
        )

        if not result:
            return

        try:
            self.log("Reverting last organization...", "INFO")
            count = self.engine.revert_last_organization(dry_run=False)

            if count > 0:
                self.log(f"Reverted {count} files", "SUCCESS")
                messagebox.showinfo("Success", f"Reverted {count} files successfully!")
            else:
                self.log("No session found to revert", "WARNING")
                messagebox.showinfo("No Session", "No organization session found to revert.")

        except Exception as e:
            self.log(f"ERROR: {e}", "ERROR")
            messagebox.showerror("Error", f"Revert failed: {e}")

    def log(self, message: str, level: str = "INFO"):
        """
        Add message to log.

        Args:
            message: Log message
            level: Log level (INFO, WARNING, ERROR, SUCCESS)
        """
        self.log_display.append(message, level)

    def _center_window(self):
        """Center window on screen."""
        self.root.update_idletasks()
        width = self.root.winfo_width()
        height = self.root.winfo_height()
        x = (self.root.winfo_screenwidth() // 2) - (width // 2)
        y = (self.root.winfo_screenheight() // 2) - (height // 2)
        self.root.geometry(f'{width}x{height}+{x}+{y}')

    def run(self):
        """Start the application."""
        self.root.mainloop()


def main():
    """Main entry point."""
    # Show splash screen
    try:
        splash = SplashScreen(duration_ms=2000)
        splash.show()
    except Exception:
        pass  # Skip splash if error

    # Launch main window
    app = MainWindow()
    app.run()


if __name__ == '__main__':
    main()
