import tkinter as tk
from tkinter import ttk
from tkinterweb import HtmlFrame
from warboard.warboard_engine import deploy_supra_warboard


def launch_dashboard():
    root = tk.Tk()
    root.title('Litigation OS')
    root.geometry('900x600')

    notebook = ttk.Notebook(root)
    notebook.pack(fill='both', expand=True)

    warboard_tab = ttk.Frame(notebook)
    notebook.add(warboard_tab, text='🗺️ Warboard Visualizer')

    frame = HtmlFrame(warboard_tab)
    frame.pack(fill='both', expand=True)

    def refresh_warboard():
        deploy_supra_warboard()
        with open('warboard/exports/SHADY_OAKS_WARBOARD.svg') as f:
            frame.set_content(f.read())

    ttk.Button(warboard_tab, text='Build Warboard', command=refresh_warboard).pack(pady=5)
    refresh_warboard()

    root.mainloop()


if __name__ == '__main__':
    launch_dashboard()
