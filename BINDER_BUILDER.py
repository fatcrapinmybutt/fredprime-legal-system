# BINDER_BUILDER.py – Create Exhibit Index & Binder TOC

exhibits = [
    ("Exhibit A", "Eviction Notice and Billing Ledger"),
    ("Exhibit B", "Photos of Damages to Mobile Home"),
    ("Exhibit C", "Emails with Nicole Browley and Shady Oaks"),
    ("Exhibit D", "EGLE Violation Documents"),
    ("Exhibit E", "Police Reports"),
    ("Exhibit F", "Previous Case Orders and Lease Docs"),
]


def generate_index():
    print("📘 STRIKEBACK BINDER – EXHIBIT INDEX")
    for tag, desc in exhibits:
        print(f"{tag} – {desc}")


if __name__ == "__main__":
    generate_index()
