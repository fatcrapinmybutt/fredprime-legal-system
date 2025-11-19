import os
import zipfile

BASE_DIR = os.getenv("LEGAL_RESULTS_DIR", os.path.join("F:/", "LegalResults"))
OUTPUT_DIR = os.path.join(BASE_DIR, "COURT_SUBMISSIONS", "MIFILE")
DEFAULT_FILES = [
    "emergency_injunction_motion_shady_oaks.docx",
    "motion_protective_order_postwrit_entry.docx",
    "federal_complaint_section1983_conversion.docx",
]


def build_bundle(files=None):
    files = files or DEFAULT_FILES
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    zip_path = os.path.join(OUTPUT_DIR, "MIFILE_HOUSING_EMERGENCY_BUNDLE_2025.zip")
    with zipfile.ZipFile(zip_path, "w") as z:
        for name in files:
            path = os.path.join(BASE_DIR, "MOTIONS", "HOUSING", name)
            if os.path.exists(path):
                z.write(path, name)
    print(f"MiFile bundle created at {zip_path}")


if __name__ == "__main__":
    build_bundle()
