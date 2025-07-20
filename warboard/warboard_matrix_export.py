import os
import zipfile

EXPORT_DIR = os.path.join('F:', os.sep, 'LegalResults', 'WARBOARD_MATRIX')
FILES_TO_INCLUDE = [
    os.path.join('warboard', 'exports', 'SHADY_OAKS_WARBOARD.docx'),
    os.path.join('warboard', 'exports', 'SHADY_OAKS_WARBOARD.svg'),
    os.path.join('warboard', 'exports', 'PPO_WARBOARD.docx'),
    os.path.join('warboard', 'exports', 'PPO_WARBOARD.svg'),
    os.path.join('warboard', 'exports', 'CUSTODY_INTERFERENCE_MAP.docx'),
    os.path.join('warboard', 'exports', 'CUSTODY_INTERFERENCE_MAP.svg'),
    os.path.join('data', 'contradiction_matrix.json'),
    os.path.join('data', 'timeline.json'),
]


def build_zip_bundle(export_dir: str = EXPORT_DIR, files: list[str] | None = None) -> str:
    """Create a ZIP bundle of warboard and data files.

    Parameters
    ----------
    export_dir : str
        Destination directory for the ZIP file.
    files : list[str] | None
        Additional files to include. When ``None``, ``FILES_TO_INCLUDE`` is used.
    Returns
    -------
    str
        Path to the created ZIP file.
    """
    os.makedirs(export_dir, exist_ok=True)
    zip_path = os.path.join(export_dir, 'WARBOARD_BUNDLE.zip')
    file_list = files if files is not None else FILES_TO_INCLUDE
    with zipfile.ZipFile(zip_path, 'w') as zipf:
        for file in file_list:
            if os.path.exists(file):
                zipf.write(file, os.path.relpath(file, start=export_dir))
    print(f'ZIP bundle created at {zip_path}')
    return zip_path


if __name__ == '__main__':
    build_zip_bundle()
