# MBP Litigation System

This repository contains a small scaffold for the "MBP Litigation OS".  The
`build_mbp_system.py` script generates a folder structure and example modules for
a local litigation toolkit.  The modules demonstrate simple document generation,
zip bundling and exhibit scanning utilities.

## Usage

1. Install Python 3 with `python-docx` available.
2. Run the setup script:

```bash
python build_mbp_system.py
```

This will create the `MBP_SYSTEM` directory with subfolders such as `ENGINE`,
`GUI`, `UPGRADES` and `VFS`.  Example engine modules will be written to these
folders.

3. After the scaffold is generated you can test a module, e.g.:

```bash
python MBP_SYSTEM/ENGINE/motion_generator.py
```

The script will generate a document inside `MBP_SYSTEM/VFS/GENERATED_ZIPS`.

## Notes

These files are simplified examples based on the requested system outline.  They
do not provide a complete litigation solution but illustrate the general
structure described in the project documentation.
