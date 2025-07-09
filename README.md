# FRED PRIME Litigation Utilities

This repository provides simple Python scripts for managing litigation documents. The tools here allow you to:

- **Auto-label exhibits** in a folder (`Exhibit_A`, `Exhibit_B`, ...)
- **Link motions to exhibits** by scanning motion text files
- **Validate signature blocks** according to Michigan Court Rule 1.109(D)(3)

These scripts are intentionally lightweight and run entirely offline. They only operate on files you provide and produce markdown summaries for easy review.

## Setup

You need Python 3.8 or newer. Clone the repo and install dependencies (only the standard library is used):

```bash
python3 fredprime.py --help
```

## Usage

### Label exhibits
Rename all files in a directory sequentially and create `Exhibit_Index.md`:

```bash
python3 fredprime.py label-exhibits path/to/exhibits
```

### Link motions to exhibits
Scan all `.txt` files in a directory for references like `Exhibit A` and build `Motion_to_Exhibit_Map.md`:

```bash
python3 fredprime.py link-motions path/to/motions
```

### Validate a signature block
Check that a document contains a line beginning with `s/` followed by a name:

```bash
python3 fredprime.py validate-signature path/to/document.txt
```

## Repository Contents

- `fredprime.py` – command‑line entry point
- `src/` – module implementations
  - `exhibit_labeler.py`
  - `motion_exhibit_linker.py`
  - `signature_validator.py`

These utilities are minimal examples and may be extended as needed.
