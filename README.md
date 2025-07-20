
[README_FRED_PRIME_OMNIA (1).md](https://github.com/user-attachments/files/21210312/README_FRED_PRIME_OMNIA.1.md)
[ruleset.json](https://github.com/user-attachments/files/21210316/ruleset.json)
[TrueFiling_User_Guide.pdf](https://github.com/user-attachments/files/21210318/TrueFiling_User_Guide.pdf)
[fredprime-legal-system-codex-develop-an-infinite-memory-litigation-ecosystem.zip](https://github.com/user-attachments/files/21210319/fredprime-legal-system-codex-develop-an-infinite-memory-litigation-ecosystem.zip)
[FRED_PRIME_LITIGATION_ENGINE.zip](https://github.com/user-attachments/files/21210321/FRED_PRIME_LITIGATION_ENGINE.zip)

## Usage

Generate a manifest of the current directory:

```bash
python cli/generate_manifest.py
# or use the console script after installation
generate_manifest
```

Use `-o` to specify the output file path:

```bash
python cli/generate_manifest.py -o path/to/manifest.json
```

### Generate the system definition

Run `firstimport.py` to create a JSON file describing the FRED PRIME system.

```bash
python firstimport.py
```

The script writes `fredprime_litigation_system.json` to `/mnt/data/`.
