# `firstimport.py` — System definition generator

This small utility builds a JSON description of the FRED PRIME system and writes it to disk.

Usage

- Run with defaults (writes to `./output/fredprime_litigation_system.json`):

```
python firstimport.py
```

- Override base path or output file:

```
python firstimport.py --base /path/to/project --out /path/to/out.json
```

Validation

- If `jsonschema` is installed and `schema/systemdef.schema.json` exists in the base path, the script will validate the generated JSON before writing. Add `jsonschema` to `requirements.txt` to enable validation.

Environment variables

- `FREDPRIME_BASE` — default base path if `--base` not provided
- `FREDPRIME_JSON` — default JSON output path if `--out` not provided
- `FREDPRIME_SCHEMA` — override the schema path for validation

New option

- `--no-validate` — skip JSON Schema validation before writing the output (useful for fast local runs)
