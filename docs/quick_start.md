# Quick Start

## Local Model Requirements

The system uses an offline text analysis model implemented in
`core/local_llm.py`. No network connectivity is required. The default
analyzer relies only on the Python standard library and runs on any
machine with Python 3.11 or newer. For advanced models, place them on
the local filesystem and modify `analyze_content()` accordingly.
