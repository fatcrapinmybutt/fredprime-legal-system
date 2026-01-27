import os

# Set OPENAI_API_KEY in your environment (e.g., `export OPENAI_API_KEY=...`) before use.
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")


def init_openai_api_key() -> None:
    """Initialize the OpenAI API key from the environment."""
    if not OPENAI_API_KEY:
        raise RuntimeError(
            "OPENAI_API_KEY is not set. Configure the environment variable before "
            "initializing OpenAI."
        )
    import openai

    openai.api_key = OPENAI_API_KEY
