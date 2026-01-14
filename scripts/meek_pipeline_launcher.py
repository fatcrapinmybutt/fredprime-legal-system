import argparse


def run_ingest() -> None:
    """Run the ingestion stage."""
    from meek_ingest_cli import main as ingest_main

    ingest_main()


def run_search(query: str) -> None:
    """Run the search stage."""
    from fts_cli import search_records

    search_records(query)


def run_post_process() -> None:
    """Run optional post-processing/report generation."""
    print("Post-processing complete.")


def main(argv: list[str] | None = None) -> None:
    """Launch the Meek ingestion and search pipeline."""
    parser = argparse.ArgumentParser(description="Meek pipeline launcher")
    parser.add_argument("--skip-ingest", action="store_true", help="Skip ingestion stage")
    parser.add_argument("--skip-search", action="store_true", help="Skip search stage")
    parser.add_argument("--search-query", help="Query string for search")
    parser.add_argument("--post-process", action="store_true", help="Run post-processing stage")
    args = parser.parse_args(argv)

    if not args.skip_ingest:
        run_ingest()

    if not args.skip_search:
        query = args.search_query or input("Enter search query: ")
        run_search(query)

    if args.post_process:
        run_post_process()


if __name__ == "__main__":
    main()
