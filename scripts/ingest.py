"""
Ingestion script — chunks corpus/ and upserts into Qdrant.
Run after prepare_data.py.

Usage:
  python scripts/ingest.py                  # uses active_chunking_strategy from settings.yaml
  python scripts/ingest.py fixed            # force fixed-size strategy
  python scripts/ingest.py recursive        # force recursive strategy
  python scripts/ingest.py both             # ingest both strategies (uses two collections)
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from rag.ingest import run_ingestion


def main():
    strategy_arg = sys.argv[1] if len(sys.argv) > 1 else None

    if strategy_arg == "both":
        print("=== Ingesting with FIXED strategy ===")
        run_ingestion("fixed")
        print("\n=== Ingesting with RECURSIVE strategy ===")
        run_ingestion("recursive")
    else:
        run_ingestion(strategy_arg)


if __name__ == "__main__":
    main()
