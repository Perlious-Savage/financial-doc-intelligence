"""One-shot database setup and health check.

Creates the pgvector extension, the documents table and the similarity index, then
reports what is actually in there. Safe to run repeatedly.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from dotenv import load_dotenv

load_dotenv()

from src.index import count_documents, init_schema  # noqa: E402


def main() -> None:
    print("creating extension, table and index ...")
    init_schema()
    print(f"ready. documents indexed: {count_documents()}")


if __name__ == "__main__":
    main()
