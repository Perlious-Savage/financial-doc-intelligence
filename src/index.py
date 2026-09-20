"""Semantic document retrieval over PostgreSQL + pgvector.

This is semantic retrieval, not retrieval-augmented generation. Nothing retrieved here
is injected into a generation step, and calling it RAG would invite a question with no
good answer. Its job is finding similar or near-duplicate documents across the corpus —
the same receipt submitted twice, or a set of documents sharing suspicious structure.

Why Postgres rather than a dedicated vector database: the corpus is small, the service
already needs a relational store for documents and results, and one datastore is less
to operate than two. At a corpus size where that stops being true, the answer changes.

Embedding requires a model and therefore runs on the GPU side (see notebooks/). Query
time needs no model at all: finding documents similar to one already indexed is a pure
SQL operation, which is why the API can serve it from the CPU container.
"""

from __future__ import annotations

import os
from typing import Any, Optional

EMBED_MODEL = "BAAI/bge-small-en-v1.5"
EMBED_DIM = 384


def _database_url() -> str:
    url = os.environ.get("DATABASE_URL", "").strip()
    if not url:
        raise RuntimeError(
            "DATABASE_URL is not set. Copy .env.example to .env and add your connection string."
        )
    return url


def connect():
    import psycopg

    return psycopg.connect(_database_url())


def init_schema() -> None:
    """Create the extension and table. Safe to run repeatedly."""
    with connect() as conn, conn.cursor() as cur:
        cur.execute("CREATE EXTENSION IF NOT EXISTS vector")
        cur.execute(
            f"""
            CREATE TABLE IF NOT EXISTS documents (
                id          bigserial PRIMARY KEY,
                doc_id      text UNIQUE NOT NULL,
                content     text NOT NULL,
                metadata    jsonb DEFAULT '{{}}'::jsonb,
                embedding   vector({EMBED_DIM})
            )
            """
        )
        # Cosine distance, matching how the embeddings are normalized.
        cur.execute(
            "CREATE INDEX IF NOT EXISTS documents_embedding_idx "
            "ON documents USING ivfflat (embedding vector_cosine_ops) WITH (lists = 50)"
        )
        conn.commit()


def embed_texts(texts: list[str]) -> list[list[float]]:
    """Encode text. Requires sentence-transformers, so this runs GPU-side."""
    from sentence_transformers import SentenceTransformer

    model = SentenceTransformer(EMBED_MODEL)
    return model.encode(texts, normalize_embeddings=True).tolist()


def upsert_documents(
    doc_ids: list[str],
    contents: list[str],
    embeddings: list[list[float]],
    metadatas: Optional[list[dict[str, Any]]] = None,
) -> int:
    import json

    metadatas = metadatas or [{} for _ in doc_ids]
    rows = list(zip(doc_ids, contents, [json.dumps(m) for m in metadatas], embeddings))
    with connect() as conn, conn.cursor() as cur:
        cur.executemany(
            """
            INSERT INTO documents (doc_id, content, metadata, embedding)
            VALUES (%s, %s, %s::jsonb, %s::vector)
            ON CONFLICT (doc_id) DO UPDATE
              SET content = EXCLUDED.content,
                  metadata = EXCLUDED.metadata,
                  embedding = EXCLUDED.embedding
            """,
            [(d, c, m, str(e)) for d, c, m, e in rows],
        )
        conn.commit()
    return len(rows)


def find_similar(doc_id: str, k: int = 5) -> list[dict[str, Any]]:
    """Documents most similar to one already indexed.

    The document excludes itself, which is not a detail: without that clause every
    query returns the query document at similarity 1.0 and the result looks perfect
    while being useless.
    """
    with connect() as conn, conn.cursor() as cur:
        cur.execute(
            """
            SELECT d.doc_id,
                   d.content,
                   1 - (d.embedding <=> q.embedding) AS similarity
            FROM documents d, (SELECT embedding FROM documents WHERE doc_id = %s) q
            WHERE d.doc_id <> %s AND d.embedding IS NOT NULL
            ORDER BY d.embedding <=> q.embedding
            LIMIT %s
            """,
            (doc_id, doc_id, k),
        )
        return [
            {"doc_id": r[0], "content": r[1][:200], "similarity": round(float(r[2]), 4)}
            for r in cur.fetchall()
        ]


def count_documents() -> int:
    with connect() as conn, conn.cursor() as cur:
        cur.execute("SELECT count(*) FROM documents")
        return int(cur.fetchone()[0])
