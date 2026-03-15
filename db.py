import logging

import asyncpg

logger = logging.getLogger(__name__)

_SCHEMA = """
CREATE EXTENSION IF NOT EXISTS vector;

CREATE TABLE IF NOT EXISTS infrastructure (
    id               UUID        PRIMARY KEY DEFAULT gen_random_uuid(),
    kind             TEXT        NOT NULL,
    name             TEXT        NOT NULL,
    namespace        TEXT        NOT NULL DEFAULT '',
    content_hash     TEXT        NOT NULL,  -- full spec hash
    structural_hash  TEXT        NOT NULL,  -- spec minus volatile fields (replicas etc.)
    content          TEXT        NOT NULL,
    enriched         BOOLEAN     NOT NULL DEFAULT FALSE,
    embedding        vector(384),
    updated_at       TIMESTAMPTZ NOT NULL DEFAULT now(),
    UNIQUE (kind, name, namespace)
);

CREATE INDEX IF NOT EXISTS infrastructure_embedding_idx
    ON infrastructure USING hnsw (embedding vector_cosine_ops);

CREATE TABLE IF NOT EXISTS incidents (
    id         UUID        PRIMARY KEY DEFAULT gen_random_uuid(),
    summary    TEXT        NOT NULL,
    content    TEXT        NOT NULL,
    severity   TEXT,
    namespace  TEXT,
    embedding  vector(384),
    created_at TIMESTAMPTZ NOT NULL DEFAULT now()
);

CREATE INDEX IF NOT EXISTS incidents_embedding_idx
    ON incidents USING hnsw (embedding vector_cosine_ops);
"""


async def create_pool(dsn: str) -> asyncpg.Pool:
    pool = await asyncpg.create_pool(dsn, min_size=1, max_size=5)
    async with pool.acquire() as conn:
        await conn.execute(_SCHEMA)
    logger.info("Database schema ready")
    return pool


async def upsert_resource(
    pool: asyncpg.Pool,
    *,
    kind: str,
    name: str,
    namespace: str,
    content_hash: str,
    structural_hash: str,
    content: str,
    embedding: list[float],
    enriched: bool = False,
) -> tuple[bool, bool]:
    """Insert or update a resource.

    Returns (content_changed, structure_changed):
      content_changed   — the row was updated (any spec change, including replicas)
      structure_changed — structural_hash changed → LLM re-enrichment needed
    """
    row = await pool.fetchrow(
        """
        INSERT INTO infrastructure
            (kind, name, namespace, content_hash, structural_hash, content, embedding, enriched)
        VALUES ($1, $2, $3, $4, $5, $6, $7::vector, $8)
        ON CONFLICT (kind, name, namespace) DO UPDATE
            SET content_hash    = EXCLUDED.content_hash,
                structural_hash = EXCLUDED.structural_hash,
                content         = EXCLUDED.content,
                embedding       = EXCLUDED.embedding,
                enriched        = EXCLUDED.enriched,
                updated_at      = now()
            WHERE infrastructure.content_hash != EXCLUDED.content_hash
        RETURNING
            (xmax = 0)                                              AS inserted,
            infrastructure.structural_hash != EXCLUDED.structural_hash AS structure_changed
        """,
        kind, name, namespace,
        content_hash, structural_hash, content, str(embedding), enriched,
    )
    if row is None:
        return False, False   # nothing changed
    return True, bool(row["inserted"] or row["structure_changed"])


async def delete_resource(
    pool: asyncpg.Pool,
    *,
    kind: str,
    name: str,
    namespace: str,
) -> None:
    await pool.execute(
        "DELETE FROM infrastructure WHERE kind=$1 AND name=$2 AND namespace=$3",
        kind, name, namespace,
    )
