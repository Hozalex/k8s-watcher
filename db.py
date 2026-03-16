import asyncio
import logging

import asyncpg

logger = logging.getLogger(__name__)

_SCHEMA = """
CREATE EXTENSION IF NOT EXISTS pgvector;

CREATE TABLE IF NOT EXISTS infrastructure (
    id               UUID        PRIMARY KEY DEFAULT gen_random_uuid(),
    cluster          TEXT        NOT NULL,
    kind             TEXT        NOT NULL,
    name             TEXT        NOT NULL,
    namespace        TEXT        NOT NULL DEFAULT '',
    content_hash     TEXT        NOT NULL,  -- full spec hash
    structural_hash  TEXT        NOT NULL,  -- spec minus volatile fields (replicas etc.)
    content          TEXT        NOT NULL,
    enriched         BOOLEAN     NOT NULL DEFAULT FALSE,
    embedding        vector(384),
    updated_at       TIMESTAMPTZ NOT NULL DEFAULT now(),
    UNIQUE (cluster, kind, name, namespace)
);

CREATE INDEX IF NOT EXISTS infrastructure_embedding_idx
    ON infrastructure USING hnsw (embedding vector_cosine_ops);

CREATE INDEX IF NOT EXISTS infrastructure_cluster_idx
    ON infrastructure (cluster);

CREATE TABLE IF NOT EXISTS incidents (
    id         UUID        PRIMARY KEY DEFAULT gen_random_uuid(),
    cluster    TEXT        NOT NULL,
    summary    TEXT        NOT NULL,
    content    TEXT        NOT NULL,
    severity   TEXT,
    namespace  TEXT,
    embedding  vector(384),
    created_at TIMESTAMPTZ NOT NULL DEFAULT now()
);

CREATE INDEX IF NOT EXISTS incidents_embedding_idx
    ON incidents USING hnsw (embedding vector_cosine_ops);

CREATE INDEX IF NOT EXISTS incidents_cluster_idx
    ON incidents (cluster);
"""


async def create_pool(dsn: str, retries: int = 10, delay: float = 5.0) -> asyncpg.Pool:
    """Create a connection pool, retrying on failure.

    Retries handle the window where Cilium FQDN policy hasn't yet resolved
    the hostname and established egress rules.
    """
    for attempt in range(1, retries + 1):
        try:
            pool = await asyncpg.create_pool(dsn, min_size=1, max_size=5)
            async with pool.acquire() as conn:
                await conn.execute(_SCHEMA)
            logger.info("Database schema ready")
            return pool
        except Exception as exc:
            if attempt == retries:
                raise
            logger.warning(
                "DB connect failed (attempt %d/%d): %s — retrying in %.0fs",
                attempt, retries, exc, delay,
            )
            await asyncio.sleep(delay)


async def fetch_hashes(
    pool: asyncpg.Pool,
    *,
    cluster: str,
    kind: str,
    name: str,
    namespace: str,
) -> tuple[str, str] | None:
    """Return (content_hash, structural_hash) for the resource, or None if it doesn't exist."""
    row = await pool.fetchrow(
        "SELECT content_hash, structural_hash FROM infrastructure"
        " WHERE cluster=$1 AND kind=$2 AND name=$3 AND namespace=$4",
        cluster, kind, name, namespace,
    )
    return (row["content_hash"], row["structural_hash"]) if row else None


async def upsert_resource(
    pool: asyncpg.Pool,
    *,
    cluster: str,
    kind: str,
    name: str,
    namespace: str,
    content_hash: str,
    structural_hash: str,
    content: str,
    embedding: list[float],
    enriched: bool = False,
    old_structural_hash: str | None,
) -> tuple[bool, bool]:
    """Insert or update a resource.

    Returns (content_changed, structure_changed):
      content_changed   — the row was inserted or updated (any spec change)
      structure_changed — structural_hash changed → LLM re-enrichment needed
    """
    row = await pool.fetchrow(
        """
        INSERT INTO infrastructure
            (cluster, kind, name, namespace, content_hash, structural_hash, content, embedding, enriched)
        VALUES ($1, $2, $3, $4, $5, $6, $7, $8::vector, $9)
        ON CONFLICT (cluster, kind, name, namespace) DO UPDATE
            SET content_hash    = EXCLUDED.content_hash,
                structural_hash = EXCLUDED.structural_hash,
                content         = EXCLUDED.content,
                embedding       = EXCLUDED.embedding,
                enriched        = EXCLUDED.enriched,
                updated_at      = now()
            WHERE infrastructure.content_hash != EXCLUDED.content_hash
        RETURNING xmax
        """,
        cluster, kind, name, namespace,
        content_hash, structural_hash, content, str(embedding), enriched,
    )
    if row is None:
        return False, False   # content_hash unchanged — nothing to do
    structure_changed = old_structural_hash is None or old_structural_hash != structural_hash
    return True, structure_changed


async def delete_resource(
    pool: asyncpg.Pool,
    *,
    cluster: str,
    kind: str,
    name: str,
    namespace: str,
) -> None:
    await pool.execute(
        "DELETE FROM infrastructure WHERE cluster=$1 AND kind=$2 AND name=$3 AND namespace=$4",
        cluster, kind, name, namespace,
    )
