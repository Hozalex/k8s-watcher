import os
from dataclasses import dataclass


@dataclass(frozen=True)
class Config:
    cluster: str          # logical cluster name, stored in every DB row
    database_url: str
    embeddings_url: str
    anthropic_api_key: str | None
    log_level: str
    enrich_enabled: bool
    enrich_concurrency: int


def load() -> Config:
    return Config(
        cluster=os.environ["CLUSTER_NAME"],
        database_url=os.environ["DATABASE_URL"],
        embeddings_url=os.environ.get("EMBEDDINGS_URL", "http://embeddings-api/embed"),
        anthropic_api_key=os.environ.get("ANTHROPIC_API_KEY"),
        log_level=os.environ.get("LOG_LEVEL", "INFO").upper(),
        enrich_enabled=os.environ.get("ENRICH_ENABLED", "true").lower() == "true",
        enrich_concurrency=int(os.environ.get("ENRICH_CONCURRENCY", "2")),
    )
