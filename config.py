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
    # Extra watched resources on top of the built-in list.
    # Format: "api_version:plural:namespaced" comma-separated
    # Example: "cert-manager.io/v1:certificates:true,monitoring.coreos.com/v1:prometheuses:true"
    extra_watched_resources: list[tuple[str, str, bool]]


def _parse_extra_resources(raw: str) -> list[tuple[str, str, bool]]:
    result = []
    for entry in raw.split(","):
        entry = entry.strip()
        if not entry:
            continue
        parts = entry.split(":")
        if len(parts) != 3:
            raise ValueError(
                f"Invalid EXTRA_WATCHED_RESOURCES entry {entry!r}. "
                "Expected format: api_version:plural:namespaced"
            )
        api_version, plural, namespaced = parts
        result.append((api_version.strip(), plural.strip(), namespaced.strip().lower() == "true"))
    return result


def load() -> Config:
    return Config(
        cluster=os.environ["CLUSTER_NAME"],
        database_url=os.environ["DATABASE_URL"],
        embeddings_url=os.environ.get("EMBEDDINGS_URL", "http://embeddings-api/embed"),
        anthropic_api_key=os.environ.get("ANTHROPIC_API_KEY"),
        log_level=os.environ.get("LOG_LEVEL", "INFO").upper(),
        enrich_enabled=os.environ.get("ENRICH_ENABLED", "true").lower() == "true",
        enrich_concurrency=int(os.environ.get("ENRICH_CONCURRENCY", "2")),
        extra_watched_resources=_parse_extra_resources(
            os.environ.get("EXTRA_WATCHED_RESOURCES", "")
        ),
    )
