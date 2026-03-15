"""
LLM enrichment for stable k8s resources (Deployments, StatefulSets, Services).

Uses Anthropic Haiku directly (single-turn call, no Agent SDK needed).
Runs as a background worker consuming from an asyncio.Queue.
"""
import asyncio
import logging
from dataclasses import dataclass

import anthropic
import asyncpg

from embedder import EmbedderClient

logger = logging.getLogger(__name__)

_PROMPT = """\
Analyze this Kubernetes resource and write a short operational description (3-5 sentences).
Cover: what it does, what services or users depend on it, and what breaks if it goes down.
Plain text only, no markdown.

{content}"""


@dataclass
class EnrichTask:
    kind: str
    name: str
    namespace: str
    content: str


async def _enrich_one(
    task: EnrichTask,
    anthropic_client: anthropic.AsyncAnthropic,
    embedder: EmbedderClient,
    pool: asyncpg.Pool,
) -> None:
    try:
        # Generate semantic description via Haiku
        msg = await anthropic_client.messages.create(
            model="claude-haiku-4-5-20251001",
            max_tokens=256,
            messages=[{"role": "user", "content": _PROMPT.format(content=task.content)}],
        )
        enriched_text = msg.content[0].text.strip()

        in_tok = msg.usage.input_tokens
        out_tok = msg.usage.output_tokens
        cost_usd = in_tok * 0.80 / 1_000_000 + out_tok * 4.00 / 1_000_000
        logger.info(
            "Haiku enriched %s/%s/%s — in=%d out=%d tokens, cost=$%.5f",
            task.kind, task.namespace, task.name, in_tok, out_tok, cost_usd,
        )

        # Embed the enriched description
        embedding = await embedder.embed(enriched_text)

        # Update the row — replace raw content with enriched, keep hash unchanged
        await pool.execute(
            """
            UPDATE infrastructure
               SET content  = $1,
                   embedding = $2::vector,
                   enriched  = TRUE,
                   updated_at = now()
             WHERE kind=$3 AND name=$4 AND namespace=$5
            """,
            enriched_text, str(embedding),
            task.kind, task.name, task.namespace,
        )

    except Exception:
        logger.exception("Failed to enrich %s/%s/%s", task.kind, task.namespace, task.name)


async def enrichment_worker(
    queue: asyncio.Queue,
    anthropic_client: anthropic.AsyncAnthropic,
    embedder: EmbedderClient,
    pool: asyncpg.Pool,
    concurrency: int = 2,
) -> None:
    """Consume EnrichTasks from the queue, running up to `concurrency` tasks at once."""
    semaphore = asyncio.Semaphore(concurrency)

    async def _bounded(task: EnrichTask) -> None:
        async with semaphore:
            await _enrich_one(task, anthropic_client, embedder, pool)

    while True:
        task: EnrichTask = await queue.get()
        asyncio.create_task(_bounded(task))
        queue.task_done()
