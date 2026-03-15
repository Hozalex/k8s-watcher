import asyncio
import logging
import signal

import anthropic

import config as cfg
from db import create_pool, upsert_resource, delete_resource
from embedder import EmbedderClient
from enricher import EnrichTask, enrichment_worker
from k8s import ResourceEvent, start_watchers


async def _process_events(
    event_queue: asyncio.Queue,
    enrich_queue: asyncio.Queue,
    pool,
    embedder: EmbedderClient,
    cluster: str,
    enrich_enabled: bool,
) -> None:
    """Main loop: consume k8s events, embed and store them."""
    while True:
        event: ResourceEvent = await event_queue.get()
        try:
            if event.event_type == "DELETED":
                await delete_resource(
                    pool,
                    cluster=cluster,
                    kind=event.kind,
                    name=event.name,
                    namespace=event.namespace,
                )
                logging.getLogger(__name__).debug(
                    "Deleted %s/%s/%s", event.kind, event.namespace, event.name
                )
                continue

            # Embed raw template text first (fast, no LLM)
            embedding = await embedder.embed(event.content)

            content_changed, structure_changed = await upsert_resource(
                pool,
                cluster=cluster,
                kind=event.kind,
                name=event.name,
                namespace=event.namespace,
                content_hash=event.content_hash,
                structural_hash=event.structural_hash,
                content=event.content,
                embedding=embedding,
                enriched=False,
            )

            if content_changed:
                logging.getLogger(__name__).info(
                    "Indexed %s/%s/%s", event.kind, event.namespace, event.name
                )
                # LLM enrichment only when structure changed (not just replica scaling)
                if enrich_enabled and event.needs_enrichment and structure_changed:
                    await enrich_queue.put(EnrichTask(
                        kind=event.kind,
                        name=event.name,
                        namespace=event.namespace,
                        content=event.content,
                    ))

        except Exception:
            logging.getLogger(__name__).exception(
                "Error processing event %s %s/%s",
                event.event_type, event.namespace, event.name,
            )
        finally:
            event_queue.task_done()


async def main() -> None:
    conf = cfg.load()
    logging.basicConfig(
        level=conf.log_level,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )
    logger = logging.getLogger(__name__)
    logger.info("Starting k8s-watcher")

    pool = await create_pool(conf.database_url)
    embedder = EmbedderClient(conf.embeddings_url)

    event_queue: asyncio.Queue = asyncio.Queue(maxsize=500)
    enrich_queue: asyncio.Queue = asyncio.Queue(maxsize=200)

    anthropic_client = (
        anthropic.AsyncAnthropic(api_key=conf.anthropic_api_key)
        if conf.anthropic_api_key and conf.enrich_enabled
        else None
    )

    loop = asyncio.get_running_loop()
    stop_event = asyncio.Event()

    def _handle_signal(sig: int) -> None:
        logger.info("Received signal %s, shutting down gracefully…", signal.Signals(sig).name)
        stop_event.set()

    for sig in (signal.SIGTERM, signal.SIGINT):
        loop.add_signal_handler(sig, _handle_signal, sig)

    tasks: list[asyncio.Task] = []
    try:
        tasks.append(asyncio.create_task(start_watchers(event_queue), name="watchers"))
        tasks.append(asyncio.create_task(
            _process_events(event_queue, enrich_queue, pool, embedder, conf.cluster, conf.enrich_enabled),
            name="process_events",
        ))
        if anthropic_client:
            tasks.append(asyncio.create_task(
                enrichment_worker(enrich_queue, anthropic_client, embedder, pool, conf.enrich_concurrency),
                name="enrichment_worker",
            ))
        else:
            logger.warning("ANTHROPIC_API_KEY not set — LLM enrichment disabled")

        await stop_event.wait()

    finally:
        logger.info("Cancelling tasks…")
        for t in tasks:
            t.cancel()
        await asyncio.gather(*tasks, return_exceptions=True)
        await embedder.close()
        await pool.close()
        logger.info("Shutdown complete")


if __name__ == "__main__":
    asyncio.run(main())
