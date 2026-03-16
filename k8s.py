"""
Kubernetes resource watcher.

Generates template-based text for every resource change.
LLM enrichment is gated on structural_hash — changes to replicas or
HPA current counts do NOT trigger a Haiku call.
"""
import asyncio
import copy
import hashlib
import json
import logging
import time
from dataclasses import dataclass
from typing import Any

from kubernetes import client, config, watch
from kubernetes.client.exceptions import ApiException

logger = logging.getLogger(__name__)

# Kinds that carry business meaning and benefit from LLM enrichment
STABLE_KINDS = {
    "Deployment", "StatefulSet", "DaemonSet",
    "Ingress", "HTTPRoute", "Gateway",
    "CronJob",
}

# (api_version, plural, namespaced)
WATCHED_RESOURCES = [
    # Core workloads
    ("apps/v1",                       "deployments",             True),
    ("apps/v1",                       "statefulsets",            True),
    ("apps/v1",                       "daemonsets",              True),
    # Networking
    ("v1",                            "services",                True),
    ("networking.k8s.io/v1",          "ingresses",               True),
    ("gateway.networking.k8s.io/v1",  "gateways",                True),
    ("gateway.networking.k8s.io/v1",  "httproutes",              True),
    # Scheduling & scaling
    ("batch/v1",                      "cronjobs",                True),
    ("autoscaling/v2",                "horizontalpodautoscalers", True),
    # Infrastructure
    ("v1",                            "nodes",                   False),
]

# Top-level spec fields that are volatile (scaling) and should NOT
# trigger LLM re-enrichment when they change.
_VOLATILE_SPEC_FIELDS: dict[str, set[str]] = {
    "Deployment":              {"replicas"},
    "StatefulSet":             {"replicas"},
    "HorizontalPodAutoscaler": {"currentReplicas", "desiredReplicas"},
}


@dataclass
class ResourceEvent:
    event_type: str       # ADDED | MODIFIED | DELETED
    kind: str
    name: str
    namespace: str
    content: str          # template text
    content_hash: str     # full spec hash — controls re-embedding
    structural_hash: str  # spec minus volatile fields — controls LLM enrichment
    needs_enrichment: bool


# ── Template generators ────────────────────────────────────────────────────────

def _labels(obj: dict) -> str:
    labels = obj.get("metadata", {}).get("labels") or {}
    return ", ".join(f"{k}={v}" for k, v in labels.items()) or "none"


def _header(kind_label: str, meta: dict) -> str:
    """Return the standard first line: '<Kind> <name> / <namespace>'."""
    return f"{kind_label} {meta.get('name', '?')} / {meta.get('namespace', '')}\n"


def _workload_images(spec: dict) -> str:
    """Return a comma-separated list of container images for pod-template workloads."""
    containers = spec.get("template", {}).get("spec", {}).get("containers", [])
    return ", ".join(c.get("image", "?") for c in containers)


def _format_conditions(conditions: list) -> str:
    """Format a list of condition dicts as 'Type=Status, ...' or 'none'."""
    return ", ".join(
        f"{c.get('type')}={c.get('status')}" for c in conditions
    ) or "none"


def _render_deployment(obj: dict) -> str:
    meta = obj["metadata"]
    spec = obj.get("spec", {})
    status = obj.get("status", {})
    selector = json.dumps(spec.get("selector", {}).get("matchLabels", {}))
    return (
        _header("Deployment", meta)
        + f"Replicas: {status.get('readyReplicas', 0)}/{spec.get('replicas', '?')} ready\n"
        f"Images: {_workload_images(spec)}\n"
        f"Selector: {selector}\n"
        f"Labels: {_labels(obj)}\n"
        f"Conditions: {_format_conditions(status.get('conditions', []))}"
    )


def _render_statefulset(obj: dict) -> str:
    meta = obj["metadata"]
    spec = obj.get("spec", {})
    status = obj.get("status", {})
    return (
        _header("StatefulSet", meta)
        + f"Replicas: {status.get('readyReplicas', 0)}/{spec.get('replicas', '?')} ready\n"
        f"Images: {_workload_images(spec)}\n"
        f"ServiceName: {spec.get('serviceName', '?')}\n"
        f"Labels: {_labels(obj)}"
    )


def _render_daemonset(obj: dict) -> str:
    meta = obj["metadata"]
    spec = obj.get("spec", {})
    status = obj.get("status", {})
    return (
        _header("DaemonSet", meta)
        + f"Desired: {status.get('desiredNumberScheduled', '?')}, "
        f"Ready: {status.get('numberReady', '?')}\n"
        f"Images: {_workload_images(spec)}\n"
        f"Labels: {_labels(obj)}"
    )


def _render_service(obj: dict) -> str:
    meta = obj["metadata"]
    spec = obj.get("spec", {})
    ports = ", ".join(
        f"{p.get('port')}->{p.get('targetPort', '?')}/{p.get('protocol', 'TCP')}"
        for p in spec.get("ports", [])
    )
    return (
        _header("Service", meta)
        + f"Type: {spec.get('type', 'ClusterIP')}\n"
        f"Ports: {ports or 'none'}\n"
        f"Selector: {json.dumps(spec.get('selector') or {})}\n"
        f"ClusterIP: {spec.get('clusterIP', '?')}\n"
        f"Labels: {_labels(obj)}"
    )


def _render_ingress(obj: dict) -> str:
    meta = obj["metadata"]
    spec = obj.get("spec", {})
    rules = spec.get("rules", [])
    hosts = ", ".join(r.get("host", "*") for r in rules) or "any"
    paths = ", ".join(
        f"{r.get('host', '*')}{p['path']}"
        for r in rules
        for p in r.get("http", {}).get("paths", [])
    )
    tls_hosts = ", ".join(
        h for t in spec.get("tls", []) for h in t.get("hosts", [])
    ) or "none"
    return (
        _header("Ingress", meta)
        + f"IngressClass: {spec.get('ingressClassName', '?')}\n"
        f"Hosts: {hosts}\n"
        f"Paths: {paths or 'none'}\n"
        f"TLS: {tls_hosts}\n"
        f"Labels: {_labels(obj)}"
    )


def _render_httproute(obj: dict) -> str:
    meta = obj["metadata"]
    spec = obj.get("spec", {})
    hostnames = ", ".join(spec.get("hostnames", [])) or "any"
    parents = ", ".join(
        p.get("name", "?") for p in spec.get("parentRefs", [])
    )
    rules_count = len(spec.get("rules", []))
    return (
        _header("HTTPRoute", meta)
        + f"Hostnames: {hostnames}\n"
        f"Gateway: {parents}\n"
        f"Rules: {rules_count}\n"
        f"Labels: {_labels(obj)}"
    )


def _render_gateway(obj: dict) -> str:
    meta = obj["metadata"]
    spec = obj.get("spec", {})
    listeners = ", ".join(
        f"{l.get('name')}:{l.get('port')}/{l.get('protocol', '?')}"
        for l in spec.get("listeners", [])
    )
    return (
        _header("Gateway", meta)
        + f"GatewayClass: {spec.get('gatewayClassName', '?')}\n"
        f"Listeners: {listeners or 'none'}\n"
        f"Labels: {_labels(obj)}"
    )


def _render_cronjob(obj: dict) -> str:
    meta = obj["metadata"]
    spec = obj.get("spec", {})
    containers = (
        spec.get("jobTemplate", {})
            .get("spec", {})
            .get("template", {})
            .get("spec", {})
            .get("containers", [])
    )
    images = ", ".join(c.get("image", "?") for c in containers)
    return (
        _header("CronJob", meta)
        + f"Schedule: {spec.get('schedule', '?')}\n"
        f"Images: {images}\n"
        f"Suspend: {spec.get('suspend', False)}\n"
        f"Labels: {_labels(obj)}"
    )


def _render_hpa(obj: dict) -> str:
    meta = obj["metadata"]
    spec = obj.get("spec", {})
    status = obj.get("status", {})
    ref = spec.get("scaleTargetRef", {})
    return (
        _header("HorizontalPodAutoscaler", meta)
        + f"Target: {ref.get('kind', '?')}/{ref.get('name', '?')}\n"
        f"Replicas: {status.get('currentReplicas', '?')} current, "
        f"min={spec.get('minReplicas', '?')} max={spec.get('maxReplicas', '?')}\n"
        f"Labels: {_labels(obj)}"
    )


def _render_node(obj: dict) -> str:
    meta = obj["metadata"]
    status = obj.get("status", {})
    alloc = status.get("allocatable", {})
    roles = ", ".join(
        k.replace("node-role.kubernetes.io/", "")
        for k in (meta.get("labels") or {})
        if k.startswith("node-role.kubernetes.io/")
    ) or "worker"
    taints = ", ".join(
        f"{t['key']}:{t['effect']}"
        for t in obj.get("spec", {}).get("taints", [])
    ) or "none"
    return (
        f"Node {meta['name']}\n"
        f"Roles: {roles}\n"
        f"CPU: {alloc.get('cpu', '?')}, Memory: {alloc.get('memory', '?')}\n"
        f"Conditions: {_format_conditions(status.get('conditions', []))}\n"
        f"Taints: {taints}"
    )


def _render_rabbitmq(obj: dict) -> str:
    meta = obj["metadata"]
    spec = obj.get("spec", {})
    status = obj.get("status", {})
    return (
        _header("RabbitmqCluster", meta)
        + f"Replicas: {spec.get('replicas', '?')}\n"
        f"Image: {spec.get('image', 'default')}\n"
        f"Conditions: {_format_conditions(status.get('conditions', []))}\n"
        f"Labels: {_labels(obj)}"
    )


_RENDERERS = {
    "Deployment":              _render_deployment,
    "StatefulSet":             _render_statefulset,
    "DaemonSet":               _render_daemonset,
    "Service":                 _render_service,
    "Ingress":                 _render_ingress,
    "HTTPRoute":               _render_httproute,
    "Gateway":                 _render_gateway,
    "CronJob":                 _render_cronjob,
    "HorizontalPodAutoscaler": _render_hpa,
    "Node":                    _render_node,
    "RabbitmqCluster":         _render_rabbitmq,
}


def _render(kind: str, obj: dict) -> str:
    renderer = _RENDERERS.get(kind)
    return renderer(obj) if renderer else json.dumps(obj.get("spec", {}), indent=2)


def _hash(data: dict) -> str:
    return hashlib.sha256(
        json.dumps(data, sort_keys=True).encode()
    ).hexdigest()[:16]


def _structural_hash(kind: str, spec: dict) -> str:
    """Hash that ignores volatile fields (replica counts).
    A change here means the resource actually changed shape → re-enrich."""
    volatile = _VOLATILE_SPEC_FIELDS.get(kind)
    if not volatile:
        return _hash(spec)
    stripped = {k: v for k, v in spec.items() if k not in volatile}
    return _hash(stripped)


# ── Watch loop ─────────────────────────────────────────────────────────────────

def _watch_resource(
    api_version: str,
    plural: str,
    namespaced: bool,
    queue: asyncio.Queue,
    loop: asyncio.AbstractEventLoop,
) -> None:
    """Runs in a thread. Watches a single resource type and puts events in queue."""
    api_client_inst = client.ApiClient()
    core_api = client.CoreV1Api(api_client=api_client_inst)
    custom_api = client.CustomObjectsApi(api_client=api_client_inst)
    w = watch.Watch()

    group, _, version = api_version.partition("/")
    if not version:
        group, version = "", api_version

    is_core = group == ""

    logger.info("Starting watch: %s/%s", api_version, plural)

    while True:
        try:
            kwargs: dict[str, Any] = {"timeout_seconds": 0}

            if is_core:
                if namespaced:
                    list_fn = getattr(core_api, f"list_{plural[:-1]}_for_all_namespaces")
                else:
                    list_fn = getattr(core_api, f"list_{plural[:-1]}")
                stream = w.stream(list_fn, **kwargs)
            else:
                if namespaced:
                    stream = w.stream(
                        custom_api.list_namespaced_custom_object,
                        group=group, version=version, plural=plural, namespace="",
                        **kwargs,
                    )
                else:
                    stream = w.stream(
                        custom_api.list_cluster_custom_object,
                        group=group, version=version, plural=plural,
                        **kwargs,
                    )

            for raw in stream:
                event_type = raw["type"]
                obj = raw["object"]
                # Core API returns typed objects; serialize to plain dict
                if not isinstance(obj, dict):
                    obj = api_client_inst.sanitize_for_serialization(obj)
                kind = obj.get("kind") or plural.rstrip("s").capitalize()
                meta = obj.get("metadata", {})
                name = meta.get("name", "")
                namespace = meta.get("namespace", "")
                spec = obj.get("spec", {})

                content = _render(kind, obj)
                c_hash = _hash(spec)
                s_hash = _structural_hash(kind, spec)

                event = ResourceEvent(
                    event_type=event_type,
                    kind=kind,
                    name=name,
                    namespace=namespace,
                    content=content,
                    content_hash=c_hash,
                    structural_hash=s_hash,
                    needs_enrichment=(
                        kind in STABLE_KINDS and event_type in ("ADDED", "MODIFIED")
                    ),
                )
                asyncio.run_coroutine_threadsafe(queue.put(event), loop)

        except ApiException as exc:
            if exc.status == 404:
                logger.warning(
                    "Resource %s/%s not found in this cluster (CRD not installed) — skipping",
                    api_version, plural,
                )
                return  # no point retrying
            logger.exception("Watch error for %s/%s, restarting in 5s", api_version, plural)
            time.sleep(5)
        except Exception:
            logger.exception("Watch error for %s/%s, restarting in 5s", api_version, plural)
            time.sleep(5)


async def start_watchers(
    queue: asyncio.Queue,
    extra: list[tuple[str, str, bool]] | None = None,
) -> None:
    try:
        config.load_incluster_config()
        logger.info("Using in-cluster kubeconfig")
    except config.ConfigException:
        config.load_kube_config()
        logger.info("Using local kubeconfig")

    from concurrent.futures import ThreadPoolExecutor

    all_resources = WATCHED_RESOURCES + (extra or [])

    # Dedicated pool so watcher threads don't starve asyncio's default pool
    # (httpx uses it for getaddrinfo / DNS resolution).
    watcher_pool = ThreadPoolExecutor(
        max_workers=len(all_resources),
        thread_name_prefix="k8s-watch",
    )
    loop = asyncio.get_running_loop()
    for api_version, plural, namespaced in all_resources:
        loop.run_in_executor(
            watcher_pool,
            _watch_resource,
            api_version, plural, namespaced, queue, loop,
        )
    logger.info("Watchers started for %d resource types", len(all_resources))
