"""Public model snapshots used by the transport service."""

EMBED_MODEL = "nvidia/llama-nemotron-embed-vl-1b-v2"
EMBED_REVISION = "582e3bf72aee355e3c59ed89de53543c5b0657ee"
RERANK_MODEL = "nvidia/llama-nemotron-rerank-vl-1b-v2"
RERANK_REVISION = "b8a9987b05b75db5ad949c825cbcbf6eb7c48b5a"
DIMENSIONS = 2048


def catalog() -> list[dict]:
    return [
        {"id": EMBED_MODEL, "revision": EMBED_REVISION, "role": "embedder", "dimensions": DIMENSIONS},
        {"id": RERANK_MODEL, "revision": RERANK_REVISION, "role": "reranker"},
    ]
