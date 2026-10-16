"""Check a running GPU service with setup text and a generated blank image."""
from __future__ import annotations

import argparse
import base64
from datetime import datetime, timezone
from io import BytesIO
import json
import math
import os
from pathlib import Path
from urllib.request import Request, urlopen

from PIL import Image

from model_pins import DIMENSIONS, EMBED_MODEL, EMBED_REVISION, RERANK_MODEL, RERANK_REVISION, catalog


def check(base_url: str) -> dict:
    base_url = base_url.rstrip("/")
    if not base_url.endswith("/v1"):
        raise ValueError("base URL must end with /v1")
    headers = {"Content-Type": "application/json"}
    if token := os.environ.get("RETRIEVAL_API_KEY"):
        headers["Authorization"] = f"Bearer {token}"

    def call(path: str, payload: dict | None = None, *, root: bool = False) -> dict:
        url = (base_url[:-3] if root else base_url) + path
        request = Request(url, data=json.dumps(payload).encode() if payload is not None else None, headers=headers)
        with urlopen(request, timeout=180) as response:
            return json.load(response)

    ready = call("/ready", root=True)
    runtime = ready.get("runtime", {})
    if ready.get("status") != "ready" or runtime.get("backend") != "torch_cuda" or runtime.get("production") is not True:
        raise RuntimeError("service did not report a ready CUDA model backend")
    if call("/models").get("data") != catalog():
        raise RuntimeError("model catalog differs from the pinned service contract")
    with BytesIO() as data:
        image = Image.new("RGB", (64, 64), "white")
        image.save(data, format="PNG")
        image.close()
        image_data = "data:image/png;base64," + base64.b64encode(data.getvalue()).decode()
    documents = ["A setup check.", {"image": image_data}, {"text": "A setup check.", "image": image_data}]
    shapes = []
    for input_type, items in (("query", ["What is on the page?"]), ("document", documents)):
        body = call("/embeddings", {"model": EMBED_MODEL, "input_type": input_type, "input": items})
        if body.get("model_revision") != EMBED_REVISION or body.get("dimensions") != DIMENSIONS or body.get("normalized") is not True:
            raise RuntimeError("embedding identity or shape metadata differs")
        rows = body.get("data", [])
        if len(rows) != len(items) or [row["index"] for row in rows] != list(range(len(items))):
            raise RuntimeError("embedding indices differ from the inputs")
        for row in rows:
            vector = row["embedding"]
            if len(vector) != DIMENSIONS or not all(math.isfinite(value) for value in vector):
                raise RuntimeError("embedding has invalid values")
            if abs(math.sqrt(sum(value * value for value in vector)) - 1.0) > 1e-4:
                raise RuntimeError("embedding does not have unit length")
        shapes.append({"input_type": input_type, "rows": len(rows), "dimensions": DIMENSIONS})
    scores = call("/rerank", {"model": RERANK_MODEL, "query": "What is on the page?", "documents": documents})
    results = scores.get("results", [])
    if scores.get("model_revision") != RERANK_REVISION or scores.get("score_type") != "raw_logit":
        raise RuntimeError("reranker identity or score type differs")
    if len(results) != len(documents) or sorted(row["index"] for row in results) != list(range(len(documents))):
        raise RuntimeError("reranker indices differ from the inputs")
    if not all(math.isfinite(row["relevance_score"]) for row in results):
        raise RuntimeError("reranker returned invalid scores")
    if any(results[index]["relevance_score"] < results[index + 1]["relevance_score"] for index in range(len(results) - 1)):
        raise RuntimeError("reranker scores are not in descending order")
    return {"checked_at": datetime.now(timezone.utc).isoformat(), "base_url": base_url,
            "runtime": runtime, "models": catalog(), "embedding_shapes": shapes,
            "rerank_count": len(results), "score_type": "raw_logit",
            "checks": "setup transport and finite model outputs; no challenge quality assessment"}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-url", default="http://127.0.0.1:8000/v1")
    parser.add_argument("--receipt", type=Path, required=True)
    args = parser.parse_args()
    receipt = check(args.base_url)
    args.receipt.parent.mkdir(parents=True, exist_ok=True)
    args.receipt.write_text(json.dumps(receipt, indent=2) + "\n")
    print(f"GPU setup checks passed. Receipt: {args.receipt}")
