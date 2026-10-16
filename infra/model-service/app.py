"""Embedding and reranking transport for two pinned public models."""
from __future__ import annotations

import asyncio
from contextlib import asynccontextmanager
from dataclasses import dataclass
import logging
import math
import os
from pathlib import Path
from typing import Any, Callable

from fastapi import FastAPI, HTTPException, Request
from fastapi.concurrency import run_in_threadpool
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse

from backend import GPUCapacityError, TorchBackend
from contracts import (EmbeddingRequest, InputError, MAX_BODY_BYTES, RerankRequest,
                       close_images, decode_items)
from model_pins import DIMENSIONS, EMBED_MODEL, EMBED_REVISION, RERANK_MODEL, RERANK_REVISION, catalog

LOGGER = logging.getLogger("model_service")


@dataclass(frozen=True)
class Settings:
    model_root: Path = Path("/opt/models")
    max_waiting: int = 4
    queue_wait_seconds: float = 1.0
    max_body_bytes: int = MAX_BODY_BYTES

    @classmethod
    def from_env(cls) -> Settings:
        return cls(model_root=Path(os.environ.get("MODEL_ROOT", "/opt/models")))


class BodyLimitMiddleware:
    def __init__(self, app: Any, max_bytes: int):
        self.app, self.max_bytes = app, max_bytes

    async def __call__(self, scope: Any, receive: Any, send: Any) -> None:
        if scope["type"] != "http":
            await self.app(scope, receive, send)
            return
        body = bytearray()
        while True:
            message = await receive()
            if message["type"] == "http.disconnect":
                return
            chunk = message.get("body", b"")
            if len(body) + len(chunk) > self.max_bytes:
                await JSONResponse({"detail": "request_body_limit"}, status_code=413)(scope, receive, send)
                return
            body.extend(chunk)
            if not message.get("more_body", False):
                break
        delivered = False

        async def replay() -> dict:
            nonlocal delivered
            if delivered:
                return await receive()
            delivered = True
            return {"type": "http.request", "body": bytes(body), "more_body": False}

        await self.app(scope, replay, send)


def embedding_result(backend: Any, body: EmbeddingRequest) -> dict:
    documents = decode_items(body.input)
    try:
        vectors = backend.embeddings(documents, body.input_type)
    finally:
        close_images(documents)
    if len(vectors) != len(body.input):
        raise RuntimeError("embedder returned the wrong vector count")
    for vector in vectors:
        if len(vector) != DIMENSIONS or not all(math.isfinite(value) for value in vector):
            raise RuntimeError("embedder returned an invalid vector")
        if abs(math.sqrt(sum(value * value for value in vector)) - 1.0) > 1e-4:
            raise RuntimeError("embedder returned a vector without unit length")
    return {"object": "list", "model": EMBED_MODEL, "model_revision": EMBED_REVISION,
            "dimensions": DIMENSIONS, "normalized": True,
            "data": [{"object": "embedding", "index": index, "embedding": vector}
                     for index, vector in enumerate(vectors)]}


def rerank_result(backend: Any, body: RerankRequest) -> dict:
    documents = decode_items(body.documents)
    try:
        scores = backend.rerank(body.query, documents)
    finally:
        close_images(documents)
    if len(scores) != len(body.documents) or not all(math.isfinite(value) for value in scores):
        raise RuntimeError("reranker returned invalid scores")
    order = sorted(range(len(scores)), key=lambda index: (-scores[index], index))
    return {"model": RERANK_MODEL, "model_revision": RERANK_REVISION, "score_type": "raw_logit",
            "results": [{"index": index, "relevance_score": scores[index]}
                        for index in order[:body.top_n or len(scores)]]}


def create_app(settings: Settings | None = None, backend_factory: Callable[[], Any] | None = None) -> FastAPI:
    settings = settings or Settings.from_env()
    factory = backend_factory or (lambda: TorchBackend(settings.model_root))

    @asynccontextmanager
    async def lifespan(api: FastAPI):
        api.state.backend = None
        api.state.slots = asyncio.Semaphore(1)
        api.state.waiting = 0
        backend = factory()
        try:
            await run_in_threadpool(backend.load)
            await run_in_threadpool(backend.warmup)
            api.state.backend = backend
        except Exception:
            LOGGER.exception("Model startup or warmup failed")
        try:
            yield
        finally:
            api.state.backend = None
            await run_in_threadpool(backend.close)

    api = FastAPI(title="Nemotron model transport", version="1.0.0", lifespan=lifespan)
    api.add_middleware(BodyLimitMiddleware, max_bytes=settings.max_body_bytes)

    def ready_backend() -> Any:
        if api.state.backend is None:
            raise HTTPException(503, detail="models_not_ready")
        return api.state.backend

    async def run_job(function: Any, body: Any) -> dict:
        backend = ready_backend()
        if api.state.waiting >= settings.max_waiting:
            raise HTTPException(429, detail="gpu_queue_full", headers={"Retry-After": "1"})
        api.state.waiting += 1
        try:
            await asyncio.wait_for(api.state.slots.acquire(), timeout=settings.queue_wait_seconds)
        except TimeoutError:
            raise HTTPException(429, detail="gpu_busy", headers={"Retry-After": "1"}) from None
        finally:
            api.state.waiting -= 1
        try:
            # The slot stays held until the synchronous GPU work returns.
            return await run_in_threadpool(function, backend, body)
        except InputError as exc:
            raise HTTPException(422, detail=str(exc)) from None
        except GPUCapacityError:
            raise HTTPException(503, detail="gpu_memory_limit", headers={"Retry-After": "1"}) from None
        except Exception:
            LOGGER.exception("Model inference failed")
            api.state.backend = None
            raise HTTPException(503, detail="model_inference_failed") from None
        finally:
            api.state.slots.release()

    @api.get("/health")
    async def health() -> dict:
        return {"status": "ok"}

    @api.get("/ready")
    async def ready() -> dict:
        backend = ready_backend()
        return {"status": "ready", "models": catalog(), "runtime": backend.proof}

    @api.get("/v1/models")
    async def models() -> dict:
        ready_backend()
        return {"object": "list", "data": catalog()}

    @api.post("/v1/embeddings")
    async def embeddings(body: EmbeddingRequest) -> dict:
        return await run_job(embedding_result, body)

    @api.post("/v1/rerank")
    async def rerank(body: RerankRequest) -> dict:
        return await run_job(rerank_result, body)

    @api.exception_handler(RequestValidationError)
    async def invalid_request(request: Request, exc: RequestValidationError) -> JSONResponse:
        def clean(value: Any) -> str:
            return str(value).encode("utf-8", errors="replace").decode("utf-8")
        return JSONResponse({"detail": [{"loc": [clean(part) for part in error["loc"]],
                                       "type": error["type"], "msg": clean(error["msg"])}
                                      for error in exc.errors()]}, status_code=422)

    return api


app = create_app()
