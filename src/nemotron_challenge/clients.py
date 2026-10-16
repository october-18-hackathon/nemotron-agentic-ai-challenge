"""Bounded transport helpers. Teams choose prompts and retrieval methods."""
from __future__ import annotations

import base64
import hashlib
import json
import math
import re
from typing import Any, Literal

import httpx
from pydantic import ValidationError

from .config import (
    ConfigurationError, DEFAULT_PARSE_PROMPT, EMBED_DIMENSIONS, EMBED_MODEL, EMBED_REVISION,
    PARSE_MODEL, RERANK_MODEL, RERANK_REVISION, SUPER_MODEL, WIKIPEDIA_REVISION, endpoint,
)
from .contracts import WikiArticle, WikiSearch

MAX_RETRIEVAL_QUERY_CHARS = 2000
MAX_RETRIEVAL_DOCUMENT_CHARS = 32000


class ServiceError(RuntimeError):
    def __init__(self, message: str, *, status_code: int | None = None, retry_after: str | None = None):
        super().__init__(message)
        self.status_code, self.retry_after = status_code, retry_after


class ProtocolError(ServiceError):
    """A server reply differs from the expected contract."""


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ProtocolError(message)


def _text(value: str, limit: int, name: str) -> str:
    if not isinstance(value, str) or not value.strip() or len(value) > limit:
        raise ValueError(f"{name} must contain 1 to {limit} characters")
    try:
        value.encode("utf-8")
    except UnicodeEncodeError:
        raise ValueError(f"{name} must contain valid UTF-8 text") from None
    return value


def _image(value: str) -> str:
    if not isinstance(value, str) or len(value) > 12_000_000:
        raise ValueError("image must be an inline PNG/JPEG data URL below 8 MiB")
    match = re.fullmatch(r"data:image/(png|jpeg);base64,([A-Za-z0-9+/]*={0,2})", value)
    if match is None:
        raise ValueError("image must be an inline PNG/JPEG data URL")
    try:
        raw = base64.b64decode(match[2], validate=True)
    except ValueError:
        raise ValueError("image contains invalid base64") from None
    if not raw or len(raw) > 8 * 1024 * 1024:
        raise ValueError("image must contain 1 byte to 8 MiB")
    return value


def _retrieval_text(value: str, limit: int, name: str) -> str:
    _text(value, limit, name)
    if any(ord(char) < 32 and char not in "\n\r\t" for char in value):
        raise ValueError(f"{name} contains an unsupported control character")
    if any(marker in value for marker in ("<image>", "<img>", "</img>", "<IMG_CONTEXT>")):
        raise ValueError(f"{name} contains reserved image control tokens")
    return value


def _contents(items: list[str | dict[str, str]], limit: int, *, allow_images: bool = True,
              text_limit: int = MAX_RETRIEVAL_DOCUMENT_CHARS) -> list[str | dict[str, str]]:
    if not isinstance(items, list) or not 1 <= len(items) <= limit:
        raise ValueError(f"input must contain 1 to {limit} items")
    for item in items:
        if isinstance(item, str):
            _retrieval_text(item, text_limit, "text")
        elif isinstance(item, dict) and item and not set(item) - {"text", "image"}:
            if not item.get("text") and not item.get("image"):
                raise ValueError("each input needs text or an image")
            if item.get("text") is not None:
                _retrieval_text(item["text"], text_limit, "text")
            if item.get("image") is not None:
                if not allow_images:
                    raise ValueError("query embeddings accept text inputs")
                _image(item["image"])
        else:
            raise ValueError("input must be text or a text/image object")
    return items


class JSONClient:
    """Reuse a connection pool. Do not print request bodies or API keys."""

    def __init__(self, base_url: str, api_key: str = "", *, client: httpx.Client | None = None,
                 timeout: float = 180, max_response_bytes: int = 8 * 1024 * 1024):
        if not base_url:
            raise ConfigurationError("set the service base URL in .env")
        self.base_url = endpoint(base_url, "service base URL")
        self.api_key = api_key
        self.timeout = httpx.Timeout(timeout, connect=10, pool=10)
        self.max_response_bytes = max_response_bytes
        self._owned = client is None
        self.client = client or httpx.Client(
            timeout=self.timeout, limits=httpx.Limits(max_connections=4, max_keepalive_connections=4),
            follow_redirects=False,
        )

    def close(self) -> None:
        if self._owned:
            self.client.close()

    def __enter__(self):
        return self

    def __exit__(self, *_: Any) -> None:
        self.close()

    def request(self, method: str, path: str, *, payload: dict[str, Any] | None = None,
                params: dict[str, Any] | None = None) -> dict[str, Any]:
        headers = {"Accept": "application/json"}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"
        body = None
        if payload is not None:
            body = json.dumps(payload, ensure_ascii=True, allow_nan=False).encode()
            if len(body) > 24 * 1024 * 1024:
                raise ValueError("request exceeds the 24 MiB client limit")
            headers["Content-Type"] = "application/json"
        try:
            with self.client.stream(method, f"{self.base_url}/{path.lstrip('/')}", content=body,
                                    params=params, headers=headers, timeout=self.timeout,
                                    follow_redirects=False) as response:
                if not 200 <= response.status_code < 300:
                    raise ServiceError(f"service returned HTTP {response.status_code}",
                                       status_code=response.status_code, retry_after=response.headers.get("Retry-After"))
                chunks, size = [], 0
                for chunk in response.iter_bytes():
                    size += len(chunk)
                    if size > self.max_response_bytes:
                        raise ProtocolError("response exceeds the client byte limit")
                    chunks.append(chunk)
                result = json.loads(b"".join(chunks))
        except httpx.TimeoutException:
            raise ServiceError("service request reached its timeout") from None
        except httpx.HTTPError:
            raise ServiceError("service connection failed") from None
        except (ValueError, UnicodeError):
            raise ProtocolError("service did not return a valid JSON object") from None
        _require(isinstance(result, dict) and "error" not in result, "service returned an error or a non-object response")
        return result

    def models(self) -> dict[str, dict[str, Any]]:
        result = self.request("GET", "models")
        rows = result.get("data")
        _require(isinstance(rows, list) and all(isinstance(row, dict) and isinstance(row.get("id"), str) for row in rows),
                 "model catalog has an invalid data list")
        _require(len({row["id"] for row in rows}) == len(rows), "model catalog repeats an ID")
        return {row["id"]: row for row in rows}


class WikiClient(JSONClient):
    def corpus(self) -> dict[str, Any]:
        ready = self.request("GET", "ready")
        _require(ready.get("status") == "ready" and ready.get("production") is True,
                 "Wikipedia service has not proven production readiness")
        result = self.request("GET", "v1/corpus")
        proof = result.get("verification", {})
        _require(result.get("dataset_revision") == WIKIPEDIA_REVISION, "Wikipedia revision differs")
        _require(proof.get("production") is True and proof.get("canonical_articles") == 7_182_883
                 and proof.get("raw_rows") == 7_597_149 and proof.get("source_shards") == 86
                 and proof.get("fts_indexed_rows") == 7_182_883 and proof.get("fts_unindexed_rows") == 0,
                 "Wikipedia coverage proof differs")
        _require(result.get("license", {}).get("spdx") == "CC-BY-SA-4.0", "Wikipedia license is missing")
        return result

    def _source(self, row: Any) -> None:
        _require(row.provenance.dataset == "wikimedia/structured-wikipedia"
                 and row.provenance.dataset_revision == WIKIPEDIA_REVISION,
                 "Wikipedia article source differs")
        _require(row.license.get("spdx") == "CC-BY-SA-4.0", "Wikipedia article license is missing")

    def search(self, query: str, top_k: int = 10, excerpt_chars: int = 1000) -> WikiSearch:
        _text(query, 512, "query")
        if not 1 <= top_k <= 100 or not 100 <= excerpt_chars <= 2000:
            raise ValueError("top_k must be 1 to 100 and excerpt_chars must be 100 to 2000")
        result = self.request("POST", "v1/search", payload={"query": query, "top_k": top_k, "excerpt_chars": excerpt_chars})
        try:
            parsed = WikiSearch.model_validate(result)
        except ValidationError:
            raise ProtocolError("Wikipedia search response differs from its schema") from None
        _require(parsed.corpus_revision == WIKIPEDIA_REVISION and parsed.count == len(parsed.hits)
                 and len(parsed.hits) <= top_k, "Wikipedia search count or revision differs")
        for hit in parsed.hits:
            self._source(hit)
            _require(len(hit.excerpt) <= excerpt_chars, "Wikipedia excerpt exceeds the requested limit")
        return parsed

    def article(self, page_id: str, offset: int = 0, max_chars: int = 10000) -> WikiArticle:
        if not re.fullmatch(r"[0-9]{1,20}", page_id) or not 0 <= offset <= 100_000_000 or not 1 <= max_chars <= 20000:
            raise ValueError("article ID or pagination limits are invalid")
        result = self.request("GET", f"v1/articles/{page_id}", params={"offset": offset, "max_chars": max_chars})
        try:
            parsed = WikiArticle.model_validate(result)
        except ValidationError:
            raise ProtocolError("Wikipedia article response differs from its schema") from None
        self._source(parsed)
        _require(parsed.page_id == page_id and parsed.offset == offset and parsed.returned_chars <= max_chars,
                 "Wikipedia article response differs from the request")
        if parsed.offset == 0 and parsed.next_offset is None:
            _require(hashlib.sha256(parsed.text.encode()).hexdigest() == parsed.provenance.text_sha256,
                     "Wikipedia article text hash differs")
        return parsed


class RetrievalClient(JSONClient):
    def __init__(self, *args: Any, **kwargs: Any):
        super().__init__(*args, **kwargs)
        self._verified = False

    def verify_models(self) -> dict[str, dict[str, Any]]:
        rows = self.models()
        for model, revision in ((EMBED_MODEL, EMBED_REVISION), (RERANK_MODEL, RERANK_REVISION)):
            _require(model in rows and rows[model].get("revision") == revision, f"model revision differs: {model}")
        _require(rows[EMBED_MODEL].get("dimensions") == EMBED_DIMENSIONS, "embedding dimensions differ")
        self._verified = True
        return rows

    def _model_reply(self, result: dict[str, Any], model: str, revision: str) -> None:
        _require(result.get("model") == model, "response model ID differs")
        if "model_revision" in result:
            _require(result["model_revision"] == revision, "response model revision differs")

    def embed(self, inputs: list[str | dict[str, str]], input_type: Literal["query", "document"]) -> list[list[float]]:
        if input_type not in {"query", "document"}:
            raise ValueError("input_type must be query or document")
        _contents(inputs, 16, allow_images=input_type == "document",
                  text_limit=MAX_RETRIEVAL_QUERY_CHARS if input_type == "query" else MAX_RETRIEVAL_DOCUMENT_CHARS)
        if not self._verified:
            self.verify_models()
        result = self.request("POST", "embeddings", payload={"model": EMBED_MODEL, "input": inputs, "input_type": input_type})
        self._model_reply(result, EMBED_MODEL, EMBED_REVISION)
        if "dimensions" in result:
            _require(result["dimensions"] == EMBED_DIMENSIONS, "response embedding dimensions differ")
        if "normalized" in result:
            _require(result["normalized"] is True, "response embedding normalization differs")
        rows = result.get("data")
        _require(isinstance(rows, list) and len(rows) == len(inputs), "embedding result count differs")
        by_index = {}
        for row in rows:
            _require(isinstance(row, dict) and type(row.get("index")) is int, "embedding index is invalid")
            index, vector = row["index"], row.get("embedding")
            _require(0 <= index < len(inputs) and index not in by_index, "embedding index is repeated or out of range")
            _require(isinstance(vector, list) and len(vector) == EMBED_DIMENSIONS
                     and all(type(value) in {int, float} and math.isfinite(value) for value in vector),
                     "embedding vector has invalid dimensions or values")
            _require(abs(sum(value * value for value in vector) - 1) <= 0.02, "embedding vector is not normalized")
            by_index[index] = vector
        return [by_index[index] for index in range(len(inputs))]

    def rerank(self, query: str, documents: list[str | dict[str, str]], top_n: int | None = None) -> list[dict[str, Any]]:
        _retrieval_text(query, MAX_RETRIEVAL_QUERY_CHARS, "query")
        _contents(documents, 32)
        if top_n is not None and (type(top_n) is not int or not 1 <= top_n <= len(documents)):
            raise ValueError("top_n must be 1 to the document count")
        if not self._verified:
            self.verify_models()
        payload = {"model": RERANK_MODEL, "query": query, "documents": documents}
        if top_n is not None:
            payload["top_n"] = top_n
        result = self.request("POST", "rerank", payload=payload)
        self._model_reply(result, RERANK_MODEL, RERANK_REVISION)
        _require(result.get("score_type") in {"logit", "raw_logit"}, "reranker score type differs")
        rows = result.get("results")
        _require(isinstance(rows, list) and len(rows) == (top_n or len(documents)), "reranker result count differs")
        seen = set()
        for row in rows:
            _require(isinstance(row, dict) and type(row.get("index")) is int, "reranker index is invalid")
            index, score = row["index"], row.get("relevance_score")
            _require(0 <= index < len(documents) and index not in seen, "reranker index is repeated or out of range")
            _require(type(score) in {int, float} and math.isfinite(score), "reranker score is not finite")
            seen.add(index)
        return rows


class ChatClient(JSONClient):
    def __init__(self, base_url: str, api_key: str, model: str = SUPER_MODEL, **kwargs: Any):
        super().__init__(base_url, api_key, **kwargs)
        if not model:
            raise ConfigurationError("set SUPER_MODEL to the required provider model ID")
        self.model, self._verified = model, False

    def verify_model(self) -> dict[str, Any]:
        rows = self.models()
        _require(self.model in rows, f"expected model is absent from the catalog: {self.model}")
        self._verified = True
        return rows[self.model]

    def complete(self, messages: list[dict[str, Any]], max_tokens: int = 4096,
                 extra_body: dict[str, Any] | None = None) -> dict[str, Any]:
        if not messages or not isinstance(messages, list) or type(max_tokens) is not int or not 1 <= max_tokens <= 16384:
            raise ValueError("provide messages and a max_tokens value from 1 to 16384")
        if extra_body and set(extra_body) & {"model", "messages", "max_tokens", "stream"}:
            raise ValueError("extra_body cannot replace the model, messages, token limit, or stream setting")
        if not self._verified:
            self.verify_model()
        payload = {"model": self.model, "messages": messages, "max_tokens": max_tokens, "stream": False}
        payload.update(extra_body or {})
        result = self.request("POST", "chat/completions", payload=payload)
        _require(result.get("model") == self.model, "chat response model ID differs")
        choices = result.get("choices")
        _require(isinstance(choices, list) and len(choices) == 1, "chat response must contain one choice")
        choice = choices[0]
        _require(isinstance(choice, dict), "chat choice has an invalid shape")
        message = choice.get("message", {})
        if choice.get("finish_reason") == "tool_calls":
            _require(bool(extra_body and extra_body.get("tools")), "chat returned tools that the caller did not supply")
            calls = message.get("tool_calls") if isinstance(message, dict) else None
            _require(isinstance(calls, list) and bool(calls), "chat tool reply has no calls")
            for call in calls:
                _require(isinstance(call, dict) and isinstance(call.get("id"), str) and bool(call["id"])
                         and call.get("type") == "function" and isinstance(call.get("function"), dict),
                         "chat tool call shape is invalid")
                function = call["function"]
                _require(isinstance(function.get("name"), str) and bool(function["name"])
                         and isinstance(function.get("arguments"), str), "chat tool function is invalid")
                try:
                    arguments = json.loads(function["arguments"])
                except ValueError:
                    raise ProtocolError("chat tool arguments are incomplete JSON") from None
                _require(isinstance(arguments, dict), "chat tool arguments must be a JSON object")
            return result
        _require(choice.get("finish_reason") == "stop",
                 "chat response is incomplete; inspect the token budget and finish_reason")
        _require(isinstance(message, dict) and isinstance(message.get("content"), str) and bool(message["content"].strip()),
                 "chat response has no final text")
        return result


class ParseClient(ChatClient):
    def __init__(self, base_url: str, api_key: str = "", **kwargs: Any):
        super().__init__(base_url, api_key, model=PARSE_MODEL, **kwargs)

    def parse(self, image_data_url: str, instruction: str = DEFAULT_PARSE_PROMPT, max_tokens: int = 8744) -> dict[str, Any]:
        _image(image_data_url)
        _text(instruction, 64000, "instruction")
        return self.complete([
            {"role": "user", "content": [
                {"type": "text", "text": instruction},
                {"type": "image_url", "image_url": {"url": image_data_url}},
            ]}
        ], max_tokens=max_tokens, extra_body={"temperature": 0, "skip_special_tokens": False, "add_special_tokens": False})
