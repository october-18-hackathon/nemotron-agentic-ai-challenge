"""Protocol tests use synthetic inputs and an HTTPX MockTransport."""
import base64
import hashlib
import json

import httpx
import pytest

from nemotron_challenge.clients import ChatClient, JSONClient, ParseClient, ProtocolError, RetrievalClient, ServiceError, WikiClient
from nemotron_challenge.config import DEFAULT_PARSE_PROMPT, EMBED_MODEL, EMBED_REVISION, PARSE_MODEL, RERANK_MODEL, RERANK_REVISION, SUPER_MODEL, WIKIPEDIA_REVISION

PNG = "data:image/png;base64," + base64.b64encode(bytes.fromhex(
    "89504e470d0a1a0a0000000d4948445200000001000000010804000000b51c0c020000000b4944415478da6364f80f00010501012718e3660000000049454e44ae426082"
)).decode()


def catalog():
    return {"data": [
        {"id": EMBED_MODEL, "revision": EMBED_REVISION, "dimensions": 2048},
        {"id": RERANK_MODEL, "revision": RERANK_REVISION},
    ]}


def transport_client(handler):
    return httpx.Client(transport=httpx.MockTransport(handler))


def test_embedding_wire_contract_and_response_order():
    requests = []

    def handle(request):
        requests.append(request)
        assert request.headers["Authorization"] == "Bearer test-key"
        if request.url.path == "/v1/models":
            return httpx.Response(200, json=catalog())
        assert request.url.path == "/v1/embeddings"
        assert json.loads(request.content) == {"model": EMBED_MODEL, "input": ["first", "second"], "input_type": "document"}
        return httpx.Response(200, json={"model": EMBED_MODEL, "model_revision": EMBED_REVISION,
                              "data": [{"index": 1, "embedding": [0, 1] + [0] * 2046},
                                       {"index": 0, "embedding": [1] + [0] * 2047}]})

    with transport_client(handle) as http:
        vectors = RetrievalClient("https://models.example/v1", "test-key", client=http).embed(["first", "second"], "document")
    assert len(requests) == 2
    assert vectors[0][0] == 1 and vectors[1][1] == 1


def test_reranker_keeps_raw_scores_and_server_order():
    def handle(request):
        if request.url.path.endswith("models"):
            return httpx.Response(200, json=catalog())
        assert request.url.path == "/v1/rerank"
        assert json.loads(request.content) == {"model": RERANK_MODEL, "query": "sample", "documents": ["one", {"text": "two", "image": PNG}], "top_n": 1}
        return httpx.Response(200, json={"model": RERANK_MODEL, "model_revision": RERANK_REVISION,
                              "score_type": "raw_logit", "results": [{"index": 1, "relevance_score": -3.25}]})

    with transport_client(handle) as http:
        rows = RetrievalClient("https://models.example/v1", client=http).rerank("sample", ["one", {"text": "two", "image": PNG}], 1)
    assert rows == [{"index": 1, "relevance_score": -3.25}]


@pytest.mark.parametrize("mutation", ["revision", "dimensions", "model_id"])
def test_wrong_catalog_stops_before_inference(mutation):
    calls = []

    def handle(request):
        calls.append(request.url.path)
        data = catalog()
        if mutation == "revision":
            data["data"][0]["revision"] = "wrong"
        elif mutation == "dimensions":
            data["data"][0]["dimensions"] = 12
        else:
            data["data"][0]["id"] = "another-model"
        return httpx.Response(200, json=data)

    with transport_client(handle) as http, pytest.raises(ProtocolError):
        RetrievalClient("https://models.example/v1", client=http).embed(["sample"], "query")
    assert calls == ["/v1/models"]


@pytest.mark.parametrize("data", [
    [{"index": 0, "embedding": [1, 0]}],
    [{"index": 1, "embedding": [1] + [0] * 2047}],
    [{"index": 0, "embedding": [0] * 2048}],
    [{"index": 0, "embedding": [True] + [0] * 2047}],
])
def test_invalid_embedding_cannot_look_successful(data):
    def handle(request):
        return httpx.Response(200, json=catalog() if request.url.path.endswith("models") else {"model": EMBED_MODEL, "data": data})
    with transport_client(handle) as http, pytest.raises(ProtocolError):
        RetrievalClient("https://models.example/v1", client=http).embed(["sample"], "query")


def test_query_images_rejected_without_network():
    with transport_client(lambda request: pytest.fail("query image must fail before a request")) as http:
        with pytest.raises(ValueError, match="text inputs"):
            RetrievalClient("https://models.example/v1", client=http).embed([{"image": PNG}], "query")


@pytest.mark.parametrize("operation", ["query", "document", "rerank_query", "rerank_document", "reserved", "control"])
def test_model_service_input_limits_fail_before_network(operation):
    with transport_client(lambda request: pytest.fail("invalid model input started a request")) as http:
        client = RetrievalClient("https://models.example/v1", client=http)
        with pytest.raises(ValueError):
            if operation == "query":
                client.embed(["x" * 2001], "query")
            elif operation == "document":
                client.embed([{"text": "x" * 32001}], "document")
            elif operation == "rerank_query":
                client.rerank("x" * 2001, ["document"])
            elif operation == "rerank_document":
                client.rerank("query", ["x" * 32001])
            elif operation == "reserved":
                client.embed(["text <image>"], "document")
            else:
                client.rerank("bad\x00query", ["document"])


@pytest.mark.parametrize("field,value", [("dimensions", 12), ("normalized", False), ("model_revision", "wrong")])
def test_contradictory_embedding_metadata_is_not_success(field, value):
    def handle(request):
        if request.url.path.endswith("models"):
            return httpx.Response(200, json=catalog())
        result = {"model": EMBED_MODEL, "data": [{"index": 0, "embedding": [1] + [0] * 2047}], field: value}
        return httpx.Response(200, json=result)
    with transport_client(handle) as http, pytest.raises(ProtocolError):
        RetrievalClient("https://models.example/v1", client=http).embed(["sample"], "query")


@pytest.mark.parametrize("code", [429, 503])
def test_busy_error_preserves_retry_header_without_key_or_body(code):
    with transport_client(lambda request: httpx.Response(code, headers={"Retry-After": "2"}, json={"error": "private response body"})) as http:
        with pytest.raises(ServiceError) as caught:
            JSONClient("https://models.example/v1", "test-secret", client=http).request("GET", "models")
    assert caught.value.status_code == code
    assert caught.value.retry_after == "2"
    assert "test-secret" not in str(caught.value) and "private response body" not in str(caught.value)


def test_redirect_does_not_forward_credentials():
    calls = []
    def handle(request):
        calls.append(str(request.url))
        return httpx.Response(302, headers={"Location": "https://other.example/models"})
    with transport_client(handle) as http, pytest.raises(ServiceError):
        JSONClient("https://models.example/v1", "key", client=http).request("GET", "models")
    assert calls == ["https://models.example/v1/models"]


@pytest.mark.parametrize("response", [httpx.Response(200, content=b"not JSON"), httpx.Response(200, json={"error": "failure"})])
def test_bad_json_or_error_payload_is_not_success(response):
    with transport_client(lambda request: response) as http, pytest.raises(ProtocolError):
        JSONClient("https://models.example/v1", client=http).request("GET", "models")


def test_response_byte_limit():
    with transport_client(lambda request: httpx.Response(200, json={"data": "x" * 200})) as http, pytest.raises(ProtocolError, match="byte limit"):
        JSONClient("https://models.example/v1", client=http, max_response_bytes=64).request("GET", "models")


def test_parse_35_preserves_transport_tokens_and_raw_reply():
    requests = []
    def handle(request):
        requests.append(request)
        if request.url.path.endswith("models"):
            return httpx.Response(200, json={"data": [{"id": PARSE_MODEL}]})
        payload = json.loads(request.content)
        assert payload["model"] == PARSE_MODEL
        assert payload["messages"][0]["content"][0]["text"] == DEFAULT_PARSE_PROMPT
        assert payload["messages"][0]["content"][1]["image_url"]["url"] == PNG
        assert payload["skip_special_tokens"] is False and payload["add_special_tokens"] is False
        assert payload["max_tokens"] == 8744
        return httpx.Response(200, json={"model": PARSE_MODEL, "choices": [{"finish_reason": "stop", "message": {"content": "synthetic raw reply"}}]})
    with transport_client(handle) as http:
        result = ParseClient("https://parse.example/v1", client=http).parse(PNG)
    assert result["choices"][0]["message"]["content"] == "synthetic raw reply"
    assert len(requests) == 2


@pytest.mark.parametrize("reason,content", [("length", "partial"), ("stop", ""), ("error", "text"), (None, "text")])
def test_truncated_or_empty_chat_response_is_not_success(reason, content):
    def handle(request):
        if request.url.path.endswith("models"):
            return httpx.Response(200, json={"data": [{"id": SUPER_MODEL}]})
        return httpx.Response(200, json={"model": SUPER_MODEL, "choices": [{"finish_reason": reason, "message": {"content": content}}]})
    with transport_client(handle) as http, pytest.raises(ProtocolError):
        ChatClient("https://chat.example/v1", "test", client=http).complete([{"role": "user", "content": "sample"}])


def test_tool_call_transport_requires_caller_tools_and_valid_arguments():
    def handle(request):
        if request.url.path.endswith("models"):
            return httpx.Response(200, json={"data": [{"id": SUPER_MODEL}]})
        return httpx.Response(200, json={"model": SUPER_MODEL, "choices": [{"finish_reason": "tool_calls", "message": {"tool_calls": [
            {"id": "call-1", "type": "function", "function": {"name": "team_tool", "arguments": "{}"}}
        ]}}]})
    with transport_client(handle) as http:
        client = ChatClient("https://chat.example/v1", "test", client=http)
        with pytest.raises(ProtocolError, match="did not supply"):
            client.complete([{"role": "user", "content": "sample"}])
        result = client.complete([{"role": "user", "content": "sample"}], extra_body={"tools": [{"type": "function", "function": {"name": "team_tool"}}]})
    assert result["choices"][0]["finish_reason"] == "tool_calls"


def wiki_identity(text):
    return {"page_id": "123", "revision_id": "456", "title": "Synthetic fixture", "url": "https://en.wikipedia.org/?curid=123",
            "license": {"spdx": "CC-BY-SA-4.0"}, "provenance": {
                "dataset": "wikimedia/structured-wikipedia", "dataset_revision": WIKIPEDIA_REVISION,
                "dataset_url": "https://huggingface.co/datasets/wikimedia/structured-wikipedia",
                "source_key": "fixture", "source_row_idx": 0, "source_shard": "fixture.parquet",
                "text_sha256": hashlib.sha256(text.encode()).hexdigest(), "extraction": "fixture",
                "revision_url": "https://en.wikipedia.org/w/index.php?oldid=456"}}


def test_wikipedia_pagination_keeps_provenance_and_validates_complete_text():
    text = "Synthetic café text"
    def handle(request):
        assert request.url.path == "/v1/articles/123"
        offset = int(request.url.params["offset"])
        size = int(request.url.params["max_chars"])
        chunk = text[offset:offset + size]
        next_offset = offset + len(chunk)
        return httpx.Response(200, json=wiki_identity(text) | {"text": chunk, "offset": offset,
                   "returned_chars": len(chunk), "total_chars": len(text), "next_offset": next_offset if next_offset < len(text) else None})
    with transport_client(handle) as http:
        client = WikiClient("https://wiki.example", client=http)
        first, second = client.article("123", max_chars=10), client.article("123", offset=10, max_chars=10)
        assert first.text + second.text == text
        assert client.article("123").provenance.text_sha256 == hashlib.sha256(text.encode()).hexdigest()


def test_wikipedia_search_rejects_foreign_revision():
    identity = wiki_identity("sample")
    identity["provenance"]["dataset_revision"] = "wrong"
    payload = {"query": "sample", "retrieval": "lexical_bm25", "corpus_revision": WIKIPEDIA_REVISION, "count": 1,
               "hits": [identity | {"rank": 1, "bm25_score": 1.0, "excerpt": "sample", "excerpt_offset": 0, "total_chars": 6}]}
    with transport_client(lambda request: httpx.Response(200, json=payload)) as http, pytest.raises(ProtocolError, match="source differs"):
        WikiClient("https://wiki.example", client=http).search("sample")


def test_fixture_wikipedia_is_not_production():
    with transport_client(lambda request: httpx.Response(200, json={"status": "ready", "production": False})) as http, pytest.raises(ProtocolError):
        WikiClient("https://wiki.example", client=http).corpus()
