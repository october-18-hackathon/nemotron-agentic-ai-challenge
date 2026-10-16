"""CPU transport tests with a stub backend. No model quality or GPU proof."""
import base64
from dataclasses import replace
from io import BytesIO
import json
from pathlib import Path
import sys
import threading

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from fastapi.testclient import TestClient
from PIL import Image
import pytest

from app import Settings, create_app
from backend import GPUCapacityError
import contracts
from contracts import InputError
from download_models import MANIFEST, PINS, file_inventory, verify_snapshot
from model_pins import DIMENSIONS, EMBED_MODEL, EMBED_REVISION, RERANK_MODEL, RERANK_REVISION


def image_url(format="PNG", *, animated=False):
    with BytesIO() as data:
        image = Image.new("RGB", (64, 64), "white")
        if animated:
            other = Image.new("RGB", (64, 64), "black")
            image.save(data, format=format, save_all=True, append_images=[other], duration=10)
            other.close()
        else:
            image.save(data, format=format)
        image.close()
        return f"data:image/{format.lower()};base64," + base64.b64encode(data.getvalue()).decode()


class StubBackend:
    proof = {"backend": "cpu_stub", "production": False}

    def __init__(self):
        self.calls = []
        self.closed = False

    def load(self):
        pass

    def warmup(self):
        pass

    def close(self):
        self.closed = True

    def embeddings(self, documents, input_type):
        self.calls.append((input_type, [(doc.text, doc.image.mode if doc.image else None) for doc in documents]))
        return [[1.0] + [0.0] * (DIMENSIONS - 1) for _ in documents]

    def rerank(self, query, documents):
        self.calls.append((query, [(doc.text, doc.image.mode if doc.image else None) for doc in documents]))
        return [-2.0, 3.0, 3.0][:len(documents)]


@pytest.fixture
def service():
    backend = StubBackend()
    with TestClient(create_app(backend_factory=lambda: backend)) as client:
        yield client, backend
    assert backend.closed


def test_catalog_and_readiness_report_fixed_pins_and_stub_proof(service):
    client, _ = service
    assert client.get("/health").json() == {"status": "ok"}
    ready = client.get("/ready").json()
    assert ready["runtime"]["production"] is False
    rows = client.get("/v1/models").json()["data"]
    assert rows == [
        {"id": EMBED_MODEL, "revision": EMBED_REVISION, "role": "embedder", "dimensions": 2048},
        {"id": RERANK_MODEL, "revision": RERANK_REVISION, "role": "reranker"},
    ]


def test_text_image_and_combined_embeddings_keep_input_indices(service):
    client, backend = service
    response = client.post("/v1/embeddings", json={
        "model": EMBED_MODEL, "input_type": "document",
        "input": ["setup text", {"image": image_url()}, {"text": "setup text", "image": image_url("JPEG")}],
    })
    assert response.status_code == 200, response.text
    body = response.json()
    assert body["model_revision"] == EMBED_REVISION
    assert body["normalized"] is True and body["dimensions"] == 2048
    assert [row["index"] for row in body["data"]] == [0, 1, 2]
    assert all(len(row["embedding"]) == 2048 for row in body["data"])
    assert backend.calls == [("document", [("setup text", None), ("", "RGB"), ("setup text", "RGB")])]


def test_query_embeddings_accept_text_records_and_strings(service):
    client, backend = service
    response = client.post("/v1/embeddings", json={"model": EMBED_MODEL, "input_type": "query",
                                                 "input": ["setup query", {"text": "another query"}]})
    assert response.status_code == 200
    assert backend.calls[0][0] == "query"


def test_rerank_returns_raw_logits_sorted_with_source_indices(service):
    client, backend = service
    response = client.post("/v1/rerank", json={"model": RERANK_MODEL, "query": "setup query",
                                              "documents": ["a", {"image": image_url()},
                                                            {"text": "c", "image": image_url()}], "top_n": 2})
    assert response.status_code == 200, response.text
    assert response.json() == {"model": RERANK_MODEL, "model_revision": RERANK_REVISION,
                               "score_type": "raw_logit", "results": [
                                   {"index": 1, "relevance_score": 3.0}, {"index": 2, "relevance_score": 3.0}]}


@pytest.mark.parametrize("changes", [
    {"model": RERANK_MODEL}, {"input": []}, {"input": ["x"] * 17}, {"input": [" "]},
    {"input": ["x" * 32001]}, {"input_type": "other"}, {"input": [{}]},
    {"input": [{"text": "x", "url": "http://169.254.169.254/"}]},
    {"input": [{"image": "https://example.com/image.png"}]},
    {"input": [{"image": "/etc/passwd"}]}, {"input": ["<IMG_CONTEXT>"]},
    {"model_revision": EMBED_REVISION},
])
def test_embedding_constraints_do_not_reach_inference(service, changes):
    client, backend = service
    payload = {"model": EMBED_MODEL, "input_type": "document", "input": ["x"]} | changes
    assert client.post("/v1/embeddings", json=payload).status_code == 422
    assert backend.calls == []
    assert client.get("/ready").status_code == 200


@pytest.mark.parametrize("item", [{"image": image_url()}, "x" * 2001])
def test_query_images_and_large_query_text_are_rejected(service, item):
    client, backend = service
    assert client.post("/v1/embeddings", json={"model": EMBED_MODEL, "input_type": "query", "input": [item]}).status_code == 422
    assert not backend.calls


@pytest.mark.parametrize("changes", [
    {"query": " "}, {"query": "x" * 2001}, {"documents": []}, {"documents": ["x"] * 33},
    {"top_n": 0}, {"top_n": 3}, {"top_n": True}, {"top_n": "1"}, {"query": "<image>"},
    {"model": EMBED_MODEL}, {"documents": [" "]},
])
def test_rerank_constraints(service, changes):
    client, backend = service
    payload = {"model": RERANK_MODEL, "query": "setup", "documents": ["a", "b"]} | changes
    assert client.post("/v1/rerank", json=payload).status_code == 422
    assert not backend.calls


@pytest.mark.parametrize("data_url", [
    "data:image/png;base64,====", "data:image/png;base64,AA==",
    image_url().replace("data:image/png", "data:image/jpeg"), image_url(animated=True),
])
def test_bad_images_keep_readiness_and_never_reach_inference(service, data_url):
    client, backend = service
    response = client.post("/v1/embeddings", json={"model": EMBED_MODEL, "input_type": "document",
                                                 "input": [{"image": data_url}]})
    assert response.status_code == 422
    assert not backend.calls
    assert client.get("/ready").status_code == 200


def test_png_parser_syntax_error_is_a_client_error(service):
    client, backend = service
    # This PNG header opens, but a broken later chunk raises Pillow SyntaxError.
    broken = ("iVBORw0KGgoAAAANSUhEUgAAAEAAAABACAIAAAAlC+aJAAAAMUlEQVR4nNXOQREAIADDsFL/nocIHlyjIGcbZRIncRIncRIncRIncRLFcRIncRIncRIncRIncRIncRIncRIncRIncRIncRIncRIncRIncRIncRIncRIncRInbRIncRIncRIncRIncRIncRIncRIncRIncRIncRIncRIncRInFP4OvLpyqgN9ZSiDrgAAAABJRU5ErkJggg==")
    response = client.post("/v1/embeddings", json={"model": EMBED_MODEL, "input_type": "document",
                                                 "input": [{"image": "data:image/png;base64," + broken}]})
    assert response.status_code == 422
    assert not backend.calls
    assert client.get("/ready").status_code == 200
    assert client.post("/v1/embeddings", json={"model": EMBED_MODEL, "input_type": "query", "input": ["setup"]}).status_code == 200


def test_image_pixel_bound_before_inference(service, monkeypatch):
    client, backend = service
    monkeypatch.setattr(contracts, "MAX_IMAGE_PIXELS", 1000)
    response = client.post("/v1/embeddings", json={"model": EMBED_MODEL, "input_type": "document",
                                                 "input": [{"image": image_url()}]})
    assert response.status_code == 422
    assert not backend.calls


def test_total_image_pixel_bound(service, monkeypatch):
    client, backend = service
    monkeypatch.setattr(contracts, "MAX_BATCH_IMAGE_PIXELS", 6000)
    response = client.post("/v1/rerank", json={"model": RERANK_MODEL, "query": "setup",
                                              "documents": [{"image": image_url()}, {"image": image_url()}]})
    assert response.status_code == 422
    assert not backend.calls


def test_unicode_error_response_does_not_echo_invalid_raw_input(service):
    client, backend = service
    response = client.post("/v1/embeddings", content=(
        '{"model":"' + EMBED_MODEL + '","input_type":"query","input":["a\\ud800"]}').encode(),
        headers={"Content-Type": "application/json"})
    assert response.status_code == 422
    assert response.json()["detail"]
    assert not backend.calls


def test_body_limit_includes_chunked_requests():
    backend = StubBackend()
    with TestClient(create_app(Settings(max_body_bytes=64), lambda: backend)) as client:
        assert client.post("/v1/embeddings", content=b"x" * 65).status_code == 413
        assert client.post("/v1/embeddings", content=iter([b"x" * 40, b"x" * 40])).status_code == 413
        assert not backend.calls


@pytest.mark.parametrize("failure", ["load", "warmup"])
def test_startup_failure_never_reports_ready(failure):
    backend = StubBackend()
    def fail():
        raise RuntimeError("setup failed")
    setattr(backend, failure, fail)
    with TestClient(create_app(backend_factory=lambda: backend)) as client:
        assert client.get("/health").status_code == 200
        assert client.get("/ready").status_code == 503
        assert client.get("/v1/models").status_code == 503
        assert client.post("/v1/embeddings", json={"model": EMBED_MODEL, "input_type": "query", "input": ["x"]}).status_code == 503


def test_input_and_gpu_capacity_errors_release_the_slot(service):
    client, backend = service
    original = backend.embeddings
    for failure, expected in [(InputError("token limit"), 422), (GPUCapacityError("GPU memory"), 503)]:
        def fail(*args):
            raise failure
        backend.embeddings = fail
        response = client.post("/v1/embeddings", json={"model": EMBED_MODEL, "input_type": "query", "input": ["x"]})
        assert response.status_code == expected
        assert client.get("/ready").status_code == 200
        backend.embeddings = original
        assert client.post("/v1/embeddings", json={"model": EMBED_MODEL, "input_type": "query", "input": ["x"]}).status_code == 200


@pytest.mark.parametrize("bad_vectors", [[[0.0] * 2048], [[float("nan")] * 2048], [[1.0]], []])
def test_invalid_model_output_clears_readiness(service, bad_vectors):
    client, backend = service
    backend.embeddings = lambda *args: bad_vectors
    assert client.post("/v1/embeddings", json={"model": EMBED_MODEL, "input_type": "query", "input": ["x"]}).status_code == 503
    assert client.get("/ready").status_code == 503


def test_one_queue_serializes_both_models_and_returns_retry_after():
    backend = StubBackend()
    entered, release = threading.Event(), threading.Event()
    original = backend.embeddings
    def slow(*args):
        entered.set()
        assert release.wait(timeout=3)
        return original(*args)
    backend.embeddings = slow
    with TestClient(create_app(replace(Settings(), queue_wait_seconds=0.05), lambda: backend)) as client:
        results = []
        first = threading.Thread(target=lambda: results.append(client.post("/v1/embeddings", json={
            "model": EMBED_MODEL, "input_type": "query", "input": ["x"]})))
        first.start()
        assert entered.wait(timeout=1)
        second = client.post("/v1/rerank", json={"model": RERANK_MODEL, "query": "x", "documents": ["x"]})
        assert second.status_code == 429
        assert second.headers["retry-after"] == "1"
        assert not backend.calls
        release.set()
        first.join(timeout=3)
        assert not first.is_alive() and results[0].status_code == 200
        assert client.post("/v1/rerank", json={"model": RERANK_MODEL, "query": "x", "documents": ["x"]}).status_code == 200


def test_snapshot_hash_check_detects_changes(tmp_path):
    directory = tmp_path / "embed"
    directory.mkdir()
    for name in ("model.safetensors", "config.json", "LICENSE", "THIRD_PARTY_NOTICES.md", "README.md",
                 "modeling_llama_nemotron_vl.py", "processing_llama_nemotron_vl.py"):
        (directory / name).write_text("CPU hash fixture; this is not a model.")
    repo, revision = PINS["embed"]
    (directory / MANIFEST).write_text(json.dumps({"repo_id": repo, "revision": revision, "files": file_inventory(directory)}))
    verify_snapshot(directory, "embed")
    (directory / "model.safetensors").write_text("changed")
    with pytest.raises(RuntimeError, match="files differ"):
        verify_snapshot(directory, "embed")
