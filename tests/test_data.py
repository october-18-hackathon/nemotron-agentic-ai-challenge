"""Source integrity checks. These tests do not measure extraction quality."""
from copy import deepcopy
import hashlib
import json
from pathlib import Path

import huggingface_hub
import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from nemotron_challenge import data


def sign(manifest):
    unsigned = {key: value for key, value in manifest.items() if key != "manifest_sha256"}
    manifest["manifest_sha256"] = hashlib.sha256(
        json.dumps(unsigned, sort_keys=True, ensure_ascii=False, separators=(",", ":")).encode()
    ).hexdigest()
    return manifest


def resource(monkeypatch, tmp_path, manifest):
    target = tmp_path / "resources"
    target.mkdir()
    (target / "britannica-manifest.json").write_text(json.dumps(manifest))
    monkeypatch.setattr(data, "files", lambda _: tmp_path)


def test_packaged_sample_has_the_event_identity():
    manifest = data.load_manifest()
    assert manifest["sample_size"] == 100
    assert manifest["source_rows"] == 2573
    assert manifest["manifest_sha256"] == "779542e078756874f23fa6971d7d8a2d8635ad6d9d9484e1deb607d94a28e396"
    assert sum(page["image_bytes"] for page in manifest["pages"]) == 14_612_248


def test_corrupt_manifest_is_rejected(monkeypatch, tmp_path):
    manifest = deepcopy(data.load_manifest())
    manifest["pages"][0]["volume_id"] = "changed"
    resource(monkeypatch, tmp_path, manifest)
    with pytest.raises(ValueError, match="hash"):
        data.load_manifest()


@pytest.mark.parametrize("fault", ["duplicate_row", "duplicate_filename", "path", "identity", "revision"])
def test_invalid_source_coordinates_are_rejected(monkeypatch, tmp_path, fault):
    manifest = deepcopy(data.load_manifest())
    first, second = manifest["pages"][:2]
    if fault == "duplicate_row":
        second["row_index"] = first["row_index"]
    elif fault == "duplicate_filename":
        second["filename"] = first["filename"]
    elif fault == "path":
        first["filename"] = "../outside.jpg"
    elif fault == "identity":
        first["page_id"] = "britannica:wrong"
    else:
        manifest["revision"] = "0" * 40
    resource(monkeypatch, tmp_path, sign(manifest))
    with pytest.raises(ValueError):
        data.load_manifest()


@pytest.fixture
def source_fixture(monkeypatch, tmp_path):
    # Encoded bytes need no image decoder for this source-preservation test.
    blobs = [b"scan bytes one", b"scan bytes two"]
    source = tmp_path / "source.parquet"
    pq.write_table(pa.Table.from_pylist([
        {"image": {"bytes": blob, "path": None}, "id": "shared-volume"} for blob in blobs
    ]), source)
    pages = [{"row_index": i, "filename": f"scan-{i}.jpg", "page_id": f"fixture:{i}",
              "volume_id": "shared-volume", "image_bytes": len(blob),
              "image_sha256": hashlib.sha256(blob).hexdigest()} for i, blob in enumerate(blobs)]
    manifest = {"source_file": "fixture.parquet", "source_file_bytes": source.stat().st_size,
                "source_file_sha256": data.file_sha256(source), "source_rows": 2,
                "pages": pages, "manifest_sha256": "fixture"}
    monkeypatch.setattr(data, "load_manifest", lambda: manifest)

    def download(**kwargs):
        assert kwargs["revision"] == data.REVISION
        assert kwargs["token"] is False
        assert kwargs["repo_type"] == "dataset"
        return str(source)

    monkeypatch.setattr(huggingface_hub, "hf_hub_download", download)
    return source, manifest, blobs


def test_fetch_preserves_source_and_repairs_bad_output(source_fixture, tmp_path):
    source, manifest, blobs = source_fixture
    before = source.read_bytes()
    destination = tmp_path / "scans"
    result = data.fetch_sample(destination, tmp_path / "cache")
    assert result["sample_size"] == 2
    assert source.read_bytes() == before
    valid = destination / manifest["pages"][0]["filename"]
    unchanged = valid.stat().st_mtime_ns
    damaged = destination / manifest["pages"][1]["filename"]
    damaged.write_bytes(b"damaged")
    data.fetch_sample(destination, tmp_path / "cache")
    assert valid.stat().st_mtime_ns == unchanged
    assert damaged.read_bytes() == blobs[1]
    assert not list(destination.glob(".scan-*"))


def test_bad_source_hash_prevents_output(source_fixture, tmp_path):
    source, _, _ = source_fixture
    source.write_bytes(b"damaged source")
    destination = tmp_path / "scans"
    with pytest.raises(ValueError, match="checksum"):
        data.fetch_sample(destination, tmp_path / "cache")
    assert not list(destination.iterdir())


def test_wrong_volume_prevents_scan_write(source_fixture, tmp_path):
    _, manifest, _ = source_fixture
    manifest["pages"][0]["volume_id"] = "wrong-volume"
    destination = tmp_path / "scans"
    with pytest.raises(ValueError, match="metadata"):
        data.fetch_sample(destination, tmp_path / "cache")
    assert not list(destination.iterdir())
