"""Fetch the frozen source scans. This module does not extract page content."""
from __future__ import annotations

import hashlib
from importlib.resources import files
import json
import os
from pathlib import Path
import re
import tempfile
from typing import Any

DATASET = "NationalLibraryOfScotland/encyclopaedia_britannica_illustrated"
REVISION = "309f765b1ecc83e0a7f7762e44b4a927a7555443"


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while block := stream.read(8 * 1024 * 1024):
            digest.update(block)
    return digest.hexdigest()


def load_manifest() -> dict[str, Any]:
    resource = files("nemotron_challenge").joinpath("resources/britannica-manifest.json")
    manifest = json.loads(resource.read_text(encoding="utf-8"))
    unsigned = {key: value for key, value in manifest.items() if key != "manifest_sha256"}
    digest = hashlib.sha256(json.dumps(unsigned, sort_keys=True, ensure_ascii=False,
                                       separators=(",", ":")).encode()).hexdigest()
    if digest != manifest.get("manifest_sha256"):
        raise ValueError("The sample manifest hash does not match.")
    if manifest.get("dataset") != DATASET or manifest.get("revision") != REVISION:
        raise ValueError("The sample source does not match the event snapshot.")
    pages = manifest.get("pages", [])
    if manifest.get("sample_size") != 100 or len(pages) != 100:
        raise ValueError("The event sample must contain 100 scans.")
    indices = [page["row_index"] for page in pages]
    if len(set(indices)) != 100 or any(type(i) is not int or i < 0 or i >= manifest["source_rows"] for i in indices):
        raise ValueError("The source row indices are invalid.")
    if len({page["filename"] for page in pages}) != 100:
        raise ValueError("The scan filenames must be unique.")
    for page in pages:
        if not re.fullmatch(r"page-[0-9]{4}-[0-9a-f]{12}\.(jpg|png|webp)", page["filename"]):
            raise ValueError("A scan filename is invalid.")
        if not re.fullmatch(r"[0-9a-f]{64}", page["image_sha256"]):
            raise ValueError("A scan checksum is invalid.")
        if page["page_id"] != f"britannica:{REVISION}:train:{page['row_index']}":
            raise ValueError("A scan identifier does not match its source row.")
    return manifest


def fetch_sample(output_dir: Path, cache_dir: Path) -> dict[str, Any]:
    """Download the pinned source and verify all 100 encoded scan images."""
    from huggingface_hub import hf_hub_download
    import pyarrow.parquet as pq

    manifest = load_manifest()
    output_dir, cache_dir = Path(output_dir), Path(cache_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    cache_dir.mkdir(parents=True, exist_ok=True)
    source = Path(hf_hub_download(repo_id=DATASET, repo_type="dataset",
                               revision=REVISION, filename=manifest["source_file"],
                               cache_dir=str(cache_dir), token=False))
    if source.stat().st_size != manifest["source_file_bytes"] or file_sha256(source) != manifest["source_file_sha256"]:
        raise ValueError("The downloaded Parquet source failed its checksum check.")
    table = pq.read_table(source, columns=["image", "id"])
    if table.num_rows != manifest["source_rows"]:
        raise ValueError("The source scan count does not match the manifest.")
    rows = table.take([page["row_index"] for page in manifest["pages"]]).to_pylist()
    total_bytes = 0
    for page, row in zip(manifest["pages"], rows, strict=True):
        encoded = row["image"]["bytes"]
        if str(row["id"]) != page["volume_id"] or len(encoded) != page["image_bytes"]:
            raise ValueError(f"Source metadata differs for {page['page_id']}.")
        if hashlib.sha256(encoded).hexdigest() != page["image_sha256"]:
            raise ValueError(f"The scan checksum differs for {page['page_id']}.")
        destination = output_dir / page["filename"]
        if not destination.is_file() or file_sha256(destination) != page["image_sha256"]:
            temporary = None
            try:
                with tempfile.NamedTemporaryFile(dir=output_dir, prefix=".scan-", delete=False) as stream:
                    temporary = Path(stream.name)
                    stream.write(encoded)
                os.replace(temporary, destination)
            finally:
                if temporary is not None:
                    temporary.unlink(missing_ok=True)
        total_bytes += len(encoded)
    return {"status": "verified", "sample_size": len(rows), "image_bytes": total_bytes,
            "dataset_revision": REVISION, "manifest_sha256": manifest["manifest_sha256"],
            "output_dir": str(output_dir)}
