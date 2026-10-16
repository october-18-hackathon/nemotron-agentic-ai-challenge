"""Download exact public snapshots and record every model file hash."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

from model_pins import EMBED_MODEL, EMBED_REVISION, RERANK_MODEL, RERANK_REVISION

PINS = {"embed": (EMBED_MODEL, EMBED_REVISION), "rerank": (RERANK_MODEL, RERANK_REVISION)}
MANIFEST = "snapshot-manifest.json"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while block := stream.read(4 * 1024 * 1024):
            digest.update(block)
    return digest.hexdigest()


def file_inventory(directory: Path) -> dict[str, str]:
    return {str(path.relative_to(directory)): sha256(path)
            for path in sorted(directory.rglob("*"))
            if path.is_file() and ".cache" not in path.relative_to(directory).parts and path.name != MANIFEST}


def verify_snapshot(directory: Path, role: str) -> None:
    state = json.loads((directory / MANIFEST).read_text())
    repo, revision = PINS[role]
    if state.get("repo_id") != repo or state.get("revision") != revision:
        raise RuntimeError(f"{role} model pin differs from the image contract")
    files = state.get("files", {})
    for required in ("model.safetensors", "config.json", "LICENSE", "THIRD_PARTY_NOTICES.md", "README.md",
                     "modeling_llama_nemotron_vl.py", "processing_llama_nemotron_vl.py"):
        if required not in files:
            raise RuntimeError(f"{role} snapshot lacks {required}")
    if file_inventory(directory) != files:
        raise RuntimeError(f"{role} model files differ from the download manifest")


def download(destination: Path) -> None:
    from huggingface_hub import snapshot_download

    destination.mkdir(parents=True, exist_ok=True)
    for role, (repo, revision) in PINS.items():
        directory = destination / role
        snapshot_download(repo_id=repo, revision=revision, local_dir=directory, token=False, max_workers=4)
        (directory / MANIFEST).write_text(json.dumps(
            {"repo_id": repo, "revision": revision, "files": file_inventory(directory)}, indent=2) + "\n")
        verify_snapshot(directory, role)
        print(f"Verified {repo} at {revision}", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--destination", type=Path, default=Path("/opt/models"))
    download(parser.parse_args().destination)
