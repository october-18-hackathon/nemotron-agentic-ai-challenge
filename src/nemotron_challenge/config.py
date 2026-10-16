"""Read endpoint settings without starting requests or printing secrets."""
from __future__ import annotations

from dataclasses import dataclass, field
import os
from pathlib import Path
from urllib.parse import urlsplit

from dotenv import load_dotenv

WIKIPEDIA_REVISION = "417c267bb457fa645c22eb3b5c77764963194c70"
PARSE_MODEL = "nvidia/nemotron-3.5-parse"
EMBED_MODEL = "nvidia/llama-nemotron-embed-vl-1b-v2"
EMBED_REVISION = "582e3bf72aee355e3c59ed89de53543c5b0657ee"
RERANK_MODEL = "nvidia/llama-nemotron-rerank-vl-1b-v2"
RERANK_REVISION = "b8a9987b05b75db5ad949c825cbcbf6eb7c48b5a"
SUPER_MODEL = "nvidia/Nemotron-3.5-Super-VL"
DEFAULT_PARSE_PROMPT = "</s><s><predict_bbox><predict_classes><output_markdown><predict_text_in_pic>"
EMBED_DIMENSIONS = 2048
DEFAULT_WIKIPEDIA_URL = "https://port8000-rks0xavparpzq59.tunnel.applications.uk-south2.nebius.cloud"
DEFAULT_PARSE_URL = "https://port8000-kh2kceraevwe0k1.tunnel.applications.uk-south2.nebius.cloud/v1"


class ConfigurationError(ValueError):
    """A required setting is missing or invalid."""


def endpoint(value: str, name: str) -> str:
    if not value:
        return ""
    parts = urlsplit(value)
    if parts.scheme not in {"http", "https"} or not parts.netloc or parts.username or parts.password:
        raise ConfigurationError(f"{name} must be an HTTP(S) URL without credentials")
    if parts.query or parts.fragment:
        raise ConfigurationError(f"{name} must not contain a query or fragment")
    return value.rstrip("/")


@dataclass(frozen=True)
class Settings:
    wikipedia_api_url: str = DEFAULT_WIKIPEDIA_URL
    parse_base_url: str = DEFAULT_PARSE_URL
    parse_model: str = PARSE_MODEL
    parse_api_key: str = field(default="", repr=False)
    retrieval_base_url: str = ""
    retrieval_api_key: str = field(default="", repr=False)
    super_base_url: str = "https://api.tokenfactory.nebius.com/v1"
    super_model: str = SUPER_MODEL
    super_api_key: str = field(default="", repr=False)
    db_path: Path = Path("data/team.lancedb")

    @classmethod
    def from_env(cls, env_file: str | Path | None = ".env") -> Settings:
        if env_file is not None:
            load_dotenv(env_file, override=False)
        get = lambda name, default="": os.environ.get(name, default).strip()
        model = get("PARSE_MODEL", PARSE_MODEL)
        if model != PARSE_MODEL:
            raise ConfigurationError(f"PARSE_MODEL must be {PARSE_MODEL}")
        super_model = get("SUPER_MODEL", SUPER_MODEL)
        if super_model and super_model != SUPER_MODEL:
            raise ConfigurationError(f"SUPER_MODEL must be {SUPER_MODEL}")
        return cls(
            wikipedia_api_url=endpoint(get("WIKIPEDIA_API_URL", DEFAULT_WIKIPEDIA_URL), "WIKIPEDIA_API_URL"),
            parse_base_url=endpoint(get("PARSE_BASE_URL", DEFAULT_PARSE_URL), "PARSE_BASE_URL"),
            parse_model=model, parse_api_key=get("PARSE_API_KEY"),
            retrieval_base_url=endpoint(get("RETRIEVAL_BASE_URL"), "RETRIEVAL_BASE_URL"),
            retrieval_api_key=get("RETRIEVAL_API_KEY"),
            super_base_url=endpoint(get("SUPER_BASE_URL", cls.super_base_url), "SUPER_BASE_URL"),
            super_model=super_model, super_api_key=get("SUPER_API_KEY"),
            db_path=Path(get("CHALLENGE_DB_PATH", "data/team.lancedb")),
        )
