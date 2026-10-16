"""Bounded input records and image decoding. No network image access."""
from __future__ import annotations

import base64
import binascii
from dataclasses import dataclass
from io import BytesIO
import re
from typing import Annotated, Literal
import warnings

from PIL import Image
from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from model_pins import EMBED_MODEL, RERANK_MODEL

MAX_TEXT_CHARS = 32_000
MAX_QUERY_CHARS = 2_000
MAX_IMAGE_BYTES = 8 * 1024 * 1024
MAX_IMAGE_PIXELS = 20_000_000
MAX_IMAGE_SIDE = 10_000
MAX_BATCH_IMAGE_PIXELS = 80_000_000
MAX_BODY_BYTES = 24 * 1024 * 1024
DATA_IMAGE = re.compile(r"^data:image/(png|jpeg);base64,([A-Za-z0-9+/]*={0,2})$")


def valid_text(value: str) -> str:
    if not value.strip():
        raise ValueError("text must contain a word, number, or symbol")
    try:
        value.encode("utf-8")
    except UnicodeEncodeError:
        raise ValueError("text must contain valid UTF-8") from None
    if any(ord(char) < 32 and char not in "\n\r\t" for char in value):
        raise ValueError("text contains an unsupported control character")
    if any(marker in value for marker in ("<image>", "<img>", "</img>", "<IMG_CONTEXT>")):
        raise ValueError("text contains reserved image control tokens")
    return value


def valid_data_image(value: str) -> str:
    # Check encoded size before the regex and base64 allocation.
    if len(value) > 4 * ((MAX_IMAGE_BYTES + 2) // 3) + 64:
        raise ValueError("image exceeds 8 MiB")
    match = DATA_IMAGE.fullmatch(value)
    if match is None or not match[2]:
        raise ValueError("image must be a PNG or JPEG base64 data URL")
    # Pixel checks run inside the bounded worker, before GPU inference.
    return value


Text = Annotated[str, Field(min_length=1, max_length=MAX_TEXT_CHARS)]


class Document(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)
    text: Text | None = None
    image: str | None = None

    @field_validator("text")
    @classmethod
    def text_is_valid(cls, value: str | None) -> str | None:
        return valid_text(value) if value is not None else None

    @field_validator("image")
    @classmethod
    def image_is_data_url(cls, value: str | None) -> str | None:
        return valid_data_image(value) if value is not None else None

    @model_validator(mode="after")
    def has_content(self) -> Document:
        if self.text is None and self.image is None:
            raise ValueError("document must contain text or an image")
        return self


Item = Text | Document


class EmbeddingRequest(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)
    model: Literal[EMBED_MODEL]
    input: list[Item] = Field(min_length=1, max_length=16)
    input_type: Literal["query", "document"]

    @model_validator(mode="after")
    def validate_items(self) -> EmbeddingRequest:
        for item in self.input:
            text = item if isinstance(item, str) else item.text
            if text is not None:
                valid_text(text)
            if self.input_type == "query":
                if isinstance(item, Document) and item.image is not None:
                    raise ValueError("query embeddings accept text")
                if text is None or len(text) > MAX_QUERY_CHARS:
                    raise ValueError("query text must contain 1 to 2000 characters")
        return self


class RerankRequest(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)
    model: Literal[RERANK_MODEL]
    query: str = Field(min_length=1, max_length=MAX_QUERY_CHARS)
    documents: list[Item] = Field(min_length=1, max_length=32)
    top_n: int | None = Field(default=None, ge=1, le=32)

    @field_validator("query")
    @classmethod
    def query_is_text(cls, value: str) -> str:
        return valid_text(value)

    @model_validator(mode="after")
    def validate_documents(self) -> RerankRequest:
        for item in self.documents:
            if isinstance(item, str):
                valid_text(item)
        if self.top_n is not None and self.top_n > len(self.documents):
            raise ValueError("top_n must not exceed the document count")
        return self


@dataclass
class DecodedDocument:
    text: str
    image: Image.Image | None


class InputError(ValueError):
    """The request has invalid image or token content."""


def decode_image(data_url: str) -> Image.Image:
    match = DATA_IMAGE.fullmatch(data_url)
    if match is None:
        raise InputError("image must be a PNG or JPEG base64 data URL")
    try:
        raw = base64.b64decode(match[2], validate=True)
    except (ValueError, binascii.Error):
        raise InputError("image base64 is invalid") from None
    if len(raw) > MAX_IMAGE_BYTES:
        raise InputError("image exceeds 8 MiB")
    expected = "PNG" if match[1] == "png" else "JPEG"
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("error", Image.DecompressionBombWarning)
            with Image.open(BytesIO(raw)) as image:
                width, height = image.size
                if (width < 1 or height < 1 or width * height > MAX_IMAGE_PIXELS
                        or max(width, height) > MAX_IMAGE_SIDE):
                    raise InputError("image exceeds the pixel or side limit")
                if image.format != expected:
                    raise InputError("image bytes do not match the data URL format")
                if getattr(image, "n_frames", 1) != 1:
                    raise InputError("image must have one frame")
                image.load()
                return image.convert("RGB")
    except InputError:
        raise
    except Exception:
        # Pillow can raise SyntaxError and other parser errors for file bytes.
        # A bad image remains an input error; it must not disable the models.
        raise InputError("image cannot be decoded within the image limits") from None


def decode_items(items: list[Item]) -> list[DecodedDocument]:
    result = []
    try:
        for item in items:
            if isinstance(item, str):
                result.append(DecodedDocument(item, None))
            else:
                result.append(DecodedDocument(item.text or "", decode_image(item.image) if item.image else None))
            if sum(document.image.width * document.image.height for document in result if document.image) > MAX_BATCH_IMAGE_PIXELS:
                raise InputError("images exceed 80 million pixels in one request")
        return result
    except BaseException:
        close_images(result)
        raise


def close_images(items: list[DecodedDocument]) -> None:
    for item in items:
        if item.image is not None:
            item.image.close()
