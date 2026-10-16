"""GPU inference through the public model APIs. No retrieval policy."""
from __future__ import annotations

import importlib.metadata
import os
from pathlib import Path
from typing import Any

from contracts import DecodedDocument, InputError
from model_pins import DIMENSIONS


class GPUCapacityError(RuntimeError):
    """A request did not fit in GPU memory."""


class TorchBackend:
    def __init__(self, model_root: Path = Path("/opt/models")):
        self.model_root = model_root
        self.proof: dict[str, Any] = {}

    def load(self) -> None:
        # No Hub access occurs after image creation. The model has custom code;
        # its exact files are checked before trust_remote_code loads them.
        os.environ["HF_HUB_OFFLINE"] = "1"
        os.environ["TRANSFORMERS_OFFLINE"] = "1"
        from download_models import verify_snapshot
        import torch
        from transformers import AutoModel, AutoModelForSequenceClassification, AutoProcessor

        if not torch.cuda.is_available():
            raise RuntimeError("an NVIDIA CUDA GPU is required")
        if not torch.cuda.is_bf16_supported():
            raise RuntimeError("the GPU must support BF16")
        if torch.__version__.split("+")[0] != "2.8.0" or torch.version.cuda != "12.8":
            raise RuntimeError("use the pinned PyTorch 2.8.0 CUDA 12.8 image")
        if importlib.metadata.version("transformers") != "4.56.2":
            raise RuntimeError("use Transformers 4.56.2 from the service lock")
        capability = torch.cuda.get_device_capability(0)
        compiled = torch.cuda.get_arch_list()
        if capability == (12, 0) and "sm_120" not in compiled:
            raise RuntimeError("this PyTorch build lacks sm_120 kernels")
        self.torch = torch
        verify_snapshot(self.model_root / "embed", "embed")
        verify_snapshot(self.model_root / "rerank", "rerank")
        kwargs = {"local_files_only": True, "trust_remote_code": True,
                  "torch_dtype": torch.bfloat16, "attn_implementation": "sdpa",
                  "use_safetensors": True, "low_cpu_mem_usage": True}
        self.embedder = AutoModel.from_pretrained(str(self.model_root / "embed"), **kwargs).eval().to("cuda:0")
        self.reranker = AutoModelForSequenceClassification.from_pretrained(
            str(self.model_root / "rerank"), **kwargs).eval().to("cuda:0")
        self.processor = AutoProcessor.from_pretrained(
            str(self.model_root / "rerank"), local_files_only=True, trust_remote_code=True,
            max_input_tiles=6, use_thumbnail=True, rerank_max_length=10240)
        self.embedder.processor.q_max_length = 512
        self.embedder.processor.p_max_length = 10240
        self.embedder.processor.max_input_tiles = 6
        self.embedder.processor.use_thumbnail = True
        self.proof = {
            "backend": "torch_cuda", "production": True, "gpu": torch.cuda.get_device_name(0),
            "gpu_memory_bytes": torch.cuda.get_device_properties(0).total_memory,
            "compute_capability": list(capability), "compiled_arches": compiled,
            "torch": torch.__version__, "cuda": torch.version.cuda,
            "transformers": importlib.metadata.version("transformers"),
            "precision": "bfloat16", "attention": "sdpa", "microbatch_size": 1,
            "weights_verified": True,
        }

    @staticmethod
    def _check_length(batch: dict, limit: int) -> None:
        if batch["input_ids"].shape[-1] > limit:
            raise InputError(f"input exceeds the {limit}-token model limit; split the input")

    def embeddings(self, documents: list[DecodedDocument], input_type: str) -> list[list[float]]:
        torch = self.torch
        rows = []
        try:
            with torch.inference_mode():
                # One item per GPU microbatch bounds memory across modalities.
                for item in documents:
                    if input_type == "query":
                        prepared = self.embedder.processor.process_queries([item.text], truncation=False)
                        self._check_length(prepared, 512)
                        del prepared
                        values = self.embedder.encode_queries([item.text])
                    else:
                        example = {"image": item.image, "text": item.text}
                        prepared = self.embedder.processor.process_documents([example], truncation=False)
                        limit = 10240 if item.image is not None and item.text else (2048 if item.image is not None else 8192)
                        self._check_length(prepared, limit)
                        del prepared
                        values = self.embedder.encode_documents(images=[item.image], texts=[item.text])
                    values = values.detach().float()
                    if values.shape != (1, DIMENSIONS) or not torch.isfinite(values).all():
                        raise RuntimeError("embedder returned invalid vectors")
                    norms = values.norm(p=2, dim=-1, keepdim=True)
                    if (norms <= 1e-12).any():
                        raise RuntimeError("embedder returned a zero vector")
                    rows.extend((values / norms).cpu().tolist())
                    del values, norms
            return rows
        except torch.OutOfMemoryError:
            torch.cuda.empty_cache()
            raise GPUCapacityError("GPU memory limit reached") from None

    def rerank(self, query: str, documents: list[DecodedDocument]) -> list[float]:
        torch = self.torch
        scores = []
        try:
            with torch.inference_mode():
                for item in documents:
                    batch = self.processor.process_queries_documents_crossencoder([
                        {"question": query, "doc_text": item.text, "doc_image": item.image}], truncation=False)
                    limit = 10240 if item.image is not None and item.text else (2048 if item.image is not None else 8192)
                    self._check_length(batch, limit)
                    batch = {key: value.to("cuda:0") if isinstance(value, torch.Tensor) else value
                             for key, value in batch.items()}
                    logits = self.reranker(**batch, return_dict=True).logits.detach().float().reshape(-1)
                    if logits.numel() != 1 or not torch.isfinite(logits).all():
                        raise RuntimeError("reranker returned invalid logits")
                    scores.append(float(logits.cpu().item()))
                    del logits, batch
            return scores
        except torch.OutOfMemoryError:
            torch.cuda.empty_cache()
            raise GPUCapacityError("GPU memory limit reached") from None

    def warmup(self) -> None:
        from PIL import Image
        image = Image.new("RGB", (64, 64), "white")
        try:
            self.embeddings([DecodedDocument("A setup check.", None)], "query")
            documents = [DecodedDocument("A setup check.", None), DecodedDocument("", image),
                         DecodedDocument("A setup check.", image)]
            self.embeddings(documents, "document")
            self.rerank("What is on the page?", documents)
            self.torch.cuda.synchronize()
            self.proof["warmup_modalities"] = ["query_text", "document_text", "image", "image_text"]
        finally:
            image.close()

    def close(self) -> None:
        for name in ("embedder", "reranker", "processor"):
            if hasattr(self, name):
                delattr(self, name)
        if hasattr(self, "torch"):
            self.torch.cuda.empty_cache()
