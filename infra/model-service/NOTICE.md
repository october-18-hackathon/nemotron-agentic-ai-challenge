# Model and service notices

The transport wrapper in this directory is original code. The repository's MIT license covers that code.

The model weights have separate terms. The image build retains each model's `LICENSE`, `THIRD_PARTY_NOTICES.md`, and model card beside its weights in `/opt/models/embed` or `/opt/models/rerank`. Those files are part of each pinned snapshot.

- [NVIDIA Open Model License](https://www.nvidia.com/en-us/agreements/enterprise-software/nvidia-open-model-license/)
- [Llama 3.2 Community License](https://raw.githubusercontent.com/meta-llama/llama-models/main/models/llama3_2/LICENSE)
- [Embedder source and model card](https://huggingface.co/nvidia/llama-nemotron-embed-vl-1b-v2/tree/582e3bf72aee355e3c59ed89de53543c5b0657ee)
- [Reranker source and model card](https://huggingface.co/nvidia/llama-nemotron-rerank-vl-1b-v2/tree/b8a9987b05b75db5ad949c825cbcbf6eb7c48b5a)

Built with Llama. NVIDIA model code and post-processing scripts use Apache 2.0 as stated in the model cards. The PyTorch base image and Python packages retain their own licenses. The service's MIT license does not change those terms.

This image has no challenge corpus. Wikipedia material has its own attribution and license. Preserve those notices when you publish source text.
