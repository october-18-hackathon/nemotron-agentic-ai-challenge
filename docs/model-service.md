# Run the embedding and reranking service

This Docker recipe serves the two retrieval models. It gives your code an HTTP interface. Teams must build segmentation, retrieval policy, matching, comparisons, and agent tools.

The service does not run Nemotron Parse 3.5. Use the Parse endpoint supplied by the organizer.

## Choose how to run it

The organizer can provide a built image. Set the image name in your run command. Use an image digest when the organizer supplies one.

To build the image, use a Linux amd64 Docker builder with internet access and at least 60 GB of free disk space for the image and build cache. The recipe uses Python 3.11, PyTorch 2.8.0, CUDA 12.8, and Transformers 4.56.2. It pins the base image digest and Python package hashes.

To run inference, use Linux, Docker, an NVIDIA driver that supports CUDA 12.8, and NVIDIA Container Toolkit. Use a GPU with BF16 support. The reference hardware is one RTX PRO 6000 Blackwell Server Edition with 96 GB of GPU memory. Other compatible GPUs can work. Their memory and response times need a test with your inputs.

The recipe checks compiled `sm_120` support for Blackwell. That check does not run a GPU kernel. A GPU smoke check is a separate step below.

## Models

| Role | Model | Fixed revision |
| --- | --- | --- |
| Embedding | `nvidia/llama-nemotron-embed-vl-1b-v2` | `582e3bf72aee355e3c59ed89de53543c5b0657ee` |
| Reranking | `nvidia/llama-nemotron-rerank-vl-1b-v2` | `b8a9987b05b75db5ad949c825cbcbf6eb7c48b5a` |

The build downloads each public model snapshot. It keeps the weights, model code, model card, license, and third party notices. It records a SHA-256 hash for each file. At startup, the service checks those hashes before it loads the model code. It uses local files and disables Hub access at runtime.

The model code comes from the pinned public NVIDIA repositories. `trust_remote_code=True` loads that code from the image. The transport wrapper is original MIT code. See the [model notices](../infra/model-service/NOTICE.md) for the separate NVIDIA and Llama terms. Built with Llama.

## Build

Run commands from the repository root:

```sh
docker buildx build --platform linux/amd64 --load \
  -t nemotron-challenge-models:local infra/model-service
```

This build includes about 6.8 GB of model files. It needs no Hub token. It does not run GPU inference.

To test the package layer before you download the weights:

```sh
docker buildx build --platform linux/amd64 --target runtime-libs --load \
  -t nemotron-challenge-model-libs:local infra/model-service
```

The package check imports PyTorch, Torchvision, and Transformers. It checks their versions, CUDA 12.8, and the compiled `sm_120` target.

## Start

The starter's `.env` file records `RETRIEVAL_IMAGE`. For the Docker shell command, export that value as shown below. Use your build tag or the image digest supplied by the organizer:

```sh
export RETRIEVAL_IMAGE=nemotron-challenge-models:local
docker run --rm --name nemotron-models --gpus device=0 \
  --read-only --tmpfs /tmp:rw,nosuid,nodev,size=1g \
  --cap-drop ALL --security-opt no-new-privileges \
  -p 127.0.0.1:8000:8000 "$RETRIEVAL_IMAGE"
```

The container runs as user 10001. Model files have read access. A temporary cache holds the loaded model code. The host port binds to loopback. For shared access, the organizer must add an HTTPS proxy, access controls, request limits, and monitoring.

Keep one service process per GPU. Both models share one queue. The service runs one inference request at a time. It accepts up to four waiting requests. A request can wait for one second. A full queue or an expired wait returns `429` with `Retry-After: 1`.

Each GPU microbatch has one item. A request can contain more items within the limits below. The service runs those items in order. These limits bound input size and memory use. GPU execution has no hard time limit. A client timeout does not cancel a GPU kernel. Restart the container if a kernel stops making progress.

## Check the service

```sh
curl --fail http://127.0.0.1:8000/health
curl --fail http://127.0.0.1:8000/ready
curl --fail http://127.0.0.1:8000/v1/models
```

`/health` checks the web process. `/ready` returns `200` after both models load and pass warmup on text, image, and combined inputs. Its runtime record includes the GPU, precision, package versions, and model file check. A startup failure leaves `/health` available. Data routes and `/ready` return `503`.

Set the starter's retrieval base URL to:

```dotenv
RETRIEVAL_BASE_URL=http://127.0.0.1:8000/v1
```

The `/v1/models` response lists the fixed model IDs and revisions. The embedder record has `dimensions: 2048`.

## HTTP inputs

Both endpoints accept text strings and document records. A document record has `text`, `image`, or both. An image must be a PNG or JPEG base64 data URL, such as `data:image/png;base64,...`. The service decodes the bytes to an RGB image. It does not fetch image URLs or read image paths from requests.

### Embeddings

```sh
curl --fail http://127.0.0.1:8000/v1/embeddings \
  -H 'Content-Type: application/json' \
  --data '{"model":"nvidia/llama-nemotron-embed-vl-1b-v2","input_type":"query","input":["A setup query"]}'
```

Use `input_type: query` for text queries. Use `input_type: document` for text, image, or combined documents. Do not add the model's query or passage prefix. The model processor adds it.

The response contains `data`, with one `embedding` and input `index` per item. Each vector has 2,048 finite values and unit L2 length. The response includes `model`, `model_revision`, `dimensions`, and `normalized: true`.

### Reranking

```sh
curl --fail http://127.0.0.1:8000/v1/rerank \
  -H 'Content-Type: application/json' \
  --data '{"model":"nvidia/llama-nemotron-rerank-vl-1b-v2","query":"A setup query","documents":["A setup document","Another setup document"],"top_n":2}'
```

The query must be text. `results` contains the original document `index` and `relevance_score`, sorted by score from high to low. Equal scores keep input order. Scores are raw logits. They are not probabilities or match decisions. The response includes the fixed model revision and `score_type: raw_logit`. Omit `top_n` to return all results.

## Limits and errors

| Input | Limit |
| --- | --- |
| Request body | 24 MiB, including JSON and base64 |
| Embedding items | 1 to 16 |
| Reranking documents | 1 to 32 |
| `top_n` | 1 to the number of documents |
| Query text | 2,000 characters |
| Document text | 32,000 characters per item |
| Encoded image | 8 MiB of decoded file bytes |
| Image dimensions | 20 million pixels and 10,000 pixels per side |
| Images in one request | 80 million pixels in total |
| Image frames | 1 |
| Embedding query context | 512 tokens |
| Text document context | 8,192 tokens |
| Image document context | 2,048 tokens |
| Combined document context | 10,240 tokens |

Context limits include model prompt and image tokens. For reranking, they include the query. The service checks the processed token count before it sends work to the GPU. It rejects overflow. It does not cut text to fit. Split long source text in your own pipeline.

Text cannot contain the reserved image markers `<image>`, `<img>`, `</img>`, or `<IMG_CONTEXT>`. Unknown fields, invalid Unicode, invalid images, and token overflow return `422`. A body above the byte limit returns `413`. These input errors leave the service ready.

A GPU memory error returns `503` with `detail: gpu_memory_limit`. Try smaller inputs. Other model failures return `503` and clear readiness. Inspect the container log, fix the cause, and restart the container.

## CPU checks

The transport tests use a stub model backend. They cover request limits, image decoding, model pins, result shape, queue limits, output checks, and model file hash checks.

```sh
uv venv --python 3.12 .venv-model-contract
uv pip install --python .venv-model-contract/bin/python -r infra/model-service/requirements-test.txt
.venv-model-contract/bin/python -m pytest -q infra/model-service/tests
```

These tests do not load the weights. They do not prove GPU execution or model quality. The stub reports `production: false` in its readiness record. No environment setting can enable that backend in the service image.

## GPU smoke check

After the container reports ready, run this check from a Python environment with Pillow installed:

```sh
.venv-model-contract/bin/python infra/model-service/smoke_gpu.py \
  --base-url http://127.0.0.1:8000/v1 \
  --receipt runs/model-service-smoke.json
```

It sends setup text and a generated blank image. It checks model pins, query and document vectors, all three document formats, raw reranking scores, and output indices. It saves the reported GPU runtime and checks in a receipt. It does not choose challenge matches or test their quality. If a shared proxy needs a Bearer token, export `RETRIEVAL_API_KEY` in the check process. The receipt does not store that token.

Keep a receipt from a real GPU run before you describe the image as tested for participants. A Docker build and CPU test result do not provide that proof.

## Sources

- [Pinned NVIDIA embedder model and code](https://huggingface.co/nvidia/llama-nemotron-embed-vl-1b-v2/tree/582e3bf72aee355e3c59ed89de53543c5b0657ee)
- [Pinned NVIDIA reranker model and code](https://huggingface.co/nvidia/llama-nemotron-rerank-vl-1b-v2/tree/b8a9987b05b75db5ad949c825cbcbf6eb7c48b5a)
- [Transformers 4.56.2 local model loading](https://huggingface.co/docs/transformers/v4.56.2/en/installation)
- [Hugging Face snapshot downloads](https://huggingface.co/docs/huggingface_hub/guides/download)
- [PyTorch 2.8.0 CUDA 12.8 packages](https://pytorch.org/get-started/previous-versions/#v280)
