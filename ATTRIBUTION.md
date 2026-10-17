# Sources and licenses

The [MIT license](LICENSE) covers original starter code and prose.
It does not change the terms for source data, model weights, or dependencies.

## Britannica scans

Source: National Library of Scotland,
[Encyclopaedia Britannica Illustrated](https://huggingface.co/datasets/NationalLibraryOfScotland/encyclopaedia_britannica_illustrated/tree/309f765b1ecc83e0a7f7762e44b4a927a7555443).
The dataset declares [CC0 1.0](https://creativecommons.org/publicdomain/zero/1.0/).
The sample manifest records the fixed source revision and each scan checksum.
Credit the National Library of Scotland when you describe the source collection.

## Wikipedia text

Source: Wikipedia contributors, through the Wikimedia Foundation's
[structured Wikipedia dataset](https://huggingface.co/datasets/wikimedia/structured-wikipedia/tree/417c267bb457fa645c22eb3b5c77764963194c70).
The API uses the English snapshot at revision `417c267bb457fa645c22eb3b5c77764963194c70`.
Article text uses [CC BY-SA 4.0](https://creativecommons.org/licenses/by-sa/4.0/).

Keep the article title, revision link, contributor credit, and license link with shared text.
State any changes you make to source text.
Retain the API provenance record with stored excerpts and comparisons.
Consult the source license when you distribute adaptations.

## Models

Built with Llama.

The two retrieval models retain the NVIDIA Open Model and Llama 3.2 terms in their pinned snapshots.
The model source headers declare Apache 2.0 for code.
The model cards state Apache 2.0 terms for post-processing scripts.
The Docker build retains model cards, license files, and third party notices beside the weights.

- [Embedder model and terms](https://huggingface.co/nvidia/llama-nemotron-embed-vl-1b-v2/tree/582e3bf72aee355e3c59ed89de53543c5b0657ee).
- [Reranker model and terms](https://huggingface.co/nvidia/llama-nemotron-rerank-vl-1b-v2/tree/b8a9987b05b75db5ad949c825cbcbf6eb7c48b5a).
- [Model service notices](infra/model-service/NOTICE.md).

Parse 3.5 and Super 3.5 VL run through separate services.
Their providers' model and service terms apply.
This repository does not contain those model weights.

## Software

Python packages, PyTorch, and the Docker base image retain their licenses.
Hermes Agent retains the license in its upstream repository.
Keep upstream notices when you distribute those components.

## Diagram font

The editable infrastructure and BERTopic SVGs embed DejaVu Sans.
The font retains its [license notice](docs/assets/DejaVu-LICENSE.txt).
