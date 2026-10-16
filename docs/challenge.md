# Challenge

Find Wikipedia articles that match the topics in 100 historical Britannica scans.
Compare how the two sources describe those topics.
Build a Hermes agent that can answer comparison questions with evidence.

## Required work

1. Extract the content of the shared 100 scans.
2. Identify source entries and find candidate Wikipedia articles.
3. Use the required embedder and reranker to support your matching method.
4. Compare vocabulary, word frequencies, or topics in the matched documents.
5. Build a static website with the extracted Britannica content as compiled LaTeX documents.
6. Set up Hermes with Super 3.5 VL and your comparison tools.
7. Record three successes and three failures from your agent runs.

A page can contain several entries.
An entry can continue onto another page.
Some scans can be blank or hard to read.
Your team must decide how to handle these cases and report source coverage.

## Required models

| Role | Model | Deployment |
| --- | --- | --- |
| Page extraction | `nvidia/nemotron-3.5-parse` | Shared organizer endpoint |
| Embedding | `nvidia/llama-nemotron-embed-vl-1b-v2` | Team deployment on Nebius AI Cloud |
| Reranking | `nvidia/llama-nemotron-rerank-vl-1b-v2` | Team deployment on Nebius AI Cloud |
| Hermes reasoning and vision | `nvidia/Nemotron-3.5-Super-VL` | Nebius Token Factory |

The embedding and reranking revisions appear in the Docker recipe.
Use LanceDB as the starting database, or replace it through the storage interface.

Nebius AI Cloud and Nebius Token Factory require separate accounts.
Your team supplies its model deployment and Token Factory key.

## Shared sources

The packaged manifest fixes the Britannica dataset revision, source rows, and image checksums.
Run `uv run nemotron-challenge data show` to inspect it.
The source `id` is a volume ID. It cannot identify a scan on its own.
Use the manifest's page ID and checksum to identify each scan.

The Wikipedia API contains 7,182,883 canonical English articles from the pinned Wikimedia snapshot.
It supplies lexical candidates and source text.
Your team chooses its semantic retrieval and matching method.

## Submission

Submit your solution repository and a video of 3 to 5 minutes.
Include the static website, compiled documents, comparison analysis, and six case studies.
Use the templates in `templates/` to record model use and evidence.

The starter has no answer key and no supplied quality score.
Show how you checked a match, a comparison, and a citation.
State what failed and what remains uncertain.

## References

- [Organizer challenge document](https://docs.google.com/document/d/132-v2EoYUcxif6GjlnfrjfD7BkmeScYstDbRwXN5F6g/edit).
- [Britannica source](https://huggingface.co/datasets/NationalLibraryOfScotland/encyclopaedia_britannica_illustrated/tree/309f765b1ecc83e0a7f7762e44b4a927a7555443).
- [Wikimedia source](https://huggingface.co/datasets/wikimedia/structured-wikipedia/tree/417c267bb457fa645c22eb3b5c77764963194c70).

The organizer used a private NVIDIA reference repository during design.
Participants do not need access to that repository.
The public starter contains original infrastructure code and unfinished project stages.
