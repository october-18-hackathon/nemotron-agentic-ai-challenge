# Wikipedia API

Set the base URL in your shell:

```sh
export WIKIPEDIA_API_URL=https://port8000-rks0xavparpzq59.tunnel.applications.uk-south2.nebius.cloud
```

This public API requires no key.
It provides lexical search over 7,182,883 canonical English articles.
It does not call embedding, reranking, or reasoning models.

## Why the index is provided

- Skip the full-corpus download, storage, and keyword-index build in each team's environment.
- Search millions of articles through one API while reserving model calls for candidate matching and comparison.
- Use the same pinned source snapshot across teams and keep article revisions and checksums with each result.

The shared BM25 index supplies lexical candidates and article text.
Your team builds its semantic matching method with the required Nemotron embedder and reranker.
Store selected passages and vectors in your own LanceDB index or chosen storage implementation.
The shared index and your team's evidence index serve separate roles.

## Use it in your project

Extract a Britannica entry with Parse 3.5.
Choose search terms from that entry and request candidate articles.
Read the candidate text through the article endpoint, following `next_offset` until the text is complete.
Use the required embedder and reranker in your matching method.
Decide whether the evidence supports a match.
Keep the source metadata with your comparison results.

Reuse the selected article text for vocabulary or topic comparisons and for cited agent answers.
For a joint BERTopic comparison, embed the retained Britannica entries and distinct selected Wikipedia articles with the same model and document settings.
You do not need to embed all articles in the shared corpus to run that comparison.

The service returns keyword candidates. It supplies no Britannica matches or answer key.
A reranker cannot recover an article that your candidate set does not contain.
Your team must test candidate coverage and match quality.

The starter includes a Python client:

```python
from nemotron_challenge.config import Settings
from nemotron_challenge.clients import WikiClient

settings = Settings.from_env()
with WikiClient(settings.wikipedia_api_url) as wiki:
    results = wiki.search("ocean currents", top_k=20)
    for hit in results.hits:
        print(hit.page_id, hit.title)
```

This example lists candidates. It does not select a match.
Cache retrieved text with its article ID, revision, and checksum.

## Inspect the corpus

```sh
curl "$WIKIPEDIA_API_URL/ready"
curl "$WIKIPEDIA_API_URL/v1/corpus"
```

The response reports source revision, row counts, text-index coverage, and license metadata.
Readiness requires the complete index. It cannot report a partial corpus as ready.

## Search

```sh
curl "$WIKIPEDIA_API_URL/v1/search" \
  -H 'Content-Type: application/json' \
  -d '{"query":"your search terms","top_k":10,"excerpt_chars":1000}'
```

The API treats the query as words for BM25 search.
It does not expose an advanced query language.
Each hit includes a page ID, title, article revision, source URL, excerpt, and provenance.
The excerpt offset counts Unicode characters.

`top_k` accepts 1 to 100.
The query accepts up to 512 characters and 64 words.
Excerpts accept 100 to 2,000 characters.

## Read an article

Use a page ID from a search hit.

```sh
curl "$WIKIPEDIA_API_URL/v1/articles/PAGE_ID?offset=0&max_chars=10000"
```

`max_chars` accepts 1 to 20,000.
If the response has a `next_offset`, request that offset to read the next part.
Concatenate the returned pages in offset order.
Verify the complete UTF-8 text against `provenance.text_sha256` before you cache it.
Keep the page ID, revision ID, source URL, and text checksum with your results.

## Handle limits

The service accepts four active reads and a bounded waiting queue.
Each database read has a ten-second deadline.
The gateway limits requests from each network peer.
Several participants can share one request limit through the HTTPS proxy.

If the service returns `429` or `503`, wait and retry with a bound on attempts.
Honor the `Retry-After` header when the service supplies it.
Do not retry invalid input responses such as `422` without changing the request.

Request bodies must not exceed 8 KiB.
The API supports CORS for participant clients and websites.

## Source and attribution

Dataset: `wikimedia/structured-wikipedia`.
Revision: `417c267bb457fa645c22eb3b5c77764963194c70`.
The source contains 86 English shards and 7,597,149 raw records.
The canonical index keeps one record per English page ID, using the latest numeric revision.

Wikipedia contributors retain attribution for article text.
Text uses [CC BY-SA 4.0](https://creativecommons.org/licenses/by-sa/4.0/).
Each response includes article source and revision links.
Retain that metadata when you share excerpts or derived reports.
