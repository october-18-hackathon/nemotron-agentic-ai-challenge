# Shared source scans

All teams use the same 100 raw scans.
The starter supplies no transcriptions, entry boundaries, matches, or comparison results.

The packaged manifest is at `src/nemotron_challenge/resources/britannica-manifest.json`.
It selects 100 rows without replacement from 2,573 source scans, with seed `20261016`.
The selected row indices appear in source order.
The selection does not use extraction output or match quality.

```sh
uv run nemotron-challenge data show
uv run nemotron-challenge data fetch
```

The first fetch downloads the pinned Parquet source, about 357 MiB.
The selected images total 14,612,248 bytes.
The loader verifies the source file, image bytes, and source volume IDs.
It writes the selected images to `data/raw/britannica`.
It replaces a damaged output image and keeps a valid image.

The source volume ID repeats across scans.
Use the manifest page ID, row index, and image checksum as the scan identity.
Keep this identity with each extraction record.

The loader keeps the Hugging Face cache under `data/cache`.
The repository ignores the cache and raw images.
Share the manifest and source citations in your solution repository.
Read [source attribution](../ATTRIBUTION.md) before you publish source material.
