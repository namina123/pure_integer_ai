# 20k Source-Bound Broad QA Preview

This is the project's second real-scale vertical slice over a frozen Chinese Wikipedia snapshot. A deterministic 100,000-page candidate selection is used as the source ceiling, and 20,000 projectable pages are indexed without reselecting pages for answerability.

## Public facts

- Source: `ZHWIKIPEDIA_20260701`; snapshot manifest SHA-256 `0e81569aaf6cf9cb688b41da27d5eff19707153ee5c74bf9bf362f34427869dd`.
- Candidate ceiling: 100,000; accepted pages: 20,000; cutoff selection ordinal: 64,236.
- Final SQLite: 109,006 passages, 3,608,002 terms, 251,494,400 bytes; SHA-256 `e18db72b090dfdfd96aac23c74a5ad0751afe17c2dcfb02fc91f1213b0f7c4da`; SQLite `integrity_check=ok`.
- Every answer retains page, revision, contributor, raw evidence span/hash, source URL, and CC BY-SA 4.0 identity. Queries without a page-title anchor fail closed instead of answering from weak relation-word co-occurrence.

## Probe boundary

The fixed 24-question set remains a development probe permanently labeled `DEVELOPMENT_VERTICAL_PROBE_NOT_HELD_OUT`. The 20k run produced `22 ANSWER / 1 UNKNOWN / 1 CLARIFY`, with zero citation reconstruction failures. Its per-question statuses and answers were unchanged from the published 10k run.

This demonstrates one real run of source-bound sparse retrieval, evidence extraction, refusal/clarification, and auditable publication at a larger index size. It does not demonstrate general QA, free-form generation, long conversation, open-domain semantic learning, permanent memory, or language weaning. External posting merge and SQLite publication remain the main measured performance hotspot.

The machine-readable receipt is [`data/ph2/broad_qa_20k_preview_receipt_v1.json`](../data/ph2/broad_qa_20k_preview_receipt_v1.json). The repository does not include the 3.5 GB source dump, the 20k SQLite file, or reproducible shards. They can be rebuilt from the official source, frozen manifest, public CLI, and receipt identities using an explicit large-data run root outside the source tree.
