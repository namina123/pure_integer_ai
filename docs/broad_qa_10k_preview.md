# Source-Bound Chinese Broad QA: 10k Development Preview

This preview is the first end-to-end, real-source vertical slice of PIDSLCA's one-week broad-QA track. It is a reproducible development result, not a held-out benchmark, a general-QA claim, or evidence of language weaning.

## What Runs Now

- A stable hash rule selects pages without inspecting their answerability.
- A 2026-07-01 Chinese Wikipedia snapshot is read by multistream block.
- Wikitext is projected into source-linked passages with raw spans and SHA-256 commitments.
- Integer character features and delta-varint postings retrieve a bounded candidate set.
- The current query path returns `ANSWER`, `CLARIFY`, or `UNKNOWN` and never emits an unreferenced answer. `CONFLICT` exists in the result contract but has not yet been connected to a multi-source conflict detector.
- Simplified/traditional conversion is a deterministic lookup aid; it does not rewrite quoted source text.

The current index contains 10,000 main-namespace, non-redirect pages and 54,026 passages. Its SQLite artifact is 135,421,952 bytes. It was built from a stable 36,000-page candidate selection because not every selected index entry resolves to an eligible, projectable page.

## Development Probe

The public CC0 probe contains 24 fixed, cross-topic Chinese questions. It was used repeatedly during development, so it is explicitly not held-out.

On the current 10k index:

- 22 questions returned `ANSWER`;
- the impossible strong constraint returned `UNKNOWN`;
- the ambiguous surname returned `CLARIFY`;
- all 22 answers contained the requested fact after manual review;
- all 22 citations passed independent reconstruction of page, revision, revision timestamp, contributor, title, URL, raw span and raw SHA-256 from 18 frozen Wikipedia blocks;
- 72 warm queries measured p50 29 ms and p95 245 ms;
- 24 separate-process cold queries measured p50 1.245 s and p95 1.501 s.

These latency values describe Windows CPython 3.11 runs on the project's 4-core/8-thread development machine. They are performance evidence for this slice, not a cross-platform guarantee.

Representative behavior:

```text
Q: 矮寨大桥何时建成通车？
A: 2012年3月31日建成通车。
   source: page 1920441, revision 92965292

Q: 火星上的矮寨大桥何时通车？
status: UNKNOWN

Q: 辛普森是谁？
status: CLARIFY
```

## Reproduce the Probe

Build the index from the public snapshot contract and official Wikimedia dump, then run:

```bash
python scripts/run_broad_qa_dev_probe.py \
  --database <broad-qa.sqlite3> \
  --selection <selection.json> \
  --xml <zhwiki-pages-articles-multistream.xml.bz2> \
  --repeat 3
```

The optional selection/XML pair enables independent citation reconstruction. Without it, the same script still emits per-question results and warm latency, while citation audit fields remain `null`.

Source identity, official URLs, upstream hashes, local hashes, license and attribution policy are frozen in [`data/ph2/manifests/zhwikipedia_20260701.multistream_snapshot.json`](../data/ph2/manifests/zhwikipedia_20260701.multistream_snapshot.json). Wikipedia-derived text remains under CC BY-SA 4.0 and every answer preserves its page and revision URL.

The compact machine-readable result is [`data/ph2/broad_qa_10k_dev_preview_receipt_v1.json`](../data/ph2/broad_qa_10k_dev_preview_receipt_v1.json). It intentionally records `DAY_1_VERTICAL_SLICE_RUNTIME_EVIDENCED_NOT_WEEK_MINIMUM_PASS` rather than a broad-QA PASS.

## What Is Not Done

The one-week minimum scale is 100,000 pages, followed by a separately frozen 200-question development set and 300-question held-out evaluation. The current builder also needs sharded checkpoints and external posting merge before responsible 100k/300k expansion. Multi-hop implicit reasoning, free-form generation, long conversation and autonomous open-source language learning are outside this preview.

The next publishable milestone is therefore the 100k indexed slice with bounded restart/resume, independent retrieval evaluation and the same exact-citation requirement, not further tuning on these 24 questions.
