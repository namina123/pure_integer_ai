# Source-Bound Retrieval and Evidence Selection Joint Evaluation

This report records the joint retrieval evaluation and its successor family. The chain retrieves a source page from an index containing the random 20k Wikipedia slice plus frozen source pages, then selects up to four independently verifiable evidence spans from the resolved terminal page.

This is not open-domain general QA, free-form generation, or language weaning. Questions, source titles, labels, and thresholds are frozen before execution. Questions, labels, source targets, and alias ledgers remain separate.

## V1 development result

`PH2_BROAD_QA_JOINT_RETRIEVAL_EVIDENCE_PACK_V1` excluded all 454 titles from the earlier external-context family and froze 200 development plus 300 held-out questions. Of 439 new titles, 437 resolved through the snapshot and two were absent.

- Without aliases: Recall@20 `59.5%`, top1 `58.0%`, evidence hit `32.5%`, `FAIL`.
- With explicit aliases: Recall@20 `88.0%`, top1 `85.5%`, evidence hit `44.0%`, `FAIL`.
- Generic question-shape and bounded within-page selection probes did not close the frozen 60% evidence gate; the final V1 probe reached `43.0%`.

The V1 held-out split was not run. Work stopped on that consumed development family before freezing a successor.

## V2 successor contract

`PH2_BROAD_QA_JOINT_RETRIEVAL_EVIDENCE_PACK_V2` excludes both the 454 external-family question titles and all 439 V1 source-target titles: 893 titles in total, with zero title overlap between the two joint families. It freezes 200 development questions, 300 held-out questions, and 451 source targets. Its manifest SHA-256 is `47f19f8a33fd9992842efb744c93862437da9faa4eb775be12dd58cbdee373e9`.

The source path resolved 449/451 targets, with two missing from the snapshot. The target index contains 449 pages, 4,330 passages, and 369,227 terms. The combined successor index contains 20,439 pages, 113,231 passages, and 3,766,159 terms; SHA-256 is `01450bfb115532e19ef3cbe43f8e6cf92c5de3900969eeb58aa76d2407777fa7`.

Queries use bounded `alias_term` lookup rather than scanning the alias table. An answer may carry up to four real passage citations from one page and revision. Each citation records its raw span/hash, source identity, and selected text. Scoring revalidates every citation and fails the complete ANSWER closed if any citation is altered.

## V2 development result

- Recall@20: `199/200 = 99.5%`.
- Top1 source hit: `198/200 = 99.0%`.
- ANSWER citation validity: `190/190 = 100%`.
- Full-denominator evidence hit: `107/200 = 53.5%`.
- Query p50/p95: `174.356/299.486 ms`.
- Status: `FAIL`; the predeclared 60% evidence gate was not reduced.
- Aggregate SHA-256: `f5455954b206dc8eb0e11af6800ba5b5f139411c94e91e8c084711c481ab016b`.

Aggregate V2 separately checks whether the frozen terminal page actually contains any legacy gold answer. Only `126/200 = 63.0%` do. Evidence hit within that source-covered subset is `107/126 = 84.9206%`: CMRC2018 covers 72 and hits 62, while DRCD covers 54 and hits 45. Failure accounting is `SOURCE_GOLD_ABSENT_FROM_SNAPSHOT=74`, `GOLD_NOT_IN_EVIDENCE=18`, and `NON_ANSWER=1`.

CMRC2018 and DRCD preserve older Wikipedia contexts, while retrieval targets the frozen July 2026 terminal pages. The exact gold answer is absent from the current page for 74 questions. More query tuning, a lower threshold, denominator changes, or item-specific rules cannot honestly repair that source-version mismatch. The conditional rate is diagnostic; it does not replace the 53.5% full-denominator result or global `FAIL`.

The 300-question V2 held-out split has not been run, and no joint PASS receipt has been published.

## Source-alignment census

The next contract has now been executed. Before any new-family QA run, the complete candidate freeze excluded 1,344 consumed titles from the external, joint V1, and joint V2 families, then retained 10,061 naturally title-anchored questions from 48,129 valid external questions. The census checked aliases, current terminal revisions, full visible page text, and the actual first-12-passage index projection.

- `SOURCE_ALIGNED=7,189`, or `71.4541%` raw-population coverage.
- `GOLD_ABSENT_FROM_TERMINAL_REVISION=1,854`.
- `GOLD_PRESENT_OUTSIDE_PROJECTED_PASSAGES=790`.
- `SOURCE_ALIAS_MISSING=159`.
- `GOLD_ONLY_IN_RAW_WIKITEXT=65`.
- `PASSAGE_PROJECTION_DIVERGES_FROM_FULL_PAGE=4`.

Only questions whose answer appears in both full visible page text and the actual passage projection enter the new family. This rule was frozen before QA execution. The uncovered 28.5459% remains public source-coverage accounting and is not replaced by the family accuracy. Census SHA-256 is `0809f96843c11bec6264065fb166498fc73e3df4a325833711d4a66bc7dc5823`.

## Source-aligned family and development gate

The new family freezes 200 development and 300 held-out questions, split evenly between CMRC2018 and DRCD. Development uses 182 titles and held-out uses 277, with zero overlap between them or any consumed predecessor title domain. Family manifest SHA-256 is `82f4d641cf44c553594c8a5610b071e1e3ec09197a6bcf562d9c838d6dfcd666`.

The 459 source pages form a target index of 4,514 passages and 390,483 terms. The combined index contains 20,449 pages, 113,431 passages, and 3,781,174 terms; SHA-256 is `17bdea8850ca6afea3637fdab2bd4f58fa90fc0c3df5ea04ebf1a697a4c31cab`. Development reaches `200/200` Recall@20, `200/200` top1, `195/195` valid ANSWER citations, and `165/200 = 82.5%` evidence hit: `PASS`, aggregate SHA-256 `bfdb15d6244ccd9a245598efb5987034a752397ceb5ed78cfcf73479bb92e9bf`.

## Unique formal held-out run

The algorithm, family, census, index, aliases, selection, questions, labels, development aggregate, and 14 algorithm files are bound to public commit `7f3d87607eedc29c69eb17f40729be39e04f9045`. A fixed-path `OUTCOME_PENDING` intent claims the single run before prediction. Ordinary prediction rejects held-out; formal prediction authorization does not read labels; formal scoring revalidates the complete freeze before parsing labels. An existing intent forbids rerunning from another output directory.

The single 300-question formal result is:

- Recall@20: `300/300 = 100%`.
- Top1 source hit: `300/300 = 100%`.
- ANSWER citation validity: `296/296 = 100%`.
- Evidence hit: `253/300 = 84.3333%`.
- CMRC2018: `126/150 = 84.0%`; DRCD: `127/150 = 84.6666%`.
- Four `UNKNOWN` results and 43 `GOLD_NOT_IN_EVIDENCE` failures.
- Query p50/p95: `177.8824/350.2957 ms`.
- Status: `PASS`; the frozen 80% Recall, 70% top1, 100% citation, and 60% evidence gates were not reduced.
- Aggregate SHA-256: `84bfeb9023ffa31386fb4dcd159af9d82d797c92393d5e83322210a3cf4d30f3`.

The compact public receipt is [`broad_qa_source_aligned_formal_receipt_v1.json`](../data/ph2/broad_qa_source_aligned_formal_receipt_v1.json). It contains no third-party questions, labels, page text, predictions, or local paths.

The accurate new claim is that page retrieval, source-bound extraction, and citation-by-citation verification pass a predeclared 300-question evaluation on a frozen Chinese Wikipedia source and source-version-aligned question population. This is not arbitrary open-source coverage, free-form generation, mature dialogue, language weaning, or general QA. The next engineering gap is the 43 correct-page evidence-window misses and four refusals, followed by a fresh unconsumed family for relational, temporal, quantitative, causal, and comparative constraints. This held-out family must not be rerun or used for item-level tuning.
