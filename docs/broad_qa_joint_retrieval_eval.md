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

## Next contract

The next step is a new source-version-alignment contract, not more tuning on this consumed 200-question development split. Before QA execution, it must freeze a complete title/alias and terminal-page answer-coverage census. Source coverage and algorithm accuracy remain separate metrics. Any new source-aligned family must exclude all consumed title domains and freeze its questions, labels, splits, thresholds, and source identities before development execution. Held-out may run once only after the new development gate passes and the algorithm is frozen.

The honest current claim is a 20k source-bound extractive preview, a formal PASS for evidence selection on provided external contexts, and a joint-retrieval successor development `FAIL`. V2 nearly closes page retrieval and citation verification while exposing source-version alignment as a distinct unresolved problem.
