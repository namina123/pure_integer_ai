# Source-Bound Retrieval and Evidence Selection Joint Evaluation

This document records the independent family `PH2_BROAD_QA_JOINT_RETRIEVAL_EVIDENCE_PACK_V1`. It connects two previously separate paths: retrieve a source page from a joint index containing the random 20k Wikipedia slice plus the frozen source pages, then select auditable evidence from that page.

This is not open-domain general QA, free-form generation, or weaning. Questions naturally contain their source title; titles and labels are frozen before system execution. The previous 300-question external-context family and its title domain are excluded.

The family froze 200 development and 300 held-out questions from CMRC2018 and DRCD. The old 454 title domains were excluded. Of 439 new titles, 437 resolve through the snapshot, including redirects, and two are absent from the snapshot. The final alias ledger records original surfaces, redirect chains, terminal page/revision identities, and the two missing-source failures.

Development results:

- v1 without aliases: Recall@20 59.5%, top1 source hit 58.0%, citation-valid ANSWER 101/101, evidence hit 32.5%, `FAIL`.
- v2 with explicit alias relations: Recall@20 88.0%, top1 source hit 85.5%, citation-valid ANSWER 140/140, evidence hit 44.0%, `FAIL`.
- v3/v4 answer-side probes changed only generic question-slot shape scoring, explicit-scope gating, and bounded within-page passage selection. v4 evidence hit was 86/200 = 43.0%; Recall@20 and top1 remained 88.0% and 85.5%.

The frozen joint gates are Recall@20 80%, top1 source hit 70%, evidence hit 60%, and 100% citation validity for ANSWER results. The current aggregate is `FAIL`, SHA-256 `917df05f60d8c8b8afa1f0f93ea5f5ccadf1044725430ca13293ed4c0f924f02`. The 300-question held-out split has not been run, and no joint PASS receipt has been published. The honest next engineering gap is page-local fact/evidence modeling, not a larger index.
