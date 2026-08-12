# PH2 authored sample data license

Files in this directory whose source key is `AUTHORED_CC0_V1` are dedicated to
the public domain under CC0 1.0 Universal.

SPDX-License-Identifier: CC0-1.0

Official legal text: https://creativecommons.org/publicdomain/zero/1.0/legalcode

This declaration applies only to the authored data samples in this directory.
It does not relicense third-party data or the project source code.

## LC-01 text-fidelity authored course

`authored_text_fidelity_seed_v1.jsonl.sample` is an original CC0 course for
raw/derived dual-track observations, normalization receipts, segmentation and
tokenization candidate lattices, controlled noise, irreversible-loss
disclosure, generation-surface checks, and retention revalidation.

SPDX-License-Identifier: CC0-1.0

The student-visible payload preserves the exact raw text and exposes only
unselected derived candidates and receipts. Expected states remain in the
physically separate teacher/evaluator owner artifacts. Course compilation and
the `COURSE_FROZEN` status do not assert runtime learning, mastery, readiness,
or semantic truth.

## LC-02 morphology and word-formation authored course

`authored_morphology_seed_v1.jsonl.sample` is an original CC0 course for
language-scoped stem, slot, affix, compound, reduplication, lexical-exception,
ambiguous-segmentation, reverse-generation, and retention candidates.

SPDX-License-Identifier: CC0-1.0

The student-visible observation keeps each candidate `UNSELECTED`; accepted
analyses and generated surfaces remain private to teacher/evaluator owners.
The evaluator split freezes stem-by-construction combinations that are absent
from the teacher split even when both individual axes occur there. A
dictionary-replay-only candidate is an explicit negative baseline and cannot
pass productive morphology merely because the whole surface is present.

The emitted reproducibility pack is published under
`ph2_dataset_artifacts/d02_language_courses_v1`. Course and manifest
publication do not assert runtime learning, held-out generalization, mastery,
readiness, or teacher-free execution.

## CC-CEDICT license evidence sample

`cc_cedict_20260725_header_and_rows_v1.txt.sample` is an attributed excerpt
from the 2026-07-25 CC-CEDICT snapshot published by MDBG. Its embedded header
declares the Creative Commons Attribution-ShareAlike 4.0 International License.

SPDX-License-Identifier: CC-BY-SA-4.0

Source: https://www.mdbg.net/chinese/export/cedict/cedict_1_0_ts_utf-8_mdbg.txt.gz

License: https://creativecommons.org/licenses/by-sa/4.0/

The sample is retained only as parser and license-audit evidence. English
glosses are not treated as authoritative Chinese logical truth.

`manifests/cc_cedict_20260725.license_reconciliation_v1.json` is
project-authored acquisition and license-audit metadata. It preserves the
historical `LICENSE_PARTITION_MISMATCH/BLOCKED` manifest by hash and records
that the CC-CEDICT project Wiki states CC-BY-SA-3.0 while the current MDBG
download page and the exact 2026-07-25 snapshot header state CC-BY-SA-4.0.

Because those official statements have different scopes and do not converge,
the source remains blocked from public PH2 packs. The reconciliation manifest
does not contain response cookies, proxy details, client identifiers, private
absolute paths, or third-party dictionary text. Its W-02/W-03 alternatives are
partial coverage records only and do not assert course completion or learned
capability.

## Universal Dependencies Chinese GSDSimp evidence sample

`ud_zh_gsdsimp_r2_18_dev_s2_v1.conllu.sample` is an attributed excerpt from
the `dev` split of Universal Dependencies Chinese GSDSimp release tag r2.18.
The r2.18 and r2.17 tags both resolve to commit
`7b61ed473f963e911788efdf1f478154bc1053e4` in the official repository.

SPDX-License-Identifier: CC-BY-SA-4.0

Source: https://github.com/UniversalDependencies/UD_Chinese-GSDSimp

License: https://creativecommons.org/licenses/by-sa/4.0/

The dependency labels remain third-party annotations. They are not treated as
authoritative project `Role` assignments or as Chinese logical truth.

## UD Chinese CFL and HK blind-private source metadata

`manifests/d03_v2/ph2_d03_v2_blind_private_source_extension_v1.json` is a
project-authored, payload-free authorization record for an isolated W-02 blind
private data owner. It pins Universal Dependencies r2.18 commits, Git blob
identities, sizes, README evidence, and license evidence for Chinese-CFL and
Chinese-HK. No `.conllu` content from either source is stored in public Git or
read by the main development session.

Chinese-CFL contains independently authored Simplified Chinese learner essays.
Chinese-HK contains directly annotated Traditional Chinese student-film
subtitles and Hong Kong legislative proceedings. Their new source identities
were absent from the frozen parent schema used by the consumed earlier private
family. Exact content/case/cluster exclusion against train, dev, and shadow is
still mandatory inside the isolated owner session.

SPDX-License-Identifier: CC-BY-SA-4.0

Sources:

- https://github.com/UniversalDependencies/UD_Chinese-CFL
- https://github.com/UniversalDependencies/UD_Chinese-HK

License: https://creativecommons.org/licenses/by-sa/4.0/

This extension authorizes only the new blind-private owner. It does not add
either source to Candidate training, development calibration, or shadow audit,
and it does not authorize the main session to inspect the source payload.

## UD Classical Chinese Kyoto public morphology probe

`manifests/d03_v2/stages/ph2_d03_v2_w02_morphology_successor_v4_public_probe_v1.json`
is a payload-free aggregate report over the fixed r2.18 `train` and `dev`
splits of Universal Dependencies Classical Chinese Kyoto. The bulk CoNLL-U
files remain outside Git. The report retains their exact size and SHA-256,
the upstream commit, source URL, license, integer learning counts, and disjoint
public dev/shadow results. The upstream `test` split was not read by this
public revision.

SPDX-License-Identifier: CC-BY-SA-4.0

Source: https://github.com/UniversalDependencies/UD_Classical_Chinese-Kyoto

License: https://creativecommons.org/licenses/by-sa/4.0/

The UD lemmas, UPOS, and FEATS remain third-party annotations rather than
authoritative project truth. The public probe demonstrates a language-scoped
morphology candidate capability; it does not establish W-02 runtime evidence,
mastery, readiness, or permission to rerun any consumed private family.

`manifests/d03_v2/ph2_d03_v2_w02_morphology_successor_v4_r6_source_feasibility_v1.json`
contains metadata only for the independent TueCL r2.18 test source. It records
100 sentences and 648 tokens from upstream `stats.xml` without reading the
CoNLL-U payload. It does not authorize an R6 formal owner. A fresh isolated
owner must still prove that 500 real, unique token-span cases can be formed
without inflating sentence counts, collapsing source clusters, or weakening
the five 100-case dimension gates.

## ConceptNet 5.7.0 evidence samples

`conceptnet_5_7_0_cc_by_4_0_zh_v1.csv.sample` and
`conceptnet_5_7_0_cc_by_sa_4_0_zh_v1.csv.sample` are attributed excerpts from
the official ConceptNet 5.7.0 assertions file. They remain physically split by
the license carried in each assertion metadata object.

Source: https://s3.amazonaws.com/conceptnet/downloads/2019/edges/conceptnet-assertions-5.7.0.csv.gz

For `conceptnet_5_7_0_cc_by_4_0_zh_v1.csv.sample`:

SPDX-License-Identifier: CC-BY-4.0

License: https://creativecommons.org/licenses/by/4.0/

For `conceptnet_5_7_0_cc_by_sa_4_0_zh_v1.csv.sample`:

SPDX-License-Identifier: CC-BY-SA-4.0

License: https://creativecommons.org/licenses/by-sa/4.0/

Dataset and contributor/process/activity attribution is retained in every
sample row. ConceptNet relation labels and weights are external annotations;
they are not authoritative project relation mappings or definitive truth.

## Wikidata fixed-revision snapshot metadata

`wikidata_revision_v1_allowlist.json`, its explicit v2 successor, and the
snapshot manifest are project-authored acquisition metadata. The referenced
Wikidata EntityData responses remain outside Git under the local raw root.

Wikidata structured data is made available under CC0 1.0 Universal.

SPDX-License-Identifier: CC0-1.0

Source: https://www.wikidata.org/wiki/Special:EntityData

License evidence: https://www.wikidata.org/wiki/Wikidata:Licensing

The manifest binds exact QID revisions and raw/header hashes. It does not
publish response cookies, client network fields, or treat Wikidata statements
as definitive project truth.

## Wikimedia multistream schema samples

The four `zhwiki*schema.sample` and `zhwiktionary*schema.sample` files are
project-authored minimal XML/index fixtures released under CC0 1.0 Universal.
They model the official multistream schema and failure boundaries but do not
copy pages from either 2026-07-01 dated dump.

SPDX-License-Identifier: CC0-1.0

The eventual dated raw dumps and page-derived packs remain separately subject
to the conservative CC-BY-SA-4.0 partition and page-level attribution rules.

## Wiktionary 2026-07-01 multistream snapshot metadata

`manifests/zhwiktionary_20260701.multistream_snapshot.json` is
project-authored acquisition metadata. The referenced compressed XML/index,
official dump status, project checksum list, and HTTP header captures remain
outside Git under the local raw root. The two full-EOF parser reports are
published under `ph2_dataset_raw/ZHWIKTIONARY_20260701` as reproducibility
evidence; they contain aggregate scan results rather than dictionary pages.

Wiktionary text is available under Creative Commons
Attribution-ShareAlike 4.0 International, subject to page/revision contributor
attribution and applicable upstream notices.

SPDX-License-Identifier: CC-BY-SA-4.0

Source: https://dumps.wikimedia.org/zhwiktionary/20260701/

License evidence: https://foundation.wikimedia.org/wiki/Policy:Terms_of_Use

The manifest binds independent dumpstatus and dated SHA-1-list evidence,
compressed size/upstream SHA-1/local SHA-256, response-header hashes, both
saved report identities, and the final parser report. Derived records must
retain page title, page id, revision id, revision timestamp, contributor
metadata, source URL, and the CC-BY-SA-4.0 notice. External dictionary text is
not treated as definitive project truth.

## Wikipedia 2026-07-01 multistream snapshot metadata

`manifests/zhwikipedia_20260701.multistream_snapshot.json` is
project-authored acquisition metadata. The referenced compressed XML/index,
official dump status, project checksum list, HTTP header captures, and two
full-EOF parser reports remain outside Git under the local raw root.

Wikipedia text is available under Creative Commons Attribution-ShareAlike
4.0 International, subject to page/revision contributor attribution and
applicable upstream notices.

SPDX-License-Identifier: CC-BY-SA-4.0

Source: https://dumps.wikimedia.org/zhwiki/20260701/

License evidence: https://foundation.wikimedia.org/wiki/Policy:Terms_of_Use

The manifest binds independent dumpstatus and dated SHA-1-list evidence,
compressed size/upstream SHA-1/local SHA-256, response-header hashes, both
saved report identities, and the final parser report. Derived records must
retain page title, page id, revision id, revision timestamp, contributor
metadata, source URL, and the CC-BY-SA-4.0 notice. Wikipedia text and links are
external observations, not definitive project truth.

## Unified D-02 source-pack coverage metadata

`manifests/d02_source_pack_coverage_v1.json` is a project-authored coverage
ledger. It binds six physically separate, single-license source packs under the
relative artifact root `ph2_dataset_artifacts/d02_source_pack_v1`: UD Chinese
GSDSimp, Wikidata, the two ConceptNet license partitions, Wiktionary, and
Wikipedia. The small formal reproducibility packs are published in Git; bulk
raw files remain outside Git.

Each frozen entry binds its raw snapshot manifest hash, pack manifest hash,
record count, split set, source-cluster count, and full-combination-cluster
count. The pack manifests retain source-specific attribution. Student reads are
restricted to raw `ObservationRecord` artifacts; source, teacher-evidence, and
evaluator-label owners remain physically and logically separate. External
observations are not definitive project truth.

CC-CEDICT remains a separate
`BLOCKED/OFFICIAL_LICENSE_EVIDENCE_DIVERGENCE` entry with no public pack. The
coverage ledger records zero D-03 publication, W-01 start, formal training,
teacher calls, learning-state writes, mastered claims, and readiness claims.
Pack publication and compiler success do not assert that any language
capability has been learned.

## FT30 Wiktionary public-definition v2 slice

`manifests/ft30_w03_public_definition_selection_v2.json`,
`manifests/ft30_w03_public_definition_census_v2.json`,
`w03_public_sense_runtime_v2.json`, and the source pack below
`ph2_ft30_dataset_artifacts/public_definition_source_v2` form one bounded
derived slice of the 2026-07-01 Chinese Wiktionary snapshot.

SPDX-License-Identifier: CC-BY-SA-4.0

Source: https://dumps.wikimedia.org/zhwiktionary/20260701/

License: https://creativecommons.org/licenses/by-sa/4.0/

The selection is fixed by snapshot identity, title-length strata, and stable
title hashing rather than parsing outcome. Source records retain page title,
page id, revision id, revision timestamp, contributor metadata, source URL,
license, and attribution. The compact runtime and census repeat the required
page/revision/contributor attribution for each included derived definition or
redirect, so their records remain attributable without the bulk dump. These
source definitions are external observations and are not project-adjudicated
truth, mastery evidence, or an open-domain capability claim.

## FT31 Wiktionary public-definition v3 slice

`manifests/ft31_w03_public_definition_selection_v3.json`,
`manifests/ft31_w03_public_definition_census_v3.json`,
`w03_public_sense_runtime_v3.json`, and the source pack below
`ph2_ft31_dataset_artifacts/public_definition_source_v3` form a second bounded
derived slice of the 2026-07-01 Chinese Wiktionary snapshot.

SPDX-License-Identifier: CC-BY-SA-4.0

Source: https://dumps.wikimedia.org/zhwiktionary/20260701/

License: https://creativecommons.org/licenses/by-sa/4.0/

The v3 selection excludes all v1 and v2 titles before ranking, then selects 64
titles from each frozen title-length stratum by the same snapshot/title hash.
Every derived definition and redirect retains page, revision, contributor,
license, URL, and attribution identity. The census records unsupported markup
and template-frequency evidence without treating frequency as authorization to
guess template semantics. These records remain external observations, not
project-adjudicated truth, training mastery, or an open-domain capability claim.
