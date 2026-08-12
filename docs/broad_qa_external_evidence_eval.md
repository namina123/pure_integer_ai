# External evidence-selection evaluation

This repository includes a reproducible, label-isolated evaluation of the broad-QA answer-side evidence selector.

## What is measured

The evaluation gives the system an official Chinese reading-comprehension context and a question. The system must select one complete evidence sentence from that context. A result is valid only when its byte range and SHA-256 identify an exact context span. The score counts whether at least one official gold answer occurs in the selected sentence.

This is an external-context evidence-selection benchmark. It does not measure retrieval from the random 20k Wikipedia index, free-form generation, truth adjudication, long-term memory, or general dialogue.

## Frozen sources

- CMRC2018, repository commit `c0eb1b6ba219847457e6af3180da722bbeb656af`, CC BY-SA 4.0.
- DRCD, repository commit `b944790de5af02c5fbb7cd9cb1473d27d169eebf`. The repository does not provide a machine-readable license file; its README states that the Wikipedia-derived content is released under CC BY-SA 3.0.

The six source-file SHA-256 values, parser anomaly counts, split rule, and artifact identities are recorded in the K-disk freeze manifest. The source copies and evaluation payloads are intentionally not committed to this repository.

## Split and isolation

The frozen pack contains 200 development questions and 300 held-out questions, with 100/150 from each source. A normalized title hash assigns a title domain to one split; item SHA-256 then chooses the quota. The two splits have zero title-domain overlap. Question JSONL contains no gold answers. Labels are stored separately and are read only by the scoring phase.

## Formal result

The algorithm, split, denominator, and threshold were frozen after development. The held-out set was run exactly once:

| measure | result |
| --- | ---: |
| questions | 300 |
| exact citation-valid | 300/300 (100%) |
| gold answer in selected evidence | 234/300 (78%) |
| CMRC2018 | 105/150 (70%) |
| DRCD | 129/150 (86%) |
| status | PASS at the predeclared 70% evidence-hit gate |

Formal aggregate SHA-256: `82bc0c5083fe5c9ce4e8f1a3bfee756e3681fbd28ee0756e0e6bbefb9957c96d`.

The result is a bounded, extractive capability result. It is evidence that the answer-side selector can follow external question wording to an auditable sentence on this frozen source mix. It is not a claim of broad-domain retrieval, open-ended language understanding, general intelligence, or weaning.

## Reproduction

After obtaining the pinned source checkouts and the K-disk freeze pack, run prediction and scoring as separate phases:

```powershell
python -m pure_integer_ai.experiments.run_ph2_broad_qa_external_runtime predict `
  --questions K:\...\external-source-pack-v1\held_out.questions.jsonl `
  --predictions K:\...\external-heldout-runtime-v1\held_out.predictions.jsonl

python -m pure_integer_ai.experiments.run_ph2_broad_qa_external_runtime score `
  --questions K:\...\external-source-pack-v1\held_out.questions.jsonl `
  --predictions K:\...\external-heldout-runtime-v1\held_out.predictions.jsonl `
  --labels K:\...\external-source-pack-v1\held_out.labels.jsonl `
  --aggregate K:\...\external-heldout-runtime-v1\held_out.aggregate.json `
  --scope FORMAL_HELD_OUT
```

The evaluator rejects changed source bytes, mismatched inventories, invalid context spans, changed hashes, and repeated publication. The source pack may be removed after its rebuild contract and compact receipts are retained.
