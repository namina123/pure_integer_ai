# GG-03 Blind Evaluator Owner Handoff

This protocol is for an evaluator owner who is independent from candidate
development. It defines the artifacts that must exist before a GG-03 formal
family can be frozen. It does not authorize a formal run.

## Isolation

Use one fresh evaluation root with isolated candidate and private directories:

```text
evaluation-root/
  public-preflight/
    public-dry-run.receipt.json
  candidate-visible/
    observations/formal-observations.jsonl
    packs/<candidate-pack>/manifest.json
  private-label-owner/
    labels/formal-labels.jsonl.gz
    owner-receipt.json
```

The candidate-visible directory must never contain accepted surfaces, rejected
surfaces, evaluator verdicts, label hashes, or the private owner receipt. The
private-label-owner directory must not be made available to candidate execution.

## Owner Inputs

The independent owner selects new `GenerationGeneralizationEvaluationObservation`
records and writes them as canonical JSONL in stable-key order. The inventory
must:

- use one explicit `GenerationGeneralizationEvaluationBudget`;
- jointly cover every frozen GG-03 requirement;
- contain no Observation used by the public preflight;
- contain no complete held-out answer or accepted/rejected surface;
- use only sources and licenses that can be published with the Observation.

The V2 public dry-run receipt exposes sorted SHA-256 identities for both public
Observation stable keys and Observation content with `episode_id` removed.
Family freeze strictly reads the published receipt file, binds its relative path
and file SHA-256, and rejects any stable-key or content overlap with the formal
inventory. This includes partial overlap and simple episode renaming.

## Private Labels

For every formal Observation, the owner independently adjudicates:

- at least two distinct accepted complete surfaces;
- at least one distinct rejected complete surface;
- the exact frozen requirement projection for that Observation.

Use `build_generation_generalization_formal_label_record` to convert the owner
surfaces to exact SHA-256 sets. Use
`publish_generation_generalization_private_labels` once to publish the canonical
label transport and metadata-only `owner-receipt.json`. The transport contains no
surface plaintext. An unregistered generated surface is `NE`; it must not be
guessed PASS or FAIL.

The owner receipt must bind:

- the formal Observation transport SHA-256 and record count;
- label transport and content sizes and SHA-256 identities;
- the label record commitment;
- the current sealed verdict contract SHA-256.

Do not overwrite or regenerate an owner receipt. A correction requires a fresh
evaluation root and a new family.

## Candidate-Side Readback

Before family freeze, the candidate-side process may only:

1. double-scan the label-free formal Observation inventory;
2. read `owner-receipt.json` metadata;
3. strictly read the published V2 public-preflight receipt;
4. verify candidate, code, policy, public-preflight, resource, path, and
   contamination identities;
5. publish `family-freeze.json` and `guard.available.json` once.

It must not open, stat, decompress, hash, or parse the private label transport.
Private labels become readable only after the unique guard is consumed and the
complete prediction seal is written and read back.

## Required Handoff Values

Return only these safe values to the candidate-side operator:

```text
formal Observation relative path
formal Observation transport SHA-256
formal Observation record count
resource budget
owner receipt relative path
owner receipt SHA-256
label commitment SHA-256
sealed verdict contract SHA-256
```

Do not return surface text, surface hashes, label file paths beyond the relative
path already sealed in the owner receipt, or per-record verdicts.
