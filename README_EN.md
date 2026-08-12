**English** | [中文](README.md)

# PIDSLCA: Pure-Integer Deterministic Self-Learning Cognitive Architecture

PIDSLCA is a fully open exploratory research project. It asks a specific question: can a cognitive system be built without floating-point computation while remaining runnable on an ordinary personal computer, reproducible bit for bit, auditable, and able to update from experience?

This repository publishes the reference implementation, current tests, continuous integration, format samples, development scripts, and the paper completed by the project author. "Self-learning" describes the research objective and mechanisms; it does not mean that the system has achieved general intelligence, autonomous understanding, mature conversation, or production readiness.

## Support independent research

PIDSLCA is independently researched and maintained, without institutional funding or commercial sponsorship. Donations sustain its development in public, including cross-platform tests and CI, experimental compute and storage, long-term code and paper archiving, and ongoing maintenance.

**[Support PIDSLCA's open research through WeChat, Alipay, or Ko-fi](DONATE_EN.md)**

Every contribution helps keep the code, tests, research record, and paper open to everyone. Support is entirely optional. It does not change the MIT terms or purchase roadmap priority, private builds, or exclusive access; public work remains available to everyone on the same terms.

## Research theme

- Represent cognitive state, relation strength, counts, evidence, and protocol data with integers.
- Make fixed inputs and fixed state produce reproducible execution for audits and controlled comparisons.
- Use graph structures to represent concepts, relations, memory, order, causality, and executable structures.
- Study how relation reinforcement, structure induction, memory updates, constructive verification, and recovery can work together.
- Validate these mechanisms on ordinary hardware and standard Python environments instead of specialized large-scale infrastructure.

## Distinctive features

- **Pure-integer core**: core computation paths avoid floating-point state, reducing cross-platform numeric variation.
- **Deterministic execution**: fixed inputs and protocol state should produce bit-identical results.
- **Count-based relation reinforcement**: relations accumulate through traceable integer counts and are promoted under explicit conditions.
- **Structure induction**: shared structures are extracted from alignable samples instead of storing only surface text.
- **Constructive verification**: executable results, inverse transformations, migrations, and recovery paths are checked independently.
- **Auditable boundaries**: implemented mechanisms, experimental abilities, and open research questions are stated separately.
- **Runs on ordinary hardware**: dependencies remain bounded, and the current indexes and probes can be built, run, and audited on a personal computer.

## What it is for

PIDSLCA currently serves as a research and engineering foundation for:

- reproducible experiments in deterministic cognitive architectures, graph reasoning, structure learning, and integer representations;
- auditable implementations where state changes can be traced to inputs, rules, and evidence;
- prototypes for memory, relation learning, generation, program execution, recovery, and evaluation;
- teaching and technical discussion supported by runnable code, public tests, and paper materials.

It is not currently a chat product, a general intelligence system, or a deployable decision service. Passing controlled engineering tests shows that implementations satisfy those test conditions; it does not replace real-world evaluation of semantics, generalization, or reliability.

## Public progress

The project is an actively developed research prototype. The public repository now includes:

- an installable pure-integer reference implementation with deterministic utilities, graph storage, memory, and recovery foundations;
- relation mechanisms, cognitive processes, training orchestration, generation, program execution, and evaluation facilities;
- regression tests aligned with the current implementation, cross-platform CI, format samples, and development helpers;
- the paper PDF, LaTeX sources, references, and permanent DOI archive information.

Current research focuses include runtime efficiency, long text and long-term context, formal training material, user interaction, and generalization and reliability in real semantic settings. Public tests describe the verified engineering scope; they do not imply that these open questions are solved.

The repository now contains a real-source Chinese broad factual QA vertical slice. It deterministically selects 100,000 candidate pages from a frozen Chinese Wikipedia snapshot and, without reselecting for answerability, publishes a compact index of 20,000 accepted pages, 109,006 passages, and 3,608,002 sparse features. Every `ANSWER` carries page, revision, contributor, raw evidence span/hash, and license identity. The fixed 24-question development probe produced `22 ANSWER / 1 UNKNOWN / 1 CLARIFY`, with zero citation reconstruction failures. The 20k SQLite artifact is 251,494,400 bytes with SHA-256 `e18db72b090dfdfd96aac23c74a5ad0751afe17c2dcfb02fc91f1213b0f7c4da`. A bounded multi-round posting merge preserved that SHA bit for bit while reducing real publication time from 840.972 to 542.421 seconds, a 35.501% reduction.

An independent external-context evidence-selection evaluation is now frozen and formally run once: 300 held-out questions, 100% exact citation validity, and `234/300 = 78%` gold-answer presence in the selected evidence sentence. The result passes the predeclared 70% evidence-selection gate (CMRC2018 `70%`, DRCD `86%`); aggregate SHA-256 is `82bc0c5083fe5c9ce4e8f1a3bfee756e3681fbd28ee0756e0e6bbefb9957c96d`. This measures answer-side evidence selection on provided contexts, not random-index retrieval, free-form generation, or general dialogue. See the [external evaluation report](docs/broad_qa_external_evidence_eval.md).

This remains a source-bound extractive preview, not free-form generation, general QA, or language weaning. Checkpointed projection, posting shards, external merge, receipt-last publication, and the one-shot held-out evidence evaluation have run in reality. The next priority is to connect evidence selection to random-index retrieval and close the remaining query/retrieval coverage gaps without weakening the frozen evaluation boundaries. See the [20k development preview](docs/broad_qa_20k_preview_en.md) and [external evaluation report](docs/broad_qa_external_evidence_eval.md) for contracts, reproduction paths, and limits.

## Quick start

```bash
git clone https://github.com/namina123/pure_integer_ai.git pure_integer_ai
cd pure_integer_ai
python -m pip install -e ".[test]"
python -m pure_integer_ai.crosscut.guards.lint
python -m pytest -q
```

Run all commands from the repository root. CPython 3.11 and later are supported, with public CI coverage on Linux and Windows. Files under `data/*.sample` are publicly distributable format examples; builds and tests do not depend on private material or archived projects.

### Experimental short-answer probe

After installation, query the experimental learned result built from the current public samples:

```bash
pure-integer-qa "什么使得河水上涨？"
```

The probe accepts a raw question and can optionally restrict it with `--source-ref 1,2,...`. It emits only the sparse short result by default; add `--audit` explicitly for complete audit traces. `--repeat N` runs warm queries on the same built runtime to check bit-identical repetition. This entry point demonstrates only the capabilities covered by the current public learned samples; it is not broad-domain QA or mature dialogue.

Startup validates and loads the repository's typed canonical snapshot by default. If the snapshot is missing, damaged, or inconsistent with the public source identities, partial loading is rejected and the runtime is rebuilt in full.

Use the long-lived JSONL mode to share one runtime build across different questions:

```bash
pure-integer-qa --jsonl
{"question":"什么使得河水上涨？"}
{"question":"河水上涨的原因是什么？","audit":false}
```

Each input object receives an immediate result record. A bad line emits a typed error without stopping later lines, and a final session probe is emitted when input ends.

### Source-bound broad-QA development preview

`pure-integer-broad-qa` selects pages from a frozen Chinese Wikipedia multistream snapshot, builds a compact integer index, and runs source-bound queries:

```bash
pure-integer-broad-qa query \
  --run-root <run-root> \
  --database <run-root>/indexes/broad-qa.sqlite3 \
  "矮寨大桥何时建成通车？"
```

Building requires the XML and index files pinned by the snapshot manifest from the official Wikimedia URLs. The repository does not commit the 3.5 GB source dump or the 20k SQLite artifact. The public fixed questions and path-independent runner are `data/ph2/broad_qa_dev_questions_v1.json` and `scripts/run_broad_qa_dev_probe.py`. This path currently performs sparse retrieval and source-bound extraction without an LLM; it is not free-form generation, mature dialogue, or completed open-domain semantic learning.

### Public-source sense probe

`pure-integer-sense` queries sense candidates compiled from bounded public Wiktionary and Wikidata slices:

```bash
pure-integer-sense "首页"
pure-integer-sense "金星" --context "距离太阳第二近的行星"
pure-integer-sense "金星" --primitive
pure-integer-sense "金星" --proposition
pure-integer-sense "什么是金星" --definition
pure-integer-sense "什么是金星" --context "{{lb|zh|astronomy}} [[太陽系]]的第二顆[[行星]]，為[[類地行星]]" --display-definition
pure-integer-sense "蘇維埃社會主義共和國聯盟" --artifact-version v2
pure-integer-sense "敗仗" --artifact-version v3
pure-integer-sense "亠" --artifact-version v4
```

Results retain traceable source information and distinguish unique, ambiguous, unknown, and unresolved cross-source conflict states. The current artifact covers only the frozen bounded public slice in this repository. A candidate is not a claim of definitive truth, and this probe is not an open-domain dictionary or broad-domain QA system.

The explicit `--primitive` mode projects the same candidates as typed source claims. `--proposition` adds structured roles, source identity, and lifecycle information. Both modes mean only that a source defines, labels, or aliases a value; neither is a project-level adjudication of truth.

The explicit `--definition` mode recognizes the general Chinese forms “什么是 X” and “X 是什么意思”. It returns source definition text only when both the sense and the active definition are unique. It refuses to select an answer for ambiguity, cross-source conflict, unknown terms, clarification cases, label-only or alias-only candidates, or multiple different definitions of one concept.

`--display-definition` applies a deterministic display projection only after that selection has produced one source definition. It retains the raw text, source, license, revision identity, and complete commitment chain. The current projection supports ordinary wiki links and `lb`/`label` domain labels. Unknown templates, nested or unbalanced structures, illegal escapes, ambiguous links, and non-unique sources preserve raw text and explicitly refuse rendering; they are not guessed, deleted, or polished by a language model. Every added mode is opt-in, so both the default output and the existing `--definition` output remain unchanged.

`--artifact-version v2` explicitly selects an expanded artifact sampled from the same public Wiktionary snapshot by title-length strata and stable title hashing. The frozen `v1` remains the default. The v2 selection is not revised according to parsing success, and its public audit retains pages with no definition, non-Chinese definitions, redirects, and unsupported templates. This expands bounded, source-constrained experimental coverage; it does not make the probe a complete dictionary or an open-domain QA system.

`--artifact-version v3` applies the same public rule to 256 titles not used by `v1` or `v2`. Its census retains every selected page and definition result and counts unknown templates in real Chinese definitions across independent pages, revisions, and occurrences. Meeting the frequency threshold does not authorize a renderer without separate public specification evidence. The default version and existing artifact bytes remain unchanged.

`--artifact-version v4` selects another 512 titles from the same 2026-07-01 Wiktionary snapshot after excluding all 293 titles already used by `v1`, `v2`, and `v3`. Selection depends only on the frozen snapshot, title-length strata, and stable hashing; parsing then reads only the selected multistream blocks. The public census records every selected page, definition, failure state, and template frequency while inheriting the public specification decisions: `place` remains blocked and `zh-div` remains unauthorized for rendering. v4 expands attributable, auditable experimental sense coverage; it is not training, truth adjudication, or an open-domain capability claim.

The repository also publishes a specification review of the six frequent template families newly qualified by v4 and a separate deterministic semantic projector. It currently supports only the evidence-closed Chinese profiles for alternative-form, synonym, Chinese alternative-form, and surname structures while retaining the raw template text, parameters, source, and commitment chain. `rfdef` is a maintenance request for a missing definition and is never presented as lexical content. The `†` call has no verifiable template identity in the frozen snapshot and remains an explicit refusal. This projector does not rewrite the frozen v4 artifact or expand default command behavior.

## Repository map

- `src/pure_integer_ai/`: installable source package
- `tests/`: public regression tests aligned with the current implementation
- `data/*.sample`: publicly distributable format samples
- `.github/workflows/ci.yml`: cross-platform tests and credential scanning
- `scripts/`: reusable public development helpers
- `paper/`: paper PDF, LaTeX sources, and references

## Paper

This repository publicly preserves and acknowledges the paper completed by the project author. The paper remains in its published form; later code status is documented by this README and the implementation.

- [Paper PDF](paper/main.pdf)
- [LaTeX sources](paper/)
- [Zenodo archive and DOI: 10.5281/zenodo.21431532](https://doi.org/10.5281/zenodo.21431532)

## Contributing

Reproducible bug reports, design discussions, and pull requests are welcome through [Issues](https://github.com/namina123/pure_integer_ai/issues). Read the [contribution guide](CONTRIBUTING.md) first, and state the behavioral impact, verification performed, and remaining coverage boundaries in each change.

## Open-source license

Original code and documentation in this repository are released under the [MIT License](LICENSE). Any person or organization may use, copy, modify, merge, publish, distribute, sublicense, or sell copies under its terms. The project has no separate commercial license, revenue threshold, field-of-use restriction, registration process, prior approval, rights assignment, or additional agreement. `LICENSE` is the sole licensing text for original repository content. Dependencies and external data retain their own licenses: the broad-QA path uses WikiTextParser under GPLv3, the OpenCC Python implementation under Apache-2.0, and Wikipedia-derived content under CC BY-SA 4.0. See the [third-party license boundary](docs/third_party_licenses.md).

## Contact

Email: 2698801855@qq.com
