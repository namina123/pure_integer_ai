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
- **Lightweight runtime**: the main package uses only the Python standard library and can be studied on an ordinary personal computer.

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

Original code and documentation in this repository are released under the [MIT License](LICENSE). Any person or organization may use, copy, modify, merge, publish, distribute, sublicense, or sell copies under its terms. The project has no separate commercial license, revenue threshold, field-of-use restriction, registration process, prior approval, rights assignment, or additional agreement. `LICENSE` is the sole licensing text.

## Contact

Email: 2698801855@qq.com
