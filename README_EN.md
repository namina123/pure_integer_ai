**English** | [中文](README.md)

# PIDSLCA: Pure-Integer Deterministic Self-Learning Cognitive Architecture

PIDSLCA is an open exploratory research project. It asks a specific question: can a cognitive system be built without floating-point computation while remaining runnable on an ordinary personal computer, reproducible bit for bit, auditable, and able to update from experience?

This repository publishes the reference implementation, current tests, continuous integration, format samples, development scripts, and paper materials. “Self-learning” describes the research objective and mechanisms; it does not mean that the system has achieved general intelligence, autonomous understanding, mature conversation, or is ready for production.

## Research theme

- Represent cognitive state, relation strength, counts, evidence, and protocol data with integers.
- Make fixed inputs and fixed state produce reproducible execution for audits and controlled comparisons.
- Use graph structures to represent concepts, relations, memory, order, causality, and executable structures.
- Study how relation reinforcement, structure induction, memory updates, constructive verification, and recovery can work together.
- Validate these mechanisms on ordinary hardware and standard Python environments instead of specialized large-scale infrastructure.

## What it is for

PIDSLCA currently serves as a research and engineering foundation for:

- reproducible experiments in deterministic cognitive architectures, graph reasoning, structure learning, and integer representations;
- auditable implementations where state changes can be traced to inputs, rules, and evidence;
- prototypes for memory, relation learning, generation, program execution, recovery, and evaluation;
- teaching and technical discussion supported by runnable code, public tests, and paper materials.

It is not currently a chat product, a general intelligence system, or a deployable decision service. Passing controlled engineering tests shows that implementations satisfy those test conditions; it does not replace real-world evaluation of semantics, generalization, or reliability.

## Distinctive features

- **Pure-integer core**: core computation paths avoid floating-point state, reducing cross-platform numeric variation.
- **Deterministic execution**: fixed inputs and protocol state should produce bit-identical results.
- **Count-based relation reinforcement**: relations accumulate through traceable integer counts and are promoted under explicit conditions.
- **Structure induction**: shared structures are extracted from alignable samples instead of storing only surface text.
- **Constructive verification**: executable results, inverse transformations, migrations, and recovery paths are checked independently.
- **Auditable boundaries**: implemented mechanisms, experimental abilities, and open research questions are stated separately.
- **Lightweight runtime**: the main package uses only the Python standard library; test dependencies are installed separately.

## Public progress

As of July 26, 2026:

- the code uses a standard `src/pure_integer_ai/` package layout and supports editable installation;
- current tests, samples, cross-platform CI, source guards, and credential scanning are public;
- pure-integer foundations, deterministic utilities, graph storage, memory and recovery, relation mechanisms, cognitive processes, training orchestration, and evaluation facilities are implemented;
- the complete local regression on CPython 3.14.3 reports `3708 passed`;
- active research remains on efficiency, long text and long-term context, formal training data, user interaction, and real semantic generalization.

## Support the project

This is independent research without institutional funding or commercial sponsorship. Donations primarily support public testing and CI, experimental compute and storage, code and paper archiving, and long-term maintenance.

**[Support through WeChat, Alipay, or Ko-fi](DONATE_EN.md)**

Support is entirely optional. It does not change the MIT terms or purchase roadmap priority, private builds, or exclusive access; public code, tests, and papers remain available on the same basis to everyone.

## Quick start

```bash
git clone https://github.com/namina123/pure_integer_ai.git pure_integer_ai
cd pure_integer_ai
python -m pip install -e ".[test]"
python -m pure_integer_ai.crosscut.guards.lint
python -m pytest -q
```

Run all commands from the repository root. CPython 3.11 and later are supported, with public CI coverage on Linux and Windows.

Files under `data/*.sample` are format examples only. Full corpora, credentials, local configuration, logs, databases, and experiment outputs are not stored in Git and are not read by the checks above. Builds, guards, and tests do not depend on unpublished documents or archived projects.

## Repository map

- `src/pure_integer_ai/`: installable source package
- `tests/`: current public regression suite
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
