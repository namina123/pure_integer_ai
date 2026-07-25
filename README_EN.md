**English** | [中文](README.md)

# PIDSLCA: Pure-Integer Deterministic Self-Learning Cognitive Architecture

PIDSLCA is an open exploratory research project. It asks a specific question: can a cognitive system be built without floating-point computation while remaining runnable on an ordinary personal computer, reproducible bit for bit, auditable, and able to update from experience?

This repository publishes the reference implementation, current tests, continuous-integration configuration, format samples, essential development scripts, and paper materials. “Self-learning” describes the research objective and implemented mechanisms; it does not mean that the system has achieved general intelligence, autonomous understanding, or mature conversational ability.

## Design focus

- **Pure-integer core**: core state, counters, strengths, and protocol keys are represented with integers.
- **Deterministic execution**: fixed inputs and protocol state should produce bit-identical results.
- **Edge-count reinforcement**: relations accumulate through traceable integer counts and are promoted under explicit conditions.
- **Structure induction**: shared skeletons are extracted from alignable structures across samples.
- **Constructive verification**: executable results, inverse transformations, and recovery paths are checked independently.
- **Auditable status**: production-live, opt-in, test-only, and incomplete mechanisms are distinguished so that code presence is not presented as achieved capability.

## Current status

Status snapshot: July 25, 2026.

| Scope | Status | Meaning |
|---|---|---|
| `PH1-CORE` | Complete | The first-phase core facilities have been assembled, forming `J-F1`. |
| `F-01` | Passed controlled assembly verification | Coverage includes source admission, Memory queries, question answering and generation, Use/outcome attribution, rollback, reparsing, migration, recovery, cloning, and parallel determinism. |
| `PH1-EXT` | Incomplete | `A-00` efficiency and surface work, `A-04` user familiarity/preferences, and `A-07` long-text and long-term context remain open. |
| `PH2` | Not entered | Formal curricula, production training data, and second-phase capability work have not begun. |

`J-F1` means only that the first-phase facilities can carry later data and training. It does not mean formal post-weaning operation, `readiness=true`, language mastery, a usable chat assistant, or production readiness. The current result is primarily controlled-fixture and deterministic engineering evidence, not empirical proof of general capability or semantic correctness.

## Verification record

The PH1 closure record from the engineering workspace is:

- T0 focused and inventory tests: `75 passed`
- T1 direct dependencies: `523 passed`
- T2 integration regression: `600 passed`
- T3 full regression, run twice: `3706 passed` each time
- Identical F-01 report SHA-256 under `PYTHONHASHSEED=0/1`: `ed7f35522053e3dcb257ee48f49f06ec742d98b5df64a7e8c465e532ca1d0905`
- Guards, source compilation, and `git diff --check` passed

These numbers are historical engineering records from PH1 closure. The current effective test suite is now published in this repository and uses the same entry points locally and in public CI. Archived implementations, obsolete tests, private design records, local corpora, and experiment outputs do not participate in the build or verification.

After the public-repository migration, an independent CPython 3.14.3 verification on July 26, 2026 passed editable installation, source compilation, all four built-in guards, and the complete suite: `3708 passed`.

## Quick start

The runtime uses only the Python standard library. The current engineering verification environment is CPython 3.14.3.

```bash
git clone https://github.com/namina123/pure_integer_ai.git pure_integer_ai
cd pure_integer_ai
python -m pip install -e ".[test]"
python -m pure_integer_ai.crosscut.guards.lint
python -m pytest -q
```

Run all commands from the repository root. Runtime code uses only the Python standard library; `.[test]` installs pytest for verification. CPython 3.11 and later are supported, with public CI coverage on Linux and Windows.

Files under `data/*.sample` are format examples only. Full corpora, credentials, local configuration, logs, databases, and experiment outputs are not stored in Git and are not read by the checks above. Builds, guards, and tests do not depend on unpublished documents or archived projects.

## Repository map

- `src/pure_integer_ai/`: installable source package
- `src/pure_integer_ai/cognition/`: cognitive objects, understanding, generation, and process mechanisms
- `src/pure_integer_ai/storage/`: events, Memory, recovery, and persistence
- `src/pure_integer_ai/numeric/`, `src/pure_integer_ai/vm/`: pure-integer numeric objects and graph-program execution
- `src/pure_integer_ai/experiments/`: training orchestration, runtime assembly, and evaluation protocols
- `src/pure_integer_ai/crosscut/`: determinism, integer constraints, and source guards
- `tests/`: current public regression suite
- `.github/workflows/ci.yml`: cross-platform tests and credential scanning
- `scripts/`: reusable public development helpers
- `paper/`: the paper PDF and LaTeX sources

## Paper

- [Paper PDF](paper/main.pdf)
- [LaTeX sources](paper/)
- [Zenodo archive and DOI: 10.5281/zenodo.21431532](https://doi.org/10.5281/zenodo.21431532)

The paper records the architecture and capability boundaries at publication time. For later code status, refer to this README and the implementation itself.

## Contributing

Reproducible bug reports, design discussions, and pull requests are welcome through [Issues](https://github.com/namina123/pure_integer_ai/issues). Read the [contribution guide](CONTRIBUTING.md) first, and state the behavioral impact, verification performed, and remaining coverage boundaries in each change.

## Open-source license

Original code and documentation in this repository are released under the [MIT License](LICENSE). Any person or organization may use, copy, modify, merge, publish, distribute, sublicense, or sell copies under its terms. The project has no separate commercial license, revenue threshold, field-of-use restriction, registration process, prior approval, rights assignment, or additional agreement. `LICENSE` is the sole licensing text.

## Support and contact

- [Support the research](DONATE_EN.md)
- Email: 2698801855@qq.com
