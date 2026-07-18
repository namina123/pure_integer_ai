**English** | [中文](README.md)

# Pure-Integer Intelligence · PIDSLCA

**PIDSLCA** (Pure-Integer Deterministic Self-Learning Cognitive Architecture) is an exploratory research project asking:

> **Can an intelligent system be built entirely on top of integers — running on an ordinary personal computer, and capable of self-learning?**

This repository is its reference implementation.

## Motivation

Mainstream intelligent systems are built almost entirely on floating-point arithmetic — neural-network weights are floats, and training is the iteration of floating-point gradients. This project explores a different path: what if the storage and computation of an intelligent system were placed entirely on top of integers?

The point is not to argue against floating point. It is to clarify one question: can a system built wholly from integers — runnable on an ordinary personal computer, reproducible bit-for-bit, and auditable — actually stand up as a self-learning system.

## What this is

The research uses pure integers (no floating-point operations) as the storage and computational substrate of cognition. Its core includes:

- **Pure-integer core**: concepts, relations, and strengths are all integers; reproducible bit-for-bit across machines.
- **Edge-count reinforcement (learning primitive)**: the strength of a relation is a monotonically non-decreasing integer, promoted when co-occurrence counts cross a threshold (tally-to-promote).
- **Six interlocking invariants**: pure-integer, bit-identical, monotone-with-overwrite, append-only-audit, teacher boundary, downward-only dependency — together constraining reproducibility and auditability.
- **Structure induction**: a shared skeleton is extracted by aligning multiple samples, as a form of template abstraction.
- **Constructive self-check**: for teacher-declared inverse transformations, B∘A is checked against identity, yielding a three-valued verdict.
- **Anti-theater discipline**: every mechanism is labeled with its real status (production-live / opt-in gate / test-only / dead), so that "the mechanism is wired" does not silently mean "the capability is achieved".

## Current status

- **Arithmetic domain**: the end-to-end demonstration is complete — given a symbolic arithmetic specification, the system compiles it into an integer COMPOSES graph, executes it on an integer virtual machine using exact rational arithmetic, self-checks the result by recomputation, and passes six falsifiable gates.
- **Language domain**: the corpus-driven learning mechanisms are wired (cue emergence, correspondence bridge, floor measurement), but coherent sentence generation and conversational ability are still under construction.

Capability boundaries are stated in full in the paper.

## Paper

The paper gives a complete account of the architecture and its capability boundaries.

The author is an **independent researcher** and currently cannot obtain the endorsement required to submit a preprint to arXiv. The paper is therefore archived on **Zenodo**, which issues a formal, citable DOI without an endorsement barrier:

- 📄 Paper PDF (in this repo): [`paper/main.pdf`](paper/main.pdf)
- 📝 Paper sources (LaTeX): [`paper/`](paper/)
- 🔗 **Zenodo archive & DOI**: [10.5281/zenodo.21431532](https://doi.org/10.5281/zenodo.21431532) (permanently citable)

To discuss the paper on [alphaXiv](https://www.alphaxiv.org), upload the same PDF there (alphaXiv supports uploading non-arXiv papers as PDFs for discussion).

## Quick start

```bash
# Three guard checks (pure-integer / downward-only dependency / bit-identical)
python -m pure_integer_ai.crosscut.guards.lint
```

> The full training entry point and test suite live in a separate working repository; this repo focuses on the architectural reference implementation. See the paper's Appendix B for full reproducibility notes.

## Support this research

This is independent research with no institutional backing and no commercial funding, carried out in the author's spare time. If you find it valuable, you are welcome to offer support → [Support the research (donate)](DONATE_EN.md).

## License

- Code copyright: [MIT License](LICENSE)
- Patent rights: granted in tiers per [PATENTS.md](PATENTS.md) (free for individuals and small organizations; commercial use by separate agreement)
- Commercial licensing: see [COMMERCIAL.md](COMMERCIAL.md)

## Contact

Email: 2698801855@qq.com
