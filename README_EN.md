**English** | [中文](README.md)

# Pure-Integer Intelligence · PIDSLCA

**PIDSLCA** (Pure-Integer Deterministic Self-Learning Cognitive Architecture) is an exploratory research project asking:

> **Can an intelligent system be built entirely on top of integers — running on an ordinary personal computer, and capable of self-learning?**

This repository is its reference implementation.

## What this is

This is an **existence exploration of an architecture**, not a finished AI product. The research attempts to use pure integers (no floating-point operations) as the storage and computational substrate of cognition, and takes "edge-count reinforcement" as the learning primitive — the strength of a relation is a monotonically non-decreasing integer, promoted when co-occurrence counts cross a threshold.

- **Pure-integer core**: concepts, relations, and strengths are all integers; reproducible bit-for-bit across machines.
- **Six interlocking invariants**: constrain reproducibility and auditability.
- **Arithmetic domain**: an end-to-end demonstration of the mechanism (compile → integer-VM execution → self-check → six gates).
- **Language domain**: a second domain under active construction — mechanisms are wired but no usable capability yet.

Honestly, it **cannot yet hold a conversation or generate coherent sentences**. That capability is still being built; the paper states these boundaries in full.

## Paper

The paper gives a complete account of the architecture and an honest statement of capability boundaries.

The author is an **independent researcher** and currently cannot obtain the endorsement required to submit a preprint to arXiv. The paper is therefore archived on **Zenodo**, which issues a formal, citable DOI without an endorsement barrier:

- 📄 Paper PDF (in this repo): [`paper/main.pdf`](paper/main.pdf)
- 📝 Paper sources (LaTeX): [`paper/`](paper/)
- 🔗 **Zenodo archive & DOI**: *(the paper will be uploaded to Zenodo shortly; the link will be added here)*

To discuss the paper on [alphaXiv](https://www.alphaxiv.org): once the paper is on Zenodo, you can upload the same PDF to alphaXiv to start an open discussion (alphaXiv supports uploading non-arXiv papers as PDFs for discussion).

## Quick start

```bash
# Three guard checks (pure-integer / downward-only dependency / bit-identical)
python -m pure_integer_ai.crosscut.guards.lint
```

> The full training entry point and test suite live in a separate working repository; this repo focuses on the architectural reference implementation. See the paper's Appendix B for full reproducibility notes.

## Support this research

This is **independent research with no institutional backing and no commercial funding**, carried out in the author's spare time. If you find it valuable, you are welcome (**entirely optional**) to offer support → [Support the research (donate)](DONATE_EN.md).

## License

- Code copyright: [MIT License](LICENSE)
- Patent rights: granted in tiers per [PATENTS.md](PATENTS.md) (free for individuals and small organizations; commercial use by separate agreement)
- Commercial licensing: see [COMMERCIAL.md](COMMERCIAL.md)

## Contact

Email: 2698801855@qq.com

---

*This project is an honest record of an exploration in progress, not a claim of achieved results.*
