"""Run the graph-only probe with a read-only integer held-out overlay."""
from __future__ import annotations

from pathlib import Path

import scripts.run_core_graph_only_consumption_20260924g as base
from pure_integer_ai.experiments.heldout_response_runtime import HeldoutResponseRuntime


COURSE = Path(
    r"K:/pure_integer_ai_work/current_runs/"
    r"core-graph-heldout-20260920r5/heldout_response_course.int.json")
MANIFEST = Path(
    r"K:/pure_integer_ai_work/current_runs/"
    r"core-graph-heldout-20260920r5/heldout_manifest.json")
ROOT = Path(
    r"K:/pure_integer_ai_work/current_runs/"
    r"core-graph-only-heldout-20260924h")

_generator = None


class _CanonicalWithHeldout:
    def __init__(self, database: Path):
        self._canonical_context = base.TrainedGenerationConnectorRuntime(database)
        self._heldout_context = None
        self._canonical = None

    def __enter__(self):
        global _generator
        self._canonical = self._canonical_context.__enter__()
        self._heldout_context = HeldoutResponseRuntime(
            self._canonical, course_path=COURSE, manifest_path=MANIFEST)
        _generator = self._heldout_context.__enter__()
        return self._canonical

    def __exit__(self, exc_type, exc, traceback):
        global _generator
        try:
            if self._heldout_context is not None:
                self._heldout_context.__exit__(exc_type, exc, traceback)
        finally:
            _generator = None
            return self._canonical_context.__exit__(exc_type, exc, traceback)


class _GraphBridgeWithHeldout(base.TrainedGraphQueryBridge):
    def __init__(self, *args, **kwargs):
        if _generator is None:
            raise RuntimeError("heldout generator context is not active")
        kwargs["surface_generator"] = _generator
        super().__init__(*args, **kwargs)


def main() -> int:
    base.DATABASE = base.DATABASE
    base.INVENTORY = base.INVENTORY
    base.ROOT = ROOT
    base.TrainedGenerationConnectorRuntime = _CanonicalWithHeldout
    base.TrainedGraphQueryBridge = _GraphBridgeWithHeldout
    return base.main()


if __name__ == "__main__":
    raise SystemExit(main())
