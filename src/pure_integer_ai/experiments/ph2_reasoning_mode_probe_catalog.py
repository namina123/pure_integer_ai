"""构建 RI-00 五类额外推理模式的文件绑定 bounded 决断。"""
from __future__ import annotations

import hashlib
from pathlib import Path

from pure_integer_ai.experiments.ph2_dataset_contract import CanonicalJsonObject
from pure_integer_ai.experiments.ph2_reasoning_mode_probe_contract import (
    ARTIFACT_STATUS,
    EXECUTION_STATE,
    FORMAT_VERSION,
    RUNTIME_STATUS,
    VERIFIER_DIMENSIONS,
    VERIFIER_NE_CONDITIONS,
    ReasoningModeEvidenceFile,
    ReasoningModeProbeDecision,
    ReasoningModeProbeManifest,
)


RI00_MANIFEST_PATH = Path(
    "data/ph2/manifests/ri00_reasoning_mode_probe_manifest_v2.json")
RI00_ARTIFACT_VERSION = (
    "RI-00-reasoning-mode-probe-manifest-v2-supersedes-v1")
BASELINE_MANIFEST_PATH = Path(
    "data/ph2/manifests/language_capability_baseline_v34.json")


class ReasoningModeProbeCatalogError(RuntimeError):
    """RI-00 基线或模式 evidence 文件身份无法闭合。"""


def _sha256_path(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while True:
            block = handle.read(1024 * 1024)
            if not block:
                break
            digest.update(block)
    return digest.hexdigest()


def _decisions() -> tuple[ReasoningModeProbeDecision, ...]:
    return (
        ReasoningModeProbeDecision(
            "ABDUCTION",
            "PARTIAL_SCAFFOLD",
            0,
            CanonicalJsonObject.from_value({
                "NO_NEW_CAUSES": "PASS",
                "TYPED_ABDUCTIVE_BRANCH": "REJECT",
            }),
            0,
            0,
            "REJECT",
            "NO_TYPED_ABDUCTIVE_BRANCH; CAUSES_MINTING_FORBIDDEN",
            (
                "src/pure_integer_ai/cognition/shared/reasoning_planner.py",
                "src/pure_integer_ai/experiments/causal_relation_runtime.py",
            ),
            ("TYPED_ABDUCTIVE_BRANCH_ABSENT",),
        ),
        ReasoningModeProbeDecision(
            "COUNTERFACTUAL",
            "ABSENT",
            0,
            CanonicalJsonObject.from_value({
                "COUNTERFACTUAL_BRANCH_RUNTIME": "REJECT",
                "CURRENT_PROJECTION_POLLUTION_ZERO": "NE",
            }),
            0,
            0,
            "REJECT",
            "NO_ISOLATED_COUNTERFACTUAL_BRANCH; CURRENT_PROJECTION_CLAIM_NE",
            (
                "src/pure_integer_ai/cognition/shared/situation_state.py",
                "src/pure_integer_ai/experiments/logic_closure_runtime.py",
            ),
            ("COUNTERFACTUAL_BRANCH_RUNTIME_ABSENT",),
        ),
        ReasoningModeProbeDecision(
            "DEFEASIBLE_DEFAULT",
            "PARTIAL_SCAFFOLD",
            0,
            CanonicalJsonObject.from_value({
                "DEFAULT_EXCEPTION_REVERSAL": "REJECT",
                "SOURCE_SCOPE_PRIORITY": "NE",
            }),
            0,
            0,
            "REJECT",
            "REVISION_EXISTS_BUT_DEFAULT_EXCEPTION_PRIORITY_RUNTIME_ABSENT",
            (
                "src/pure_integer_ai/cognition/shared/memory_event_log.py",
                "src/pure_integer_ai/experiments/ph2_authored_discourse_information_course.py",
            ),
            ("DEFEASIBLE_PRIORITY_RUNTIME_ABSENT",),
        ),
        ReasoningModeProbeDecision(
            "DEONTIC_NORMATIVE",
            "PARTIAL_SCAFFOLD",
            0,
            CanonicalJsonObject.from_value({
                "NORMATIVE_FACT_PROJECTION_SEPARATION": "REJECT",
                "NORM_CONFLICT_STATUS": "NE",
            }),
            0,
            0,
            "REJECT",
            "MODAL_COURSE_EXISTS_BUT_NORMATIVE_FACT_LEDGER_IS_NOT_SEPARATE",
            (
                "src/pure_integer_ai/cognition/shared/semantic_object.py",
                "src/pure_integer_ai/experiments/logic_closure_runtime.py",
                "src/pure_integer_ai/experiments/ph2_authored_modal_course.py",
            ),
            ("DEONTIC_PROJECTION_RUNTIME_ABSENT",),
        ),
        ReasoningModeProbeDecision(
            "TEMPORAL",
            "AVAILABLE_NOT_EXECUTED",
            1,
            CanonicalJsonObject.from_value({
                "SOURCE_SCOPE_PRESERVED": "PASS",
                "SURFACE_ORDER_NOT_TRUTH": "PASS",
                "TEMPORAL_FOUR_STATE": "PASS",
            }),
            0,
            0,
            "PASS",
            "BOUNDED_TYPED_PRECEDENCE_EVENT_TIME_ONLY",
            (
                "src/pure_integer_ai/cognition/shared/event_time.py",
                "src/pure_integer_ai/experiments/event_time_runtime.py",
                "src/pure_integer_ai/experiments/precedence_relation_runtime.py",
                "tests/test_r06_precedence_relation_runtime.py",
                "tests/test_r06b_event_time_production_runtime.py",
            ),
            (),
        ),
    )


def _evidence_inventory(
        repository_root: Path,
        decisions: tuple[ReasoningModeProbeDecision, ...],
        ) -> tuple[ReasoningModeEvidenceFile, ...]:
    result = []
    for relative_path in sorted({
            path for decision in decisions for path in decision.evidence_refs}):
        path = repository_root / Path(*relative_path.split("/"))
        if not path.is_file():
            raise ReasoningModeProbeCatalogError(
                f"RI-00 evidence 文件缺失: {relative_path}")
        result.append(ReasoningModeEvidenceFile(
            relative_path, path.stat().st_size, _sha256_path(path)))
    return tuple(result)


def build_reasoning_mode_probe_manifest(
        repository_root: str | Path,
        ) -> ReasoningModeProbeManifest:
    """构建五模式 bounded probe 和不扩张范围的直接裁决。"""
    repository = Path(repository_root).resolve()
    baseline_path = repository / BASELINE_MANIFEST_PATH
    if not baseline_path.is_file():
        raise ReasoningModeProbeCatalogError("RI-00 基线文件缺失")
    decisions = _decisions()
    return ReasoningModeProbeManifest(
        FORMAT_VERSION,
        RI00_ARTIFACT_VERSION,
        ARTIFACT_STATUS,
        RUNTIME_STATUS,
        "RI-00",
        BASELINE_MANIFEST_PATH.as_posix(),
        _sha256_path(baseline_path),
        decisions,
        _evidence_inventory(repository, decisions),
        sum(item.verdict == "PASS" for item in decisions),
        sum(item.verdict == "REJECT" for item in decisions),
        sum(item.verdict == "NE" for item in decisions),
        0,
        VERIFIER_DIMENSIONS,
        VERIFIER_NE_CONDITIONS,
        CanonicalJsonObject.from_value(EXECUTION_STATE),
    )


__all__ = [
    "BASELINE_MANIFEST_PATH",
    "RI00_ARTIFACT_VERSION",
    "RI00_MANIFEST_PATH",
    "ReasoningModeProbeCatalogError",
    "build_reasoning_mode_probe_manifest",
]
