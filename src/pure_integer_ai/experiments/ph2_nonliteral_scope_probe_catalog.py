"""构建 NL-00 五层非字面与文化依赖语言的 bounded 范围裁决。"""
from __future__ import annotations

import hashlib
from pathlib import Path

from pure_integer_ai.experiments.ph2_dataset_contract import CanonicalJsonObject
from pure_integer_ai.experiments.ph2_nonliteral_scope_probe_contract import (
    ARTIFACT_STATUS,
    EXECUTION_STATE,
    FORMAT_VERSION,
    RUNTIME_STATUS,
    UNRESOLVED_DECISION_KEYS,
    VERIFIER_DIMENSIONS,
    VERIFIER_NE_CONDITIONS,
    NonliteralEvidenceFile,
    NonliteralLayerDecision,
    NonliteralScopeProbeManifest,
)


NL00_MANIFEST_PATH = Path(
    "data/ph2/manifests/nl00_nonliteral_scope_probe_manifest_v1.json")
NL00_ARTIFACT_VERSION = "NL-00-nonliteral-scope-probe-manifest-v1"
BASELINE_MANIFEST_PATH = Path(
    "data/ph2/manifests/language_capability_baseline_v35.json")


class NonliteralScopeProbeCatalogError(RuntimeError):
    """NL-00 基线或现有 evidence 文件身份无法闭合。"""


def _sha256_path(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while True:
            block = handle.read(1024 * 1024)
            if not block:
                break
            digest.update(block)
    return digest.hexdigest()


def _decisions() -> tuple[NonliteralLayerDecision, ...]:
    return (
        NonliteralLayerDecision(
            "CONVENTIONAL_IMPLICATURE",
            "PARTIAL_SCAFFOLD",
            0,
            (
                "CONTEXT_SCOPED_PROPOSITION_REUSE",
                "DISCOURSE_INFORMATION_CANDIDATE_REUSE",
                "SOURCE_REF_EVIDENCE_REUSE",
            ),
            (
                "LC07_DISCOURSE_INFORMATION_COURSE",
                "LC08_OPEN_SET_CLARIFICATION_COURSE",
                "SOURCE_SCOPED_OBSERVATION",
            ),
            (
                "ATTRIBUTED_CONTENT_NOT_CURRENT_FACT",
                "EXPLICIT_CANCELLATION",
                "SAME_DIRECT_CONTENT_DIFFERENT_CONVENTION",
            ),
            CanonicalJsonObject.from_value({
                "CANCELLABILITY_DISTINCT_FROM_TRUTH": "REJECT",
                "DIRECT_CONTENT_PRESERVED": "PASS",
                "SOURCE_SCOPE_PRESERVED": "PASS",
            }),
            "INDEPENDENT_EVALUATOR_ABSENT",
            "SOURCE_CONDITIONAL",
            0,
            "REJECT",
            "DEFER_DEDICATED_CONVENTIONAL_IMPLICATURE_OBJECT_AND_EVALUATOR",
            "LANGUAGE_INTERNAL_BUT_DISC08_DISC12_UNDECIDED",
            (
                "src/pure_integer_ai/cognition/shared/semantic_object.py",
                "src/pure_integer_ai/experiments/ph2_authored_discourse_information_course.py",
                "src/pure_integer_ai/experiments/ph2_authored_open_set_clarification_course.py",
            ),
            (
                "CONVENTIONAL_IMPLICATURE_RUNTIME_ABSENT",
                "DISC08_DEPTH_UNDECIDED",
                "DISC12_EVALUATOR_SIGNAL_UNDECIDED",
            ),
        ),
        NonliteralLayerDecision(
            "CULTURAL_ALLUSION",
            "PARTIAL_SCAFFOLD",
            0,
            (
                "CANDIDATE_LEARNING_RUNTIME_REUSE",
                "SOURCE_REF_EVIDENCE_REUSE",
                "UNKNOWN_GROUNDING_STATUS_REUSE",
            ),
            (
                "LC08_ACTIVE_EVIDENCE_REQUEST",
                "SOURCE_SCOPED_OBSERVATION",
            ),
            (
                "CONFLICTING_CULTURAL_SOURCE",
                "SAME_ALLUSION_DIFFERENT_ERA_DOMAIN",
                "UNKNOWN_REFERENCE_NO_GUESS",
            ),
            CanonicalJsonObject.from_value({
                "ALLUSION_SOURCE_IDENTITY": "PASS",
                "CULTURAL_GROUNDING_AUTHORIZED": "NE",
                "UNKNOWN_NOT_GUESSED": "PASS",
            }),
            "INDEPENDENT_EVALUATOR_ABSENT",
            "EXTERNAL_GROUNDING_NE",
            0,
            "NE",
            "PRESERVE_SOURCE_CANDIDATE_AND_REQUEST_EVIDENCE_WITHOUT_INTERPRETIVE_PASS",
            "W1_OR_EXTERNAL_CULTURAL_GROUNDING_REQUIRED_FOR_DEFINITIVE_CLAIM",
            (
                "src/pure_integer_ai/cognition/shared/candidate_runtime.py",
                "src/pure_integer_ai/cognition/shared/identity.py",
                "src/pure_integer_ai/experiments/ph2_authored_open_set_clarification_course.py",
            ),
            (
                "CULTURAL_GROUNDING_EVALUATOR_UNAUTHORIZED",
                "DISC08_DEPTH_UNDECIDED",
                "DISC12_EVALUATOR_SIGNAL_UNDECIDED",
                "W1_EXTERNAL_GROUNDING_NOT_AVAILABLE",
            ),
        ),
        NonliteralLayerDecision(
            "IRONY_HUMOR",
            "PARTIAL_SCAFFOLD",
            0,
            (
                "CONTEXT_SCOPED_PROPOSITION_REUSE",
                "DISCOURSE_INFORMATION_CANDIDATE_REUSE",
                "SOURCE_REF_EVIDENCE_REUSE",
            ),
            (
                "LC07_DISCOURSE_INFORMATION_COURSE",
                "SOURCE_SCOPED_OBSERVATION",
            ),
            (
                "NO_IRONY_CUE_UNKNOWN",
                "SAME_WORDS_SINCERE_CONTEXT",
                "SPEAKER_STANCE_NOT_MIND_READABLE",
            ),
            CanonicalJsonObject.from_value({
                "LITERAL_CONTENT_PRESERVED": "PASS",
                "MIND_READING_FORBIDDEN": "PASS",
                "STANCE_EXPECTATION_CONTRAST": "REJECT",
            }),
            "INDEPENDENT_EVALUATOR_ABSENT",
            "SOURCE_CONDITIONAL",
            0,
            "REJECT",
            "DEFER_STANCE_EXPECTATION_CONTRAST_RUNTIME_AND_INDEPENDENT_EVALUATOR",
            "LANGUAGE_INTERNAL_CANDIDATE_ONLY; DEFINITIVE_SPEAKER_STATE_FORBIDDEN",
            (
                "src/pure_integer_ai/cognition/shared/hypothesis.py",
                "src/pure_integer_ai/cognition/shared/semantic_object.py",
                "src/pure_integer_ai/experiments/ph2_authored_discourse_information_course.py",
            ),
            (
                "DISC08_DEPTH_UNDECIDED",
                "DISC12_EVALUATOR_SIGNAL_UNDECIDED",
                "IRONY_HUMOR_RUNTIME_ABSENT",
            ),
        ),
        NonliteralLayerDecision(
            "LEXICALIZED_IDIOM",
            "AVAILABLE_NOT_EXECUTED",
            1,
            (
                "CONSTRUCTION_IDENTITY_V1",
                "LEXICALIZATION_STATE_V1",
                "REGISTER_SCOPED_SURFACE_V1",
            ),
            (
                "LC03_AUTHORED_CONSTRUCTION_COURSE",
                "SOURCE_SCOPED_OBSERVATION",
            ),
            (
                "LITERAL_TOKEN_SUM_BASELINE",
                "SAME_SURFACE_DIFFERENT_CONSTRUCTION",
                "WHOLE_VS_PARTIAL_LEXICALIZATION",
            ),
            CanonicalJsonObject.from_value({
                "ANTI_LITERAL_BASELINE": "PASS",
                "LEXICALIZATION_IDENTITY": "PASS",
                "REGISTER_SCOPE": "PASS",
            }),
            "STRUCTURAL_ONLY",
            "LANGUAGE_INTERNAL",
            0,
            "PASS",
            "FIRST_PHASE_LEXICALIZED_IDIOM_REPRESENTABILITY_IN_SCOPE",
            "NO_EXTERNAL_GROUNDING_REQUIRED_FOR_FROZEN_LEXICAL_IDENTITY",
            (
                "data/ph2/manifests/lc03_construction_course_v1.json",
                "src/pure_integer_ai/experiments/ph2_authored_construction_course.py",
                "tests/test_d02_lc03_construction_course.py",
            ),
            (),
        ),
        NonliteralLayerDecision(
            "PRODUCTIVE_METAPHOR_METONYMY",
            "PARTIAL_SCAFFOLD",
            0,
            (
                "CANDIDATE_LEARNING_RUNTIME_REUSE",
                "CONTEXT_SCOPED_PROPOSITION_REUSE",
                "SOURCE_REF_EVIDENCE_REUSE",
            ),
            (
                "LC03_CONSTRUCTION_CANDIDATE_SOURCE",
                "LC08_ACTIVE_EVIDENCE_REQUEST",
                "SOURCE_SCOPED_OBSERVATION",
            ),
            (
                "LITERAL_AND_FIGURATIVE_SAME_SURFACE",
                "NOVEL_SOURCE_TARGET_PAIR",
                "REVERSED_OR_UNSUPPORTED_MAPPING",
            ),
            CanonicalJsonObject.from_value({
                "GROUNDING_SCOPE_EXPLICIT": "PASS",
                "LITERAL_FIGURATIVE_COMPETITION": "NE",
                "SOURCE_TARGET_MAPPING_TYPED": "REJECT",
            }),
            "INDEPENDENT_EVALUATOR_ABSENT",
            "SOURCE_CONDITIONAL",
            0,
            "REJECT",
            "DEFER_PRODUCTIVE_MAPPING_OBJECT_COMPETITION_AND_EVALUATOR",
            "LANGUAGE_INTERNAL_CANDIDATE_POSSIBLE; EXTERNAL_GROUNDING_CONDITIONAL",
            (
                "src/pure_integer_ai/cognition/shared/candidate_runtime.py",
                "src/pure_integer_ai/cognition/shared/semantic_object.py",
                "src/pure_integer_ai/experiments/ph2_authored_construction_course.py",
            ),
            (
                "DISC08_DEPTH_UNDECIDED",
                "DISC12_EVALUATOR_SIGNAL_UNDECIDED",
                "PRODUCTIVE_FIGURATIVE_MAPPING_RUNTIME_ABSENT",
            ),
        ),
    )


def _evidence_inventory(
        repository_root: Path,
        decisions: tuple[NonliteralLayerDecision, ...],
        ) -> tuple[NonliteralEvidenceFile, ...]:
    result = []
    for relative_path in sorted({
            path for decision in decisions for path in decision.evidence_refs}):
        path = repository_root / Path(*relative_path.split("/"))
        if not path.is_file():
            raise NonliteralScopeProbeCatalogError(
                f"NL-00 evidence 文件缺失: {relative_path}")
        result.append(NonliteralEvidenceFile(
            relative_path, path.stat().st_size, _sha256_path(path)))
    return tuple(result)


def build_nonliteral_scope_probe_manifest(
        repository_root: str | Path,
        ) -> NonliteralScopeProbeManifest:
    """构建五层 bounded scope probe，不扩张深层语用和文化结论。"""
    repository = Path(repository_root).resolve()
    baseline_path = repository / BASELINE_MANIFEST_PATH
    if not baseline_path.is_file():
        raise NonliteralScopeProbeCatalogError("NL-00 基线文件缺失")
    decisions = _decisions()
    return NonliteralScopeProbeManifest(
        FORMAT_VERSION,
        NL00_ARTIFACT_VERSION,
        ARTIFACT_STATUS,
        RUNTIME_STATUS,
        "NL-00",
        BASELINE_MANIFEST_PATH.as_posix(),
        _sha256_path(baseline_path),
        decisions,
        _evidence_inventory(repository, decisions),
        sum(item.verdict == "PASS" for item in decisions),
        sum(item.verdict == "REJECT" for item in decisions),
        sum(item.verdict == "NE" for item in decisions),
        1,
        0,
        0,
        UNRESOLVED_DECISION_KEYS,
        VERIFIER_DIMENSIONS,
        VERIFIER_NE_CONDITIONS,
        CanonicalJsonObject.from_value(EXECUTION_STATE),
    )


__all__ = [
    "BASELINE_MANIFEST_PATH",
    "NL00_ARTIFACT_VERSION",
    "NL00_MANIFEST_PATH",
    "NonliteralScopeProbeCatalogError",
    "build_nonliteral_scope_probe_manifest",
]
