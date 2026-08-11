"""把两个已学问题构造族交叉应用到两个来源绑定 Proposition。"""
from __future__ import annotations

from dataclasses import dataclass
import hashlib
from pathlib import Path

from pure_integer_ai.experiments.ph2_d03_contract_core import (
    canonical_json_bytes,
)
from pure_integer_ai.experiments.ph2_w03_w04_w05_raw_question import (
    build_raw_question_catalog,
    compile_raw_question_pattern,
)
from pure_integer_ai.experiments.ph2_w03_w04_w05_raw_question_contract import (
    RawQuestionConstruction,
    RawQuestionPattern,
)
from pure_integer_ai.experiments.ph2_w03_w04_w05_vertical import (
    run_w03_w04_w05_vertical_queries,
)
from pure_integer_ai.experiments.ph2_w03_w04_w05_vertical_contract import (
    W03W04W05VerticalQuery,
    W03W04W05VerticalResult,
)
from pure_integer_ai.experiments.ph2_w03_w04_w05_vertical_generalization import (
    VERTICAL_GENERALIZATION_TARGETS,
    W03W04W05VerticalGeneralizationOverlay,
)


RAW_QUESTION_CAUSE_GENERALIZATION_SAMPLE_SHA256 = (
    "dd70024336d9dd6a493d7e36bce93f908512b75ae57ef050eddfdf9324b2aa5f")
RAW_QUESTION_EFFECT_GENERALIZATION_SAMPLE_SHA256 = (
    "b3b71546de9dc46b243ef2299b4be1b06fa6c58c5f183794c846cdef1f91d897")
RAW_QUESTION_GENERALIZATION_VERTICAL_SHA256S = (
    "73e42a13ffea503eb8b37079a8f98812de9865f0b397552b84e25a5a1065e903",
    "a45e601145072d5e2e714076a61867fef083e37606d6d3f07a0b6798e38778ba",
)
RAW_QUESTION_GENERALIZATION_BUNDLE_SHA256 = (
    "b48d4c304cd17945b9a4dda0dd84f4b42dc9acb01ef6ab9f5f364fd85a97a514")
RAW_QUESTION_GENERALIZATION_EXPRESSION_BOUNDARY = (
    ("different_segment_order", "SUPPORTED_BY_LEARNED_CONSTRUCTION"),
    ("explicit_predicate", "SUPPORTED_BY_SOURCE_OCCURRENCE"),
    ("implicit_predicate", "UNKNOWN_UNTIL_A_LEARNED_STRUCTURE_EXISTS"),
    ("predicate_alias", "UNKNOWN_UNTIL_A_LEARNED_LEXICAL_LINK_EXISTS"),
    ("role_inventory", "CURRENTLY_PROVEN_FOR_TWO_ROLE_PROPOSITIONS"),
)


# object-model: exception
class W03W04W05RawQuestionGeneralizationError(ValueError):
    """问题构造泛化资料、纵向结果或交叉目录发生漂移。"""


def _sha(value: object) -> str:
    return hashlib.sha256(canonical_json_bytes(value)).hexdigest()


def _vertical_results(
        overlay: W03W04W05VerticalGeneralizationOverlay,
        ) -> tuple[W03W04W05VerticalResult, ...]:
    results = run_w03_w04_w05_vertical_queries(
        overlay.w03_batch,
        overlay.w04_batch,
        overlay.w05_batch,
        tuple(W03W04W05VerticalQuery(
            spec.surface,
            spec.context,
            spec.proposition_surface,
        )
        for spec in VERTICAL_GENERALIZATION_TARGETS
        ),
        overlay_validation_sha256=overlay.validation_sha256,
    )
    if (tuple(item.sha256() for item in results)
            != RAW_QUESTION_GENERALIZATION_VERTICAL_SHA256S
            or any(item.status != "BRIDGED" or item.link is None
                   for item in results)):
        raise W03W04W05RawQuestionGeneralizationError(
            "generalization vertical results drifted")
    return results


def _identity_payload(
        overlay: W03W04W05VerticalGeneralizationOverlay,
        vertical_results: tuple[W03W04W05VerticalResult, ...],
        patterns: tuple[RawQuestionPattern, ...],
        catalog: tuple[RawQuestionConstruction, ...],
        ) -> dict[str, object]:
    return {
        "catalog": [item.to_dict() for item in catalog],
        "expression_boundary": [
            {"capability": key, "status": value}
            for key, value in RAW_QUESTION_GENERALIZATION_EXPRESSION_BOUNDARY
        ],
        "overlay_validation_sha256": overlay.validation_sha256,
        "patterns": [item.to_dict() for item in patterns],
        "vertical_results": [
            item.to_dict() for item in vertical_results
        ],
    }


# object-model: value; representation=struct; interop=pending
@dataclass(frozen=True, slots=True)
class W03W04W05RawQuestionGeneralization:
    """双构造×双内容的四个来源化问题实例。"""

    overlay: W03W04W05VerticalGeneralizationOverlay
    vertical_results: tuple[W03W04W05VerticalResult, ...]
    patterns: tuple[RawQuestionPattern, ...]
    catalog: tuple[RawQuestionConstruction, ...]
    identity_sha256: str

    def __post_init__(self) -> None:
        if (not isinstance(
                self.overlay, W03W04W05VerticalGeneralizationOverlay)
                or len(self.vertical_results) != 2
                or any(not isinstance(item, W03W04W05VerticalResult)
                       for item in self.vertical_results)
                or len(self.patterns) != 2
                or any(not isinstance(item, RawQuestionPattern)
                       for item in self.patterns)
                or len(self.catalog) != 4
                or any(not isinstance(item, RawQuestionConstruction)
                       for item in self.catalog)
                or not isinstance(self.identity_sha256, str)
                or len(self.identity_sha256) != 64):
            raise W03W04W05RawQuestionGeneralizationError(
                "raw question generalization inventory drifted")
        pattern_ids = {item.pattern.sha256() for item in self.catalog}
        source_ids = {item.source_record_key for item in self.catalog}
        pairs = {
            (item.pattern.sha256(), item.source_record_key)
            for item in self.catalog
        }
        if (len(pattern_ids) != 2 or len(source_ids) != 2
                or len(pairs) != 4
                or self.identity_sha256 != self.sha256()):
            raise W03W04W05RawQuestionGeneralizationError(
                "raw question construction/content cross product drifted")

    def to_dict(self) -> dict[str, object]:
        return _identity_payload(
            self.overlay,
            self.vertical_results,
            self.patterns,
            self.catalog,
        )

    def sha256(self) -> str:
        return _sha(self.to_dict())


def build_raw_question_generalization(
        overlay: W03W04W05VerticalGeneralizationOverlay,
        cause_sample_path: str | Path,
        effect_sample_path: str | Path,
        ) -> W03W04W05RawQuestionGeneralization:
    """学习两种段序，并在两个独立来源命题上形成完整交叉目录。"""
    if not isinstance(overlay, W03W04W05VerticalGeneralizationOverlay):
        raise TypeError("raw question generalization overlay is invalid")
    verticals = _vertical_results(overlay)
    patterns = tuple(sorted(
        (
            compile_raw_question_pattern(
                cause_sample_path,
                verticals[0],
                expected_sample_sha256=(
                    RAW_QUESTION_CAUSE_GENERALIZATION_SAMPLE_SHA256),
            ),
            compile_raw_question_pattern(
                effect_sample_path,
                verticals[1],
                expected_sample_sha256=(
                    RAW_QUESTION_EFFECT_GENERALIZATION_SAMPLE_SHA256),
            ),
        ),
        key=lambda item: item.sha256(),
    ))
    catalog = build_raw_question_catalog(patterns, verticals)
    identity = _sha(_identity_payload(
        overlay,
        verticals,
        patterns,
        catalog,
    ))
    value = W03W04W05RawQuestionGeneralization(
        overlay,
        verticals,
        patterns,
        catalog,
        identity,
    )
    if identity != RAW_QUESTION_GENERALIZATION_BUNDLE_SHA256:
        raise W03W04W05RawQuestionGeneralizationError(
            "raw question generalization identity drifted")
    return value


__all__ = [
    "RAW_QUESTION_CAUSE_GENERALIZATION_SAMPLE_SHA256",
    "RAW_QUESTION_EFFECT_GENERALIZATION_SAMPLE_SHA256",
    "RAW_QUESTION_GENERALIZATION_BUNDLE_SHA256",
    "RAW_QUESTION_GENERALIZATION_EXPRESSION_BOUNDARY",
    "RAW_QUESTION_GENERALIZATION_VERTICAL_SHA256S",
    "W03W04W05RawQuestionGeneralization",
    "W03W04W05RawQuestionGeneralizationError",
    "build_raw_question_generalization",
]
