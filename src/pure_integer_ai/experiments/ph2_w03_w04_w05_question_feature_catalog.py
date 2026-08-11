"""显式问题与派生问题特征共用的已学目录边界。"""
from __future__ import annotations

from dataclasses import dataclass
import hashlib

from pure_integer_ai.experiments.ph2_d03_contract_core import (
    canonical_json_bytes,
)
from pure_integer_ai.experiments.ph2_evaluation_public_source import (
    EvaluationPublicBatch,
)
from pure_integer_ai.experiments.ph2_w03_w04_w05_raw_question import (
    run_raw_question_answer,
)
from pure_integer_ai.experiments.ph2_w03_w04_w05_raw_question_contract import (
    RawQuestionAnswerResult,
    RawQuestionConstruction,
    RawQuestionPattern,
    RawQuestionRequest,
)
from pure_integer_ai.experiments.ph2_w03_w04_w05_vertical_contract import (
    W03W04W05VerticalResult,
)


# object-model: exception
class W03W04W05QuestionFeatureCatalogError(ValueError):
    """共享问题目录或公开运行输入发生漂移。"""


def _sha(value: object) -> str:
    return hashlib.sha256(canonical_json_bytes(value)).hexdigest()


def _sha256(value: object, *, where: str) -> str:
    if (not isinstance(value, str) or len(value) != 64
            or any(item not in "0123456789abcdef" for item in value)):
        raise W03W04W05QuestionFeatureCatalogError(
            f"{where} is not a canonical SHA-256")
    return value


# object-model: value; representation=struct; interop=pending
@dataclass(frozen=True, slots=True)
class RawQuestionFeatureCatalog:
    """一个显式构造目录及其完整公开执行输入。"""

    bundle_identity_sha256: str
    overlay_validation_sha256: str
    vertical_results: tuple[W03W04W05VerticalResult, ...]
    patterns: tuple[RawQuestionPattern, ...]
    catalog: tuple[RawQuestionConstruction, ...]
    w03_batch: EvaluationPublicBatch
    w04_batch: EvaluationPublicBatch
    w05_batch: EvaluationPublicBatch

    def __post_init__(self) -> None:
        _sha256(
            self.bundle_identity_sha256,
            where="question feature bundle identity",
        )
        _sha256(
            self.overlay_validation_sha256,
            where="question feature overlay validation",
        )
        if (not isinstance(self.vertical_results, tuple)
                or not self.vertical_results
                or any(not isinstance(item, W03W04W05VerticalResult)
                       for item in self.vertical_results)
                or not isinstance(self.patterns, tuple)
                or not self.patterns
                or any(not isinstance(item, RawQuestionPattern)
                       for item in self.patterns)
                or not isinstance(self.catalog, tuple)
                or not self.catalog
                or any(not isinstance(item, RawQuestionConstruction)
                       for item in self.catalog)
                or any(not isinstance(item, EvaluationPublicBatch) for item in (
                    self.w03_batch, self.w04_batch, self.w05_batch))):
            raise W03W04W05QuestionFeatureCatalogError(
                "question feature inventory is invalid")
        vertical_ids = {item.sha256() for item in self.vertical_results}
        pattern_ids = {item.sha256() for item in self.patterns}
        if (any(item.vertical_result.sha256() not in vertical_ids
                for item in self.catalog)
                or any(item.pattern.sha256() not in pattern_ids
                       for item in self.catalog)):
            raise W03W04W05QuestionFeatureCatalogError(
                "question feature catalog escaped its learned inventory")

    def to_dict(self) -> dict[str, object]:
        return {
            "bundle_identity_sha256": self.bundle_identity_sha256,
            "catalog_sha256s": [item.sha256() for item in self.catalog],
            "overlay_validation_sha256": self.overlay_validation_sha256,
            "pattern_sha256s": [item.sha256() for item in self.patterns],
            "vertical_result_sha256s": [
                item.sha256() for item in self.vertical_results
            ],
            "w03_source_binding_sha256": (
                self.w03_batch.source_binding.sha256()),
            "w04_source_binding_sha256": (
                self.w04_batch.source_binding.sha256()),
            "w05_source_binding_sha256": (
                self.w05_batch.source_binding.sha256()),
        }

    def sha256(self) -> str:
        return _sha(self.to_dict())


def raw_question_feature_catalog(
        value: object,
        *,
        expected_overlay: object | None = None,
    ) -> RawQuestionFeatureCatalog:
    """把两 Role 或三 Role 已学 bundle 适配到同一共享边界。"""
    if isinstance(value, RawQuestionFeatureCatalog):
        if expected_overlay is not None:
            raise W03W04W05QuestionFeatureCatalogError(
                "a materialized feature catalog cannot accept an overlay")
        return value
    overlay = getattr(value, "overlay", None)
    if overlay is None:
        raise TypeError("question feature source lacks a public overlay")
    if expected_overlay is not None and overlay != expected_overlay:
        raise W03W04W05QuestionFeatureCatalogError(
            "question feature bundle escaped its public overlay")
    try:
        return RawQuestionFeatureCatalog(
            value.identity_sha256,
            overlay.validation_sha256,
            value.vertical_results,
            value.patterns,
            value.catalog,
            overlay.w03_batch,
            overlay.w04_batch,
            overlay.w05_batch,
        )
    except AttributeError as exc:
        raise TypeError(
            "question feature source does not expose the shared boundary") from exc


def run_raw_question_feature_answer(
        feature_catalog: RawQuestionFeatureCatalog,
        request: RawQuestionRequest,
        *,
        state_sha256: str | None = None,
    ) -> RawQuestionAnswerResult:
    """通过共享目录分派精确的已学构造匹配。"""
    if (not isinstance(feature_catalog, RawQuestionFeatureCatalog)
            or not isinstance(request, RawQuestionRequest)):
        raise TypeError("question feature dispatch inputs are invalid")
    return run_raw_question_answer(
        feature_catalog.catalog,
        feature_catalog.w03_batch,
        feature_catalog.w04_batch,
        feature_catalog.w05_batch,
        request,
        overlay_validation_sha256=(
            feature_catalog.overlay_validation_sha256),
        state_sha256=state_sha256,
    )


def run_raw_question_feature_candidate_answer(
        feature_catalog: RawQuestionFeatureCatalog,
        request: RawQuestionRequest,
        candidate_constructions: tuple[RawQuestionConstruction, ...],
        *,
        state_sha256: str | None = None,
    ) -> RawQuestionAnswerResult:
    """只在已由上游不可变索引验证的显式构式候选中执行 FT11。"""
    if (not isinstance(feature_catalog, RawQuestionFeatureCatalog)
            or not isinstance(request, RawQuestionRequest)
            or not isinstance(candidate_constructions, tuple)
            or any(not isinstance(item, RawQuestionConstruction)
                   for item in candidate_constructions)):
        raise TypeError("question feature candidate dispatch inputs are invalid")
    if not candidate_constructions:
        return RawQuestionAnswerResult(
            request,
            "UNKNOWN",
            None,
            (),
            None,
            None,
        )
    return run_raw_question_answer(
        candidate_constructions,
        feature_catalog.w03_batch,
        feature_catalog.w04_batch,
        feature_catalog.w05_batch,
        request,
        overlay_validation_sha256=(
            feature_catalog.overlay_validation_sha256),
        state_sha256=state_sha256,
    )


__all__ = [
    "RawQuestionFeatureCatalog",
    "W03W04W05QuestionFeatureCatalogError",
    "raw_question_feature_catalog",
    "run_raw_question_feature_answer",
    "run_raw_question_feature_candidate_answer",
]
